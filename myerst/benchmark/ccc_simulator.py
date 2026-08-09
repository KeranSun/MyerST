"""CCCSimulator v0: cooperative ligand-receptor signaling with ground truth.

Geometry: sender cells (type 0) left, receiver cells (type 1) right, with an
interleaved interface band in the middle (tumor-immune-like boundary).

Signal model (COOPERATIVE — the key property):
    activation_p(i) = mean(L_p over sender neighbors of i) * R_p(i)
Downstream targets of pair p are upregulated in receiver i proportional to
activation_p(i). A downstream response requires BOTH the ligand (from a
spatial neighbor) AND the receptor (in the cell itself) — this is the
cooperative target semantics that edge-attribution methods should recover
(and where the E5 lesson applies: cross-type edges should SYNERGIZE here).

Ground truth returned: LR pairs, downstream targets, communicating edges
(sender-receiver edges where signaling actually flows), and the per-spot
activation matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from myerst.data.spadata import SpaData
from myerst.data.graph import build_knn_graph, adjacency_list


@dataclass
class CCCResult:
    data: SpaData
    lr_pairs: list[tuple[int, int]]            # (ligand_gene, receptor_gene)
    targets: dict[int, np.ndarray]             # pair idx -> downstream gene idx
    comm_edges: np.ndarray                     # (E_c, 2) true signaling edges
    activation: np.ndarray                     # (n_spots, n_pairs) activation strength
    sender_mask: np.ndarray
    lig_on: np.ndarray | None = None           # (n, n_pairs) sender expresses ligand p
    competent: np.ndarray | None = None        # (n, n_pairs) receiver competent for p


class CCCSimulator:
    def __init__(
        self,
        grid_size: int = 40,
        n_genes: int = 300,
        n_pairs: int = 4,
        n_target_per_pair: int = 4,
        interface_frac: float = 0.25,
        signal_strength: float = 4.0,
        base_mean: float = 0.5,
        nb_dispersion: float = 2.0,
        dropout_rate: float = 0.25,
        seed: int | None = 0,
    ) -> None:
        self.p = dict(grid_size=grid_size, n_genes=n_genes, n_pairs=n_pairs,
                      n_target_per_pair=n_target_per_pair,
                      interface_frac=interface_frac,
                      signal_strength=signal_strength, base_mean=base_mean,
                      nb_dispersion=nb_dispersion, dropout_rate=dropout_rate)
        self.rng = np.random.default_rng(seed)

    def simulate(self) -> CCCResult:
        p = self.p
        g, G, P = p["grid_size"], p["n_genes"], p["n_pairs"]
        T = p["n_target_per_pair"]
        rng = self.rng

        # --- geometry: senders left, receivers right, mixed interface band
        xs, ys = np.meshgrid(np.arange(g), np.arange(g))
        coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
        n = g * g
        x_rel = coords[:, 0] / g
        band = p["interface_frac"]
        # interface zone: |x_rel - 0.5| < band/2 -> random type; else by side
        in_interface = np.abs(x_rel - 0.5) < band / 2
        labels = np.where(in_interface, rng.integers(0, 2, n),
                          (x_rel > 0.5).astype(int))   # 0 sender (left), 1 receiver
        sender_mask = labels == 0

        # --- gene allocation: [markers, L1..LP, R1..RP, T_11.., rest noise]
        idx = 0
        markers = {0: np.arange(idx, idx + 4), 1: np.arange(idx + 4, idx + 8)}
        idx += 8
        lig = np.arange(idx, idx + P); idx += P
        rec = np.arange(idx, idx + P); idx += P
        targets = {pi: np.arange(idx + pi * T, idx + (pi + 1) * T) for pi in range(P)}
        idx += P * T
        assert idx < G

        # --- spatial graph (radius ~1 hop on grid)
        edges = build_knn_graph(coords, k=6)
        adj = adjacency_list(edges, n)

        # --- mean structure (np.ix_ — chained fancy indexing would silently copy)
        s_idx = np.where(sender_mask)[0]
        r_idx = np.where(~sender_mask)[0]
        mean = np.full((n, G), p["base_mean"])
        mean[np.ix_(s_idx, markers[0])] *= 5
        mean[np.ix_(r_idx, markers[1])] *= 5
        mean[np.ix_(r_idx, rec)] *= 4            # receptors in receivers
        # ligand heterogeneity: each sender expresses a random SUBSET (~50%)
        # of ligands — senders are NOT exchangeable, so individual sender
        # edges carry unique information (E9b lesson: uniform senders make
        # single-edge marginals vanish under neighbor averaging).
        lig_on = rng.random((n, P)) < 0.5
        lig_on[~sender_mask] = False
        for pi in range(P):
            mean[lig_on[:, pi], lig[pi]] *= 4

        # --- cooperative activation: receiver targets ON only with L(neighbor)*R(self),
        # inside the pair's signaling zone, AND if the receiver is signaling-
        # competent for that pair (~50%). Competence creates QUIET interface
        # receivers with sender neighbors — so "cross-type edge" and
        # "communicating edge" are no longer synonymous.
        rows = coords[:, 1]
        competent = (rng.random((n, P)) < 0.5) & (~sender_mask)[:, None]
        activation = np.zeros((n, P))
        for i in range(n):
            if sender_mask[i]:
                continue
            send_nb = [j for j in adj[i] if sender_mask[j]]
            if not send_nb:
                continue
            for pi in range(P):
                in_zone = (rows[i] < g / 2) if pi % 2 == 0 else (rows[i] >= g / 2)
                has_lig = any(lig_on[j, pi] for j in send_nb)
                if in_zone and competent[i, pi] and has_lig:
                    activation[i, pi] = 4.0 * 4.0 / 4.0  # mean L level * normalized
        # scale: L mean ~ 4*base, R mean ~ 4*base; activation ~ signal_strength * base
        activation *= p["signal_strength"] * p["base_mean"] / 4.0
        for pi in range(P):
            act_rows = activation[:, pi] > 0
            mean[np.ix_(act_rows, targets[pi])] += activation[act_rows, pi, None]

        # --- sample counts: NB + dropout
        r = p["nb_dispersion"]
        rate = rng.gamma(shape=r, scale=mean / r)
        X = rng.poisson(rate).astype(np.float32)
        drop = rng.random(X.shape) < p["dropout_rate"]
        X[drop] = 0.0

        # --- communicating edges: sender-receiver edges where receiver activates
        comm = []
        for u, v in edges:
            if sender_mask[u] == sender_mask[v]:
                continue
            recv = v if sender_mask[u] else u
            if activation[recv].sum() > 0:
                comm.append((u, v))
        comm_edges = np.array(comm, dtype=np.int64)

        data = SpaData(X=X, coords=coords, labels=labels,
                       gene_names=[f"G{i:03d}" for i in range(G)])
        data.edges = np.asarray(edges, dtype=np.int64)   # graph already built above
        names = data.gene_names
        for c, mks in markers.items():
            for j, gi in enumerate(mks):
                names[gi] = f"Mk{'S' if c == 0 else 'R'}{j+1}"
        for pi in range(P):
            names[lig[pi]] = f"LIG{pi+1}"
            names[rec[pi]] = f"REC{pi+1}"
            for j, gi in enumerate(targets[pi]):
                names[gi] = f"TGT{pi+1}_{j+1}"
        lr_pairs = [(int(lig[pi]), int(rec[pi])) for pi in range(P)]
        return CCCResult(data=data, lr_pairs=lr_pairs, targets=targets,
                         comm_edges=comm_edges, activation=activation,
                         sender_mask=sender_mask, lig_on=lig_on,
                         competent=competent)
