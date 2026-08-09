"""MyersonExplainer: topology-constrained game-theoretic node attribution.

Players are spots (typically the boundary band + 1-hop neighbors). The
characteristic function v(C) is the host model's scalar target when players
outside coalition C are masked to the domain-mean baseline (context nodes
outside the player set are always kept). The Myerson value restricts
cooperation to spatially connected coalitions:

    v_g(S) = sum over connected components C of S (in the player subgraph) v(C)

Estimated by Monte Carlo permutation sampling, batched across permutations:
B permutations advance in lock-step, and all component evaluations of a step
are fused into one batched forward pass.
"""

from __future__ import annotations

import numpy as np
import torch

from myerst.adapters.base import ExplanationTarget
from myerst.data.graph import adjacency_list, connected_components
from myerst.explainers.base import Explainer, Explanation
from myerst.explainers.occlusion import _target_from_logits


class MyersonExplainer(Explainer):
    def __init__(self, n_samples: int = 256, perm_batch: int = 64,
                 fwd_chunk: int = 16, seed: int = 0, return_cache: bool = False):
        self.n_samples = n_samples
        self.perm_batch = perm_batch
        self.fwd_chunk = fwd_chunk
        self.seed = seed
        self.return_cache = return_cache

    @torch.no_grad()
    def explain(self, adapter, x: torch.Tensor, target: ExplanationTarget,
                players: np.ndarray, edges: np.ndarray, n_spots: int,
                baseline: torch.Tensor | None = None,
                boundary_idx: np.ndarray | None = None) -> Explanation:
        x = x.detach()
        x0 = baseline.detach() if baseline is not None else torch.zeros_like(x)
        players = np.asarray(players, dtype=np.int64)
        P = len(players)
        p_index = {int(p): i for i, p in enumerate(players)}          # node -> player slot
        pset = set(p_index)
        padj_full = adjacency_list(edges, n_spots)
        padj = [(nb & pset) if i in p_index else set() for i, nb in enumerate(padj_full)]

        if boundary_idx is None:
            boundary_idx = self._boundary_idx(adapter, target)
        rng = np.random.default_rng(self.seed)
        # v(empty) = model value with ALL players masked — the correct Shapley
        # reference so that efficiency holds: sum(phi) = v(N) - v(empty).
        x_all_masked = x.clone()
        x_all_masked[players] = x0[players]
        v_empty = float(_target_from_logits(adapter.forward(x_all_masked),
                                            target, boundary_idx).item())
        v_cache: dict[frozenset[int], float] = {frozenset(): v_empty}

        accum = np.zeros(P)
        accum_sq = np.zeros(P)
        n_done = 0
        while n_done < self.n_samples:
            B = min(self.perm_batch, self.n_samples - n_done)
            perms = np.stack([rng.permutation(P) for _ in range(B)])   # player slots
            prefixes: list[set[int]] = [set() for _ in range(B)]       # node ids
            prev_vg = np.full(B, v_empty)
            for t in range(P):
                coalitions: list[list[int]] = []
                owner: list[int] = []
                keys: list[frozenset[int]] = []
                for b in range(B):
                    node = int(players[perms[b, t]])
                    prefixes[b].add(node)
                    for comp in connected_components(prefixes[b], padj):
                        keys.append(comp)
                        owner.append(b)
                todo = [k for k in keys if k not in v_cache]
                if todo:
                    vals = self._v_batch(adapter, x, x0, players, p_index,
                                         [sorted(k) for k in todo], target, boundary_idx)
                    for k, val in zip(todo, vals):
                        v_cache[k] = float(val)
                vg = np.zeros(B)
                for k, b in zip(keys, owner):
                    vg[b] += v_cache[k]
                for b in range(B):
                    slot = perms[b, t]
                    contrib = vg[b] - prev_vg[b]
                    accum[slot] += contrib
                    accum_sq[slot] += contrib * contrib
                prev_vg = vg
            n_done += B

        phi = accum / n_done
        var = accum_sq / n_done - phi ** 2
        sem = np.sqrt(np.maximum(var, 0) / n_done)
        meta = {"method": "myerson_mc", "players": players,
                "n_samples": n_done, "sem": sem,
                "player_set_size": P, "v_empty": v_empty}
        if self.return_cache:
            meta["v_cache"] = v_cache
        return Explanation(node_scores=phi, edge_scores=None, meta=meta)

    @torch.no_grad()
    def edge_synergy(self, adapter, x: torch.Tensor, target: ExplanationTarget,
                     players: np.ndarray, edges: np.ndarray,
                     v_cache: dict, boundary_idx: np.ndarray,
                     baseline: torch.Tensor | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Pairwise edge synergy (first-order Shapley interaction on graph edges):

            psi_ij = v({i,j}) - v({i}) - v({j}) + v(empty)

        This is the first-order term of the Myerson fairness difference
        phi_i(g) - phi_i(g - ij); missing coalitions are evaluated directly.
        Returns (edges_in_player_graph, synergy_scores).
        """
        x = x.detach()
        x0 = baseline.detach() if baseline is not None else torch.zeros_like(x)
        players = np.asarray(players, dtype=np.int64)
        p_index = {int(p): i for i, p in enumerate(players)}
        pset = set(p_index)
        pedges = np.array([(u, v) for u, v in np.asarray(edges)
                           if int(u) in pset and int(v) in pset], dtype=np.int64)

        missing: list[frozenset[int]] = []
        for u, v in pedges:
            for c in (frozenset([int(u)]), frozenset([int(v)]), frozenset([int(u), int(v)])):
                if c not in v_cache:
                    missing.append(c)
        missing = list(set(missing))
        if missing:
            vals = self._v_batch(adapter, x, x0, players, p_index,
                                 [sorted(c) for c in missing], target, boundary_idx)
            for c, val in zip(missing, vals):
                v_cache[c] = float(val)

        v_empty = v_cache[frozenset()]
        syn = np.array([
            v_cache[frozenset([int(u), int(v)])] - v_cache[frozenset([int(u)])]
            - v_cache[frozenset([int(v)])] + v_empty
            for u, v in pedges
        ])
        return pedges, syn

    def _v_batch(self, adapter, x, x0, players, p_index, coalitions, target, boundary_idx):
        """v(C) for a list of coalitions: mask players not in C to baseline."""
        out = np.empty(len(coalitions))
        for s in range(0, len(coalitions), self.fwd_chunk):
            chunk = coalitions[s:s + self.fwd_chunk]
            B = len(chunk)
            xb = x.unsqueeze(0).repeat(B, 1, 1)
            for i, comp in enumerate(chunk):
                active_slots = np.zeros(len(players), dtype=bool)
                for c in comp:
                    active_slots[p_index[c]] = True
                mask_nodes = players[~active_slots]
                xb[i, mask_nodes] = x0[mask_nodes]
            logits = adapter.forward(xb)
            tgt = _target_from_logits(logits, target, boundary_idx)
            out[s:s + B] = tgt.numpy()
        return out

    @staticmethod
    def _boundary_idx(adapter, target) -> np.ndarray:
        if target.kind == "domain_boundary":
            spots = getattr(adapter, "boundary_spots", {}).get(tuple(target.payload))
            if spots is None or len(spots) == 0:
                raise ValueError(f"no boundary spots for {target.payload}")
            return np.asarray(spots, dtype=np.int64)
        return np.empty(0, dtype=np.int64)
