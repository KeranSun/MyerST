"""Fast local CCC synergy via receptive-field subgraph forwards.

For a 2-layer GCN, the output at receiver j depends ONLY on j's 2-hop
neighborhood and the edges within it. So the local game can be evaluated on
the small 2-hop subgraph (~30-80 nodes) instead of the full tissue graph —
exactly equivalent, orders of magnitude faster (E10: 66k full-graph forwards
estimated at hours -> seconds).

Per receiver j, all (sender, variant) combinations are evaluated in ONE
batched forward on the subgraph: variants = for each sender i: {i,j}, {i},
plus shared {j}, empty.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def local_ccc_synergy_fast(model, x: torch.Tensor, adj_norm: torch.Tensor,
                           receivers: np.ndarray,
                           sender_neighbors: dict[int, list[int]],
                           baseline: torch.Tensor, target_cls: int,
                           local_nodes: dict[int, np.ndarray],
                           pos_in_zone: dict[int, int] | None = None) -> dict:
    """Same semantics as local_ccc_synergy, evaluated on 2-hop subgraphs."""
    x = x.detach()
    x0 = baseline.detach()
    all_pairs: list[tuple[int, int]] = []
    all_psi: list[float] = []

    for j in receivers:
        j = int(j)
        senders = sender_neighbors.get(j, [])
        if not senders:
            continue
        zone = np.asarray(local_nodes[j], dtype=np.int64)
        zidx = {int(n): k for k, n in enumerate(zone)}
        xz = x[zone]                                   # (Z, F)
        x0z = x0[zone]
        adj_sub = adj_norm[zone][:, zone]              # (Z, Z) exact for 2-layer GCN

        # variants: for each sender i: {i,j}, {i}; plus {j}, {}
        n_v = 2 * len(senders) + 2
        xb = xz.unsqueeze(0).repeat(n_v, 1, 1)
        xb[:, :, :] = x0z                              # start fully masked
        for k, i in enumerate(senders):
            xb[2 * k, zidx[j]] = xz[zidx[j]]           # {i, j}
            xb[2 * k, zidx[i]] = xz[zidx[i]]
            xb[2 * k + 1, zidx[i]] = xz[zidx[i]]       # {i} only
        xb[-2, zidx[j]] = xz[zidx[j]]                  # {j} only
        # xb[-1] stays fully masked = empty
        logits = model(xb, adj_sub)                    # (n_v, Z, C)
        tgt = logits[:, zidx[j], target_cls].numpy()

        v_j, v_0 = tgt[-2], tgt[-1]
        for k, i in enumerate(senders):
            v_ij, v_i = tgt[2 * k], tgt[2 * k + 1]
            all_pairs.append((j, int(i)))
            all_psi.append(float(v_ij - v_i - v_j + v_0))

    return {"pairs": np.asarray(all_pairs, dtype=np.int64).reshape(-1, 2),
            "psi": np.asarray(all_psi)}
