"""Local CCC synergy: per-receiver communication games (E9 lesson).

First-order pairwise synergy computed against an EMPTY background carries no
signal (a ligand-receiver pair only matters in the presence of its
neighborhood). The fix is a LOCAL game per receiver j:

    players = {j} ∪ sender-neighbors of j
    v(C)    = target class logit at j when players outside C (within this
              local neighborhood) are masked to baseline
    psi_i   = v({i, j}) - v({i}) - v({j}) + v(empty)

psi_i measures the cooperative gain of sender i's ligand with receiver j's
receptor — exactly the LR cooperativity that communication edges should have.
All coalitions needed are singletons/pairs, so evaluation is exact (no MC)
and batched.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def local_ccc_synergy(model, x: torch.Tensor, adj_norm: torch.Tensor,
                      receivers: np.ndarray, sender_neighbors: dict[int, list[int]],
                      baseline: torch.Tensor, target_cls: int,
                      local_nodes: dict[int, np.ndarray],
                      chunk: int = 64) -> dict:
    """Pairwise LR synergy for each (receiver, sender-neighbor) pair.

    model(x, adj_norm) -> logits (n, C). local_nodes[j] = nodes to mask for
    receiver j's local game (typically {j} ∪ 2-hop neighborhood).
    Evaluated in chunks of `chunk` variants (4 per pair) to bound memory.
    Returns dict with per-pair psi and the components.
    """
    x = x.detach()
    x0 = baseline.detach()
    pairs: list[tuple[int, int]] = []       # (receiver j, sender i)
    for j in receivers:
        for i in sender_neighbors.get(int(j), []):
            pairs.append((int(j), i))
    if not pairs:
        return {"pairs": np.empty((0, 2), dtype=np.int64), "psi": np.empty(0)}

    # four masked variants per pair: {i,j}, {i}, {j}, empty
    variants = []
    for j, i in pairs:
        variants += [(j, (i, j)), (j, (i,)), (j, (j,)), (j, ())]

    tgt = np.empty(len(variants))
    for s in range(0, len(variants), chunk):
        part = variants[s:s + chunk]
        xb = x.unsqueeze(0).repeat(len(part), 1, 1)
        js = np.empty(len(part), dtype=np.int64)
        for k, (j, active) in enumerate(part):
            loc = np.asarray(local_nodes[j], dtype=np.int64)
            keep = np.isin(loc, np.asarray(active, dtype=np.int64))
            xb[k, loc[~keep]] = x0[loc[~keep]]
            js[k] = j
        logits = model(xb, adj_norm)                      # (B, n, C)
        tgt[s:s + len(part)] = logits[torch.arange(len(part)),
                                      torch.from_numpy(js), target_cls].numpy()

    psi = np.empty(len(pairs))
    for pi in range(len(pairs)):
        v_ij, v_i, v_j, v_0 = tgt[4 * pi: 4 * pi + 4]
        psi[pi] = v_ij - v_i - v_j + v_0
    return {"pairs": np.asarray(pairs, dtype=np.int64), "psi": psi,
            "v_ij": tgt[0::4], "v_i": tgt[1::4], "v_j": tgt[2::4], "v_0": tgt[3::4]}
