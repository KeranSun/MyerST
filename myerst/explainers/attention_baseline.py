"""Attention-weight baseline explainer — the falsification arm.

The classic "attention as explanation" heuristic: a node's importance is the
attention mass it receives from the spots whose prediction we explain (here:
boundary spots). Node score_i = sum over boundary spots j of attn[j -> i],
averaged over heads. This is the baseline our benchmark tests (and, per prior
work, attention weights are expected to fail faithfulness checks).
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def attention_node_scores(model, x: torch.Tensor, adj_bin: torch.Tensor,
                          boundary_idx: np.ndarray) -> np.ndarray:
    """attention received by each node FROM boundary spots (mean over heads)."""
    model.eval()
    _, attn = model(x, adj_bin, return_attention=True)   # (n_i, n_j) mean over heads
    if attn.dim() == 3:                                   # (n, n, H) -> mean heads
        attn = attn.mean(-1)
    idx = np.asarray(boundary_idx, dtype=np.int64)
    received = attn[idx].sum(dim=0)                       # column sum over boundary rows
    return received.numpy()
