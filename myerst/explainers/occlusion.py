"""Spatial occlusion attribution: gene-wise masking against a domain-mean baseline.

For each gene g, replace its column with the per-domain mean and measure the
change in the scalar target. Attribution = target(x) - target(x_occluded):
positive means the gene *supports* the target prediction.

Occluded inputs are evaluated in batches (batch_size genes per forward) —
naive per-gene loops are prohibitively slow for attention hosts on real
Visium-scale data (3000 genes x ~2s/forward).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from myerst.adapters.base import ExplanationTarget
from myerst.explainers.base import Explainer, Explanation


def _target_from_logits(logits: torch.Tensor, target: ExplanationTarget,
                        boundary_idx: np.ndarray) -> torch.Tensor:
    """Per-batch-element scalar targets from (B, n, C) or (n, C) logits."""
    if target.kind == "domain_boundary":
        a, b = target.payload
        idx = torch.from_numpy(np.asarray(boundary_idx, dtype=np.int64))
        diff = logits[..., idx, a] - logits[..., idx, b]
        return diff.mean(dim=-1)
    if target.kind == "class_score":
        cls = target.payload[0] if isinstance(target.payload, tuple) else target.payload
        return logits[..., cls].mean(dim=-1)
    if target.kind == "class_score_at":
        cls, spots = target.payload
        idx = torch.from_numpy(np.asarray(spots, dtype=np.int64))
        return logits[..., idx, cls].mean(dim=-1)
    if target.kind == "domain_boundary_margin":
        # class-signed probability margin: mean over boundary spots of
        # sign * (p_a - p_b), sign=+1 for true-a spots, -1 for true-b.
        # Bounded in [-1,1], always positive for a good host — unlike the raw
        # logit difference, whose scale/sign is ill-conditioned (E5/E8 lesson).
        a, b, signs = target.payload
        idx = torch.from_numpy(np.asarray(boundary_idx, dtype=np.int64))
        sg = torch.from_numpy(np.asarray(signs, dtype=np.float32))
        p = F.softmax(logits, dim=-1)
        diff = (p[..., idx, a] - p[..., idx, b]) * sg
        return diff.mean(dim=-1)
    raise ValueError(f"unsupported target kind: {target.kind}")


class SpatialOcclusion(Explainer):
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size

    def explain(self, adapter, x: torch.Tensor, target: ExplanationTarget,
                baseline: torch.Tensor | None = None) -> Explanation:
        x = x.detach()
        x0 = baseline.detach() if baseline is not None else torch.zeros_like(x)
        bidx = self._boundary_idx(adapter, target)

        with torch.no_grad():
            ref_logits = adapter.forward(x)
            ref = _target_from_logits(ref_logits, target, bidx).item()

            n_genes = x.shape[1]
            scores = np.empty(n_genes)
            for s in range(0, n_genes, self.batch_size):
                genes = np.arange(s, min(s + self.batch_size, n_genes))
                xb = x.unsqueeze(0).repeat(len(genes), 1, 1)          # (B, n, F)
                rows = np.arange(len(genes))
                xb[rows, :, genes] = x0[:, genes].T                   # row i: only gene g_i occluded
                logits_b = adapter.forward(xb)                        # (B, n, C)
                tgt = _target_from_logits(logits_b, target, bidx)     # (B,)
                scores[genes] = ref - tgt.numpy()

        return Explanation(
            node_scores=np.abs(scores),
            edge_scores=None,
            meta={"method": "spatial_occlusion", "signed_scores": scores,
                  "batch_size": self.batch_size,
                  "baseline": "domain_mean" if baseline is not None else "zero"},
        )

    @staticmethod
    def _boundary_idx(adapter, target) -> np.ndarray:
        if target.kind == "domain_boundary":
            spots = getattr(adapter, "boundary_spots", {}).get(tuple(target.payload))
            if spots is None or len(spots) == 0:
                raise ValueError(f"no boundary spots for {target.payload}")
            return np.asarray(spots, dtype=np.int64)
        return np.empty(0, dtype=np.int64)
