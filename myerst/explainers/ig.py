"""Integrated Gradients with a spatially coherent baseline.

Naive IG uses a zero baseline, which is a *false* reference under single-cell
dropout: a zero count often means "not detected", not "not expressed". We
integrate from the per-domain mean expression instead (SpaData.domain_mean),
so attribution measures deviation from the spot's own tissue context.
"""

from __future__ import annotations

import numpy as np
import torch

from myerst.adapters.base import ExplanationTarget
from myerst.explainers.base import Explainer, Explanation


class IGExplainer(Explainer):
    def __init__(self, n_steps: int = 50):
        self.n_steps = n_steps

    def explain(self, adapter, x: torch.Tensor, target: ExplanationTarget,
                baseline: torch.Tensor | None = None,
                aggregate: str = "sum_abs") -> Explanation:
        x = x.detach()
        x0 = baseline.detach() if baseline is not None else torch.zeros_like(x)

        total_grad = torch.zeros_like(x)
        for k in range(1, self.n_steps + 1):
            xk = (x0 + (k / self.n_steps) * (x - x0)).requires_grad_(True)
            out = adapter.target_output(xk, target)
            grad = torch.autograd.grad(out, xk)[0]
            total_grad += grad
        attr = (x - x0) * total_grad / self.n_steps        # (n_spots, n_genes)

        node_scores = attr.detach().numpy()
        if aggregate == "sum_abs":
            gene_scores = np.abs(node_scores).sum(axis=0)
        else:
            gene_scores = node_scores.sum(axis=0)

        return Explanation(
            node_scores=gene_scores,
            edge_scores=None,
            meta={"method": "integrated_gradients", "n_steps": self.n_steps,
                  "baseline": "domain_mean" if baseline is not None else "zero",
                  "node_level": node_scores},
        )
