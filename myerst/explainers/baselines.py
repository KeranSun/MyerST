"""Baseline explainers: GraphLIME (perturb-and-fit) and GNNExplainerST
(mask learning). Both produce gene-level scores against the scalar target,
comparable to IG / SpatialOcclusion in the benchmark matrix.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from myerst.explainers.base import Explainer, Explanation
from myerst.explainers.occlusion import _target_from_logits


class GraphLIME(Explainer):
    """LIME-style local surrogate: perturb genes, fit Lasso on target change.

    Samples M noisy versions of the input (Gaussian noise on all genes),
    records the scalar target change, and fits a sparse linear model
    delta_target ~ noise. Coefficients are per-gene importances.
    """

    def __init__(self, n_samples: int = 512, noise_scale: float = 0.5,
                 alpha: float = 0.01, seed: int = 0):
        self.n_samples = n_samples
        self.noise_scale = noise_scale
        self.alpha = alpha
        self.seed = seed

    @torch.no_grad()
    def explain(self, adapter, x: torch.Tensor, target, baseline=None,
                boundary_idx=None) -> Explanation:
        from sklearn.linear_model import Lasso
        if boundary_idx is None:
            boundary_idx = np.arange(x.shape[0])
        x = x.detach()
        rng = np.random.default_rng(self.seed)
        ref = _target_from_logits(adapter.forward(x), target, boundary_idx).item()
        noise = torch.from_numpy(
            rng.normal(0, self.noise_scale, size=(self.n_samples, *x.shape[1:]))
            .astype(np.float32))
        deltas = np.empty(self.n_samples)
        B = 32
        for s in range(0, self.n_samples, B):
            xb = x.unsqueeze(0) + noise[s:s + B].unsqueeze(1)
            tgt = _target_from_logits(adapter.forward(xb), target, boundary_idx)
            deltas[s:s + B] = (tgt - ref).numpy()
        X_fit = noise.numpy()
        model = Lasso(alpha=self.alpha)
        model.fit(X_fit, deltas)
        return Explanation(node_scores=np.abs(model.coef_),
                           edge_scores=None,
                           meta={"method": "graphlime", "n_samples": self.n_samples})


class GNNExplainerST(Explainer):
    """GNNExplainer-style feature mask learning (spatial task adaptation).

    Learns a per-gene mask m in [0,1]^G applied as x * sigmoid(m) so that the
    masked input preserves the scalar target, with L1 sparsity pressure.
    sigmoid(m) after optimization = per-gene importance.
    """

    def __init__(self, epochs: int = 300, lr: float = 0.05, l1: float = 0.01,
                 seed: int = 0):
        self.epochs = epochs
        self.lr = lr
        self.l1 = l1
        self.seed = seed

    def explain(self, adapter, x: torch.Tensor, target, baseline=None,
                boundary_idx=None) -> Explanation:
        torch.manual_seed(self.seed)
        if boundary_idx is None:
            boundary_idx = np.arange(x.shape[0])
        x = x.detach()
        for p in adapter.model.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            ref = _target_from_logits(adapter.forward(x), target, boundary_idx).item()
        m = torch.zeros(x.shape[1], requires_grad=True)
        opt = torch.optim.Adam([m], lr=self.lr)
        for _ in range(self.epochs):
            opt.zero_grad()
            mask = torch.sigmoid(m)
            tgt = _target_from_logits(adapter.forward(x * mask), target,
                                      boundary_idx).item()
            loss = (tgt - ref) ** 2 + self.l1 * mask.sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            scores = torch.sigmoid(m).numpy()
        for p in adapter.model.parameters():
            p.requires_grad_(True)
        return Explanation(node_scores=scores, edge_scores=None,
                           meta={"method": "gnnexplainer_st",
                                 "epochs": self.epochs})
