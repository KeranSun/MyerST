"""Torch adapter: wraps any (x, adj_norm) -> logits model behind the
HostModelAdapter contract, with scalar targets for attribution."""

from __future__ import annotations

import numpy as np
import torch

from myerst.adapters.base import HostModelAdapter, ExplanationTarget


class TorchModelAdapter(HostModelAdapter):
    """Adapter for dense-graph torch host models.

    boundary_spots : dict mapping (classA, classB) -> indices of spots that
        straddle the A/B interface (a spot in A or B with >=1 neighbor in the
        other class). Precomputed from the spatial graph + labels.
    """

    def __init__(self, model: torch.nn.Module, adj_norm: torch.Tensor,
                 boundary_spots: dict[tuple[int, int], np.ndarray] | None = None):
        self.model = model
        self.adj_norm = adj_norm
        self.boundary_spots = boundary_spots or {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, self.adj_norm)

    def target_output(self, x: torch.Tensor, target: ExplanationTarget) -> torch.Tensor:
        logits = self.forward(x)
        if target.kind == "domain_boundary":
            a, b = target.payload
            spots = self.boundary_spots.get((a, b))
            if spots is None or len(spots) == 0:
                raise ValueError(f"no boundary spots for {(a, b)}")
            idx = torch.from_numpy(np.asarray(spots, dtype=np.int64))
            return (logits[idx, a] - logits[idx, b]).mean()
        if target.kind == "class_score":
            (cls,) = target.payload if isinstance(target.payload, tuple) else (target.payload,)
            return logits[:, cls].mean()
        if target.kind == "class_score_at":
            cls, spots = target.payload
            idx = torch.from_numpy(np.asarray(spots, dtype=np.int64))
            return logits[idx, cls].mean()
        if target.kind == "domain_boundary_margin":
            a, b, signs = target.payload
            spots = self.boundary_spots.get((a, b))
            idx = torch.from_numpy(np.asarray(spots, dtype=np.int64))
            sg = torch.from_numpy(np.asarray(signs, dtype=np.float32))
            p = torch.softmax(logits, dim=-1)
            return ((p[idx, a] - p[idx, b]) * sg).mean()
        raise ValueError(f"unsupported target kind: {target.kind}")

    def parameters_trainable(self) -> bool:
        return True
