"""FaithfulnessEvaluator: cumulative masking curves (W2 calibration design).

Single-shot masking fails under redundancy (see DESIGN.md section 6.5): with
many correlated features carrying the same signal, removing the top-20 changes
nothing. The remedy is the *cumulative* curve: mask the top-k ranked items
progressively (k = 1..K) and watch the scalar target decay. A faithful
ranking puts model-critical items first, so its curve decays faster.

Metric: normalized area under the decay curve (AUC in [0,1]-ish; LOWER =
more faithful). Also returns the half-life k50 (items to mask before the
target drops to 50%) — smaller is more faithful.
"""

from __future__ import annotations

import numpy as np
import torch

from myerst.adapters.base import ExplanationTarget
from myerst.explainers.occlusion import _target_from_logits


class FaithfulnessEvaluator:
    def __init__(self, adapter, x: torch.Tensor, baseline: torch.Tensor,
                 target: ExplanationTarget, boundary_idx: np.ndarray):
        self.adapter = adapter
        self.x = x.detach()
        self.x0 = baseline.detach()
        self.target = target
        self.bidx = np.asarray(boundary_idx, dtype=np.int64)
        with torch.no_grad():
            self.ref = float(_target_from_logits(
                adapter.forward(self.x), target, self.bidx).item())

    @torch.no_grad()
    def gene_curve(self, gene_scores: np.ndarray, n_steps: int = 25) -> dict:
        """Cumulative top-k GENE masking -> target decay curve."""
        order = np.ascontiguousarray(np.argsort(gene_scores)[::-1])
        n_genes = len(order)
        ks = np.unique(np.linspace(1, n_genes, n_steps).astype(int))
        vals = np.empty(len(ks))
        x_m = self.x.clone()
        prev = 0
        for i, k in enumerate(ks):
            cols = order[prev:k]                      # incrementally mask
            x_m[:, cols] = self.x0[:, cols]
            vals[i] = _target_from_logits(self.adapter.forward(x_m),
                                          self.target, self.bidx).item()
            prev = k
        v_end = float(vals[-1])                       # everything masked
        return self._summarize(ks / n_genes, vals, ks, v_end)

    @torch.no_grad()
    def node_curve(self, node_scores: np.ndarray, players: np.ndarray,
                   n_steps: int = 25) -> dict:
        """Cumulative top-k SPOT masking (players) -> target decay curve."""
        order = np.ascontiguousarray(np.argsort(node_scores)[::-1])
        players = np.asarray(players, dtype=np.int64)
        P = len(order)
        ks = np.unique(np.linspace(1, P, n_steps).astype(int))
        vals = np.empty(len(ks))
        x_m = self.x.clone()
        prev = 0
        for i, k in enumerate(ks):
            rows = players[order[prev:k]]
            x_m[rows] = self.x0[rows]
            vals[i] = _target_from_logits(self.adapter.forward(x_m),
                                          self.target, self.bidx).item()
            prev = k
        v_end = float(vals[-1])
        return self._summarize(ks / P, vals, ks, v_end)

    def _summarize(self, frac: np.ndarray, vals: np.ndarray, ks: np.ndarray,
                   v_end: float) -> dict:
        # normalize by the FULL dynamic range (ref -> all-masked), not by |ref|:
        # dividing by a near-zero ref explodes the scale (E5 lesson).
        scale = abs(self.ref - v_end) + 1e-12
        decay = (self.ref - vals) / scale                  # 0 -> ~1 (ideally)
        decay = np.concatenate([[0.0], decay])
        frac = np.concatenate([[0.0], frac])
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        auc = float(trapezoid(decay, frac))                # higher = more faithful
        thr = self.ref - 0.5 * (self.ref - v_end)
        k50 = int(ks[np.argmax(vals <= thr)]) if np.any(vals <= thr) else -1
        return {"frac": frac, "values": np.concatenate([[self.ref], vals]),
                "decay_auc": auc, "k50": k50, "ref": self.ref}
