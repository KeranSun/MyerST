"""ROAR: RemOve And Retrain faithfulness evaluation (Hooker et al., 2019).

The non-circular alternative to masking-based fidelity: physically REMOVE the
top-k ranked items from the data (drop gene columns / drop spots and their
graph edges), RETRAIN the host model from scratch, and measure performance
degradation. Retraining is a third-party operator that no explainer used
internally, so no explainer is advantaged by operator matching.

Performance metric: probability margin at boundary spots
    mean( softmax_p(layerA) - softmax_p(layerA) ) ... see prob_margin target
(bounded in [-1, 1], stable across retrains — unlike raw logit differences).

Reported: retention curve (metric vs k, mean+-sd over host seeds) and its
AUC. LOWER retention AUC = more faithful ranking (the model collapses sooner).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def prob_margin(logits: torch.Tensor, boundary_idx: np.ndarray,
                a: int, b: int) -> float:
    """mean( p_a - p_b ) over boundary spots; bounded, scale-stable."""
    p = F.softmax(logits, dim=-1)
    idx = torch.from_numpy(np.asarray(boundary_idx, dtype=np.int64))
    return float((p[idx, a] - p[idx, b]).mean().item())


def prob_margin_signed(logits: torch.Tensor, labels: np.ndarray,
                       boundary_idx: np.ndarray, a: int, b: int) -> float:
    """Class-signed margin: mean over boundary spots of p_true - p_other.

    The unsigned margin is ZERO BY CONSTRUCTION on a mixed two-class boundary
    set (layerA spots give +1, layerB spots give -1 and they cancel). The
    signed version measures each spot's confidence toward its own side.
    """
    p = F.softmax(logits, dim=-1).numpy()
    idx = np.asarray(boundary_idx, dtype=np.int64)
    lab = np.asarray(labels)[idx]
    own = np.where(lab == a, p[idx, a] - p[idx, b], p[idx, b] - p[idx, a])
    return float(own.mean())


def boundary_accuracy(logits: torch.Tensor, labels: np.ndarray,
                      boundary_idx: np.ndarray) -> float:
    idx = np.asarray(boundary_idx, dtype=np.int64)
    pred = logits.argmax(dim=-1).numpy()[idx]
    return float((pred == labels[idx]).mean())


class ROAREvaluator:
    """remove-and-retrain evaluation over ranking methods.

    train_fn(X, coords, labels, seed) -> callable producing logits for the
    given (possibly reduced) data. The caller supplies everything data-
    dependent; ROAR stays host-agnostic.
    """

    def __init__(self, train_fn, ks, seeds=(0, 1, 2)):
        self.train_fn = train_fn
        self.ks = list(ks)
        self.seeds = list(seeds)

    def gene_roar(self, X, coords, labels, boundary_idx, a, b,
                  rankings: dict[str, np.ndarray]) -> dict:
        """Drop top-k gene columns per ranking, retrain, measure margin+acc.

        train_fn must return (logits, eval_mask); metrics are computed ONLY on
        held-out eval spots — in-sample accuracy is meaningless because an
        overparameterized host memorizes residual features (E6/E7 lesson).
        """
        out = {}
        for name, scores in rankings.items():
            order = np.argsort(scores)[::-1]
            curves_margin, curves_acc = [], []
            for seed in self.seeds:
                ms, acs = [], []
                for k in self.ks:
                    keep = np.ones(X.shape[1], dtype=bool)
                    if k > 0:
                        keep[order[:k]] = False
                    logits, eval_mask = self.train_fn(X[:, keep], coords, labels, seed)
                    bidx = np.array([i for i in boundary_idx if eval_mask[i]],
                                    dtype=np.int64)
                    if len(bidx) < 5:
                        ms.append(np.nan); acs.append(np.nan); continue
                    ms.append(prob_margin_signed(logits, labels, bidx, a, b))
                    acs.append(boundary_accuracy(logits, labels, bidx))
                curves_margin.append(ms)
                curves_acc.append(acs)
            out[name] = self._pack(curves_margin, curves_acc)
        return out

    def node_roar(self, X, coords, labels, boundary_idx, a, b, players,
                  rankings: dict[str, np.ndarray],
                  eval_by_labels: bool = True) -> dict:
        """Remove top-k spots (with their edges), retrain, measure on held-out rest.

        eval_by_labels=True evaluates on ALL held-out spots with labels in
        {a, b} (excluding removed ones) instead of the shrinking boundary set —
        avoids the evaluation-set contamination where one method's top-k IS
        the evaluation set (E6 lesson).
        """
        players = np.asarray(players, dtype=np.int64)
        out = {}
        for name, scores in rankings.items():
            order = np.argsort(scores)[::-1]
            curves_margin, curves_acc = [], []
            for seed in self.seeds:
                ms, acs = [], []
                for k in self.ks:
                    drop = set(players[order[:k]].tolist()) if k > 0 else set()
                    keep_mask = np.array([i not in drop for i in range(X.shape[0])])
                    keep_old = np.where(keep_mask)[0]
                    Xr = X[keep_mask]
                    coords_r = coords[keep_mask]
                    labels_r = labels[keep_mask]
                    remap = {old: new for new, old in enumerate(keep_old)}
                    logits, eval_mask = self.train_fn(Xr, coords_r, labels_r, seed)
                    if eval_by_labels:
                        bidx_r = np.array([i for i in range(len(labels_r))
                                           if labels_r[i] in (a, b) and eval_mask[i]],
                                          dtype=np.int64)
                    else:
                        bidx_r = np.array([remap[i] for i in boundary_idx
                                           if i in remap and eval_mask[remap[i]]],
                                          dtype=np.int64)
                    if len(bidx_r) < 5:
                        ms.append(np.nan); acs.append(np.nan); continue
                    ms.append(prob_margin_signed(logits, labels_r, bidx_r, a, b))
                    acs.append(boundary_accuracy(logits, labels_r, bidx_r))
                curves_margin.append(ms)
                curves_acc.append(acs)
            out[name] = self._pack(curves_margin, curves_acc)
        return out

    def _pack(self, curves_margin, curves_acc):
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        M = np.array(curves_margin, dtype=float)     # (seeds, ks)
        Ac = np.array(curves_acc, dtype=float)
        x = np.linspace(0, 1, M.shape[1])
        return {
            "margin_mean": M.mean(0), "margin_sd": M.std(0),
            "acc_mean": Ac.mean(0), "acc_sd": Ac.std(0),
            "acc_auc": float(trapezoid(np.nanmean(Ac, 0), x)),
            "margin_auc": float(trapezoid(np.nanmean(M, 0), x)),
        }
