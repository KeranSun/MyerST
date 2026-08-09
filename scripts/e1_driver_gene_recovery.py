"""E1 prototype: driver-gene recovery on simulated layered tissue.

Pipeline: simulate -> log1p/z-score -> kNN graph -> train GCN host ->
explain boundary discrimination with IG (domain-mean vs zero baseline) and
SpatialOcclusion -> AUROC against ground-truth driver genes.

The zero-baseline IG arm exists to demonstrate the dropout-baseline failure
mode (paper R2 evidence). A naive differential-expression baseline is added
for context.
"""

import time

import numpy as np
import torch

from myerst.benchmark.simulator import LayeredTissueSimulator
from myerst.models.gcn import GCN, build_norm_adj, train_gcn
from myerst.adapters.torch_adapter import TorchModelAdapter
from myerst.adapters.base import ExplanationTarget
from myerst.explainers.ig import IGExplainer
from myerst.explainers.occlusion import SpatialOcclusion


def auroc(scores: np.ndarray, truth: np.ndarray) -> float:
    """Rank-based AUROC (truth: boolean array)."""
    order = np.argsort(np.argsort(scores))
    ranks = order.astype(float) + 1
    pos = ranks[truth]
    n_pos, n_neg = truth.sum(), (~truth).sum()
    return float((pos.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def boundary_spots(labels: np.ndarray, adj: list[set[int]], a: int, b: int) -> np.ndarray:
    out = []
    for i in range(len(labels)):
        if labels[i] in (a, b) and any(labels[j] in (a, b) and labels[j] != labels[i] for j in adj[i]):
            out.append(i)
    return np.array(out, dtype=np.int64)


def main():
    t0 = time.perf_counter()
    print("=" * 66)
    print("E1 prototype: driver-gene recovery on simulated layered tissue")
    print("=" * 66)

    # --- 1. simulate
    sim = LayeredTissueSimulator(grid_size=60, n_layers=3, n_genes=200,
                                 n_driver_per_layer=10, dropout_rate=0.25, seed=0)
    res = sim.simulate()
    data = res.data
    data.build_graph(k=6)
    print(f"simulated: {data.n_spots} spots x {data.n_genes} genes, "
          f"{len(data.edges)} graph edges, layers={np.unique(data.labels).tolist()}")

    # --- 2. preprocess
    X = np.log1p(data.X)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    x = torch.from_numpy(X.astype(np.float32))
    baseline_x = torch.from_numpy(
        ((np.log1p(data.domain_mean()) - X.mean(0)) / (X.std(0) + 1e-6)).astype(np.float32))

    # --- 3. host model
    adj_norm = build_norm_adj(data.edges, data.n_spots)
    gcn = GCN(n_feat=data.n_genes, n_hidden=32, n_classes=3)
    train_gcn(gcn, x, adj_norm, data.labels, epochs=200, seed=0)
    with torch.no_grad():
        acc = (gcn(x, adj_norm).argmax(1) == torch.from_numpy(data.labels.astype(np.int64))).float().mean().item()
    print(f"GCN host trained, full-data accuracy = {acc:.4f}")

    # --- 4. boundary (0, 1) explanation
    A, B = 0, 1
    bspots = boundary_spots(data.labels, data.adj, A, B)
    print(f"boundary spots ({A}|{B}): {len(bspots)}")
    adapter = TorchModelAdapter(gcn, adj_norm, boundary_spots={(A, B): bspots})
    target = ExplanationTarget(kind="domain_boundary", payload=(A, B))
    truth = np.zeros(data.n_genes, dtype=bool)
    truth[res.boundary_genes[(A, B)]] = True
    print(f"ground-truth driver genes: {truth.sum()}")

    # --- 5. attribution arms
    arms = {}
    arms["IG (domain-mean baseline)"] = IGExplainer(n_steps=50).explain(
        adapter, x, target, baseline=baseline_x).node_scores
    arms["IG (zero baseline)"] = IGExplainer(n_steps=50).explain(
        adapter, x, target, baseline=None).node_scores
    arms["SpatialOcclusion"] = SpatialOcclusion().explain(
        adapter, x, target, baseline=baseline_x).node_scores
    # naive DE context baseline on raw counts
    de = np.abs(data.X[data.labels == A].mean(0) - data.X[data.labels == B].mean(0))
    arms["Naive DE (context)"] = de

    # --- 6. evaluation
    print("-" * 66)
    print(f"{'method':<30} {'AUROC':>8} {'top-20 precision':>16}")
    for name, scores in arms.items():
        au = auroc(scores, truth)
        top20 = np.argsort(scores)[::-1][:20]
        prec = truth[top20].mean()
        print(f"{name:<30} {au:>8.3f} {prec:>16.2f}")
    print(f"{'random expectation':<30} {0.5:>8.3f} {truth.mean():>16.2f}")
    print("-" * 66)
    print(f"total time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
