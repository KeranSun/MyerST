"""E1b: ranking recovery vs fidelity under correlated passenger genes.

Setup: each driver gene gets noisy "passenger" copies that correlate with the
layer label but have no direct effect. Univariate DE cannot separate drivers
from passengers; we show that (a) AUROC rankings degrade for everyone, and
(b) fidelity@k (accuracy drop after masking top-k genes) is the metric that
actually separates model-faithful attribution from naive DE.

This is the core argument of paper section R2: ranking metrics are
insufficient, fidelity is necessary.
"""

import numpy as np
import torch

from myerst.benchmark.simulator import LayeredTissueSimulator
from myerst.models.gcn import GCN, build_norm_adj, train_gcn
from myerst.adapters.torch_adapter import TorchModelAdapter
from myerst.adapters.base import ExplanationTarget
from myerst.explainers.ig import IGExplainer
from myerst.explainers.occlusion import SpatialOcclusion
from scripts.e1_driver_gene_recovery import auroc, boundary_spots


def main():
    sim = LayeredTissueSimulator(grid_size=60, n_layers=3, n_genes=500,
                                 n_driver_per_layer=8, driver_fold=3.0,
                                 dropout_rate=0.3, n_passenger_per_driver=2,
                                 passenger_noise=0.3, seed=0)
    res = sim.simulate()
    data = res.data
    data.build_graph(k=6)
    n_drivers = sum(len(v) for v in res.driver_genes.values())
    n_pass = sum(len(v) for v in (res.passenger_genes or {}).values())
    print(f"spots={data.n_spots} genes={data.n_genes} drivers={n_drivers} passengers={n_pass}")

    X = np.log1p(data.X)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    x = torch.from_numpy(((X - mu) / sd).astype(np.float32))
    baseline_np = (np.log1p(data.domain_mean()) - mu) / sd
    baseline_x = torch.from_numpy(baseline_np.astype(np.float32))

    adj_norm = build_norm_adj(data.edges, data.n_spots)
    gcn = GCN(n_feat=data.n_genes, n_hidden=32, n_classes=3)
    train_gcn(gcn, x, adj_norm, data.labels, epochs=200, seed=0)
    y = torch.from_numpy(data.labels.astype(np.int64))
    with torch.no_grad():
        acc_full = (gcn(x, adj_norm).argmax(1) == y).float().mean().item()
    print(f"GCN accuracy (all genes) = {acc_full:.4f}")

    A, B = 0, 1
    bspots = boundary_spots(data.labels, data.adj, A, B)
    adapter = TorchModelAdapter(gcn, adj_norm, boundary_spots={(A, B): bspots})
    target = ExplanationTarget(kind="domain_boundary", payload=(A, B))
    truth = np.zeros(data.n_genes, dtype=bool)
    truth[res.boundary_genes[(A, B)]] = True
    pass_mask = np.zeros(data.n_genes, dtype=bool)
    for ell in (A, B):
        pass_mask[res.passenger_genes[ell]] = True

    arms = {
        "IG (domain-mean)": IGExplainer(50).explain(adapter, x, target, baseline=baseline_x).node_scores,
        "IG (zero)": IGExplainer(50).explain(adapter, x, target, baseline=None).node_scores,
        "SpatialOcclusion": SpatialOcclusion().explain(adapter, x, target, baseline=baseline_x).node_scores,
        "Naive DE": np.abs(data.X[data.labels == A].mean(0) - data.X[data.labels == B].mean(0)),
    }

    K = 20
    print("-" * 74)
    print(f"{'method':<20} {'AUROC':>7} {'top20 driver':>12} {'top20 passgr':>12} {'fidelity@20 (dAcc)':>18}")
    rng = np.random.default_rng(0)
    rand_drop = []
    for _ in range(5):
        rk = rng.choice(data.n_genes, K, replace=False)
        rand_drop.append(mask_and_accuracy(gcn, x, adj_norm, y, rk, baseline_np))
    print(f"{'random control':<20} {0.5:>7.3f} {'-':>12} {'-':>12} {np.mean(rand_drop):>18.4f}")
    for name, scores in arms.items():
        au = auroc(scores, truth)
        topk = np.argsort(scores)[::-1][:K]
        d_acc = mask_and_accuracy(gcn, x, adj_norm, y, topk, baseline_np)
        print(f"{name:<20} {au:>7.3f} {truth[topk].mean():>12.2f} {pass_mask[topk].mean():>12.2f} {d_acc:>18.4f}")
    print("-" * 74)
    print("fidelity@20 (dAcc) = accuracy drop after masking the method's top-20 genes;")
    print("a faithful ranking collapses the model, an unfaithful one leaves it standing.")


def mask_and_accuracy(model, x, adj_norm, y, gene_idx, baseline_np) -> float:
    gene_idx = np.ascontiguousarray(gene_idx)
    x_m = x.clone()
    x_m[:, gene_idx] = torch.from_numpy(np.ascontiguousarray(baseline_np[:, gene_idx]).astype(np.float32))
    with torch.no_grad():
        acc_m = (model(x_m, adj_norm).argmax(1) == y).float().mean().item()
    acc_full = (model(x, adj_norm).argmax(1) == y).float().mean().item()
    return acc_full - acc_m


if __name__ == "__main__":
    main()
