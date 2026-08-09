"""E1 calibration sweep: find the simulation regime where attribution methods
actually differentiate (a benchmark where everything scores 1.0 is useless).

Grid over driver_fold x dropout_rate. For each config, report GCN accuracy,
AUROC per method, and the zero-vs-domain-mean IG gap (baseline-design evidence).
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


def run_config(driver_fold, dropout_rate, seed=0):
    sim = LayeredTissueSimulator(grid_size=60, n_layers=3, n_genes=500,
                                 n_driver_per_layer=8, driver_fold=driver_fold,
                                 dropout_rate=dropout_rate, seed=seed)
    res = sim.simulate()
    data = res.data
    data.build_graph(k=6)

    X = np.log1p(data.X)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    x = torch.from_numpy(((X - mu) / sd).astype(np.float32))
    baseline_x = torch.from_numpy(((np.log1p(data.domain_mean()) - mu) / sd).astype(np.float32))

    adj_norm = build_norm_adj(data.edges, data.n_spots)
    gcn = GCN(n_feat=data.n_genes, n_hidden=32, n_classes=3)
    train_gcn(gcn, x, adj_norm, data.labels, epochs=200, seed=seed)
    with torch.no_grad():
        acc = (gcn(x, adj_norm).argmax(1) == torch.from_numpy(data.labels.astype(np.int64))).float().mean().item()

    A, B = 0, 1
    bspots = boundary_spots(data.labels, data.adj, A, B)
    adapter = TorchModelAdapter(gcn, adj_norm, boundary_spots={(A, B): bspots})
    target = ExplanationTarget(kind="domain_boundary", payload=(A, B))
    truth = np.zeros(data.n_genes, dtype=bool)
    truth[res.boundary_genes[(A, B)]] = True

    ig_dm = IGExplainer(50).explain(adapter, x, target, baseline=baseline_x).node_scores
    ig_zero = IGExplainer(50).explain(adapter, x, target, baseline=None).node_scores
    occ = SpatialOcclusion().explain(adapter, x, target, baseline=baseline_x).node_scores
    de = np.abs(data.X[data.labels == A].mean(0) - data.X[data.labels == B].mean(0))

    return {
        "acc": acc,
        "IG_dm": auroc(ig_dm, truth),
        "IG_zero": auroc(ig_zero, truth),
        "Occ": auroc(occ, truth),
        "DE": auroc(de, truth),
        "gap_dm_zero": auroc(ig_dm, truth) - auroc(ig_zero, truth),
    }


def main():
    print(f"{'fold':>5} {'drop':>5} | {'GCNacc':>7} {'IG_dm':>6} {'IG_zero':>8} {'Occ':>6} {'DE':>6} {'dm-zero':>8}")
    print("-" * 64)
    for fold in [1.5, 2.0, 3.0]:
        for drop in [0.3, 0.5]:
            r = run_config(fold, drop)
            print(f"{fold:>5.1f} {drop:>5.1f} | {r['acc']:>7.3f} {r['IG_dm']:>6.3f} "
                  f"{r['IG_zero']:>8.3f} {r['Occ']:>6.3f} {r['DE']:>6.3f} {r['gap_dm_zero']:>+8.3f}")


if __name__ == "__main__":
    main()
