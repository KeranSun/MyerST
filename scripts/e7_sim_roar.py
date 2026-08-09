"""E7: ROAR on redundancy-controlled simulations — where methods SHOULD differ.

E6 showed gene-level ROAR saturates on real DLPFC (3000 redundant HVGs: even
random removal of 1000 genes doesn't hurt). The benchmark therefore needs
redundancy-controlled regimes:

  Sim SPARSE   : 6 drivers/layer, no passengers -> model relies on few genes
  Sim REDUNDANT: +2 passenger copies/driver     -> E1b-style redundancy

Expectation: in SPARSE, removing IG/Occ top genes collapses boundary accuracy
while random removal does not; in REDUNDANT, all rankings degrade toward
random — demonstrating the benchmark's dynamic range and why faithfulness
evaluation must control redundancy.
"""

import time

import numpy as np
import torch

from myerst.benchmark.simulator import LayeredTissueSimulator
from myerst.models.gcn import GCN, build_norm_adj, train_gcn
from myerst.data.graph import build_knn_graph
from myerst.adapters.torch_adapter import TorchModelAdapter
from myerst.adapters.base import ExplanationTarget
from myerst.explainers.ig import IGExplainer
from myerst.explainers.occlusion import SpatialOcclusion
from myerst.benchmark.roar import ROAREvaluator
from scripts.e1_driver_gene_recovery import boundary_spots

SEEDS = (0, 1)
EPOCHS = 100
KS = [0, 4, 8, 16, 32, 64, 128]


def train_fn(X, coords, labels, seed):
    edges = build_knn_graph(coords, k=6)
    adj_norm = build_norm_adj(edges, X.shape[0])
    x = torch.from_numpy(np.ascontiguousarray(X).astype(np.float32))
    model = GCN(n_feat=X.shape[1], n_hidden=32, n_classes=3)
    model, train_mask = train_gcn(model, x, adj_norm, labels, epochs=EPOCHS,
                                  seed=seed, return_mask=True)
    model.eval()
    with torch.no_grad():
        logits = model(x, adj_norm)
    return logits, ~train_mask          # evaluate on held-out spots only


def run_sim(name, sim_kw):
    print(f"--- {name} ---", flush=True)
    sim = LayeredTissueSimulator(grid_size=40, n_layers=3, n_genes=300, seed=0, **sim_kw)
    res = sim.simulate()
    data = res.data
    data.build_graph(k=6)

    X = np.log1p(data.X)
    X = ((X - X.mean(0)) / (X.std(0) + 1e-6)).astype(np.float32)
    x = torch.from_numpy(X)
    baseline_x = torch.from_numpy(
        ((np.log1p(data.domain_mean()) - X.mean(0)) / (X.std(0) + 1e-6)).astype(np.float32))

    adj_norm = build_norm_adj(data.edges, data.n_spots)
    ref = GCN(n_feat=data.n_genes, n_hidden=32, n_classes=3)
    train_gcn(ref, x, adj_norm, data.labels, epochs=150, seed=0)
    A, B = 0, 1
    interface = boundary_spots(data.labels, data.adj, A, B)
    adapter = TorchModelAdapter(ref, adj_norm, boundary_spots={(A, B): interface})
    target = ExplanationTarget(kind="domain_boundary", payload=(A, B))

    ig = IGExplainer(40).explain(adapter, x, target, baseline=baseline_x).node_scores
    occ = SpatialOcclusion(batch_size=16).explain(adapter, x, target, baseline=baseline_x).node_scores
    de = np.abs(data.X[data.labels == A].mean(0) - data.X[data.labels == B].mean(0))
    rng = np.random.default_rng(0)
    rankings = {"IG": ig, "Occlusion": occ, "Naive DE": de, "random": rng.random(data.n_genes)}

    roar = ROAREvaluator(train_fn, ks=KS, seeds=SEEDS)
    res_roar = roar.gene_roar(X, data.coords, data.labels, interface, A, B, rankings)
    for m, r in res_roar.items():
        print(f"  {m:<12} margin_auc={r['margin_auc']:.3f}  "
              f"margin@k: {np.round(r['margin_mean'], 3)}", flush=True)
        print(f"  {'':<12} acc___auc={r['acc_auc']:.3f}  "
              f"acc___@k: {np.round(r['acc_mean'], 3)}", flush=True)


def main():
    t0 = time.perf_counter()
    print("=" * 72)
    print("E7: ROAR on redundancy-controlled simulations")
    print("=" * 72)
    run_sim("SPARSE (6 drivers/layer, no passengers)",
            dict(n_driver_per_layer=6, driver_fold=3.0, dropout_rate=0.3,
                 n_passenger_per_driver=0))
    run_sim("REDUNDANT (+2 passengers/driver)",
            dict(n_driver_per_layer=6, driver_fold=3.0, dropout_rate=0.3,
                 n_passenger_per_driver=2, passenger_noise=0.3))
    print(f"total time {(time.perf_counter() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
