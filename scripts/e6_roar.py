"""E6: ROAR (remove-and-retrain) faithfulness — the non-circular protocol.

Gene-level: IG vs Occlusion vs naive DE vs random rankings; drop top-k genes,
retrain GCN, measure boundary probability margin + boundary accuracy.
Node-level: Myerson phi vs IG node scores vs random (rankings loaded from the
E5 cache); remove top-k spots, retrain, measure the same.

Faithful ranking => performance collapses FAST => low retention AUC.
3 host seeds, mean+-sd. This is the publishable version of the E5 question.
"""

import os
import time

import numpy as np
import torch

from myerst.data.spadata import SpaData
from myerst.models.gcn import GCN, build_norm_adj, train_gcn
from myerst.data.graph import build_knn_graph
from myerst.adapters.torch_adapter import TorchModelAdapter
from myerst.adapters.base import ExplanationTarget
from myerst.explainers.ig import IGExplainer
from myerst.explainers.occlusion import SpatialOcclusion
from myerst.benchmark.roar import ROAREvaluator
from scripts.e1_driver_gene_recovery import boundary_spots
from scripts.tune_stagate_seurat_hvg import load_seurat_hvg, ORDER

BOUNDARY = ("L5", "L6")
KS_GENE = [0, 10, 50, 150, 400, 1000]
KS_NODE = [0, 10, 30, 60, 120, 240]
SEEDS = (0, 1, 2)
EPOCHS = 120


def train_fn(X, coords, labels, seed):
    """GCN host factory for ROAR: rebuild graph, train, return logits + eval mask."""
    edges = build_knn_graph(coords, k=6)
    adj_norm = build_norm_adj(edges, X.shape[0])
    x = torch.from_numpy(np.ascontiguousarray(X).astype(np.float32))
    model = GCN(n_feat=X.shape[1], n_hidden=64, n_classes=7)
    model, train_mask = train_gcn(model, x, adj_norm, labels, epochs=EPOCHS,
                                  seed=seed, return_mask=True)
    model.eval()
    with torch.no_grad():
        logits = model(x, adj_norm)
    return logits, ~train_mask          # held-out spots only


def main():
    t0 = time.perf_counter()
    print("=" * 72)
    print("E6: ROAR faithfulness (gene-level + node-level), DLPFC 151673 L5|L6")
    print("=" * 72)

    x_t, coords, labels = load_seurat_hvg()
    X = x_t.numpy()
    data = SpaData(X=X, coords=coords, labels=labels)
    data.build_graph(k=6)
    A, B = ORDER.index(BOUNDARY[0]), ORDER.index(BOUNDARY[1])
    interface = boundary_spots(data.labels, data.adj, A, B)

    # ---- reference host for generating rankings
    print("training reference host for rankings...", flush=True)
    adj_norm = build_norm_adj(data.edges, data.n_spots)
    ref = GCN(n_feat=data.n_genes, n_hidden=64, n_classes=7)
    train_gcn(ref, x_t, adj_norm, data.labels, epochs=200, seed=0)
    adapter = TorchModelAdapter(ref, adj_norm, boundary_spots={(A, B): interface})
    target = ExplanationTarget(kind="domain_boundary", payload=(A, B))
    baseline_x = torch.from_numpy(data.domain_mean().astype(np.float32))
    ig_gene = IGExplainer(40).explain(adapter, x_t, target, baseline=baseline_x).node_scores
    occ_gene = SpatialOcclusion(batch_size=8).explain(adapter, x_t, target, baseline=baseline_x).node_scores
    de_gene = np.abs(X[labels == A].mean(0) - X[labels == B].mean(0))
    rng = np.random.default_rng(0)
    print("rankings ready", flush=True)

    roar = ROAREvaluator(train_fn, ks=KS_GENE, seeds=SEEDS)

    # ---- gene-level ROAR
    print("-" * 72)
    print("GENE-level ROAR (acc retention AUC; lower = more faithful)")
    gene_rankings = {"IG": ig_gene, "Occlusion": occ_gene,
                     "Naive DE": de_gene, "random": rng.random(data.n_genes)}
    res_g = roar.gene_roar(X, coords, labels, interface, A, B, gene_rankings)
    for name, r in res_g.items():
        print(f"  {name:<12} acc_auc={r['acc_auc']:.3f}  "
              f"acc@k: {np.round(r['acc_mean'], 3)}")

    # ---- node-level ROAR (rankings from E5 cache)
    print("-" * 72)
    print("NODE-level ROAR (acc retention AUC; lower = more faithful)")
    cache = np.load("data/processed/e5_fidelity_edges.npz")
    players = cache["players"]
    node_rankings = {"Myerson phi": cache["phi"], "IG node": cache["ig_node"],
                     "random": rng.random(len(players))}
    roar.ks = KS_NODE
    res_n = roar.node_roar(X, coords, labels, interface, A, B, players, node_rankings)
    for name, r in res_n.items():
        print(f"  {name:<12} acc_auc={r['acc_auc']:.3f}  "
              f"acc@k: {np.round(r['acc_mean'], 3)}")

    np.savez("data/processed/e6_roar.npz",
             gene={k: v for k, v in res_g.items()}, node={k: v for k, v in res_n.items()},
             ks_gene=KS_GENE, ks_node=KS_NODE)
    print(f"total time {(time.perf_counter() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
