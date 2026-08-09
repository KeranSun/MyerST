"""E5: the fidelity experiments — where Myerson is supposed to beat ranking.

Three parts on DLPFC 151673 L5|L6 (GCN host):
  A. node-level cumulative masking: Myerson phi vs IG node scores vs random.
     If Myerson identifies the spots the MODEL actually relies on, its decay
     curve is steeper even though its ranking AUROC is lower.
  B. gene-level cumulative masking: IG vs Occlusion vs naive DE vs random.
  C. edge synergy: pairwise Myerson synergy on player-graph edges, validated
     against "cross-layer edge" (L5-L6 links) ground truth.
"""

import os
import time

import numpy as np
import torch

from myerst.data.spadata import SpaData
from myerst.models.gcn import GCN, build_norm_adj, train_gcn
from myerst.adapters.torch_adapter import TorchModelAdapter
from myerst.adapters.base import ExplanationTarget
from myerst.explainers.ig import IGExplainer
from myerst.explainers.occlusion import SpatialOcclusion
from myerst.explainers.myerson_explainer import MyersonExplainer
from myerst.benchmark.faithfulness import FaithfulnessEvaluator
from scripts.e1_driver_gene_recovery import auroc, boundary_spots
from scripts.tune_stagate_seurat_hvg import load_seurat_hvg, ORDER

BOUNDARY = ("L5", "L6")


def main():
    t0 = time.perf_counter()
    print("=" * 70)
    print("E5: fidelity curves + edge synergy (DLPFC 151673 L5|L6, GCN host)")
    print("=" * 70)

    x_t, coords, labels = load_seurat_hvg()
    data = SpaData(X=x_t.numpy(), coords=coords, labels=labels)
    data.build_graph(k=6)
    A, B = ORDER.index(BOUNDARY[0]), ORDER.index(BOUNDARY[1])
    interface = boundary_spots(data.labels, data.adj, A, B)
    neigh = set()
    for i in interface:
        neigh |= data.adj[i]
    players = np.array(sorted(neigh | set(interface.tolist())))
    truth_node = np.isin(players, interface)

    x = x_t
    baseline_x = torch.from_numpy(data.domain_mean().astype(np.float32))
    adj_norm = build_norm_adj(data.edges, data.n_spots)
    gcn = GCN(n_feat=data.n_genes, n_hidden=64, n_classes=7)
    train_gcn(gcn, x, adj_norm, data.labels, epochs=200, seed=0)
    adapter = TorchModelAdapter(gcn, adj_norm, boundary_spots={(A, B): interface})
    target = ExplanationTarget(kind="domain_boundary", payload=(A, B))
    ev = FaithfulnessEvaluator(adapter, x, baseline_x, target, interface)
    print(f"players={len(players)}, ref target = {ev.ref:.4f}", flush=True)

    # ---- Myerson (with cache for edge synergy; result cached to disk)
    import pickle
    cache_file = "data/processed/e5_myerson_cache.pkl"
    os.makedirs("data/processed", exist_ok=True)
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            exp = pickle.load(f)
        phi = exp.node_scores
        print(f"Myerson loaded from cache ({len(exp.meta['v_cache'])} coalitions)", flush=True)
    else:
        t1 = time.perf_counter()
        mye = MyersonExplainer(n_samples=128, perm_batch=32, fwd_chunk=16,
                               seed=0, return_cache=True)
        exp = mye.explain(adapter, x, target, players=players, edges=data.edges,
                          n_spots=data.n_spots, baseline=baseline_x)
        phi = exp.node_scores
        print(f"Myerson done ({(time.perf_counter()-t1)/60:.1f} min), "
              f"cache {len(exp.meta['v_cache'])} coalitions", flush=True)
        with open(cache_file, "wb") as f:
            pickle.dump(exp, f)

    # ---- IG / Occlusion gene-level + IG node-level
    ig_exp = IGExplainer(n_steps=40).explain(adapter, x, target, baseline=baseline_x)
    ig_gene = ig_exp.node_scores
    ig_node = np.abs(ig_exp.meta["node_level"]).sum(axis=1)[players]
    occ_gene = SpatialOcclusion(batch_size=8).explain(adapter, x, target, baseline=baseline_x).node_scores
    de_gene = np.abs(data.X[data.labels == A].mean(0) - data.X[data.labels == B].mean(0))
    print("IG/Occ/DE done", flush=True)

    # ---- A. node-level fidelity
    print("-" * 70)
    print("A. node-level cumulative masking (decay AUC, higher = more faithful)")
    rng = np.random.default_rng(0)
    rand_node = rng.random(len(players))
    for name, sc in [("Myerson phi", phi), ("IG node-level", ig_node),
                     ("random", rand_node)]:
        r = ev.node_curve(sc, players)
        print(f"  {name:<18} AUC={r['decay_auc']:.3f}  k50={r['k50']}")

    # ---- B. gene-level fidelity
    print("B. gene-level cumulative masking")
    rand_gene = rng.random(data.n_genes)
    for name, sc in [("IG", ig_gene), ("Occlusion", occ_gene),
                     ("Naive DE", de_gene), ("random", rand_gene)]:
        r = ev.gene_curve(sc)
        print(f"  {name:<18} AUC={r['decay_auc']:.3f}  k50={r['k50']}")

    # ---- C. edge synergy vs cross-layer ground truth
    print("C. edge synergy (pairwise Myerson interaction)")
    syn_helper = MyersonExplainer()  # only used for the edge_synergy method
    pedges, syn = syn_helper.edge_synergy(adapter, x, target, players, data.edges,
                                          exp.meta["v_cache"], interface,
                                          baseline=baseline_x)
    cross = labels[pedges[:, 0]] != labels[pedges[:, 1]]
    print(f"  player edges: {len(pedges)} (cross-layer {cross.sum()})")
    print(f"  synergy AUROC vs cross-layer: {auroc(syn, cross):.3f} (random 0.5)")
    top_e = np.argsort(syn)[::-1][:8]
    print("  top-8 synergy edges (u[layer] - v[layer], psi):")
    for e in top_e:
        u, v = pedges[e]
        print(f"    {u}({ORDER[labels[u]]}) -- {v}({ORDER[labels[v]]})  psi={syn[e]:+.4f}")

    os.makedirs("data/processed", exist_ok=True)
    np.savez("data/processed/e5_fidelity_edges.npz", players=players, phi=phi,
             ig_node=ig_node, pedges=pedges, syn=syn, cross=cross)
    print(f"total time {(time.perf_counter() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
