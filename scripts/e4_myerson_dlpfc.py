"""E4: MyersonExplainer on the real DLPFC L5|L6 boundary (GCN host).

Players = interface spots + 1-hop neighbors. Reports the efficiency identity,
MC standard errors, node-level AUROC (interface vs 1-hop), and the top
attributed spots; saves per-spot phi for downstream plotting/analysis.
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
from myerst.explainers.myerson_explainer import MyersonExplainer
from myerst.explainers.occlusion import _target_from_logits
from scripts.e1_driver_gene_recovery import auroc, boundary_spots
from scripts.tune_stagate_seurat_hvg import load_seurat_hvg, ORDER

BOUNDARY = ("L5", "L6")


def main():
    t0 = time.perf_counter()
    print("=" * 68)
    print("E4: MyersonExplainer on DLPFC 151673 L5|L6 (GCN host)")
    print("=" * 68)

    x_t, coords, labels = load_seurat_hvg()
    data = SpaData(X=x_t.numpy(), coords=coords, labels=labels)
    data.build_graph(k=6)
    A, B = ORDER.index(BOUNDARY[0]), ORDER.index(BOUNDARY[1])
    interface = boundary_spots(data.labels, data.adj, A, B)
    neigh = set()
    for i in interface:
        neigh |= data.adj[i]
    players = np.array(sorted(neigh | set(interface.tolist())))
    truth = np.isin(players, interface)
    print(f"players = {len(players)} (interface {truth.sum()} / 1-hop {(~truth).sum()})")

    x = x_t
    baseline_x = torch.from_numpy(data.domain_mean().astype(np.float32))
    adj_norm = build_norm_adj(data.edges, data.n_spots)
    gcn = GCN(n_feat=data.n_genes, n_hidden=64, n_classes=7)
    train_gcn(gcn, x, adj_norm, data.labels, epochs=200, seed=0)
    with torch.no_grad():
        acc = (gcn(x, adj_norm).argmax(1) == torch.from_numpy(data.labels.astype(np.int64))).float().mean().item()
    print(f"GCN host acc = {acc:.4f}")

    adapter = TorchModelAdapter(gcn, adj_norm, boundary_spots={(A, B): interface})
    target = ExplanationTarget(kind="domain_boundary", payload=(A, B))

    t1 = time.perf_counter()
    mye = MyersonExplainer(n_samples=256, perm_batch=32, fwd_chunk=16, seed=0)
    exp = mye.explain(adapter, x, target, players=players, edges=data.edges,
                      n_spots=data.n_spots, baseline=baseline_x)
    phi, sem = exp.node_scores, exp.meta["sem"]
    t_my = time.perf_counter() - t1
    print(f"Myerson: {t_my/60:.1f} min, mean SEM = {sem.mean():.4f}, "
          f"|phi| mean = {np.abs(phi).mean():.4f}")

    # efficiency identity
    with torch.no_grad():
        v_full = _target_from_logits(adapter.forward(x), target, interface).item()
        x_m = x.clone()
        x_m[players] = baseline_x[players]
        v_empty = _target_from_logits(adapter.forward(x_m), target, interface).item()
    print(f"efficiency: sum(phi) = {phi.sum():.4f} vs v_g(N)-v(0) = {v_full - v_empty:.4f}"
          f" (equal iff player graph connected)")

    # baseline comparison: IG node-level on the same players
    ig_exp = IGExplainer(n_steps=40).explain(adapter, x, target, baseline=baseline_x)
    ig_node = np.abs(ig_exp.meta["node_level"]).sum(axis=1)[players]

    print("-" * 68)
    print(f"{'method':<24} {'node AUROC':>10}")
    print(f"{'Myerson (flagship)':<24} {auroc(phi, truth):>10.3f}")
    print(f"{'IG node-level':<24} {auroc(ig_node, truth):>10.3f}")
    print(f"{'random':<24} {0.5:>10.3f}")

    top = np.argsort(phi)[::-1][:10]
    print("-" * 68)
    print("top-10 spots by Myerson phi (node, layer, phi, SEM, is_interface):")
    for t in top:
        node = players[t]
        print(f"  spot {node:5d}  {ORDER[data.labels[node]]:>3}  "
              f"phi {phi[t]:+.4f}  sem {sem[t]:.4f}  {truth[t]}")

    os.makedirs("data/processed", exist_ok=True)
    np.savez("data/processed/e4_myerson_dlpfc_L5L6.npz",
             players=players, phi=phi, sem=sem, truth=truth,
             labels=data.labels, coords=data.coords)
    print(f"saved: data/processed/e4_myerson_dlpfc_L5L6.npz")
    print(f"total time {(time.perf_counter() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
