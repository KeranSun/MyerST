"""E2 prototype: MyersonExplainer node-level validation on synthetic tissue.

Setup: layered simulation + GCN host. Players = boundary-interface spots
(+ 1-hop neighbors). Ground truth at node level: interface spots (which
directly straddle the domain boundary) should receive higher attribution
than spots that are merely adjacent to the interface.

Arms: MyersonExplainer (flagship) vs IG node-level scores (baseline).
Also reports the efficiency consistency check: sum(phi) ~= v(N).
"""

import time

import numpy as np
import torch

from myerst.benchmark.simulator import LayeredTissueSimulator
from myerst.models.gcn import GCN, build_norm_adj, train_gcn
from myerst.adapters.torch_adapter import TorchModelAdapter
from myerst.adapters.base import ExplanationTarget
from myerst.explainers.ig import IGExplainer
from myerst.explainers.myerson_explainer import MyersonExplainer
from scripts.e1_driver_gene_recovery import auroc, boundary_spots


def main():
    t0 = time.perf_counter()
    print("=" * 66)
    print("E2 prototype: MyersonExplainer node-level validation (synthetic)")
    print("=" * 66)

    sim = LayeredTissueSimulator(grid_size=60, n_layers=3, n_genes=500,
                                 n_driver_per_layer=8, driver_fold=3.0,
                                 dropout_rate=0.3, seed=0)
    res = sim.simulate()
    data = res.data
    data.build_graph(k=6)

    X = np.log1p(data.X)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    x = torch.from_numpy(((X - mu) / sd).astype(np.float32))
    baseline_x = torch.from_numpy(((np.log1p(data.domain_mean()) - mu) / sd).astype(np.float32))

    adj_norm = build_norm_adj(data.edges, data.n_spots)
    gcn = GCN(n_feat=data.n_genes, n_hidden=32, n_classes=3)
    train_gcn(gcn, x, adj_norm, data.labels, epochs=200, seed=0)
    with torch.no_grad():
        acc = (gcn(x, adj_norm).argmax(1) == torch.from_numpy(data.labels.astype(np.int64))).float().mean().item()
    print(f"sim {data.n_spots} spots; GCN acc = {acc:.4f}")

    A, B = 0, 1
    interface = boundary_spots(data.labels, data.adj, A, B)
    padj = data.adj
    neigh = set()
    for i in interface:
        neigh |= padj[i]
    players = np.array(sorted((neigh | set(interface.tolist()))))
    truth = np.isin(players, interface)          # interface = positive
    print(f"players = {len(players)} (interface {truth.sum()} / 1-hop {(~truth).sum()})")

    adapter = TorchModelAdapter(gcn, adj_norm, boundary_spots={(A, B): interface})
    target = ExplanationTarget(kind="domain_boundary", payload=(A, B))

    # --- flagship: Myerson
    t1 = time.perf_counter()
    mye = MyersonExplainer(n_samples=128, perm_batch=32, fwd_chunk=16, seed=0)
    exp_m = mye.explain(adapter, x, target, players=players, edges=data.edges,
                        n_spots=data.n_spots, baseline=baseline_x)
    phi = exp_m.node_scores
    sem = exp_m.meta["sem"]
    t_my = time.perf_counter() - t1
    print(f"Myerson: {t_my:.1f}s, mean SEM = {sem.mean():.4f}")

    # --- baseline: IG node-level
    ig_exp = IGExplainer(n_steps=40).explain(adapter, x, target, baseline=baseline_x)
    node_level = np.abs(ig_exp.meta["node_level"]).sum(axis=1)   # (n_spots,)
    ig_scores = node_level[players]

    au_m = auroc(phi, truth)
    au_ig = auroc(ig_scores, truth)
    print("-" * 66)
    print(f"{'method':<22} {'node AUROC':>10}")
    print(f"{'Myerson (flagship)':<22} {au_m:>10.3f}")
    print(f"{'IG node-level (baseline)':<22} {au_ig:>10.3f}")
    print(f"{'random':<22} {0.5:>10.3f}")

    # --- efficiency consistency: sum(phi) ~ v(N) with all players active
    from myerst.explainers.occlusion import _target_from_logits
    with torch.no_grad():
        v_full = _target_from_logits(adapter.forward(x), target, interface).item()
        x_masked = x.clone()
        x_masked[players] = baseline_x[players]
        v_empty = _target_from_logits(adapter.forward(x_masked), target, interface).item()
    print("-" * 66)
    print(f"efficiency check: sum(phi) = {phi.sum():.4f} vs v(N)-v(0) = {v_full - v_empty:.4f}")
    print(f"total time {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
