"""E9: edge synergy on a COOPERATIVE target — the right validation ground.

E5 showed edge synergy under a contrast target is meaningless (cross-layer
edges are antagonistic by design). The CCC simulator provides the correct
setting: sender-receiver edges COOPERATE (ligand x receptor), so ground-truth
communicating edges should carry high pairwise synergy.

Host: 3-class GCN (sender / quiet receiver / activating receiver).
Target: class_score_at = mean logit of 'activating receiver' over activating
receivers. Players: interface-zone spots.
Checks: (a) edge synergy AUROC vs ground-truth communicating edges;
(b) gene-level IG recovers LIG/REC/TGT genes; (c) efficiency identity.
"""

import os
import time

import numpy as np
import torch

from myerst.benchmark.ccc_simulator import CCCSimulator
from myerst.models.gcn import GCN, build_norm_adj, train_gcn
from myerst.adapters.torch_adapter import TorchModelAdapter
from myerst.adapters.base import ExplanationTarget
from myerst.explainers.ig import IGExplainer
from myerst.explainers.myerson_explainer import MyersonExplainer
from myerst.explainers.occlusion import _target_from_logits
from scripts.e1_driver_gene_recovery import auroc


def main():
    t0 = time.perf_counter()
    print("=" * 70)
    print("E9: edge synergy on cooperative CCC target (synthetic)")
    print("=" * 70)

    res = CCCSimulator(grid_size=40, n_genes=300, seed=0).simulate()
    data = res.data
    g = 40
    band = 6
    # labels: 0 sender / 1 quiet receiver / 2 activating receiver
    active_recv = res.activation.sum(1) > 0
    labels3 = np.where(res.sender_mask, 0, np.where(active_recv, 2, 1)).astype(int)
    print(f"senders {(labels3==0).sum()}, quiet recv {(labels3==1).sum()}, "
          f"active recv {(labels3==2).sum()}")

    X = np.log1p(data.X)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    x = torch.from_numpy(((X - mu) / sd).astype(np.float32))
    # 3-class domain-mean baseline
    base_np = np.empty_like(data.X)
    for c in range(3):
        m = labels3 == c
        base_np[m] = data.X[m].mean(0)
    baseline_x = torch.from_numpy(((np.log1p(base_np) - mu) / sd).astype(np.float32))

    adj_norm = build_norm_adj(data.edges, data.n_spots)
    gcn = GCN(n_feat=data.n_genes, n_hidden=32, n_classes=3)
    train_gcn(gcn, x, adj_norm, labels3, epochs=200, seed=0)
    with torch.no_grad():
        acc = (gcn(x, adj_norm).argmax(1) == torch.from_numpy(labels3)).float().mean().item()
    print(f"GCN host acc = {acc:.4f}")

    recv_spots = np.where(active_recv)[0]
    adapter = TorchModelAdapter(gcn, adj_norm)
    target = ExplanationTarget(kind="class_score_at", payload=(2, recv_spots))

    # players: interface band (both types)
    cols = data.coords[:, 0]
    players = np.where(np.abs(cols - (g - 1) / 2) <= band / 2)[0]
    print(f"players (interface zone): {len(players)}")

    mye = MyersonExplainer(n_samples=96, perm_batch=32, fwd_chunk=16,
                           seed=0, return_cache=True)
    t1 = time.perf_counter()
    exp = mye.explain(adapter, x, target, players=players, edges=data.edges,
                      n_spots=data.n_spots, baseline=baseline_x,
                      boundary_idx=recv_spots)
    print(f"Myerson: {(time.perf_counter()-t1):.0f}s, mean SEM {exp.meta['sem'].mean():.4f}")

    # (a) edge synergy vs ground-truth communicating edges
    pedges, syn = mye.edge_synergy(adapter, x, target, players, data.edges,
                                   exp.meta["v_cache"], recv_spots,
                                   baseline=baseline_x)
    comm_set = {tuple(e) for e in np.asarray(res.comm_edges)}
    pedge_tuples = [tuple(e) for e in pedges]
    truth_edge = np.array([pe in comm_set for pe in pedge_tuples])
    print("-" * 70)
    print(f"(a) player edges {len(pedges)} | communicating {truth_edge.sum()}")
    print(f"    edge synergy AUROC vs comm ground truth: {auroc(syn, truth_edge):.3f} (random 0.5)")
    same_type = data.labels[pedges[:, 0]] == data.labels[pedges[:, 1]]
    print(f"    sanity: synergy on cross-type edges {syn[~same_type].mean():+.4f} "
          f"vs same-type {syn[same_type].mean():+.4f}")

    # (b) gene-level: IG recovers LIG/REC/TGT
    ig = IGExplainer(40).explain(adapter, x, target, baseline=baseline_x,
                                 ).node_scores
    truth_gene = np.zeros(data.n_genes, dtype=bool)
    for l, r in res.lr_pairs:
        truth_gene[l] = truth_gene[r] = True
    for t in np.concatenate(list(res.targets.values())):
        truth_gene[t] = True
    au = auroc(ig, truth_gene)
    top = np.argsort(ig)[::-1][:15]
    pretty = ", ".join(data.gene_names[i] for i in top)
    print(f"(b) IG gene AUROC vs LIG/REC/TGT: {au:.3f}")
    print(f"    IG top-15: {pretty}")

    # (c) efficiency
    with torch.no_grad():
        v_full = _target_from_logits(adapter.forward(x), target, recv_spots).item()
        x_m = x.clone()
        x_m[players] = baseline_x[players]
        v_empty = _target_from_logits(adapter.forward(x_m), target, recv_spots).item()
    print(f"(c) efficiency: sum(phi)={exp.node_scores.sum():.4f} "
          f"vs v_g(N)-v(0)={v_full - v_empty:.4f}")

    os.makedirs("data/processed", exist_ok=True)
    np.savez("data/processed/e9_ccc_edges.npz", players=players, phi=exp.node_scores,
             pedges=pedges, syn=syn, truth_edge=truth_edge,
             ig_gene=ig, truth_gene=truth_gene, labels3=labels3,
             coords=data.coords)
    print(f"total {(time.perf_counter()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
