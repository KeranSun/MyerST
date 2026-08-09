"""Diagnose E8 multi-seed instability: why does Myerson node AUROC collapse
on sim seeds 1/2, and why does P2 decay AUC explode?

For seeds 0,1,2 (sparse regime): print host acc, target ref / v_end scale,
Myerson & IG node AUROC vs interface, and the mean phi split
(interface vs 1-hop) to see where credit actually goes.
"""

import numpy as np
import torch

from myerst.benchmark.simulator import LayeredTissueSimulator
from myerst.benchmark.faithfulness import FaithfulnessEvaluator
from myerst.models.gcn import GCN, build_norm_adj, train_gcn
from myerst.adapters.torch_adapter import TorchModelAdapter
from myerst.adapters.base import ExplanationTarget
from myerst.explainers.ig import IGExplainer
from myerst.explainers.myerson_explainer import MyersonExplainer
from myerst.explainers.occlusion import _target_from_logits
from scripts.e1_driver_gene_recovery import auroc, boundary_spots


def run(seed):
    print(f"\n{'='*66}\nSIM SEED {seed}\n{'='*66}", flush=True)
    sim = LayeredTissueSimulator(grid_size=40, n_layers=3, n_genes=300, seed=seed,
                                 n_driver_per_layer=6, driver_fold=3.0,
                                 dropout_rate=0.3, n_passenger_per_driver=0)
    res = sim.simulate()
    data = res.data
    data.build_graph(k=6)
    X = np.log1p(data.X)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    x = torch.from_numpy(((X - mu) / sd).astype(np.float32))
    baseline_x = torch.from_numpy(
        ((np.log1p(data.domain_mean()) - mu) / sd).astype(np.float32))

    adj_norm = build_norm_adj(data.edges, data.n_spots)
    gcn = GCN(n_feat=data.n_genes, n_hidden=32, n_classes=3)
    train_gcn(gcn, x, adj_norm, data.labels, epochs=150, seed=0)
    with torch.no_grad():
        acc = (gcn(x, adj_norm).argmax(1) == torch.from_numpy(
            data.labels.astype(np.int64))).float().mean().item()

    A, B = 0, 1
    interface = boundary_spots(data.labels, data.adj, A, B)
    adapter = TorchModelAdapter(gcn, adj_norm, boundary_spots={(A, B): interface})
    target = ExplanationTarget(kind="domain_boundary", payload=(A, B))
    with torch.no_grad():
        ref = _target_from_logits(adapter.forward(x), target, interface).item()
        x_m = x.clone()
        x_m[:] = baseline_x
        v_end = _target_from_logits(adapter.forward(x_m), target, interface).item()
    print(f"host acc={acc:.3f} | target ref={ref:+.4f} v_allmasked={v_end:+.4f} "
          f"|ref-v_end|={abs(ref-v_end):.4f}")

    neigh = set()
    for i in interface:
        neigh |= data.adj[i]
    players = np.array(sorted(neigh | set(interface.tolist())))
    truth = np.isin(players, interface)

    ig_node = np.abs(IGExplainer(40).explain(
        adapter, x, target, baseline=baseline_x).meta["node_level"]).sum(1)[players]
    phi = MyersonExplainer(n_samples=128, perm_batch=32, fwd_chunk=16,
                           seed=0).explain(
        adapter, x, target, players=players, edges=data.edges,
        n_spots=data.n_spots, baseline=baseline_x).node_scores

    print(f"IG-node AUROC      = {auroc(ig_node, truth):.3f}")
    print(f"Myerson phi AUROC  = {auroc(phi, truth):.3f}")
    print(f"mean |phi|: interface {np.abs(phi)[truth].mean():.4f} | "
          f"1-hop {np.abs(phi)[~truth].mean():.4f}")
    print(f"mean phi:  interface {phi[truth].mean():+.4f} | "
          f"1-hop {phi[~truth].mean():+.4f}")
    print(f"phi sign: {(phi > 0).mean():.2f} positive; target ref sign "
          f"{'+' if ref > 0 else '-'}")


for s in [0, 1, 2]:
    run(s)
