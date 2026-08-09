"""E8c: multi-seed node-level benchmark with FIXED evaluation semantics.

Fixes diagnosed in diag_e8_seeds:
1. Myerson phi is scored sign-aligned: credit = sign(ref) * phi. Raw-phi AUROC
   inverts whenever the host learns a negative target direction (seed 2 lesson).
2. P2 masking target is the class-signed probability margin (bounded, always
   positive scale) instead of the raw logit difference (ill-conditioned scale).

3 sim seeds x 2 regimes (sparse/medium) x {Myerson, IG-node, attention, random}.
"""

import pickle
import time

import numpy as np
import torch

from myerst.benchmark.simulator import LayeredTissueSimulator
from myerst.benchmark.faithfulness import FaithfulnessEvaluator
from myerst.models.gcn import GCN, build_norm_adj, train_gcn
from myerst.models.gat_classifier import GATClassifier, train_gat
from myerst.models.stagate_lite import adj_binary
from myerst.adapters.torch_adapter import TorchModelAdapter
from myerst.adapters.base import ExplanationTarget
from myerst.explainers.ig import IGExplainer
from myerst.explainers.myerson_explainer import MyersonExplainer
from myerst.explainers.attention_baseline import attention_node_scores
from myerst.explainers.occlusion import _target_from_logits
from scripts.e1_driver_gene_recovery import auroc, boundary_spots

REGIMES = {
    "sparse": dict(n_driver_per_layer=6, driver_fold=3.0, dropout_rate=0.3,
                   n_passenger_per_driver=0),
    "medium": dict(n_driver_per_layer=6, driver_fold=3.0, dropout_rate=0.3,
                   n_passenger_per_driver=2, passenger_noise=0.3),
}
SEEDS = [0, 1, 2]


def run(regime, kw, sim_seed):
    sim = LayeredTissueSimulator(grid_size=40, n_layers=3, n_genes=300,
                                 seed=sim_seed, **kw)
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
    adj_bin = adj_binary(data.edges, data.n_spots)
    gat = GATClassifier(n_feat=data.n_genes, n_hidden=32, n_classes=3, heads=4)
    train_gat(gat, x, adj_bin, data.labels, epochs=150, seed=0)

    A, B = 0, 1
    interface = boundary_spots(data.labels, data.adj, A, B)
    signs = np.where(data.labels[interface] == A, 1.0, -1.0)
    adapter = TorchModelAdapter(gcn, adj_norm, boundary_spots={(A, B): interface})
    target_raw = ExplanationTarget(kind="domain_boundary", payload=(A, B))
    target_margin = ExplanationTarget(kind="domain_boundary_margin",
                                      payload=(A, B, signs))

    neigh = set()
    for i in interface:
        neigh |= data.adj[i]
    players = np.array(sorted(neigh | set(interface.tolist())))
    truth = np.isin(players, interface)

    ig_node = np.abs(IGExplainer(40).explain(
        adapter, x, target_raw, baseline=baseline_x).meta["node_level"]).sum(1)[players]
    attn = attention_node_scores(gat, x, adj_bin, interface)[players]
    phi = MyersonExplainer(n_samples=256, perm_batch=32, fwd_chunk=16,
                           seed=0).explain(
        adapter, x, target_raw, players=players, edges=data.edges,
        n_spots=data.n_spots, baseline=baseline_x).node_scores
    rng = np.random.default_rng(0)

    with torch.no_grad():
        ref = _target_from_logits(adapter.forward(x), target_raw, interface).item()
    sign = 1.0 if ref > 0 else -1.0
    rank = {"Myerson": sign * phi, "IG-node": ig_node, "attention": attn,
            "random": rng.random(len(players))}
    p1 = {m: auroc(s, truth) for m, s in rank.items()}

    ev = FaithfulnessEvaluator(adapter, x, baseline_x, target_margin, interface)
    p2 = {m: ev.node_curve(s, players)["decay_auc"] for m, s in rank.items()}

    print(f"{regime}/seed{sim_seed}: ref={ref:+.3f} | "
          f"P1 " + " ".join(f"{m}={v:.3f}" for m, v in p1.items()), flush=True)
    print(f"{'':>16}P2 " + " ".join(f"{m}={v:.3f}" for m, v in p2.items()), flush=True)
    return {"P1": p1, "P2": p2}


def main():
    t0 = time.perf_counter()
    out = {}
    for regime, kw in REGIMES.items():
        for s in SEEDS:
            out[(regime, s)] = run(regime, kw, s)
    with open("data/processed/e8c_node_multiseed.pkl", "wb") as f:
        pickle.dump(out, f)
    print("\n--- aggregated (mean +- sd over seeds) ---")
    for regime in REGIMES:
        for proto in ["P1", "P2"]:
            for m in ["Myerson", "IG-node", "attention", "random"]:
                vals = [out[(regime, s)][proto][m] for s in SEEDS]
                print(f"{regime}/{proto}/{m}: {np.mean(vals):.3f} +- {np.std(vals):.3f}")
    print(f"total {(time.perf_counter()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
