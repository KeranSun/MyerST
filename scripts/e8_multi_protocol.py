"""E8: the multi-protocol benchmark matrix (paper Fig 2 core evidence).

Matrix: 3 redundancy regimes x 2 levels (gene/node) x 3 protocols
  protocols: ground-truth recovery AUROC | masking decay AUC | ROAR held-out
  gene methods: IG, Occlusion, Naive DE, random
  node methods: Myerson, IG-node, attention-weight baseline, random

The point: method rankings should FLIP across protocols (masking favors
masking-based explainers, ROAR favors marginal-signal rankings, recovery is
protocol-free) — the "no free lunch in explanation fidelity" result.
"""

import os
import time

import numpy as np
import torch

from myerst.benchmark.simulator import LayeredTissueSimulator
from myerst.benchmark.faithfulness import FaithfulnessEvaluator
from myerst.benchmark.roar import ROAREvaluator
from myerst.models.gcn import GCN, build_norm_adj, train_gcn
from myerst.models.gat_classifier import GATClassifier, train_gat
from myerst.models.stagate_lite import adj_binary
from myerst.data.graph import build_knn_graph
from myerst.adapters.torch_adapter import TorchModelAdapter
from myerst.adapters.base import ExplanationTarget
from myerst.explainers.ig import IGExplainer
from myerst.explainers.occlusion import SpatialOcclusion
from myerst.explainers.myerson_explainer import MyersonExplainer
from myerst.explainers.attention_baseline import attention_node_scores
from scripts.e1_driver_gene_recovery import auroc, boundary_spots

REGIMES = {
    "sparse":   dict(n_driver_per_layer=6, driver_fold=3.0, dropout_rate=0.3,
                     n_passenger_per_driver=0),
    "medium":   dict(n_driver_per_layer=6, driver_fold=3.0, dropout_rate=0.3,
                     n_passenger_per_driver=2, passenger_noise=0.3),
    "high":     dict(n_driver_per_layer=6, driver_fold=3.0, dropout_rate=0.3,
                     n_passenger_per_driver=4, passenger_noise=0.3),
}
KS_GENE = [0, 4, 8, 16, 32, 64, 128]
KS_NODE = [0, 10, 30, 60, 120, 200]
SEEDS = (0, 1)


def train_fn_gcn(X, coords, labels, seed):
    edges = build_knn_graph(coords, k=6)
    adj_norm = build_norm_adj(edges, X.shape[0])
    x = torch.from_numpy(np.ascontiguousarray(X).astype(np.float32))
    m = GCN(n_feat=X.shape[1], n_hidden=32, n_classes=3)
    m, tr = train_gcn(m, x, adj_norm, labels, epochs=100, seed=seed, return_mask=True)
    m.eval()
    with torch.no_grad():
        return m(x, adj_norm), ~tr


def run_regime(name, kw, sim_seed=0):
    t0 = time.perf_counter()
    print(f"\n{'='*74}\nREGIME: {name}\n{'='*74}", flush=True)
    sim = LayeredTissueSimulator(grid_size=40, n_layers=3, n_genes=300, seed=sim_seed, **kw)
    res = sim.simulate()
    data = res.data
    data.build_graph(k=6)
    X = np.log1p(data.X)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)
    x = torch.from_numpy(Xn)
    baseline_x = torch.from_numpy(((np.log1p(data.domain_mean()) - mu) / sd).astype(np.float32))

    # hosts
    adj_norm = build_norm_adj(data.edges, data.n_spots)
    gcn = GCN(n_feat=data.n_genes, n_hidden=32, n_classes=3)
    train_gcn(gcn, x, adj_norm, data.labels, epochs=150, seed=0)
    adj_bin = adj_binary(data.edges, data.n_spots)
    gat = GATClassifier(n_feat=data.n_genes, n_hidden=32, n_classes=3, heads=4)
    train_gat(gat, x, adj_bin, data.labels, epochs=150, seed=0)

    A, B = 0, 1
    interface = boundary_spots(data.labels, data.adj, A, B)
    adapter = TorchModelAdapter(gcn, adj_norm, boundary_spots={(A, B): interface})
    target = ExplanationTarget(kind="domain_boundary", payload=(A, B))
    truth_gene = np.zeros(data.n_genes, dtype=bool)
    truth_gene[res.boundary_genes[(A, B)]] = True

    neigh = set()
    for i in interface:
        neigh |= data.adj[i]
    players = np.array(sorted(neigh | set(interface.tolist())))
    truth_node = np.isin(players, interface)

    # ---------- rankings
    ig_exp = IGExplainer(40).explain(adapter, x, target, baseline=baseline_x)
    ig_gene = ig_exp.node_scores
    ig_node = np.abs(ig_exp.meta["node_level"]).sum(1)[players]
    occ_gene = SpatialOcclusion(batch_size=16).explain(adapter, x, target, baseline=baseline_x).node_scores
    de_gene = np.abs(data.X[data.labels == A].mean(0) - data.X[data.labels == B].mean(0))
    attn_node = attention_node_scores(gat, x, adj_bin, interface)[players]
    rng = np.random.default_rng(0)
    mye = MyersonExplainer(n_samples=128, perm_batch=32, fwd_chunk=16, seed=0)
    phi = mye.explain(adapter, x, target, players=players, edges=data.edges,
                      n_spots=data.n_spots, baseline=baseline_x).node_scores

    gene_rank = {"IG": ig_gene, "Occ": occ_gene, "DE": de_gene,
                 "random": rng.random(data.n_genes)}
    node_rank = {"Myerson": phi, "IG-node": ig_node, "attention": attn_node,
                 "random": rng.random(len(players))}

    # ---------- protocol 1: recovery AUROC (protocol-free)
    print("[P1 recovery AUROC]  gene: " + "  ".join(
        f"{m}={auroc(s, truth_gene):.3f}" for m, s in gene_rank.items()))
    print("                     node: " + "  ".join(
        f"{m}={auroc(s, truth_node):.3f}" for m, s in node_rank.items()), flush=True)

    # ---------- protocol 2: masking decay AUC
    ev = FaithfulnessEvaluator(adapter, x, baseline_x, target, interface)
    p2_gene = {m: ev.gene_curve(s)['decay_auc'] for m, s in gene_rank.items()}
    p2_node = {m: ev.node_curve(s, players)['decay_auc'] for m, s in node_rank.items()}
    print("[P2 masking decayAUC] gene: " + "  ".join(f"{m}={v:.3f}" for m, v in p2_gene.items()))
    print("                     node: " + "  ".join(f"{m}={v:.3f}" for m, v in p2_node.items()), flush=True)

    # ---------- protocol 3: ROAR held-out
    roar = ROAREvaluator(train_fn_gcn, ks=KS_GENE, seeds=SEEDS)
    res_g = roar.gene_roar(Xn, data.coords, data.labels, interface, A, B, gene_rank)
    roar.ks = KS_NODE
    res_n = roar.node_roar(Xn, data.coords, data.labels, interface, A, B,
                           players, node_rank, eval_by_labels=True)
    print("[P3 ROAR acc_auc]   gene: " + "  ".join(
        f"{m}={r['acc_auc']:.3f}" for m, r in res_g.items()))
    print("                     node: " + "  ".join(
        f"{m}={r['acc_auc']:.3f}" for m, r in res_n.items()), flush=True)
    print(f"(regime {name} took {(time.perf_counter()-t0)/60:.1f} min)", flush=True)
    return {"P1_gene": {m: auroc(s, truth_gene) for m, s in gene_rank.items()},
            "P1_node": {m: auroc(s, truth_node) for m, s in node_rank.items()},
            "P2_gene": p2_gene, "P2_node": p2_node,
            "P3_gene": {m: r['acc_auc'] for m, r in res_g.items()},
            "P3_node": {m: r['acc_auc'] for m, r in res_n.items()}}


def main():
    import pickle
    t0 = time.perf_counter()
    print("E8: multi-protocol benchmark matrix")
    import sys, json
    seeds = [int(s) for s in sys.argv[1:]] or [0]
    all_res = {}
    for name, kw in REGIMES.items():
        for s in seeds:
            r = run_regime(name, kw, sim_seed=s)
            all_res.setdefault(name, []).append(r)
    with open("data/processed/e8_matrix_multiseed.pkl", "wb") as f:
        import pickle
        pickle.dump(all_res, f)
    # aggregated mean+-sd report
    for name in REGIMES:
        runs = all_res[name]
        for key in ["P1_node", "P2_node"]:
            for m in ["Myerson", "IG-node", "attention", "random"]:
                vals = [r[key][m] for r in runs]
                print(f"{name}/{key}/{m}: {np.mean(vals):.3f} +- {np.std(vals):.3f}")
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/e8_matrix.pkl", "wb") as f:
        pickle.dump(all_res, f)
    print(f"\nsaved data/processed/e8_matrix.pkl; TOTAL {(time.perf_counter()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
