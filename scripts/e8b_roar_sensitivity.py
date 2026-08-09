"""E8b: can ROAR differentiate at all? Sparser-data sensitivity test.

E8 showed ROAR held-out accuracy saturates at grid-40 scale (1600 spots give
the retrained model enough data to learn residual offsets). Hypothesis: with
less data (grid 20 = 400 spots) and weaker signal (fold 1.5), ROAR retention
curves should spread across rankings. Gene-level + node-level, 2 seeds.
"""

import time

import numpy as np
import torch

from myerst.benchmark.simulator import LayeredTissueSimulator
from myerst.benchmark.roar import ROAREvaluator
from myerst.models.gcn import GCN, build_norm_adj, train_gcn
from myerst.data.graph import build_knn_graph
from myerst.adapters.torch_adapter import TorchModelAdapter
from myerst.adapters.base import ExplanationTarget
from myerst.explainers.ig import IGExplainer
from myerst.explainers.occlusion import SpatialOcclusion
from myerst.explainers.myerson_explainer import MyersonExplainer
from scripts.e1_driver_gene_recovery import boundary_spots

KS_GENE = [0, 2, 4, 8, 16, 32, 64]
KS_NODE = [0, 5, 15, 30, 60, 100]
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


RESULTS = {}

def run(tag, grid, fold):
    print(f"--- {tag} (grid={grid}, fold={fold}) ---", flush=True)
    sim = LayeredTissueSimulator(grid_size=grid, n_layers=3, n_genes=300, seed=0,
                                 n_driver_per_layer=6, driver_fold=fold,
                                 dropout_rate=0.3, n_passenger_per_driver=0)
    res = sim.simulate()
    data = res.data
    data.build_graph(k=6)
    X = np.log1p(data.X)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)
    x = torch.from_numpy(Xn)
    baseline_x = torch.from_numpy(((np.log1p(data.domain_mean()) - mu) / sd).astype(np.float32))
    adj_norm = build_norm_adj(data.edges, data.n_spots)
    gcn = GCN(n_feat=data.n_genes, n_hidden=32, n_classes=3)
    train_gcn(gcn, x, adj_norm, data.labels, epochs=150, seed=0)
    A, B = 0, 1
    interface = boundary_spots(data.labels, data.adj, A, B)
    adapter = TorchModelAdapter(gcn, adj_norm, boundary_spots={(A, B): interface})
    target = ExplanationTarget(kind="domain_boundary", payload=(A, B))

    ig_exp = IGExplainer(40).explain(adapter, x, target, baseline=baseline_x)
    ig_node = np.abs(ig_exp.meta["node_level"]).sum(1)
    occ = SpatialOcclusion(batch_size=16).explain(adapter, x, target, baseline=baseline_x).node_scores
    de = np.abs(data.X[data.labels == A].mean(0) - data.X[data.labels == B].mean(0))
    rng = np.random.default_rng(0)

    neigh = set()
    for i in interface:
        neigh |= data.adj[i]
    players = np.array(sorted(neigh | set(interface.tolist())))
    phi = MyersonExplainer(n_samples=96, perm_batch=32, fwd_chunk=16, seed=0).explain(
        adapter, x, target, players=players, edges=data.edges,
        n_spots=data.n_spots, baseline=baseline_x).node_scores

    roar = ROAREvaluator(train_fn_gcn, ks=KS_GENE, seeds=SEEDS)
    res_g = roar.gene_roar(Xn, data.coords, data.labels, interface, A, B,
                           {"IG": ig_exp.node_scores, "Occ": occ, "DE": de,
                            "random": rng.random(data.n_genes)})
    print("  gene acc_auc: " + "  ".join(f"{m}={r['acc_auc']:.3f}" for m, r in res_g.items()))
    for m, r in res_g.items():
        print(f"    {m:<8} acc@k {np.round(r['acc_mean'], 3)}")
    roar.ks = KS_NODE
    res_n = roar.node_roar(Xn, data.coords, data.labels, interface, A, B, players,
                           {"Myerson": phi, "IG-node": ig_node[players],
                            "random": rng.random(len(players))}, eval_by_labels=True)
    print("  node acc_auc: " + "  ".join(f"{m}={r['acc_auc']:.3f}" for m, r in res_n.items()))
    for m, r in res_n.items():
        print(f"    {m:<8} acc@k {np.round(r['acc_mean'], 3)}", flush=True)
    RESULTS[tag] = {"gene": {m: r["acc_mean"].tolist() for m, r in res_g.items()},
                    "node": {m: r["acc_mean"].tolist() for m, r in res_n.items()}}


def main():
    t0 = time.perf_counter()
    run("denser+strong", 40, 3.0)
    run("sparse-grid+weak", 20, 1.5)
    np.savez("data/processed/e8b_roar_curves.npz",
             **{f"{tag}/{lvl}/{m}": np.array(v)
                for tag, rr in RESULTS.items()
                for lvl, dd in rr.items() for m, v in dd.items()})
    print(f"total {(time.perf_counter()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
