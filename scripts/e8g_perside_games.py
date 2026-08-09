"""E8g: per-side own-class games — target semantics matching the ground truth.

E8c-f lesson chain: raw logit-diff target is degenerate (ref~0 by class
cancellation); mixed signed margin has friendly fire (interface spots hurt
opposite-class neighbors' margin, so credit flows to interior supporters).
The semantics matching "which spots define the boundary" is PER-SIDE:
for side A, target = mean p_A over A-side interface spots only.

For each side we run its own game (Myerson/IG), then AUROC per side:
interface spots of that side vs 1-hop background. Pooled across sides.
"""

import pickle
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
    gm = np.log1p(data.X).mean(0)
    baseline_x = torch.from_numpy(
        np.tile((gm - mu) / sd, (data.n_spots, 1)).astype(np.float32))

    adj_norm = build_norm_adj(data.edges, data.n_spots)
    gcn = GCN(n_feat=data.n_genes, n_hidden=32, n_classes=3)
    train_gcn(gcn, x, adj_norm, data.labels, epochs=150, seed=0)
    adapter = TorchModelAdapter(gcn, adj_norm)

    A, B = 0, 1
    interface = boundary_spots(data.labels, data.adj, A, B)
    neigh = set()
    for i in interface:
        neigh |= data.adj[i]
    players = np.array(sorted(neigh | set(interface.tolist())))
    is_iface = np.isin(players, interface)

    aucs = {"Myerson": [], "IG-node": [], "random": []}
    rng = np.random.default_rng(0)
    for side in [A, B]:
        side_iface = interface[data.labels[interface] == side]
        target = ExplanationTarget(kind="class_score_at",
                                   payload=(side, side_iface))
        truth_side = np.isin(players, side_iface)
        if truth_side.sum() < 5:
            continue
        phi = MyersonExplainer(n_samples=128, perm_batch=32, fwd_chunk=16,
                               seed=0).explain(
            adapter, x, target, players=players, edges=data.edges,
            n_spots=data.n_spots, baseline=baseline_x,
            boundary_idx=side_iface).node_scores
        ig_node = np.abs(IGExplainer(40).explain(
            adapter, x, target, baseline=baseline_x).meta["node_level"]).sum(1)[players]
        aucs["Myerson"].append(auroc(np.abs(phi), truth_side))
        aucs["IG-node"].append(auroc(ig_node, truth_side))
        aucs["random"].append(auroc(rng.random(len(players)), truth_side))
    row = {m: float(np.mean(v)) for m, v in aucs.items()}
    print(f"{regime}/seed{sim_seed}: " +
          " ".join(f"{m}={v:.3f}" for m, v in row.items()), flush=True)
    return row


def main():
    t0 = time.perf_counter()
    out = {}
    for regime, kw in REGIMES.items():
        for s in SEEDS:
            out[(regime, s)] = run(regime, kw, s)
    with open("data/processed/e8g_perside.pkl", "wb") as f:
        pickle.dump(out, f)
    print("\n--- aggregated ---")
    for regime in REGIMES:
        for m in ["Myerson", "IG-node", "random"]:
            vals = [out[(regime, s)][m] for s in SEEDS]
            print(f"{regime}/{m}: {np.mean(vals):.3f} +- {np.std(vals):.3f}")
    print(f"total {(time.perf_counter()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
