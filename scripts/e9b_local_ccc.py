"""E9b: local CCC synergy on the ZONED simulator — the decisive edge test.

Zoned activation => interface now contains BOTH activating receivers (true
communication) and quiet receivers (cross-type edges but NO signaling). A good
edge score must separate these. Compare local LR synergy psi against:
  ground truth = cross-type edge to an ACTIVATING receiver.
"""

import os
import time

import numpy as np
import torch

from myerst.benchmark.ccc_simulator import CCCSimulator
from myerst.models.gcn import GCN, build_norm_adj, train_gcn
from myerst.explainers.local_ccc import local_ccc_synergy
from scripts.e1_driver_gene_recovery import auroc


def main():
    t0 = time.perf_counter()
    print("=" * 70)
    print("E9b: local CCC synergy on zoned simulator")
    print("=" * 70)

    res = CCCSimulator(grid_size=40, n_genes=300, seed=0).simulate()
    data = res.data
    active_recv = res.activation.sum(1) > 0
    labels3 = np.where(res.sender_mask, 0, np.where(active_recv, 2, 1)).astype(int)
    print(f"senders {(labels3==0).sum()}, quiet recv {(labels3==1).sum()}, "
          f"active recv {(labels3==2).sum()}")

    # Hold out downstream targets from the host INPUT. Otherwise the model
    # classifies 'activating receiver' by reading j's own TGT genes and never
    # uses neighbor ligands — and there would be no communication to explain.
    tgt_idx = np.concatenate(list(res.targets.values()))
    keep_genes = np.ones(data.n_genes, dtype=bool)
    keep_genes[tgt_idx] = False
    print(f"host input genes: {keep_genes.sum()} (TGT held out: {len(tgt_idx)})")

    X = np.log1p(data.X[:, keep_genes])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    x = torch.from_numpy(((X - mu) / sd).astype(np.float32))
    base_np = np.empty((data.n_spots, int(keep_genes.sum())), dtype=np.float32)
    for c in range(3):
        m = labels3 == c
        base_np[m] = data.X[m][:, keep_genes].mean(0)
    baseline_x = torch.from_numpy(((np.log1p(base_np) - mu) / sd).astype(np.float32))

    n_feat = int(keep_genes.sum())
    adj_norm = build_norm_adj(data.edges, data.n_spots)
    gcn = GCN(n_feat=n_feat, n_hidden=32, n_classes=3)
    train_gcn(gcn, x, adj_norm, labels3, epochs=200, seed=0)
    with torch.no_grad():
        acc = (gcn(x, adj_norm).argmax(1) == torch.from_numpy(labels3)).float().mean().item()
    print(f"GCN host acc = {acc:.4f}")

    # local games: receivers in interface (have >=1 sender neighbor), both classes
    padj = data.adj
    recv_all = [i for i in np.where(~res.sender_mask)[0]
                if any(res.sender_mask[j] for j in padj[i])]
    sender_nb = {int(j): [int(i) for i in padj[j] if res.sender_mask[i]] for j in recv_all}
    local_nodes = {}
    for j in recv_all:
        one_hop = set(padj[j]) | {j}
        two_hop = set(one_hop)
        for u in one_hop:
            two_hop |= padj[u]
        local_nodes[int(j)] = np.array(sorted(two_hop), dtype=np.int64)
    print(f"interface receivers (local games): {len(recv_all)}")

    out = local_ccc_synergy(gcn, x, adj_norm, np.asarray(recv_all), sender_nb,
                            baseline_x, target_cls=2, local_nodes=local_nodes)
    pairs, psi = out["pairs"], out["psi"]
    # pair-level ground truth: receiver j activates via some pair p for which
    # THIS sender i supplies the ligand (lig_on[i,p] and activation[j,p] > 0)
    n_pairs = res.activation.shape[1]
    truth = np.zeros(len(pairs), dtype=bool)
    for k, (j, i) in enumerate(pairs):
        truth[k] = any(res.activation[j, p] > 0 and res.lig_on[i, p]
                       for p in range(n_pairs))
    print("-" * 70)
    print(f"(receiver, sender) pairs: {len(pairs)} | true communicating: {truth.sum()}")
    print(f"local synergy AUROC vs true communication: {auroc(psi, truth):.3f} (random 0.5)")
    print(f"mean psi: communicating {psi[truth].mean():+.4f} vs not {psi[~truth].mean():+.4f}")

    # same-type negative control: pair each receiver with a RECEIVER neighbor
    recv_nb = {}
    for j in recv_all:
        rn = [int(i) for i in padj[j] if not res.sender_mask[i] and int(i) != j]
        if rn:
            recv_nb[int(j)] = rn[:1]
    out_ctl = local_ccc_synergy(gcn, x, adj_norm,
                                np.asarray(list(recv_nb.keys())), recv_nb,
                                baseline_x, target_cls=2, local_nodes=local_nodes)
    print(f"negative control (receiver-receiver pairs): mean psi {out_ctl['psi'].mean():+.4f}")

    os.makedirs("data/processed", exist_ok=True)
    np.savez("data/processed/e9b_local_ccc.npz", pairs=pairs, psi=psi, truth=truth,
             labels3=labels3, coords=data.coords, active_recv=active_recv,
             sender_mask=res.sender_mask)
    print(f"total {(time.perf_counter()-t0):.0f}s")


if __name__ == "__main__":
    main()
