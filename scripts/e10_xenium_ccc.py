"""E10: flagship CCC analysis on Xenium breast cancer (tumor-immune interface).

Host: GCN predicting T-cell cytotoxic state (GZMB/PRF1/NKG7/GNLY) on the full
cell graph. Explanation: local CCC synergy per T-cell, aggregated per
ligand-receptor pathway with permutation controls.

Known biology to recover: CXCL12(CAF) -> CXCR4(T-cell) retention axis.
Also tested: CD274(PD-L1)->PDCD1, CD80/CD86->CTLA4, PTN->SDC4.
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import anndata as ad

from myerst.models.gcn import GCN, build_norm_adj
from myerst.data.graph import build_knn_graph, adjacency_list
from myerst.explainers.local_ccc_fast import local_ccc_synergy_fast
from scripts.e1_driver_gene_recovery import auroc

CYTO = ["GZMB", "PRF1", "NKG7", "GNLY"]
LR_PAIRS = [("CXCL12", "CXCR4"), ("CD274", "PDCD1"), ("CD80", "CTLA4"),
            ("CD86", "CTLA4"), ("PTN", "SDC4")]
SUBWINDOW = 1500.0
K_GRAPH = 8
N_PERM = 100


def main():
    t0 = time.perf_counter()
    print("=" * 72)
    print("E10: Xenium breast cancer CCC (local LR synergy, pathway-aggregated)")
    print("=" * 72)

    a = ad.read_h5ad("data/processed/xenium_crop.h5ad")
    # ---- pick dense sub-window with best trio mixing
    xy = a.obsm["spatial"]
    labs = a.obs["celltype"].to_numpy()
    best = (-1.0, None)
    for x0 in np.arange(xy[:, 0].min(), xy[:, 0].max() - SUBWINDOW, 500):
        for y0 in np.arange(xy[:, 1].min(), xy[:, 1].max() - SUBWINDOW, 500):
            m = ((xy[:, 0] >= x0) & (xy[:, 0] < x0 + SUBWINDOW)
                 & (xy[:, 1] >= y0) & (xy[:, 1] < y0 + SUBWINDOW))
            if m.sum() < 4000:
                continue
            vc = pd.Series(labs[m]).value_counts(normalize=True)
            trio = min(vc.get("Cancer Epithelial", 0), vc.get("T-cells", 0),
                       vc.get("CAFs", 0))
            if trio > best[0]:
                best = (trio, (x0, y0))
    x0, y0 = best[1]
    m = ((xy[:, 0] >= x0) & (xy[:, 0] < x0 + SUBWINDOW)
         & (xy[:, 1] >= y0) & (xy[:, 1] < y0 + SUBWINDOW))
    sub = a[m].copy()
    print(f"subwindow ({x0:.0f},{y0:.0f}): {sub.n_obs} cells; "
          f"{sub.obs['celltype'].value_counts().head(4).to_dict()}")

    # ---- normalize
    X = sub.X.toarray().astype(np.float32)
    lib = X.sum(1, keepdims=True)
    Xn = np.log1p(X / np.maximum(lib, 1) * np.median(lib))
    mu, sd = Xn.mean(0), Xn.std(0) + 1e-6
    Xz = ((Xn - mu) / sd).astype(np.float32)
    genes = np.array(sub.var_names)
    labs = sub.obs["celltype"].to_numpy()
    coords = sub.obsm["spatial"]

    # ---- T-cell cytotoxic label
    gidx = {g: i for i, g in enumerate(genes)}
    cyto_score = Xz[:, [gidx[g] for g in CYTO]].mean(1)
    is_t = labs == "T-cells"
    t_act = np.zeros(len(labs), dtype=int)
    thr = np.median(cyto_score[is_t])
    t_act[is_t] = (cyto_score[is_t] > thr).astype(int)
    print(f"T-cells: {is_t.sum()}, cytotoxic (>median): {t_act[is_t].sum()}")

    # ---- host: GCN, loss on T-cells only
    edges = build_knn_graph(coords, k=K_GRAPH)
    adj_norm = build_norm_adj(edges, len(labs))
    x = torch.from_numpy(Xz)
    y = torch.from_numpy(t_act.astype(np.int64))
    tmask = torch.from_numpy(is_t)
    torch.manual_seed(0)
    gcn = GCN(n_feat=len(genes), n_hidden=64, n_classes=2)
    opt = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)
    gcn.train()
    for ep in range(200):
        opt.zero_grad()
        logits = gcn(x, adj_norm)
        loss = F.cross_entropy(logits[tmask], y[tmask])
        loss.backward()
        opt.step()
    gcn.eval()
    with torch.no_grad():
        pred = gcn(x, adj_norm).argmax(1)
        acc_t = (pred[is_t] == y[is_t]).float().mean().item()
    print(f"GCN host trained, T-cell accuracy = {acc_t:.4f}")

    # ---- local CCC synergy
    padj = adjacency_list(edges, len(labs))
    receivers = np.array([i for i in np.where(is_t)[0]
                          if any(labs[j] != "T-cells" for j in padj[i])])
    sender_nb = {int(j): [int(i) for i in padj[j] if labs[i] != "T-cells"]
                 for j in receivers}
    local_nodes = {}
    for j in receivers:
        one = set(padj[j]) | {int(j)}
        two = set(one)
        for u in one:
            two |= padj[u]
        local_nodes[int(j)] = np.array(sorted(two), dtype=np.int64)
    # baseline: mean T-cell profile (z-scored)
    base = np.tile(Xz[is_t].mean(0), (len(labs), 1)).astype(np.float32)
    baseline_x = torch.from_numpy(base)

    n_pairs = sum(len(v) for v in sender_nb.values())
    print(f"receivers: {len(receivers)}, (T-cell, sender) pairs: {n_pairs}")
    t1 = time.perf_counter()
    out = local_ccc_synergy_fast(gcn, x, adj_norm, receivers, sender_nb,
                                 baseline_x, target_cls=1,
                                 local_nodes=local_nodes)
    pairs, psi = out["pairs"], out["psi"]
    print(f"local synergy computed ({(time.perf_counter()-t1)/60:.1f} min)")

    # ---- pathway aggregation with permutation control
    Xraw = X  # raw counts for expression presence
    expr_L = {g: Xraw[:, gidx[g]] > 0 for g, _ in LR_PAIRS if g in gidx}
    expr_R = {g: Xraw[:, gidx[g]] > 0 for _, g in LR_PAIRS if g in gidx}
    print("-" * 72)
    print(f"{'pathway':<22} {'n_edges':>8} {'mean_psi':>9} {'perm_mean':>10} {'z':>6}")
    rng = np.random.default_rng(0)
    rows = []
    for lig, rec in LR_PAIRS:
        if lig not in gidx or rec not in gidx:
            continue
        sel = np.array([expr_L[lig][i] and expr_R[rec][j] for j, i in pairs])
        if sel.sum() < 10:
            continue
        obs = psi[sel].mean()
        # permutation: shuffle sender-ligand presence
        null = []
        for _ in range(N_PERM):
            perm_L = rng.permutation(expr_L[lig])
            sel_p = np.array([perm_L[i] and expr_R[rec][j] for j, i in pairs])
            if sel_p.sum() >= 10:
                null.append(psi[sel_p].mean())
        null = np.array(null)
        z = (obs - null.mean()) / (null.std() + 1e-12)
        rows.append((lig, rec, sel.sum(), obs, null.mean(), z))
        print(f"{lig+'->'+rec:<22} {sel.sum():>8} {obs:>+9.4f} {null.mean():>+10.4f} {z:>6.2f}")

    # sender-type breakdown for the top pathway
    if rows:
        top = max(rows, key=lambda r: r[5])
        lig, rec = top[0], top[1]
        sel = np.array([expr_L[lig][i] and expr_R[rec][j] for j, i in pairs])
        st = pd.Series([labs[i] for j, i in pairs[sel]])
        print(f"\ntop pathway {lig}->{rec}: sender composition of high-psi edges")
        print(st.value_counts().to_string())

    os.makedirs("data/processed", exist_ok=True)
    np.savez("data/processed/e10_xenium_ccc.npz", pairs=pairs, psi=psi,
             labs=labs, coords=coords, is_t=is_t, t_act=t_act,
             cyto_score=cyto_score, genes=genes)
    print(f"\ntotal {(time.perf_counter()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
