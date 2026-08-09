"""E10e: gene-level occlusion in the local game (the cleanest perturbation).

E10b/d masked whole CELLS (profile -> T-mean baseline), confounding ligand
removal with cell-type removal. Here only the LIGAND GENE is zeroed (count=0)
in the receiver's 2-hop zone; everything else untouched. Null = zeroing
frequency-matched random genes. Host = LR-restricted GCN (36 signaling genes),
target = T-cell infiltration (as in E10c/d).

psi_gene(j) = v_full(j) - v_{ligand zeroed in zone}(j)  (>0 = ligand supports)
"""

import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import anndata as ad
from scipy.spatial import cKDTree

from myerst.models.gcn import GCN, build_norm_adj
from myerst.data.graph import build_knn_graph, adjacency_list

LR_PAIRS = [("CXCL12", "CXCR4"), ("CD274", "PDCD1"), ("CD80", "CTLA4"),
            ("CD86", "CTLA4"), ("PTN", "SDC4")]
SUBWINDOW = 1500.0
K_GRAPH = 8


def main():
    t0 = time.perf_counter()
    print("=" * 72)
    print("E10e: gene-level ligand occlusion (LR-restricted host)")
    print("=" * 72)

    a = ad.read_h5ad("data/processed/xenium_crop.h5ad")
    xy = a.obsm["spatial"]
    labs_full = a.obs["celltype"].to_numpy()
    best = (-1.0, None)
    for x0 in np.arange(xy[:, 0].min(), xy[:, 0].max() - SUBWINDOW, 500):
        for y0 in np.arange(xy[:, 1].min(), xy[:, 1].max() - SUBWINDOW, 500):
            m = ((xy[:, 0] >= x0) & (xy[:, 0] < x0 + SUBWINDOW)
                 & (xy[:, 1] >= y0) & (xy[:, 1] < y0 + SUBWINDOW))
            if m.sum() < 4000:
                continue
            vc = pd.Series(labs_full[m]).value_counts(normalize=True)
            trio = min(vc.get("Cancer Epithelial", 0), vc.get("T-cells", 0),
                       vc.get("CAFs", 0))
            if trio > best[0]:
                best = (trio, (x0, y0))
    x0, y0 = best[1]
    m = ((xy[:, 0] >= x0) & (xy[:, 0] < x0 + SUBWINDOW)
         & (xy[:, 1] >= y0) & (xy[:, 1] < y0 + SUBWINDOW))
    sub = a[m].copy()

    X_raw = sub.X.toarray().astype(np.float32)
    lib = X_raw.sum(1, keepdims=True)
    Xn_full = np.log1p(X_raw / np.maximum(lib, 1) * np.median(lib))
    genes_full = np.array(sub.var_names)
    labs = sub.obs["celltype"].to_numpy()
    coords = sub.obsm["spatial"]
    gidx_full = {g: i for i, g in enumerate(genes_full)}

    SIGNAL = ["CXCL12", "CXCR4", "CD274", "PDCD1", "PDCD1LG2", "CD80", "CD86",
              "CTLA4", "TIGIT", "LAG3", "HAVCR2", "CCL5", "CCR7", "CXCL16",
              "CCL20", "CCL8", "CXCL5", "LTB", "PTN", "SDC4", "IL7R", "IL2RA",
              "IL2RG", "IL3RA", "KIT", "IGF1", "EGFR", "ERBB2", "KDR",
              "TNFRSF17", "SLAMF1", "SLAMF7", "GPR183", "CX3CR1", "FCER1G", "TYROBP"]
    sig_idx = [gidx_full[g] for g in SIGNAL if g in gidx_full]
    Xn = Xn_full[:, sig_idx]
    genes = genes_full[sig_idx]
    gidx = {g: i for i, g in enumerate(genes)}
    mu, sd = Xn.mean(0), Xn.std(0) + 1e-6
    Xz = ((Xn - mu) / sd).astype(np.float32)
    zero_val = (0.0 - mu) / sd                       # z-scored value of count=0

    is_t = labs == "T-cells"
    tree = cKDTree(coords[labs == "Cancer Epithelial"])
    d_near, _ = tree.query(coords)
    t_act = np.zeros(len(labs), dtype=int)
    t_act[is_t] = (d_near[is_t] < 30.0).astype(int)

    edges = build_knn_graph(coords, k=K_GRAPH)
    adj_norm = build_norm_adj(edges, len(labs))
    x = torch.from_numpy(Xz)
    y = torch.from_numpy(t_act.astype(np.int64))
    tmask = torch.from_numpy(is_t)
    HOST_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    torch.manual_seed(HOST_SEED)
    gcn = GCN(n_feat=len(genes), n_hidden=64, n_classes=2)
    opt = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)
    gcn.train()
    for ep in range(200):
        opt.zero_grad()
        loss = F.cross_entropy(gcn(x, adj_norm)[tmask], y[tmask])
        loss.backward()
        opt.step()
    gcn.eval()
    acc = (gcn(x, adj_norm).argmax(1)[is_t] == y[is_t]).float().mean().item()
    print(f"LR-restricted host ({len(genes)} genes), T acc = {acc:.4f}")

    padj = adjacency_list(edges, len(labs))
    receivers = np.array([i for i in np.where(is_t)[0]
                          if any(labs[j] != "T-cells" for j in padj[i])])
    rng = np.random.default_rng(0)

    @torch.no_grad()
    def logit_at(j, gene_to_zero=None):
        one = set(padj[j]) | {j}
        two = set(one)
        for u in one:
            two |= padj[u]
        zone = np.array(sorted(two), dtype=np.int64)
        xz = x[zone].clone()
        if gene_to_zero is not None:
            xz[:, gene_to_zero] = torch.from_numpy(
                np.full(len(zone), zero_val[gene_to_zero], dtype=np.float32))
        adj_sub = adj_norm[zone][:, zone]
        zidx = {int(n): k for k, n in enumerate(zone)}
        return float(gcn(xz.unsqueeze(0), adj_sub)[0, zidx[int(j)], 1].item())

    print(f"{'pathway':<18} {'n_recv':>7} {'psi_gene':>9} {'null_mean':>10} {'t(paired)':>9}")
    saved = {}
    for lig, rec in LR_PAIRS:
        if lig not in gidx or rec not in gidx:
            continue
        rec_expr = Xn_full[:, gidx_full[rec]] > 0
        use = [int(j) for j in receivers if rec_expr[j]
               and any(Xn_full[i, gidx_full[lig]] > 0 for i in padj[j])]
        if len(use) < 20:
            print(f"{lig+'->'+rec:<18} (only {len(use)} receivers, skipped)")
            continue
        obs = np.array([logit_at(j) - logit_at(j, gidx[lig]) for j in use])
        freq = (Xn_full[:, gidx_full[lig]] > 0).mean()
        cand = [c for c in range(len(genes))
                if genes[c] != lig and abs((Xn_full[:, gidx_full[genes[c]]] > 0).mean() - freq) < 0.05]
        nulls = rng.choice(cand, size=min(8, len(cand)), replace=False)
        null_mat = np.array([[logit_at(j) - logit_at(j, c) for j in use] for c in nulls])
        null_per_recv = null_mat.mean(0)                      # per-receiver null mean
        diff = obs - null_per_recv                            # paired differences
        t_stat = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)) + 1e-12)
        print(f"{lig+'->'+rec:<18} {len(use):>7} {obs.mean():>+9.4f} "
              f"{null_per_recv.mean():>+10.4f} {t_stat:>6.1f}")
        saved[lig] = {"use": np.array(use), "obs": obs,
                      "null": null_per_recv, "t": t_stat}

    # ---- CXCL12 sender-type breakdown: zero CXCL12 only in a given sender type
    lig = "CXCL12"
    if lig in saved:
        use = saved[lig]["use"]
        print("-" * 72)
        print("CXCL12 effect by sender cell type (zero CXCL12 only in that type):")
        for st in ["CAFs", "Cancer Epithelial", "Myeloid", "Normal Epithelial"]:
            st_mask = labs == st
            obs_st = []
            for j in use:
                one = set(padj[j]) | {j}
                two = set(one)
                for u in one:
                    two |= padj[u]
                zone = np.array(sorted(two), dtype=np.int64)
                if not st_mask[zone].any():
                    continue
                xz = x[zone].clone()
                xz[st_mask[zone], gidx[lig]] = torch.from_numpy(
                    np.full(int(st_mask[zone].sum()), zero_val[gidx[lig]],
                            dtype=np.float32))
                adj_sub = adj_norm[zone][:, zone]
                zidx = {int(n): k for k, n in enumerate(zone)}
                with torch.no_grad():
                    v0 = logit_at(j)
                    v1 = float(gcn(xz.unsqueeze(0), adj_sub)[0, zidx[int(j)], 1].item())
                obs_st.append(v0 - v1)
            if len(obs_st) >= 20:
                obs_st = np.array(obs_st)
                t_st = obs_st.mean() / (obs_st.std(ddof=1) / np.sqrt(len(obs_st)) + 1e-12)
                print(f"  {st:<20} n={len(obs_st):>5}  mean_psi={obs_st.mean():+.4f}  t={t_st:+.1f}")
        saved["sender_breakdown"] = True

    import os
    os.makedirs("data/processed", exist_ok=True)
    np.savez("data/processed/e10e_ccxcl12.npz",
             use=saved["CXCL12"]["use"], obs=saved["CXCL12"]["obs"],
             null=saved["CXCL12"]["null"], labs=labs, coords=coords,
             is_t=is_t, t_act=t_act, genes=genes, d_near=d_near,
             x_cxcl12=Xn_full[:, gidx_full["CXCL12"]])
    print(f"\nsaved data/processed/e10e_ccxcl12.npz; total {(time.perf_counter()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
