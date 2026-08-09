"""E11: multi-cohort CCC replication — breast Rep2 + lung (R6 adaptability).

Same LR-restricted host + infiltration target + gene-level ligand occlusion
pipeline as E10e, applied unchanged to:
  1. breast_rep2 (official 20-type annotation, role-mapped)
  2. lung (marker-based annotation; immuno-oncology panel)
Saves per-cohort npz for the multi-cohort figure.
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import scanpy as sc
import anndata as ad
from scipy.spatial import cKDTree

from myerst.models.gcn import GCN, build_norm_adj
from myerst.data.graph import build_knn_graph, adjacency_list

SIGNAL = ["CXCL12", "CXCR4", "CD274", "PDCD1", "PDCD1LG2", "CD80", "CD86",
          "CTLA4", "TIGIT", "LAG3", "HAVCR2", "CCL5", "CCR7", "CXCL16",
          "CCL20", "CCL8", "CXCL5", "LTB", "PTN", "SDC4", "IL7R", "IL2RA",
          "IL2RG", "IL3RA", "KIT", "IGF1", "EGFR", "ERBB2", "KDR",
          "TNFRSF17", "SLAMF1", "SLAMF7", "GPR183", "CX3CR1", "FCER1G", "TYROBP"]
LR_PAIRS = [("CXCL12", "CXCR4"), ("CD274", "PDCD1"), ("PDCD1LG2", "PDCD1"),
            ("CD80", "CTLA4"), ("CD86", "CTLA4"), ("PTN", "SDC4")]
MARKER_RULES = {  # role -> marker genes (lung annotation)
    "Cancer Epithelial": ["EPCAM", "KRT8", "KRT7", "KRT18"],
    "T-cells": ["CD3D", "CD3E", "CD3G"],
    "CAFs": ["ACTA2", "COL1A1", "COL1A2", "FAP"],
    "Myeloid": ["AIF1", "CD68", "CD14", "ITGAM"],
    "Endothelial": ["PECAM1", "VWF"],
    "B-cells": ["MS4A1", "CD79A"],
}
K_GRAPH = 8
SUBW = 1200.0


def marker_annotate(Xn, genes):
    gidx = {g: i for i, g in enumerate(genes)}
    scores = {}
    for role, marks in MARKER_RULES.items():
        present = [gidx[m] for m in marks if m in gidx]
        scores[role] = Xn[:, present].mean(1) if present else np.zeros(Xn.shape[0])
    S = np.stack(list(scores.values()), 1)
    roles = list(scores)
    lab = np.array([roles[i] for i in S.argmax(1)], dtype=object)
    lab[S.max(1) < 0.5] = "Other"
    return lab


def best_window(xy, labs, roles=("Cancer Epithelial", "T-cells", "CAFs"),
                subw=SUBW, min_cells=3000):
    best = (-1.0, None)
    step = subw * 0.4
    for x0 in np.arange(xy[:, 0].min(), xy[:, 0].max() - subw, step):
        for y0 in np.arange(xy[:, 1].min(), xy[:, 1].max() - subw, step):
            m = ((xy[:, 0] >= x0) & (xy[:, 0] < x0 + subw)
                 & (xy[:, 1] >= y0) & (xy[:, 1] < y0 + subw))
            if m.sum() < min_cells:
                continue
            vc = pd.Series(labs[m]).value_counts(normalize=True)
            trio = min(vc.get(r, 0) for r in roles)
            if trio > best[0]:
                best = (trio, (x0, y0))
    return best[1]


def run_cohort(name, adata, labs, coords, genes_full,
               subw=SUBW, min_cells=3000, infil_dist=30.0, k_graph=K_GRAPH):
    t0 = time.perf_counter()
    print(f"\n{'='*72}\nCOHORT: {name}\n{'='*72}", flush=True)
    w = best_window(coords, labs, subw=subw, min_cells=min_cells)
    m = ((coords[:, 0] >= w[0]) & (coords[:, 0] < w[0] + subw)
         & (coords[:, 1] >= w[1]) & (coords[:, 1] < w[1] + subw))
    labs, coords = labs[m], coords[m]
    X_raw = adata.X[m].toarray().astype(np.float32)
    lib = X_raw.sum(1, keepdims=True)
    Xn_full = np.log1p(X_raw / np.maximum(lib, 1) * np.median(lib))
    gidx_full = {g: i for i, g in enumerate(genes_full)}
    sig_idx = [gidx_full[g] for g in SIGNAL if g in gidx_full]
    Xn = Xn_full[:, sig_idx]
    genes = genes_full[sig_idx]
    gidx = {g: i for i, g in enumerate(genes)}
    mu, sd = Xn.mean(0), Xn.std(0) + 1e-6
    Xz = ((Xn - mu) / sd).astype(np.float32)
    zero_val = (0.0 - mu) / sd
    print(f"window ({w[0]:.0f},{w[1]:.0f}): {len(labs)} cells; "
          f"{pd.Series(labs).value_counts().head(3).to_dict()}; "
          f"signal genes {len(sig_idx)}")

    is_t = labs == "T-cells"
    is_cancer = labs == "Cancer Epithelial"
    tree = cKDTree(coords[is_cancer])
    d_near, _ = tree.query(coords)
    t_act = np.zeros(len(labs), dtype=int)
    t_act[is_t] = (d_near[is_t] < infil_dist).astype(int)
    print(f"T-cells {is_t.sum()}, infiltrated {t_act[is_t].sum()}")

    edges = build_knn_graph(coords, k=k_graph)
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
        loss = F.cross_entropy(gcn(x, adj_norm)[tmask], y[tmask])
        loss.backward()
        opt.step()
    gcn.eval()
    acc = (gcn(x, adj_norm).argmax(1)[is_t] == y[is_t]).float().mean().item()
    print(f"LR-restricted host T acc = {acc:.4f}")

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
        zidx = {int(n): k for k, n in enumerate(zone)}
        return float(gcn(xz.unsqueeze(0), adj_norm[zone][:, zone])[0, zidx[int(j)], 1].item())

    print(f"{'pathway':<18} {'n_recv':>7} {'psi_gene':>9} {'null_mean':>10} {'t(paired)':>9}")
    rows = []
    for lig, rec in LR_PAIRS:
        if lig not in gidx or rec not in gidx:
            continue
        rec_expr = Xn_full[:, gidx_full[rec]] > 0
        use = [int(j) for j in receivers if rec_expr[j]
               and any(Xn_full[i, gidx_full[lig]] > 0 for i in padj[j])]
        if len(use) < 20:
            continue
        obs = np.array([logit_at(j) - logit_at(j, gidx[lig]) for j in use])
        freq = (Xn_full[:, gidx_full[lig]] > 0).mean()
        cand = [c for c in range(len(genes))
                if genes[c] != lig and abs((Xn_full[:, gidx_full[genes[c]]] > 0).mean() - freq) < 0.05]
        nulls = rng.choice(cand, size=min(8, len(cand)), replace=False)
        null_mat = np.array([[logit_at(j) - logit_at(j, c) for j in use] for c in nulls])
        null_pr = null_mat.mean(0)
        diff = obs - null_pr
        t_stat = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)) + 1e-12)
        rows.append((f"{lig}->{rec}", len(use), obs.mean(), null_pr.mean(), t_stat))
        print(f"{lig+'->'+rec:<18} {len(use):>7} {obs.mean():>+9.4f} "
              f"{null_pr.mean():>+10.4f} {t_stat:>6.1f}")

    # CXCL12 details for the figure
    saved = {}
    if any(r[0] == "CXCL12->CXCR4" for r in rows):
        lig, rec = "CXCL12", "CXCR4"
        rec_expr = Xn_full[:, gidx_full[rec]] > 0
        use = np.array([int(j) for j in receivers if rec_expr[j]
                        and any(Xn_full[i, gidx_full[lig]] > 0 for i in padj[j])])
        obs = np.array([logit_at(j) - logit_at(j, gidx[lig]) for j in use])
        saved = dict(use=use, obs=obs)
    np.savez(f"data/processed/e11_{name}.npz", labs=labs, coords=coords,
             is_t=is_t, t_act=t_act, genes=genes,
             x_cxcl12=Xn_full[:, gidx_full["CXCL12"]] if "CXCL12" in gidx_full else np.zeros(len(labs)),
             pathways=np.array([(r[0], r[2], r[4]) for r in rows], dtype=object),
             **saved)
    print(f"saved data/processed/e11_{name}.npz ({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
    return rows


def load_cohort(name):
    raw = f"data/raw/{name}"
    adata = sc.read_10x_h5(f"{raw}/cell_feature_matrix.h5")
    cells = pd.read_csv(f"{raw}/cells.csv.gz")
    adata.obs["cell_id"] = adata.obs_names.astype(str)
    cells_idx = cells.set_index(cells["cell_id"].astype(str))
    coords = cells_idx.loc[adata.obs["cell_id"], ["x_centroid", "y_centroid"]].to_numpy()
    genes_full = np.array(adata.var_names)
    Xn_for_annot = None
    if name == "xenium_breast_rep2":
        ct = pd.read_csv(f"{raw}/xenium_rep2_celltype_major.csv")
        ct.columns = [c.strip() for c in ct.columns]
        id_c = [c for c in ct.columns if "barcode" in c.lower() or "cell" in c.lower()][0]
        lab_c = [c for c in ct.columns if c != id_c][-1]
        lab_map = dict(zip(ct[id_c].astype(str), ct[lab_c]))
        labs = adata.obs["cell_id"].map(lab_map).to_numpy(dtype=object)
        # role mapping to canonical names
        role_map = {"Invasive_Tumor": "Cancer Epithelial", "DCIS_1": "Cancer Epithelial",
                    "DCIS_2": "Cancer Epithelial",
                    "Prolif_Invasive_Tumor": "Cancer Epithelial",
                    "CD4+_T_Cells": "T-cells", "CD8+_T_Cells": "T-cells",
                    "T cells": "T-cells", "T_cells": "T-cells",
                    "Stromal": "CAFs",
                    "Macrophages_1": "Myeloid", "Macrophages_2": "Myeloid",
                    "IRF7+_DCs": "Myeloid", "LAMP3+_DCs": "Myeloid",
                    "Mast_Cells": "Myeloid",
                    "Macrophages": "Myeloid", "Myeloid": "Myeloid",
                    "Endothelial": "Endothelial",
                    "B_Cells": "B-cells", "B cells": "B-cells", "B_cells": "B-cells"}
        labs = np.array([role_map.get(str(l), "Other") for l in labs], dtype=object)
    else:  # lung: marker annotation
        X = adata.X.toarray().astype(np.float32)
        lib = X.sum(1, keepdims=True)
        Xn = np.log1p(X / np.maximum(lib, 1) * np.median(lib))
        labs = marker_annotate(Xn, genes_full)
    return adata, labs, coords, genes_full


def main():
    for name in ["xenium_breast_rep2", "xenium_lung"]:
        adata, labs, coords, genes_full = load_cohort(name)
        print(f"{name}: {adata.shape}, annotation: {pd.Series(labs).value_counts().head(6).to_dict()}")
        run_cohort(name, adata, labs, coords, genes_full)


if __name__ == "__main__":
    main()
