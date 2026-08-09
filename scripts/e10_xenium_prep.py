"""E10 prep: load Xenium breast cancer Rep1, merge cell types, crop the
tumor-immune interface region for the flagship CCC case.

Outputs data/processed/xenium_crop.h5ad (raw counts, coords, celltype).
"""

import os

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

RAW = "data/raw/xenium_breast"
OUT = "data/processed/xenium_crop.h5ad"
WINDOW = 2500.0     # um
STEP = 1000.0       # um


def main():
    print("loading expression matrix...", flush=True)
    adata = sc.read_10x_h5(f"{RAW}/cell_feature_matrix.h5")
    cells = pd.read_csv(f"{RAW}/cells.csv.gz")
    ct = pd.read_csv(f"{RAW}/xenium_rep1_celltype_major.csv")
    print(f"matrix {adata.shape}; cells.csv {cells.shape}; celltype table {ct.shape}")

    # ---- harmonize ids
    print("cells.csv cols:", list(cells.columns)[:12])
    print("celltype cols:", list(ct.columns))
    id_col = "cell_id" if "cell_id" in cells.columns else cells.columns[0]
    ct_id = "cell_id" if "cell_id" in ct.columns else ct.columns[0]
    ct_label = "celltype_major" if "celltype_major" in ct.columns else \
        [c for c in ct.columns if c != ct_id and "type" in c.lower()][0]
    cells[id_col] = cells[id_col].astype(str)
    ct[ct_id] = ct[ct_id].astype(str)
    lab_map = dict(zip(ct[ct_id], ct[ct_label]))

    # Xenium h5 obs_names are typically the integer cell ids as strings
    adata.obs["cell_id"] = adata.obs_names.astype(str)
    adata.obs["celltype"] = adata.obs["cell_id"].map(lab_map)
    cells_indexed = cells.set_index(cells[id_col].astype(str))
    adata.obs["x"] = cells_indexed.loc[adata.obs["cell_id"], "x_centroid"].to_numpy()
    adata.obs["y"] = cells_indexed.loc[adata.obs["cell_id"], "y_centroid"].to_numpy()
    adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy()
    n_lab = adata.obs["celltype"].notna().sum()
    print(f"celltype matched: {n_lab}/{adata.n_obs}")
    print(adata.obs["celltype"].value_counts().to_string())

    # ---- pick the window with richest cancer/immune/CAF mixing
    keep = adata.obs["celltype"].notna() & np.isfinite(adata.obs["x"])
    xy = adata.obsm["spatial"][keep.to_numpy()]
    labs = adata.obs.loc[keep, "celltype"].to_numpy()
    best = (-1.0, None)
    for x0 in np.arange(xy[:, 0].min(), xy[:, 0].max() - WINDOW, STEP):
        for y0 in np.arange(xy[:, 1].min(), xy[:, 1].max() - WINDOW, STEP):
            m = ((xy[:, 0] >= x0) & (xy[:, 0] < x0 + WINDOW)
                 & (xy[:, 1] >= y0) & (xy[:, 1] < y0 + WINDOW))
            if m.sum() < 3000:
                continue
            vc = pd.Series(labs[m]).value_counts(normalize=True)
            def frac(t):
                return vc.get(t, 0.0)
            # need all three players present; maximize min-share * mixing entropy
            trio = min(frac("Cancer Epithelial"), frac("T-cells"), frac("CAFs"))
            ent = -(vc * np.log(vc + 1e-12)).sum()
            score = trio * ent
            if score > best[0]:
                best = (score, (x0, y0, m.sum(), vc.to_dict()))
    score, (x0, y0, n_cells, dist) = best
    print(f"best window ({x0:.0f},{y0:.0f}) cells={n_cells} score={score:.3f}")
    print("  composition:", {k: round(v, 3) for k, v in dist.items()})

    m_all = ((adata.obs["x"] >= x0) & (adata.obs["x"] < x0 + WINDOW)
             & (adata.obs["y"] >= y0) & (adata.obs["y"] < y0 + WINDOW)
             & adata.obs["celltype"].notna())
    crop = adata[m_all].copy()
    crop.obsm["spatial"] = crop.obs[["x", "y"]].to_numpy()
    os.makedirs("data/processed", exist_ok=True)
    crop.write_h5ad(OUT)
    print(f"saved {OUT}: {crop.shape}")
    # key genes sanity
    for g in ["CXCL12", "CXCR4", "EPCAM", "CD3D", "CD8A", "CD68", "ERBB2"]:
        print(f"  {g}: {'present' if g in crop.var_names else 'MISSING'}")


if __name__ == "__main__":
    main()
