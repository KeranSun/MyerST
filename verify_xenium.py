import os
import h5py
import numpy as np
import pandas as pd

KEY_GENES = ["CXCL12", "CXCR4", "CD3D", "CD8A", "EPCAM", "CD68", "ACTA2", "MS4A1", "PECAM1"]

DATASETS = {
    "breast_rep2": "C:/Users/Sun/WorkBuddy/2026-08-03-11-16-19/MyerST/data/raw/xenium_breast_rep2",
    "lung": "C:/Users/Sun/WorkBuddy/2026-08-03-11-16-19/MyerST/data/raw/xenium_lung",
}


def read_10x_h5(path):
    with h5py.File(path, "r") as f:
        grp = f["matrix"]
        data = grp["data"][:]
        shape = grp["shape"][:]
        genes = [g.decode() for g in grp["features"]["name"][:]]
        barcodes = [b.decode() for b in grp["barcodes"][:]]
    # 10x h5 matrix is CSC: shape = (n_genes, n_cells)
    n_genes, n_cells = int(shape[0]), int(shape[1])
    return n_cells, n_genes, genes, barcodes, data


def report(name, d):
    print("=" * 70)
    print(f"DATASET: {name}  ({d})")
    print("-" * 70)
    for fn in sorted(os.listdir(d)):
        fp = os.path.join(d, fn)
        print(f"  {fp}  {os.path.getsize(fp):,} bytes")

    h5 = os.path.join(d, "cell_feature_matrix.h5")
    n_cells, n_genes, genes, barcodes, data = read_10x_h5(h5)
    print(f"\n[expression] cells x genes = {n_cells} x {n_genes}, nonzero entries = {data.size:,}")

    cells = pd.read_csv(os.path.join(d, "cells.csv.gz"))
    print(f"[cells.csv.gz] rows = {len(cells)}, columns = {cells.columns.tolist()}")
    coord_cols = [c for c in cells.columns if "centroid" in c.lower() or c.lower() in ("x", "y")]
    print(f"[coords] coordinate columns: {coord_cols}")

    print(f"[genes] first 20: {genes[:20]}")
    present = [g for g in KEY_GENES if g in genes]
    missing = [g for g in KEY_GENES if g not in genes]
    print(f"[key genes] present: {present}")
    print(f"[key genes] MISSING: {missing}")

    # annotation
    anno_csv = os.path.join(d, "xenium_rep2_celltype_major.csv")
    if os.path.exists(anno_csv):
        anno = pd.read_csv(anno_csv)
        print(f"[annotation] {anno.shape[0]} rows, cols={anno.columns.tolist()}, "
              f"n_types={anno['Cluster'].nunique()}")
        # check overlap with cell ids
        cid = cells.columns[0]
        bc_digits = set()
        for b in barcodes[:2000]:
            digits = "".join(ch for ch in b if ch.isdigit())
            bc_digits.add(int(digits) if digits else -1)
        overlap = len(bc_digits & set(anno["Barcode"].head(200000)))
        print(f"[annotation] barcode overlap (sample of 2000): {overlap}/2000")
        print(f"[annotation] top types: {anno['Cluster'].value_counts().head(8).to_dict()}")
    else:
        print("[annotation] no cell-type annotation file (will use marker-based annotation)")
    print()


for name, d in DATASETS.items():
    report(name, d)
