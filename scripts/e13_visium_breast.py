"""E13: Visium breast cancer — cross-platform CCC validation.

V1_Breast_Cancer_Block_A_Section_1 (fresh frozen, 3798 spots x 36601 genes).
No official annotation -> marker-based spot annotation (spots are multi-cell
mixtures, so labels are coarse "dominant type" assignments).

Coordinates are normalized so median nearest-neighbor distance = 1 (Visium
spot spacing ~100um), then the SAME run_cohort pipeline from E11 runs with
spot-scale parameters (infiltration = adjacent cancer spot, dist 1.5).
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import cKDTree

from scripts.e11_multi_cohort import run_cohort, marker_annotate


def main():
    print("=" * 72)
    print("E13: Visium breast cancer cross-platform CCC")
    print("=" * 72)

    a = sc.read_visium("data/raw/visium_breast",
                       count_file="filtered_feature_bc_matrix.h5")
    a = a[a.obs["in_tissue"].astype(int) == 1].copy()
    coords = np.asarray(a.obsm["spatial"], dtype=float)
    # normalize coords: median NN distance = 1 (one Visium spot spacing ~100um)
    tree0 = cKDTree(coords)
    d_nn, _ = tree0.query(coords, k=2)
    scale = np.median(d_nn[:, 1])
    coords = coords / scale
    print(f"spots {a.n_obs}, genes {a.n_vars}, NN spacing = {scale:.1f} px")

    X = a.X.toarray().astype(np.float32)
    lib = X.sum(1, keepdims=True)
    Xn = np.log1p(X / np.maximum(lib, 1) * np.median(lib))
    genes_full = np.array(a.var_names)

    # spot-level dominant-type annotation: z-scored marker means, argmax;
    # no absolute threshold (spots are multi-cell mixtures)
    from scripts.e11_multi_cohort import MARKER_RULES
    gidx = {g: i for i, g in enumerate(genes_full)}
    z = (Xn - Xn.mean(0)) / (Xn.std(0) + 1e-6)
    roles = list(MARKER_RULES)
    S = np.stack([
        z[:, [gidx[m] for m in MARKER_RULES[r] if m in gidx]].mean(1)
        if any(m in gidx for m in MARKER_RULES[r]) else np.full(a.n_obs, -9.0)
        for r in roles], 1)
    labs = np.array([roles[i] for i in S.argmax(1)], dtype=object)
    labs[S.max(1) < 0.0] = "Other"
    print("annotation:", pd.Series(labs).value_counts().to_dict())

    run_cohort("visium_breast", a, labs, coords, genes_full,
               subw=22.0, min_cells=200, infil_dist=1.5, k_graph=6)


if __name__ == "__main__":
    main()
