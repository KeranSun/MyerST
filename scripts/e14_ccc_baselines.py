"""E14: CCC method comparison — MyerST vs established pipelines (Rep1 crop).

Baselines on the SAME receivers and LR pairs:
  CPDB-style   : mean(L in sender-type) x mean(R in receivers), cell-type
                 label permutation p-value (100 permutations)
  NicheNet-lite: ligand activity = Pearson corr between a receiver's
                 neighborhood ligand expression and its target state
                 (cytotoxic score), across receivers
  MyerST       : E10e gene-level occlusion psi (host-based, per-receiver)

Question: do the baselines detect the CXCL12 exclusion signal, and at what
resolution? (They cannot be sign-informative about retention vs attraction
without a host model — that is the comparison's point.)
"""

import os
import time

import numpy as np
import pandas as pd
import anndata as ad
from scipy.spatial import cKDTree

CYTO = ["GZMB", "PRF1", "NKG7", "GNLY"]
LR_PAIRS = [("CXCL12", "CXCR4"), ("CD274", "PDCD1"), ("CD80", "CTLA4"),
            ("CD86", "CTLA4"), ("PTN", "SDC4")]
N_PERM = 100


def main():
    t0 = time.perf_counter()
    print("=" * 72)
    print("E14: CCC method comparison on Xenium breast Rep1 interface")
    print("=" * 72)

    a = ad.read_h5ad("data/processed/xenium_crop.h5ad")
    xy = a.obsm["spatial"]
    labs = a.obs["celltype"].to_numpy()
    # same window-finding as E10
    best = (-1.0, None)
    for x0 in np.arange(xy[:, 0].min(), xy[:, 0].max() - 1500, 500):
        for y0 in np.arange(xy[:, 1].min(), xy[:, 1].max() - 1500, 500):
            m = ((xy[:, 0] >= x0) & (xy[:, 0] < x0 + 1500)
                 & (xy[:, 1] >= y0) & (xy[:, 1] < y0 + 1500))
            if m.sum() < 4000:
                continue
            vc = pd.Series(labs[m]).value_counts(normalize=True)
            trio = min(vc.get("Cancer Epithelial", 0), vc.get("T-cells", 0),
                       vc.get("CAFs", 0))
            if trio > best[0]:
                best = (trio, (x0, y0))
    x0, y0 = best[1]
    m = ((xy[:, 0] >= x0) & (xy[:, 0] < x0 + 1500)
         & (xy[:, 1] >= y0) & (xy[:, 1] < y0 + 1500))
    X = a.X[m].toarray().astype(np.float32)
    labs = labs[m]
    coords = xy[m]
    genes = np.array(a.var_names)
    gidx = {g: i for i, g in enumerate(genes)}
    lib = X.sum(1, keepdims=True)
    Xn = np.log1p(X / np.maximum(lib, 1) * np.median(lib))
    print(f"window cells {len(labs)}; {pd.Series(labs).value_counts().head(4).to_dict()}")

    is_t = labs == "T-cells"
    tree = cKDTree(coords[labs == "Cancer Epithelial"])
    d_near, _ = tree.query(coords)
    infil = (d_near < 30.0).astype(float)

    # neighborhood ligand level per receiver (kNN mean over non-T neighbors)
    tree_all = cKDTree(coords)
    _, nn = tree_all.query(coords, k=9)
    nn = nn[:, 1:]
    cyto = Xn[:, [gidx[g] for g in CYTO]].mean(1)

    print(f"\n{'pathway':<18} {'CPDB p':>8} {'NicheNet r':>11} {'MyerST psi':>11} {'MyerST t':>9}")
    e10 = np.load("data/processed/e10e_ccxcl12.npz", allow_pickle=True)
    myerst_psi = float(e10["obs"].mean())
    rng = np.random.default_rng(0)

    for lig, rec in LR_PAIRS:
        if lig not in gidx or rec not in gidx:
            continue
        # ---- CPDB-style: mean L in CAFs x mean R in T, permutation over labels
        senders = labs == "CAFs"
        obs_score = Xn[senders, gidx[lig]].mean() * Xn[is_t, gidx[rec]].mean()
        null = []
        for _ in range(N_PERM):
            perm = rng.permutation(labs)
            null.append(Xn[perm == "CAFs", gidx[lig]].mean()
                        * Xn[perm == "T-cells", gidx[rec]].mean())
        null = np.array(null)
        p_cpdb = (np.sum(null >= obs_score) + 1) / (N_PERM + 1)

        # ---- NicheNet-lite: corr(neighborhood L, receiver cytotoxicity)
        nb_lig = Xn[nn, gidx[lig]].mean(1)
        rec_expr = Xn[:, gidx[rec]] > 0
        sel = is_t & rec_expr
        if sel.sum() < 20:
            print(f"{lig+'->'+rec:<18} (receivers < 20, skipped)")
            continue
        r_nn = np.corrcoef(nb_lig[sel], cyto[sel])[0, 1]
        r_loc = np.corrcoef(nb_lig[sel], infil[sel])[0, 1]

        print(f"{lig+'->'+rec:<18} {p_cpdb:>8.3f} {r_nn:>+11.3f} "
              f"{'(r_loc %+.3f)' % r_loc:>11} ", flush=True)

    print(f"\nMyerST reference (E10e): CXCL12 psi = {myerst_psi:+.4f} (t=-38.9)")
    print("note: CPDB/NicheNet scores are sign-ambiguous for retention vs "
          "attraction; only host-based psi distinguishes direction.")
    print(f"\ntotal {(time.perf_counter()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
