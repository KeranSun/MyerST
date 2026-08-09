"""E12: cross-sample boundary attribution on all 12 DLPFC slices (R6 depth).

Per slice: seurat_v3 HVG -> GCN host -> IG + Occlusion gene attribution on the
L5|L6 boundary -> known-marker hits + cross-slice Spearman consistency.
This upgrades Fig 3 from "one slice" to "12 slices across 3 donors".
"""

import os
import time

import numpy as np
import anndata as ad
import scanpy as sc
import torch
from scipy.stats import spearmanr

from myerst.data.spadata import SpaData
from myerst.models.gcn import GCN, build_norm_adj, train_gcn
from myerst.adapters.torch_adapter import TorchModelAdapter
from myerst.adapters.base import ExplanationTarget
from myerst.explainers.ig import IGExplainer
from myerst.explainers.occlusion import SpatialOcclusion
from scripts.e1_driver_gene_recovery import boundary_spots
from scripts.tune_stagate_seurat_hvg import ORDER

SLICES = ["151507", "151508", "151509", "151510", "151669", "151670",
          "151671", "151672", "151673", "151674", "151675", "151676"]
MARKERS = {"PCP4", "KRT17", "RORB", "AQP4", "GFAP", "CUX2", "NEFH",
           "KRT5", "PLP1", "MBP", "CCK", "HOPX"}
BOUNDARY = ("L5", "L6")


def load_slice(sid):
    a = ad.read_h5ad(f"data/raw/DLPFC_{sid}.h5ad")
    layer = a.obs["layer"].astype(str).to_numpy()
    keep = np.isin(layer, ORDER)          # drops 'NA', 'nan', unknown
    a = a[keep].copy()
    layer = layer[keep]
    sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=3000,
                                batch_key=None)
    a = a[:, a.var["highly_variable"]].copy()
    X = a.X.toarray() if not isinstance(a.X, np.ndarray) else a.X
    X = np.log1p(X)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xz = ((X - mu) / sd).astype(np.float32)
    return Xz, np.asarray(a.obsm["spatial"], dtype=float), layer, \
        np.array(a.var_names), ORDER


def main():
    t0 = time.perf_counter()
    print("=" * 72)
    print("E12: 12-slice DLPFC boundary attribution (IG + Occlusion)")
    print("=" * 72)
    all_ig = {}
    rows = []
    os.makedirs("data/processed", exist_ok=True)
    for sid in SLICES:
        t1 = time.perf_counter()
        cache_path = f"data/processed/e12_{sid}.npz"
        if os.path.exists(cache_path):
            c = np.load(cache_path, allow_pickle=True)
            all_ig[sid] = (c["genes"], c["ig"])
            rows.append((sid, float(c["acc"]), int(c["n_boundary"]),
                         list(c["hits"])))
            print(f"{sid}: loaded from cache")
            continue
        path = f"data/raw/DLPFC_{sid}.h5ad"
        if not os.path.exists(path):
            print(f"{sid}: missing, skipped")
            continue
        Xz, coords, layer, genes, order = load_slice(sid)
        if not (BOUNDARY[0] in layer and BOUNDARY[1] in layer):
            print(f"{sid}: boundary layers absent, skipped")
            continue
        labels = np.array([order.index(l) for l in layer])
        data = SpaData(X=Xz, coords=coords, labels=labels,
                       gene_names=genes.tolist())
        data.build_graph(k=6)
        A, B = order.index(BOUNDARY[0]), order.index(BOUNDARY[1])
        bspots = boundary_spots(data.labels, data.adj, A, B)
        x = torch.from_numpy(Xz)
        adj_norm = build_norm_adj(data.edges, data.n_spots)
        n_cls = int(labels.max()) + 1           # some slices lack layers
        gcn = GCN(n_feat=data.n_genes, n_hidden=64, n_classes=n_cls)
        train_gcn(gcn, x, adj_norm, labels, epochs=150, seed=0)
        with torch.no_grad():
            acc = (gcn(x, adj_norm).argmax(1) == torch.from_numpy(labels)).float().mean().item()
        adapter = TorchModelAdapter(gcn, adj_norm, boundary_spots={(A, B): bspots})
        target = ExplanationTarget(kind="domain_boundary", payload=(A, B))
        baseline_x = torch.from_numpy(data.domain_mean().astype(np.float32))
        ig = IGExplainer(30).explain(adapter, x, target, baseline=baseline_x).node_scores
        occ = SpatialOcclusion(batch_size=16).explain(adapter, x, target, baseline=baseline_x).node_scores
        top20 = genes[np.argsort(ig)[::-1][:20]]
        hits = sorted(set(top20) & MARKERS)
        all_ig[sid] = (genes, ig)
        rows.append((sid, acc, len(bspots), hits))
        np.savez(cache_path, genes=genes, ig=ig, occ=occ, acc=acc,
                 n_boundary=len(bspots), hits=np.array(hits, dtype=object))
        print(f"{sid}: acc={acc:.3f} boundary={len(bspots)} marker hits "
              f"{len(hits)} {hits} ({(time.perf_counter()-t1)/60:.1f} min)",
              flush=True)

    # cross-slice consistency on shared genes
    print("-" * 72)
    common = set(all_ig[SLICES[0]][0])
    for g, _ in all_ig.values():
        common &= set(g)
    common = np.array(sorted(common))
    vecs = {}
    for sid, (g, s) in all_ig.items():
        idx = {gg: i for i, gg in enumerate(g)}
        vecs[sid] = np.array([s[idx[c]] for c in common])
    sids = list(vecs)
    rhos = []
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            rhos.append(spearmanr(vecs[sids[i]], vecs[sids[j]])[0])
    rhos = np.array(rhos)
    print(f"cross-slice IG Spearman: mean={rhos.mean():.3f} "
          f"min={rhos.min():.3f} max={rhos.max():.3f} (n={len(rhos)} pairs)")

    np.savez("data/processed/e12_12slices.npz",
             common_genes=common,
             **{sid: vecs[sid] for sid in sids})
    print(f"total {(time.perf_counter()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
