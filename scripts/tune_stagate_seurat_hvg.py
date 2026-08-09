"""Test the last preprocessing difference: seurat_v3 HVG selection (official
STAGATE DLPFC tutorial) vs our variance-on-log selection.

Exact official pipeline: seurat_v3 HVG on raw counts -> normalize_total(1e4)
-> log1p -> subset HVG -> official-port STAGATE 500 epochs -> ARI with
KMeans/GMM over 5 cluster seeds (mean +- sd).
"""

import time

import numpy as np
import torch
import scanpy as sc
import anndata as ad
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

from myerst.models.stagate_official_pt import STAGATEOfficialPT, train_stagate_official
from myerst.data.graph import build_knn_graph
from myerst.models.stagate_lite import adj_binary

H5AD = "data/raw/DLPFC_151673.h5ad"
ORDER = ["L1", "L2", "L3", "L4", "L5", "L6", "WM"]


def load_seurat_hvg():
    adata = ad.read_h5ad(H5AD)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    X = adata[:, adata.var["highly_variable"]].X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    layer = np.array(adata.obs["layer"].astype(str))
    name2int = {n: i for i, n in enumerate(ORDER)}
    labels = np.array([name2int[v] for v in layer], dtype=int)
    return torch.from_numpy(X.astype(np.float32)), coords, labels


def ari_seeds(z, labels, seeds=range(5)):
    kms, gms = [], []
    z64 = z.astype(np.float64)
    for s in seeds:
        kms.append(adjusted_rand_score(labels, KMeans(7, n_init=20, random_state=s).fit(z).labels_))
        gms.append(adjusted_rand_score(labels, GaussianMixture(7, reg_covar=1e-4, n_init=5,
                                       random_state=s).fit(z64).predict(z64)))
    return np.array(kms), np.array(gms)


def main():
    x, coords, labels = load_seurat_hvg()
    edges = build_knn_graph(coords, k=6)
    adj_bin = adj_binary(edges, x.shape[0])
    print(f"seurat_v3 HVG data {tuple(x.shape)}, edges {len(edges)}", flush=True)

    model = STAGATEOfficialPT(n_feat=x.shape[1], hidden=512, latent=30)
    t0 = time.perf_counter()
    train_stagate_official(model, x, adj_bin, epochs=500, verbose=True)
    model.eval()
    with torch.no_grad():
        z = model.embed(x, adj_bin).numpy()
    kms, gms = ari_seeds(z, labels)
    print(f"epoch 500 ({(time.perf_counter()-t0)/60:.1f} min):", flush=True)
    print(f"  KMeans ARI = {kms.mean():.3f} +- {kms.std():.3f}  (max {kms.max():.3f})")
    print(f"  GMM    ARI = {gms.mean():.3f} +- {gms.std():.3f}  (max {gms.max():.3f})")
    np.save("data/processed/stagate_official_seurat_embedding.npy", z)


if __name__ == "__main__":
    main()
