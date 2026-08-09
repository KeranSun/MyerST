"""Tune STAGATE-lite clustering ARI on DLPFC 151673.

Levers tested (informed by the original STAGATE setup):
  A. preprocessing: per-gene z-score vs raw log1p-normalized input
  B. capacity: hidden 128 vs 256 (latent 32/30)
  C. graph: kNN(k=6) vs radius graph matching the Visium hex neighborhood
Each config reports KMeans ARI and GaussianMixture ARI (GMM ~ mclust used by
the original paper). Best embeddings are saved to data/processed/.
"""

import time

import numpy as np
import torch
import anndata as ad
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

from myerst.models.stagate_lite import STAGATELite, train_stagate, adj_binary
from myerst.data.graph import build_knn_graph

H5AD = "data/raw/DLPFC_151673.h5ad"
N_HVG = 3000
ORDER = ["L1", "L2", "L3", "L4", "L5", "L6", "WM"]


def load(zscore: bool):
    adata = ad.read_h5ad(H5AD)
    X = adata.X.toarray().astype(np.float32)
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    layer = np.array(adata.obs["layer"].astype(str))
    name2int = {n: i for i, n in enumerate(ORDER)}
    labels = np.array([name2int[v] for v in layer], dtype=int)

    lib = X.sum(1, keepdims=True)
    Xn = np.log1p(X / np.maximum(lib, 1) * 1e4)
    var = Xn.var(0)
    hvg = np.sort(np.argsort(var)[::-1][:N_HVG])
    Xn = Xn[:, hvg]
    if zscore:
        Xn = (Xn - Xn.mean(0)) / (Xn.std(0) + 1e-6)
    return (torch.from_numpy(Xn.astype(np.float32)), coords, labels)


def radius_graph(coords: np.ndarray, factor: float = 1.2) -> np.ndarray:
    """Connect spots within factor x median nearest-neighbor distance."""
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    d, _ = tree.query(coords, k=2)
    r = np.median(d[:, 1]) * factor
    pairs = tree.query_pairs(r)
    edges = np.array(sorted(pairs), dtype=np.int64)
    return edges


CONFIGS = [
    dict(name="h128_zscore_knn", hidden=128, latent=32, zscore=True, graph="knn"),
    dict(name="h128_nozscore_knn", hidden=128, latent=32, zscore=False, graph="knn"),
    dict(name="h256_nozscore_knn", hidden=256, latent=30, zscore=False, graph="knn"),
    dict(name="h256_nozscore_radius", hidden=256, latent=30, zscore=False, graph="radius"),
]


def main():
    torch.set_num_threads(max(1, torch.get_num_threads()))
    results = []
    best = (-1.0, None, None)
    for cfg in CONFIGS:
        t0 = time.perf_counter()
        x, coords, labels = load(cfg["zscore"])
        edges = (build_knn_graph(coords, k=6) if cfg["graph"] == "knn"
                 else radius_graph(coords))
        adj_bin = adj_binary(edges, x.shape[0])
        model = STAGATELite(n_feat=x.shape[1], hidden=cfg["hidden"],
                            latent=cfg["latent"], heads=2)
        train_stagate(model, x, adj_bin, epochs=250, lr=1e-3, seed=0)
        with torch.no_grad():
            z = model.embed(x, adj_bin).numpy()
        ari_km = adjusted_rand_score(labels, KMeans(7, n_init=20, random_state=0).fit(z).labels_)
        z64 = z.astype(np.float64)
        ari_gm = adjusted_rand_score(labels, GaussianMixture(
            7, covariance_type="full", reg_covar=1e-4,
            n_init=5, random_state=0).fit(z64).predict(z64))
        dt = time.perf_counter() - t0
        results.append((cfg["name"], len(edges), ari_km, ari_gm, dt))
        print(f"{cfg['name']:<22} edges={len(edges):<6} KMeans={ari_km:.3f} "
              f"GMM={ari_gm:.3f}  ({dt/60:.1f} min)", flush=True)
        if ari_gm > best[0]:
            best = (ari_gm, cfg["name"], z)

    print("-" * 60)
    print(f"BEST: {best[1]}  GMM ARI = {best[0]:.3f}")
    if best[2] is not None:
        import os
        os.makedirs("data/processed", exist_ok=True)
        np.save("data/processed/stagate_best_embedding.npy", best[2])
        print("saved: data/processed/stagate_best_embedding.npy")


if __name__ == "__main__":
    main()
