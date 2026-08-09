"""Validate the faithful STAGATE port on DLPFC 151673.

Expectation: paper-grade ARI (~0.5 with GMM/mclust) if the port is correct.
Runs 500 official-recipe epochs with ARI checkpoints at 250/500.
"""

import time

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

from myerst.models.stagate_official_pt import STAGATEOfficialPT, train_stagate_official
from myerst.data.graph import build_knn_graph
from myerst.models.stagate_lite import adj_binary
from scripts.tune_stagate_ari import load


def ari_both(z, labels):
    km = adjusted_rand_score(labels, KMeans(7, n_init=20, random_state=0).fit(z).labels_)
    z64 = z.astype(np.float64)
    gm = adjusted_rand_score(labels, GaussianMixture(7, reg_covar=1e-4, n_init=5,
                                                     random_state=0).fit(z64).predict(z64))
    return km, gm


def main():
    x, coords, labels = load(zscore=False)
    edges = build_knn_graph(coords, k=6)
    adj_bin = adj_binary(edges, x.shape[0])
    print(f"data {tuple(x.shape)}, edges {len(edges)}", flush=True)

    model = STAGATEOfficialPT(n_feat=x.shape[1], hidden=512, latent=30)
    t0 = time.perf_counter()
    # train with intermediate ARI check at 250
    train_stagate_official(model, x, adj_bin, epochs=250, verbose=True)
    model.eval()
    with torch.no_grad():
        z = model.embed(x, adj_bin).numpy()
    km, gm = ari_both(z, labels)
    print(f"epoch 250: KMeans={km:.3f} GMM={gm:.3f} ({(time.perf_counter()-t0)/60:.1f} min)", flush=True)

    train_stagate_official(model, x, adj_bin, epochs=250, verbose=True)
    model.eval()
    with torch.no_grad():
        z = model.embed(x, adj_bin).numpy()
    km, gm = ari_both(z, labels)
    print(f"epoch 500: KMeans={km:.3f} GMM={gm:.3f} ({(time.perf_counter()-t0)/60:.1f} min)", flush=True)

    np.save("data/processed/stagate_official_embedding.npy", z)
    print("saved: data/processed/stagate_official_embedding.npy")


if __name__ == "__main__":
    main()
