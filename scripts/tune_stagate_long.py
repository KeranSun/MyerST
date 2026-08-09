"""Long-training checkpoint experiment: does ARI climb with training length?

Motivation: at 250 epochs recon MSE ~0.90 (i.e. ~10% variance explained) —
the autoencoder is severely undertrained. Original STAGATE trains 1000 epochs.
This run trains 800 epochs (hidden 256, latent 30, no z-score, kNN-6) and
reports recon MSE + KMeans/GMM ARI at checkpoints to map the training curve.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

from myerst.models.stagate_lite import STAGATELite, adj_binary
from myerst.data.graph import build_knn_graph
from scripts.tune_stagate_ari import load

EPOCHS = 800
CHECKPOINTS = [250, 500, 800]


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
    torch.manual_seed(0)
    model = STAGATELite(n_feat=x.shape[1], hidden=256, latent=30, heads=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"training {EPOCHS} epochs, checkpoints at {CHECKPOINTS}", flush=True)
    t0 = time.perf_counter()
    model.train()
    best = (-1.0, None)
    for ep in range(1, EPOCHS + 1):
        opt.zero_grad()
        loss = F.mse_loss(model(x, adj_bin), x)
        loss.backward()
        opt.step()
        if ep in CHECKPOINTS:
            model.eval()
            with torch.no_grad():
                z = model.embed(x, adj_bin).numpy()
            km, gm = ari_both(z, labels)
            el = (time.perf_counter() - t0) / 60
            print(f"  epoch {ep:4d}  recon_mse {loss.item():.4f}  "
                  f"KMeans={km:.3f}  GMM={gm:.3f}  ({el:.1f} min)", flush=True)
            if gm > best[0]:
                best = (gm, ep)
                np.save("data/processed/stagate_best_embedding.npy", z)
            model.train()
    print("-" * 60)
    print(f"BEST GMM ARI = {best[0]:.3f} at epoch {best[1]}")


if __name__ == "__main__":
    main()
