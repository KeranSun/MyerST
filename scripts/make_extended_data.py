"""Extended Data figures (all from archived artifacts, no new experiments).

ED1: benchmark robustness suite (gene matrix, MC convergence, ROAR curves)
ED2: DLPFC 12-slice suite (marker hits, cross-slice correlation, host quality)
ED3: CCC suite (iteration ladder, synthetic validation, cohort stats, runtime)
"""

import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from myerst.data.graph import build_knn_graph
from myerst.attribution.myerson import exact_myerson, mc_myerson

OUT = "outputs"
C = {"Myerson": "#c2352b", "IG-node": "#3b6ea5", "attention": "#e8a13c",
     "random": "#9a9a9a", "IG": "#3b6ea5", "Occ": "#5a9e6f", "DE": "#7a5195"}
SLICES = ["151507", "151508", "151509", "151510", "151669", "151670",
          "151671", "151672", "151673", "151674", "151675", "151676"]
MARKERS = ["PCP4", "KRT17", "RORB", "AQP4", "GFAP", "CUX2", "NEFH",
           "KRT5", "PLP1", "MBP", "CCK", "HOPX"]


def ed1():
    e8ms = pickle.load(open("data/processed/e8_matrix_multiseed.pkl", "rb"))
    e8b = np.load("data/processed/e8b_roar_curves.npz")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    protos = ["P2_gene", "P3_gene"]
    methods = ["IG", "Occ", "DE", "random"]
    M = np.array([[np.mean([r[p][m] for r in e8ms["medium"]]) for p in protos]
                  for m in methods])
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    ax.set_xticks(range(2)); ax.set_xticklabels(["P2 masking", "P3 ROAR"], fontsize=9)
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods, fontsize=9)
    for i in range(len(methods)):
        for j in range(2):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if M[i, j] < M.max() * 0.7 else "black")
    ax.set_title("a  Gene level, medium regime (3 seeds)", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    rng = np.random.default_rng(2026)
    coords = rng.uniform(0, 20, size=(10, 2))
    edges10 = build_knn_graph(coords, k=4)
    v = lambda s: float(len(s)) ** 2
    phi_exact = exact_myerson(10, edges10, v)
    ns = [64, 256, 1024, 4096, 16384, 65536]
    errs = [np.max(np.abs(mc_myerson(10, edges10, v, n_samples=n_s, seed=0)
                          - phi_exact)) for n_s in ns]
    ax.loglog(ns, errs, "o-", color=C["Myerson"], lw=2)
    ax.loglog(ns, errs[2] * (np.array(ns) / ns[2]) ** -0.5, "--",
              color=C["random"], label="1/sqrt(M)")
    ax.set_xlabel("MC samples", fontsize=9); ax.set_ylabel("max |error|", fontsize=9)
    ax.set_title("b  Myerson sampler convergence", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    ks = [0, 10, 20, 40, 80, 160]
    for m, col in [("Myerson", C["Myerson"]), ("IG-node", C["IG-node"]),
                   ("random", C["random"])]:
        v_ = e8b[f"sparse-grid+weak/node/{m}"]
        ax.plot(range(len(v_)), v_, "-o", ms=3.5, lw=1.6, color=col, label=m)
    ax.set_xlabel("top-k nodes removed", fontsize=9)
    ax.set_ylabel("held-out boundary acc", fontsize=9)
    ax.set_ylim(0.9, 1.01)
    ax.set_title("c  Node-level ROAR curves (weak regime)", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Extended Data Fig. 1 | Benchmark robustness suite", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{OUT}/ed1_benchmark.png", dpi=200)
    fig.savefig(f"{OUT}/ed1_benchmark.pdf")
    print("saved ed1")


def ed2():
    hits = np.zeros((len(SLICES), len(MARKERS)))
    igs = {}
    for i, sid in enumerate(SLICES):
        c = np.load(f"data/processed/e12_{sid}.npz", allow_pickle=True)
        genes, ig = c["genes"], c["ig"]
        igs[sid] = (genes, ig)
        top20 = set(genes[np.argsort(ig)[::-1][:20]])
        for j, mk in enumerate(MARKERS):
            hits[i, j] = mk in top20

    common = set(igs[SLICES[0]][0])
    for sid in SLICES[1:]:
        common &= set(igs[sid][0])
    common = sorted(common)
    vecs = {sid: np.array([igs[sid][1][list(igs[sid][0]).index(g)] for g in common])
            for sid in SLICES}
    R = np.eye(len(SLICES))
    for i in range(len(SLICES)):
        for j in range(i + 1, len(SLICES)):
            R[i, j] = R[j, i] = spearmanr(vecs[SLICES[i]], vecs[SLICES[j]]).statistic

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    im = ax.imshow(hits, cmap="Greens", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(MARKERS)))
    ax.set_xticklabels(MARKERS, fontsize=7.5, rotation=45, ha="right")
    ax.set_yticks(range(len(SLICES)))
    ax.set_yticklabels(SLICES, fontsize=7.5)
    ax.set_title("a  Known markers in top-20 (12 slices)", fontsize=11)

    ax = axes[1]
    im = ax.imshow(R, cmap="RdBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(SLICES)))
    ax.set_xticklabels(SLICES, fontsize=6.5, rotation=45, ha="right")
    ax.set_yticks(range(len(SLICES)))
    ax.set_yticklabels(SLICES, fontsize=6.5)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman")
    ax.set_title("b  Cross-slice attribution correlation", fontsize=11)

    ax = axes[2]
    hosts = ["lite\n(ARI~0.30)", "official\n(ARI 0.52)"]
    agree = [0.087, 0.807]
    ax.bar(hosts, agree, color=["#9a9a9a", "#3b6ea5"], alpha=0.85)
    for i, v_ in enumerate(agree):
        ax.text(i, v_ + 0.03, f"{v_:.3f}", ha="center", fontsize=11)
    ax.set_ylabel("Spearman with GCN IG", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title("c  Host quality gates attribution", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Extended Data Fig. 2 | DLPFC twelve-slice suite", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{OUT}/ed2_dlpfc.png", dpi=200)
    fig.savefig(f"{OUT}/ed2_dlpfc.pdf")
    print("saved ed2")


def ed3():
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))

    ax = axes[0]
    iters = ["E10\nsingle-edge", "E10b\ncell-group", "E10c\nlocation",
             "E10d\nLR-host", "E10e\ngene-occ"]
    zs = [0.25, -0.12, -0.21, -0.16, -38.9]
    ax.bar(range(len(iters)), zs, color=["#9a9a9a"] * 4 + ["#c2352b"], alpha=0.85)
    ax.axhline(0, c="black", lw=0.8)
    ax.set_xticks(range(len(iters)))
    ax.set_xticklabels(iters, fontsize=7.5)
    ax.set_ylabel("CXCL12 effect (z or paired t)", fontsize=8.5)
    ax.set_title("a  Five iterations to the right design", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    e9b = np.load("data/processed/e9b_local_ccc.npz", allow_pickle=True)
    psi, truth = e9b["psi"], e9b["truth"]
    bp = ax.boxplot([psi[truth], psi[~truth]], showfliers=False, widths=0.5,
                    patch_artist=True,
                    tick_labels=["communicating", "other"])
    for patch, c in zip(bp["boxes"], ["#c2352b", "#9a9a9a"]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.axhline(0, ls=":", c="#9a9a9a", lw=1)
    ax.set_ylabel("local synergy", fontsize=9)
    ax.set_title("b  Synthetic CCC validation (AUROC=0.57)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    cohorts = ["Rep1", "Rep2", "Lung", "Visium"]
    n_recv = [1806, 1917, 60, 35]
    accs = [0.926, 0.985, 0.973, 1.000]
    ax2 = ax.twinx()
    ax.bar(range(len(cohorts)), n_recv, color="#3b6ea5", alpha=0.7, width=0.5)
    ax2.plot(range(len(cohorts)), accs, "o-", color="#c2352b", lw=2, ms=6)
    ax.set_xticks(range(len(cohorts)))
    ax.set_xticklabels(cohorts, fontsize=9)
    ax.set_ylabel("receivers (n)", fontsize=9, color="#3b6ea5")
    ax2.set_ylabel("host T acc", fontsize=9, color="#c2352b")
    ax2.set_ylim(0.9, 1.02)
    ax.set_title("c  Cohort statistics", fontsize=11)
    ax.spines[["top"]].set_visible(False)

    ax = axes[3]
    tasks = ["E4 Myerson\nDLPFC", "E12 per\nslice", "E10e\nRep1",
             "E11 per\ncohort", "E8g per\nregime-seed"]
    mins = [26, 1.9, 2.4, 1.0, 1.0]
    ax.barh(range(len(tasks)), mins, color="#5a9e6f", alpha=0.8)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=8)
    ax.set_xlabel("runtime (min, single CPU)", fontsize=9)
    ax.set_title("d  Runtime profile", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Extended Data Fig. 3 | CCC analysis suite", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{OUT}/ed3_ccc.png", dpi=200)
    fig.savefig(f"{OUT}/ed3_ccc.pdf")
    print("saved ed3")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ed1()
    ed2()
    ed3()
