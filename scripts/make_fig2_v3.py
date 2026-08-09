"""Fig 2 v3: corrected benchmark figure (post multi-seed review).

Data: e8g (per-side recovery), e8f (margin-decay fidelity),
e8_matrix_multiseed (gene recovery), e8b (ROAR curves).
Panels:
a  node-level recovery AUROC (per-side games; IG dominates, honest)
b  node-level masking fidelity (raw margin-decay AUC; attention < random)
c  gene-level recovery across redundancy (mean+-sd, 3 seeds)
d  ROAR insensitivity (accuracy retention curves, all methods flat)
e  Myerson's unique value: exact self-audit + 1/sqrt(M) convergence
f  evaluation-hygiene checklist (the four failure modes we caught)
"""

import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from myerst.data.graph import build_knn_graph
from myerst.attribution.myerson import exact_myerson, mc_myerson

OUT = "outputs"
C = {"Myerson": "#c2352b", "IG-node": "#3b6ea5", "attention": "#e8a13c",
     "random": "#9a9a9a", "IG": "#3b6ea5", "Occ": "#5a9e6f", "DE": "#7a5195"}
REGIMES = ["sparse", "medium"]
SEEDS = [0, 1, 2]


def bars_ms(ax, data, methods, metric, title, ylabel, regimes=REGIMES):
    x = np.arange(len(methods))
    w = 0.35
    for ri, reg in enumerate(regimes):
        vals = [np.mean([data[(reg, s)][metric][m] if metric in data[(reg, s)]
                         else data[(reg, s)][m] for s in SEEDS]) for m in methods]
        sds = [np.std([data[(reg, s)][metric][m] if metric in data[(reg, s)]
                       else data[(reg, s)][m] for s in SEEDS]) for m in methods]
        ax.bar(x + (ri - 0.5) * w, vals, w, yerr=sds, capsize=3,
               color=[C[m] for m in methods], alpha=0.5 + 0.35 * ri,
               edgecolor="white", linewidth=0.5, label=reg,
               error_kw=dict(lw=1, ecolor="#555"))
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8.5)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    os.makedirs(OUT, exist_ok=True)
    e8g = pickle.load(open("data/processed/e8g_perside.pkl", "rb"))
    e8f = pickle.load(open("data/processed/e8f_node_multiseed.pkl", "rb"))
    e8ms = pickle.load(open("data/processed/e8_matrix_multiseed.pkl", "rb"))
    e8b = np.load("data/processed/e8b_roar_curves.npz")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # a: node recovery (per-side games)
    bars_ms(axes[0, 0], e8g, ["Myerson", "IG-node", "random"], None,
            "a  Node recovery (per-side games)", "AUROC (mean±sd)")
    axes[0, 0].axhline(0.5, ls="--", c="#9a9a9a", lw=1)

    # b: node masking fidelity (margin decay)
    bars_ms(axes[0, 1], e8f, ["Myerson", "IG-node", "attention", "random"],
            "P2", "b  Masking fidelity (margin-decay AUC)", "AUC (mean±sd)")
    axes[0, 1].annotate("attention < random",
                        xy=(2, 0.17), xytext=(1.1, 0.34), fontsize=9,
                        color="#c2352b",
                        arrowprops=dict(arrowstyle="->", color="#c2352b", lw=1))

    # c: gene recovery across redundancy (multiseed E8)
    ax = axes[0, 2]
    gene_methods = ["IG", "Occ", "DE", "random"]
    regimes3 = ["sparse", "medium", "high"]
    x = np.arange(len(gene_methods))
    w = 0.25
    for ri, reg in enumerate(regimes3):
        vals = [np.mean([r["P1_gene"][m] for r in e8ms[reg]]) for m in gene_methods]
        sds = [np.std([r["P1_gene"][m] for r in e8ms[reg]]) for m in gene_methods]
        ax.bar(x + (ri - 1) * w, vals, w, yerr=sds, capsize=2.5,
               color=[C[m] for m in gene_methods], alpha=0.4 + 0.25 * ri,
               edgecolor="white", linewidth=0.5, label=reg,
               error_kw=dict(lw=0.8, ecolor="#555"))
    ax.set_xticks(x)
    ax.set_xticklabels(gene_methods, fontsize=8.5)
    ax.axhline(0.5, ls="--", c="#9a9a9a", lw=1)
    ax.set_title("c  Gene recovery vs redundancy (3 seeds)", fontsize=11)
    ax.set_ylabel("AUROC", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # d: ROAR insensitivity curves
    ax = axes[1, 0]
    ks_g = [0, 2, 4, 8, 16, 32, 64]
    for m, col in [("IG", C["IG"]), ("Occ", C["Occ"]), ("DE", C["DE"]),
                   ("random", C["random"])]:
        v = e8b[f"sparse-grid+weak/gene/{m}"]
        ax.plot(ks_g, v, "-o", ms=3.5, lw=1.6, color=col, label=m)
    ax.set_xlabel("top-k genes removed", fontsize=9)
    ax.set_ylabel("held-out boundary acc", fontsize=9)
    ax.set_ylim(0.8, 1.02)
    ax.set_title("d  ROAR is insensitive on spatial GNNs", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # e: Myerson unique value — audit + convergence
    ax = axes[1, 1]
    rng = np.random.default_rng(2026)
    coords = rng.uniform(0, 20, size=(10, 2))
    edges10 = build_knn_graph(coords, k=4)
    v = lambda s: float(len(s)) ** 2
    phi_exact = exact_myerson(10, edges10, v)
    ns = [64, 256, 1024, 4096, 16384, 65536]
    errs = [np.max(np.abs(mc_myerson(10, edges10, v, n_samples=n_s, seed=0)
                          - phi_exact)) for n_s in ns]
    ax.loglog(ns, errs, "o-", color=C["Myerson"], lw=2, label="max |MC - exact|")
    ax.loglog(ns, errs[2] * (np.array(ns) / ns[2]) ** -0.5, "--",
              color=C["random"], label="1/sqrt(M)")
    ax.set_xlabel("MC samples", fontsize=9)
    ax.set_ylabel("error", fontsize=9)
    ax.set_title("e  Myerson: self-auditing + convergent", fontsize=11)
    ax.text(0.03, 0.05, r"$\sum\phi=v(N)-v(\varnothing)$ exact (DLPFC: 1.2787)",
            transform=ax.transAxes, fontsize=9, color=C["Myerson"])
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # f: hygiene checklist
    ax = axes[1, 2]
    ax.axis("off")
    txt = [
        "f  Evaluation hygiene (caught by multi-seed review)", "",
        "1. Raw logit-diff targets cancel by class",
        "   (ref ~ 0) -> game is meaningless", "",
        "2. Mixed signed margins cause friendly fire", "",
        "3. Class-mean baselines inject prototype",
        "   signal -> use global-mean baselines", "",
        "4. Sign conventions must match across",
        "   explainers (magnitude vs signed)",
    ]
    ax.text(0.02, 0.97, "\n".join(txt), va="top", fontsize=9.5,
            family="monospace")

    fig.suptitle("Benchmarking spatial explanations: operator semantics decide "
                 "what 'importance' means", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(f"{OUT}/fig2_v3.png", dpi=200)
    fig.savefig(f"{OUT}/fig2_v3.pdf")
    print(f"saved {OUT}/fig2_v3.png / .pdf")


if __name__ == "__main__":
    main()
