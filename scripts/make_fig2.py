"""Fig 2 draft: multi-protocol benchmark matrix visualization.

Panels:
  a  node-level recovery AUROC by method (per regime)
  b  node-level masking decay AUC by method (per regime) — Myerson dominance,
     attention-below-random falsification
  c  rank-flip slopegraph: P1 vs P2 (node level, mean over regimes)
  d  gene-level protocol heatmap (mean over regimes)
"""

import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs"
NODE_METHODS = ["Myerson", "IG-node", "attention", "random"]
GENE_METHODS = ["IG", "Occ", "DE", "random"]
COLORS = {"Myerson": "#c2352b", "IG-node": "#3b6ea5", "attention": "#e8a13c",
          "random": "#9a9a9a", "IG": "#3b6ea5", "Occ": "#5a9e6f", "DE": "#7a5195"}
REGIMES = ["sparse", "medium", "high"]


def grouped_bars(ax, res, key, methods, title, ylabel):
    x = np.arange(len(methods))
    w = 0.25
    for ri, reg in enumerate(REGIMES):
        vals = [res[reg][key][m] for m in methods]
        ax.bar(x + (ri - 1) * w, vals, w, label=reg,
               color=[COLORS[m] for m in methods], alpha=0.35 + 0.25 * ri,
               edgecolor="white", linewidth=0.5)
    rand_vals = [res[reg][key]["random"] for reg in REGIMES]
    ax.axhline(np.mean(rand_vals), ls="--", c="#9a9a9a", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    with open("data/processed/e8_matrix.pkl", "rb") as f:
        res = pickle.load(f)
    os.makedirs(OUT, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    grouped_bars(axes[0, 0], res, "P1_node", NODE_METHODS,
                 "a  Node-level: ground-truth recovery (P1)", "AUROC")
    grouped_bars(axes[0, 1], res, "P2_node", NODE_METHODS,
                 "b  Node-level: masking fidelity (P2)", "decay AUC")

    # ---- c: rank-flip slopegraph (mean over regimes)
    ax = axes[1, 0]
    p1 = {m: np.mean([res[r]["P1_node"][m] for r in REGIMES]) for m in NODE_METHODS}
    p2 = {m: np.mean([res[r]["P2_node"][m] for r in REGIMES]) for m in NODE_METHODS}
    rank_p1 = {m: r for r, m in enumerate(sorted(NODE_METHODS, key=lambda z: -p1[z]))}
    rank_p2 = {m: r for r, m in enumerate(sorted(NODE_METHODS, key=lambda z: -p2[z]))}
    for m in NODE_METHODS:
        ax.plot([0, 1], [rank_p1[m], rank_p2[m]], "-o", color=COLORS[m], lw=2.5,
                ms=7, label=m)
        ax.text(-0.06, rank_p1[m], f"{m} ({p1[m]:.2f})", ha="right", va="center",
                fontsize=9, color=COLORS[m])
        ax.text(1.06, rank_p2[m], f"{m} ({p2[m]:.2f})", ha="left", va="center",
                fontsize=9, color=COLORS[m])
    ax.set_xlim(-0.9, 1.9)
    ax.set_ylim(3.6, -0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["P1 recovery", "P2 masking fidelity"], fontsize=10)
    ax.set_yticks([])
    ax.set_title("c  Rank flip across protocols (node level)", fontsize=11)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)

    # ---- d: gene-level heatmap (mean over regimes)
    ax = axes[1, 1]
    protos = ["P1_gene", "P2_gene", "P3_gene"]
    labels_p = ["P1 recovery", "P2 masking", "P3 ROAR"]
    M = np.array([[np.mean([res[r][p][m] for r in REGIMES]) for p in protos]
                  for m in GENE_METHODS])
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0.4, vmax=1.1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels_p, fontsize=9)
    ax.set_yticks(range(len(GENE_METHODS)))
    ax.set_yticklabels(GENE_METHODS, fontsize=9)
    for i in range(len(GENE_METHODS)):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if M[i, j] < 0.85 else "black")
    ax.set_title("d  Gene-level: method x protocol (mean over regimes)", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("MyerST benchmark draft: no free lunch in explanation fidelity",
                 fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{OUT}/fig2_draft.png", dpi=200)
    fig.savefig(f"{OUT}/fig2_draft.pdf")
    print(f"saved {OUT}/fig2_draft.png / .pdf")


if __name__ == "__main__":
    main()
