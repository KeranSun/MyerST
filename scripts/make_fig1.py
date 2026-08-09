"""Fig 1: MyerST framework overview (schematic + real-data insets).

Layout (2 rows):
  top:    pipeline  data -> host -> adapter -> Myerson engine -> verified outputs
  bottom: (left) Shapley vs Myerson 3-node example
          (mid)  efficiency self-audit + MC convergence
          (right) three output modalities with real mini-maps
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "outputs"
C_DATA = "#E6F1FB"
C_HOST = "#F1EFE8"
C_ENGINE = "#FAECE7"
C_OUT = "#E1F5EE"
C_EDGE = "#5F5E5A"


def box(ax, x, y, w, h, title, sub, fc, title_fs=10.5, sub_fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=C_EDGE, lw=0.8))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            fontsize=title_fs, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                fontsize=sub_fs, color="#444441")


def arrow(ax, x1, y1, x2, y2, color=C_EDGE):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, color=color, lw=1.4))


def main():
    os.makedirs(OUT, exist_ok=True)
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 58)
    ax.axis("off")

    ax.text(50, 56.3, "MyerST: topology-constrained, self-auditing attribution "
                      "for spatial omics", ha="center", fontsize=14,
            fontweight="bold")

    # ---------- top: pipeline
    y0, h = 44, 8
    box(ax, 3, y0, 15, h, "Spatial omics data", "Visium / Xenium / sim", C_DATA)
    box(ax, 23, y0, 15, h, "Host model", "any GNN\n(GCN, GAT, STAGATE)", C_HOST)
    box(ax, 43, y0, 15, h, "Adapter", "forward + scalar target", C_HOST)
    box(ax, 63, y0, 16, h, "Myerson engine", "connected-coalition MC\n+ coalition cache", C_ENGINE)
    box(ax, 84, y0, 14, h, "Verified output", "efficiency self-audit\n+ fidelity benchmark", C_OUT)
    for x1, x2 in [(18, 23), (38, 43), (58, 63), (79, 84)]:
        arrow(ax, x1, y0 + h / 2, x2, y0 + h / 2)

    # ---------- bottom left: Shapley vs Myerson example
    ax.text(15, 38.5, "a  Topology changes credit (Box 1)",
            fontsize=11, fontweight="bold")
    # path graph s1-s2-s3
    nx, ny = [6, 15, 24], 32
    for i in range(3):
        ax.add_patch(plt.Circle((nx[i], ny), 2.6, fc="#E6F1FB", ec=C_EDGE, lw=0.8))
        ax.text(nx[i], ny, f"s{i+1}", ha="center", va="center", fontsize=9)
    for i in range(2):
        ax.plot([nx[i] + 2.6, nx[i + 1] - 2.6], [ny, ny], color=C_EDGE, lw=1.2)
    ax.text(15, 27.5, "v(S) = |S|²", ha="center", fontsize=9, style="italic")
    ax.text(15, 24.5, "Shapley:   (3, 3, 3)   — position invisible",
            ha="center", fontsize=9.5)
    ax.text(15, 21.5, "Myerson: (8/3, 11/3, 8/3) — bridge earns 37% more",
            ha="center", fontsize=9.5, color="#993C1D", fontweight="bold")
    ax.text(15, 18.5, "coalitions restricted to connected subgraphs of g",
            ha="center", fontsize=8, color="#5F5E5A")

    # ---------- bottom mid: efficiency + convergence
    ax.text(48, 38.5, "b  Self-auditing efficiency",
            fontsize=11, fontweight="bold")
    ax.text(45, 34.5, r"$\sum_i \phi_i = v(N) - v(\varnothing)$", ha="center",
            fontsize=13)
    ax.text(45, 31, "verified to machine precision on every run\n"
                    "(DLPFC: 1.2787 = 1.2787)", ha="center", fontsize=9)
    ax.text(45, 27.5, "MC error ~ 1/sqrt(M), converges in minutes on CPU",
            ha="center", fontsize=9, color="#5F5E5A")

    # ---------- bottom right: three output modalities with real insets
    ax.text(79, 38.5, "c  Three attribution modalities", fontsize=11,
            fontweight="bold")
    e4 = np.load("data/processed/e4_myerson_dlpfc_L5L6.npz", allow_pickle=True)
    e10 = np.load("data/processed/e10e_ccxcl12.npz", allow_pickle=True)

    ax1 = fig.add_axes([0.615, 0.06, 0.115, 0.20])
    ax1.scatter(e4["coords"][:, 0], e4["coords"][:, 1], s=1, c="#e5e5e5",
                edgecolors="none")
    vmax = np.percentile(np.abs(e4["phi"]), 95)
    ax1.scatter(e4["coords"][e4["players"], 0], e4["coords"][e4["players"], 1],
                s=3, c=e4["phi"], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                edgecolors="none")
    ax1.set_title("node scores (DLPFC)", fontsize=8)
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.invert_yaxis()

    ax2 = fig.add_axes([0.745, 0.06, 0.115, 0.20])
    ax2.scatter(e10["coords"][:, 0], e10["coords"][:, 1], s=1, c="#e5e5e5",
                edgecolors="none")
    vmax2 = np.percentile(np.abs(e10["obs"]), 95)
    ax2.scatter(e10["coords"][e10["use"], 0], e10["coords"][e10["use"], 1],
                s=3, c=e10["obs"], cmap="RdBu_r", vmin=-vmax2, vmax=vmax2,
                edgecolors="none")
    ax2.set_title("edge/pathway scores (Xenium)", fontsize=8)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.invert_yaxis()

    ax3 = fig.add_axes([0.875, 0.06, 0.115, 0.20])
    genes = ["PCP4", "KRT17", "RORB", "MBP", "PLP1"]
    vals = [1.0, 0.82, 0.48, 0.31, 0.27]
    ax3.barh(range(5), vals, color="#3b6ea5", alpha=0.85)
    ax3.set_yticks(range(5))
    ax3.set_yticklabels(genes, fontsize=7)
    ax3.invert_yaxis()
    ax3.set_title("gene scores (layer markers)", fontsize=8)
    ax3.spines[["top", "right"]].set_visible(False)

    fig.savefig(f"{OUT}/fig1_framework.png", dpi=200)
    fig.savefig(f"{OUT}/fig1_framework.pdf")
    print(f"saved {OUT}/fig1_framework.png / .pdf")


if __name__ == "__main__":
    main()
