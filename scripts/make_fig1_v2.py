"""Fig 1 v2: carefully-typeset framework figure with REAL result insets.

Top: pipeline spine (data -> host -> adapter -> Myerson engine -> outputs).
Bottom: three real-data panels (DLPFC Myerson map, Xenium psi map,
rank-flip slopegraph from the benchmark).
All vector, consistent palette with Fig 2-5.
"""

import os
import pickle

import numpy as np
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "outputs"
C = {"red": "#c2352b", "blue": "#3b6ea5", "amber": "#e8a13c",
     "green": "#5a9e6f", "gray": "#9a9a9a", "ink": "#2c2c2a"}
NODE_METHODS = ["Myerson", "IG-node", "attention", "random"]
COLORS = {"Myerson": C["red"], "IG-node": C["blue"], "attention": C["amber"],
          "random": C["gray"]}
REGIMES = ["sparse", "medium", "high"]


def box(ax, x, y, w, h, title, sub, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=C["ink"])
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
            fontsize=8.5, color="#555552")


def arrow(ax, x1, y1, x2, y2, color="#777770"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=16, lw=1.6, color=color))


def main():
    os.makedirs(OUT, exist_ok=True)
    fig = plt.figure(figsize=(13.4, 8.4))
    ax = fig.add_axes([0, 0.42, 1, 0.58])
    ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")

    # ---------------- pipeline spine
    ax.text(2, 41, "a", fontsize=13, fontweight="bold", color=C["ink"])
    box(ax, 3, 22, 16, 13, "Spatial omics", "Visium / Xenium\nmulti-cohort", "#eef3f9", C["blue"])
    box(ax, 23, 22, 16, 13, "Host model", "any GNN\n(GCN / STAGATE / GAT)", "#eef6f0", C["green"])
    box(ax, 43, 22, 16, 13, "Adapter", "targets &\nperturbation semantics", "#f5f2ec", C["gray"])
    box(ax, 63, 22, 16, 13, "Myerson engine", "connected-coalition MC\n+ coalition cache", "#faecea", C["red"])
    box(ax, 83, 22, 15, 13, "Verified\nexplanations", "node / edge / gene", "#f9f4e8", C["amber"])
    for x1, x2 in [(19.5, 22.5), (39.5, 42.5), (59.5, 62.5), (79.5, 82.5)]:
        arrow(ax, x1, 28.5, x2, 28.5)

    # self-audit ribbon
    ax.add_patch(FancyBboxPatch((63, 6.5), 35, 9.5, boxstyle="round,pad=0.35",
                                fc="#faecea", ec=C["red"], lw=1.0))
    ax.text(80.5, 12.6, "self-auditing by construction", ha="center",
            fontsize=9.5, fontweight="bold", color=C["red"])
    ax.text(80.5, 9.2, r"$\sum_i \phi_i = v(N) - v(\varnothing)$"
            "   (holds exactly: 1.2787 = 1.2787)",
            ha="center", fontsize=9, color=C["ink"])
    arrow(ax, 71, 21.2, 73, 16.6, C["red"])

    ax.text(3, 16.5, "topology changes credit (Box 1):", fontsize=9.5,
            fontweight="bold", color=C["ink"])
    ax.text(3, 12.4, "path graph  s1 - s2 - s3", fontsize=9, color=C["ink"])
    ax.text(3, 9.6, "Shapley:  (3.00, 3.00, 3.00)  - position-blind",
            fontsize=9, color=C["gray"])
    ax.text(3, 6.8, "Myerson: (2.67, 3.67, 2.67)  - bridge earns more",
            fontsize=9, color=C["red"], fontweight="bold")

    # ---------------- real-data panels
    e4 = np.load("data/processed/e4_myerson_dlpfc_L5L6.npz", allow_pickle=True)
    adata = ad.read_h5ad("data/raw/DLPFC_151673.h5ad")
    coords_all = np.asarray(adata.obsm["spatial"], dtype=float)
    layer = np.array(adata.obs["layer"].astype(str))
    e10 = np.load("data/processed/e10e_ccxcl12.npz", allow_pickle=True)
    with open("data/processed/e8_matrix.pkl", "rb") as f:
        e8 = pickle.load(f)

    # b: DLPFC layers + Myerson phi
    axb = fig.add_axes([0.035, 0.045, 0.28, 0.33])
    for l, col in zip(["L1", "L2", "L3", "L4", "L5", "L6", "WM"],
                      ["#e8c547", "#e8963c", "#c2352b", "#8e44ad", "#3b6ea5",
                       "#2c6e5f", "#b9b9b2"]):
        m = layer == l
        axb.scatter(coords_all[m, 0], coords_all[m, 1], s=4, c=col,
                    edgecolors="none", rasterized=True)
    players, phi = e4["players"], e4["phi"]
    pc = e4["coords"]
    vmax = np.percentile(np.abs(phi), 95)
    axb.scatter(pc[players, 0], pc[players, 1], s=13, c=phi, cmap="RdBu_r",
                vmin=-vmax, vmax=vmax, edgecolors="black", linewidths=0.25,
                rasterized=True)
    axb.set_title("b  Myerson phi on DLPFC L5|L6 boundary", fontsize=10)
    axb.set_xticks([]); axb.set_yticks([])
    axb.set_aspect("equal"); axb.invert_yaxis()

    # c: Xenium psi map
    axc = fig.add_axes([0.375, 0.045, 0.28, 0.33])
    cc = e10["coords"]
    axc.scatter(cc[:, 0], cc[:, 1], s=2, c="#eeeeea", edgecolors="none",
                rasterized=True)
    obs = e10["obs"]; use = e10["use"]
    vm = np.percentile(np.abs(obs), 95)
    sc = axc.scatter(cc[use, 0], cc[use, 1], s=7, c=obs, cmap="RdBu_r",
                     vmin=-vm, vmax=vm, edgecolors="none", rasterized=True)
    axc.set_title("c  CXCL12 exclusion effect (Xenium breast)", fontsize=10)
    axc.set_xticks([]); axc.set_yticks([])
    axc.set_aspect("equal"); axc.invert_yaxis()
    cb = fig.colorbar(sc, ax=axc, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("psi", fontsize=8)

    # d: rank-flip slopegraph
    axd = fig.add_axes([0.73, 0.045, 0.24, 0.33])
    p1 = {m: np.mean([e8[r]["P1_node"][m] for r in REGIMES]) for m in NODE_METHODS}
    p2 = {m: np.mean([e8[r]["P2_node"][m] for r in REGIMES]) for m in NODE_METHODS}
    rank_p1 = {m: r for r, m in enumerate(sorted(NODE_METHODS, key=lambda z: -p1[z]))}
    rank_p2 = {m: r for r, m in enumerate(sorted(NODE_METHODS, key=lambda z: -p2[z]))}
    for m in NODE_METHODS:
        axd.plot([0, 1], [rank_p1[m], rank_p2[m]], "-o", color=COLORS[m],
                 lw=2.2, ms=5)
        axd.text(-0.05, rank_p1[m], m, ha="right", va="center", fontsize=8,
                 color=COLORS[m])
        axd.text(1.05, rank_p2[m], m, ha="left", va="center", fontsize=8,
                 color=COLORS[m])
    axd.set_xlim(-0.9, 1.9)
    axd.set_ylim(3.6, -0.6)
    axd.set_xticks([0, 1])
    axd.set_xticklabels(["recovery", "fidelity"], fontsize=8.5)
    axd.set_yticks([])
    axd.set_title("d  Ranks flip across protocols", fontsize=10)
    for s in ["top", "right", "left"]:
        axd.spines[s].set_visible(False)

    fig.suptitle("MyerST: topology-constrained attribution for spatial omics, "
                 "with verifiable fidelity", fontsize=14, y=0.975)
    fig.savefig(f"{OUT}/fig1_v2.png", dpi=220)
    fig.savefig(f"{OUT}/fig1_v2.pdf")
    print(f"saved {OUT}/fig1_v2.png / .pdf")


if __name__ == "__main__":
    main()
