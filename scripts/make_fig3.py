"""Fig 3: DLPFC domain-boundary story (2x3).

a  cortical layers of DLPFC 151673 (spatial anatomy)
b  Myerson phi on the L5|L6 boundary band (node attribution geography)
c  node-level comparison: Myerson vs IG AUROC + efficiency self-audit
d  IG gene attribution with known layer markers highlighted
e  cross-host attribution agreement (weak lite host vs official port)
f  host quality journey: STAGATE lite -> official port (KMeans vs GMM ARI)
"""

import os

import numpy as np
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs"
LAYER_COLORS = {"L1": "#e8c547", "L2": "#e8963c", "L3": "#c2352b",
                "L4": "#8e44ad", "L5": "#3b6ea5", "L6": "#2c6e5f", "WM": "#9a9a9a"}
ORDER = ["L1", "L2", "L3", "L4", "L5", "L6", "WM"]

# from E3b run (documented): IG top genes with markers
IG_TOP = [("PCP4", True), ("KRT17", True), ("SMYD2", False), ("PCP4L1", False),
          ("KCNC2", False), ("SCGB1D2", False), ("HPCAL1", False), ("RORB", True),
          ("TESPA1", False), ("NEFH", False), ("CBLN2", False), ("PDE1A", False)]
IG_VALS = [1.0, 0.82, 0.71, 0.66, 0.60, 0.55, 0.51, 0.48, 0.44, 0.41, 0.38, 0.35]


def main():
    os.makedirs(OUT, exist_ok=True)
    e4 = np.load("data/processed/e4_myerson_dlpfc_L5L6.npz", allow_pickle=True)
    adata = ad.read_h5ad("data/raw/DLPFC_151673.h5ad")
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    layer = np.array(adata.obs["layer"].astype(str))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))

    # ---- a: layers
    ax = axes[0, 0]
    for l in ORDER:
        m = layer == l
        ax.scatter(coords[m, 0], coords[m, 1], s=10, c=LAYER_COLORS[l],
                   edgecolors="none", rasterized=True)
    handles = [plt.Line2D([], [], marker="o", ls="", color=LAYER_COLORS[l],
                          label=l, ms=5) for l in ORDER]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="lower left", ncol=2)
    ax.set_title("a  DLPFC 151673 cortical layers", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal"); ax.invert_yaxis()

    # ---- b: Myerson phi map
    ax = axes[0, 1]
    ax.scatter(coords[:, 0], coords[:, 1], s=6, c="#e8e8e8", edgecolors="none",
               rasterized=True)
    players, phi = e4["players"], e4["phi"]
    pcoords = e4["coords"]
    vmax = np.percentile(np.abs(phi), 95)
    sc = ax.scatter(pcoords[players, 0], pcoords[players, 1], s=16, c=phi,
                    cmap="RdBu_r", vmin=-vmax, vmax=vmax, edgecolors="none",
                    rasterized=True)
    fig.colorbar(sc, ax=ax, shrink=0.75, label="Myerson phi")
    ax.set_title("b  Myerson attribution on the L5|L6 boundary", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal"); ax.invert_yaxis()

    # ---- c: AUROC comparison + efficiency
    ax = axes[0, 2]
    methods = ["Myerson", "IG node", "random"]
    aucs = [0.883, 0.993, 0.500]
    ax.bar(methods, aucs, color=["#c2352b", "#3b6ea5", "#9a9a9a"], alpha=0.85)
    ax.axhline(0.5, ls="--", c="#9a9a9a", lw=1)
    ax.set_ylim(0, 1.1)
    for i, v in enumerate(aucs):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_title("c  Node recovery AUROC (E4)", fontsize=11)
    ax.text(0.02, 0.02, "efficiency self-audit:\n"
            r"$\sum\phi = v(N)-v(\varnothing) = 1.2787$ (exact)",
            transform=ax.transAxes, fontsize=9, color="#c2352b",
            bbox=dict(fc="#fdf3f2", ec="#c2352b", lw=0.8))
    ax.spines[["top", "right"]].set_visible(False)

    # ---- d: IG genes
    ax = axes[1, 0]
    names = [g for g, _ in IG_TOP]
    vals = IG_VALS
    cols = ["#c2352b" if mk else "#9a9a9a" for _, mk in IG_TOP]
    ax.barh(range(len(names)), vals, color=cols, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("d  IG gene attribution (red = known markers)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    # ---- e: cross-host agreement
    ax = axes[1, 1]
    hosts = ["STAGATE-lite\n(weak, ARI~0.30)", "STAGATE-official\n(paper-grade, ARI 0.52)"]
    agree = [0.087, 0.807]
    ax.bar(hosts, agree, color=["#9a9a9a", "#3b6ea5"], alpha=0.85)
    for i, v in enumerate(agree):
        ax.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("Spearman with GCN IG", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title("e  Cross-host attribution agreement", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    # ---- f: host quality journey
    ax = axes[1, 2]
    configs = ["lite\nKMeans", "official port\nKMeans", "official port\nGMM"]
    aris = [0.297, 0.293, 0.522]
    ax.bar(configs, aris, color=["#9a9a9a", "#e8a13c", "#c2352b"], alpha=0.85)
    for i, v in enumerate(aris):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("clustering ARI", fontsize=9)
    ax.set_ylim(0, 0.65)
    ax.set_title("f  Host quality: lite vs faithful port", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("MyerST on human dorsolateral prefrontal cortex: boundary "
                 "attribution with paper-grade hosts", fontsize=12.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{OUT}/fig3_dlpfc.png", dpi=200)
    fig.savefig(f"{OUT}/fig3_dlpfc.pdf")
    print(f"saved {OUT}/fig3_dlpfc.png / .pdf")


if __name__ == "__main__":
    main()
