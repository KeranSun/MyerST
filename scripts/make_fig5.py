"""Fig 5: multi-cohort CCC replication (rows = cohorts).

Columns: a cell-type map | b T-cell states + CXCL12 sources | c pathway effects.
Rep1 numbers from E10e; Rep2/Lung from E11 npz files.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs"
CT_COLORS = {"Cancer Epithelial": "#c2352b", "T-cells": "#3b6ea5",
             "CAFs": "#e8a13c", "Myeloid": "#5a9e6f", "Normal Epithelial": "#b087c7",
             "Endothelial": "#7fb3d5", "B-cells": "#f4d03f",
             "Plasmablasts": "#d5a6bd", "PVL": "#808b96", "Other": "#d9d9d9"}
COHORTS = [
    ("Xenium Breast Rep1", "data/processed/e10e_ccxcl12.npz",
     [("CXCL12->CXCR4", -1.068, -38.9), ("CD80->CTLA4", 0.208, 3.3),
      ("CD86->CTLA4", -0.226, -8.8), ("PTN->SDC4", -0.216, -4.4)]),
    ("Xenium Breast Rep2", "data/processed/e11_xenium_breast_rep2.npz", None),
    ("Xenium Lung (NSCLC)", "data/processed/e11_xenium_lung.npz", None),
    ("Visium Breast", "data/processed/e11_visium_breast.npz", None),
]


def cellmap(ax, labs, coords):
    present = [l for l in CT_COLORS if l in set(labs)]
    s_pt = 2.5 if len(labs) > 2000 else 14.0
    for l in present:
        m = labs == l
        ax.scatter(coords[m, 0], coords[m, 1], s=s_pt, c=CT_COLORS[l],
                   edgecolors="none", rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal"); ax.invert_yaxis()
    handles = [plt.Line2D([], [], marker="o", ls="", color=CT_COLORS[l], label=l, ms=4)
               for l in present]
    ax.legend(handles=handles, fontsize=6, frameon=False, loc="lower left",
              ncol=2, markerscale=1.1)


def statemap(ax, d):
    labs, coords = d["labs"], d["coords"]
    is_t, t_act, cx = d["is_t"], d["t_act"], d["x_cxcl12"]
    s_bg = 2.5 if len(labs) > 2000 else 10.0
    ax.scatter(coords[:, 0], coords[:, 1], s=s_bg, c="#ececec",
               edgecolors="none", rasterized=True)
    m = cx > 0
    ax.scatter(coords[m, 0], coords[m, 1], s=s_bg * 1.6, c="#5a9e6f", alpha=0.55,
               edgecolors="none", rasterized=True)
    for st, c in [(1, "#c2352b"), (0, "#3b6ea5")]:
        mm = is_t & (t_act == st)
        ax.scatter(coords[mm, 0], coords[mm, 1], s=s_bg * 2.0, c=c, alpha=0.85,
                   edgecolors="none", rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal"); ax.invert_yaxis()
    handles = [plt.Line2D([], [], marker="o", ls="", color="#5a9e6f",
                          label="CXCL12+", ms=5),
               plt.Line2D([], [], marker="o", ls="", color="#c2352b",
                          label="infiltrated T", ms=5),
               plt.Line2D([], [], marker="o", ls="", color="#3b6ea5",
                          label="stromal T", ms=5)]
    ax.legend(handles=handles, fontsize=6.5, frameon=False, loc="lower left")


def pathbars(ax, rows):
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    ts = [r[2] for r in rows]
    cols = ["#c2352b" if abs(t) > 5 else "#9a9a9a" for t in ts]
    ax.barh(range(len(names)), vals, color=cols, alpha=0.85, height=0.6)
    for i, (v, t) in enumerate(zip(vals, ts)):
        ax.text(v / 2 if abs(v) > 0.3 else v + 0.03 * np.sign(v), i,
                f"t={t:+.1f}", va="center",
                ha="center" if abs(v) > 0.3 else ("left" if v >= 0 else "right"),
                fontsize=8, color="white" if abs(v) > 0.3 else "black")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, c="black", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(min(vals + [0]) * 1.3, max(vals + [0.35]) * 1.3)


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(len(COHORTS), 3, figsize=(14, 4.1 * len(COHORTS) + 1))
    if len(COHORTS) == 1:
        axes = axes[None, :]
    for ri, (name, path, hard_rows) in enumerate(COHORTS):
        d = np.load(path, allow_pickle=True)
        cellmap(axes[ri, 0], d["labs"], d["coords"])
        L = "abcdefghijklmnop"
        axes[ri, 0].set_title(f"{L[ri*3]}  {name}: cell types", fontsize=11)
        statemap(axes[ri, 1], d)
        axes[ri, 1].set_title(f"{L[ri*3+1]}  {name}: T-cell states vs CXCL12",
                              fontsize=11)
        if hard_rows is not None:
            rows = hard_rows
        else:
            rows = [(str(p[0]), float(p[1]), float(p[2])) for p in d["pathways"]]
        pathbars(axes[ri, 2], rows)
        axes[ri, 2].set_title(f"{L[ri*3+2]}  {name}: pathway effects", fontsize=11)

    fig.suptitle("MyerST CCC analysis replicates across cohorts: one pipeline, "
                 "four datasets, two platforms", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(f"{OUT}/fig5_multi_cohort.png", dpi=200)
    fig.savefig(f"{OUT}/fig5_multi_cohort.pdf")
    print(f"saved {OUT}/fig5_multi_cohort.png / .pdf")


if __name__ == "__main__":
    main()
