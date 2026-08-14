"""Fig 4 v3 (REAL): Xenium CCC flagship with H&E zoom chain.

Row 1: a full-section H&E + ROI | b H&E window | c cell types in window
Row 2: d T-cell states + CXCL12 | e per-receiver psi map | f pathway effects
Row 3: g sender decomposition | h method comparison (E14) | i PTN cross-cohort
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

OUT = "outputs"
CT_COLORS = {"Cancer Epithelial": "#c2352b", "T-cells": "#3b6ea5",
             "CAFs": "#e8a13c", "Myeloid": "#5a9e6f", "Normal Epithelial": "#b087c7",
             "Endothelial": "#7fb3d5", "B-cells": "#f4d03f",
             "Plasmablasts": "#d5a6bd", "PVL": "#808b96"}
# E10e (Rep1) pathway effects, paired t
PATHWAYS = [("CXCL12->CXCR4", -1.068, -38.9), ("CD80->CTLA4", 0.208, 3.3),
            ("CD86->CTLA4", -0.226, -8.8), ("PTN->SDC4", -0.216, -4.4)]
SENDER_FX = [("CAFs", -0.1493, -22.8), ("Cancer Ep.", -0.1293, -16.1),
             ("Myeloid", -0.1342, -28.3), ("Normal Ep.", -0.0658, -12.5)]
# E14 method comparison: (pathway, CPDB p, NicheNet r_loc, MyerST psi, MyerST t)
E14 = [("CXCL12->CXCR4", 0.010, -0.463, -1.068, -38.9),
       ("CD80->CTLA4", 0.050, 0.090, 0.208, 3.3),
       ("CD86->CTLA4", 0.010, -0.047, -0.226, -8.8),
       ("PTN->SDC4", 0.832, -0.219, -0.216, -4.4)]
PTN_COHORTS = [("Xenium Rep1", -4.4), ("Xenium Rep2", 6.8), ("Visium", 12.6)]


def cellmap(ax, labs, coords, s=6):
    for l, c in CT_COLORS.items():
        m = labs == l
        if m.sum():
            ax.scatter(coords[m, 0], coords[m, 1], s=s, c=c, edgecolors="none",
                       rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal"); ax.invert_yaxis()


def main():
    os.makedirs(OUT, exist_ok=True)
    d = np.load("data/processed/e10e_ccxcl12.npz", allow_pickle=True)
    labs, coords = d["labs"], d["coords"]
    is_t, t_act, cx = d["is_t"], d["t_act"], d["x_cxcl12"]
    use, obs = d["use"], d["obs"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 14))

    # a: full-section H&E with ROI
    ax = axes[0, 0]
    ax.imshow(mpimg.imread("data/processed/he_full_roi.png"))
    ax.set_title("a  Xenium breast Rep1, H&E (ROI boxed)", fontsize=11)
    ax.axis("off")

    # b: H&E analysis window
    ax = axes[0, 1]
    ax.imshow(mpimg.imread("data/processed/he_window.png"))
    ax.set_title("b  H&E, analysis window (1.5 mm)", fontsize=11)
    ax.axis("off")

    # c: cell types in window
    ax = axes[0, 2]
    cellmap(ax, labs, coords)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l, ms=5)
               for l, c in CT_COLORS.items() if (labs == l).sum()]
    ax.legend(handles=handles, fontsize=6.5, frameon=False, loc="lower left",
              ncol=2, markerscale=1.1)
    ax.set_title("c  Cell types (12,706 cells)", fontsize=11)

    # d: T-cell states + CXCL12
    ax = axes[1, 0]
    ax.scatter(coords[:, 0], coords[:, 1], s=4, c="#ececec", edgecolors="none",
               rasterized=True)
    m = cx > 0
    ax.scatter(coords[m, 0], coords[m, 1], s=6, c="#5a9e6f", alpha=0.6,
               edgecolors="none", rasterized=True)
    for st, c, lab in [(1, "#c2352b", "infiltrated T"), (0, "#3b6ea5", "stromal T")]:
        mm = is_t & (t_act == st)
        ax.scatter(coords[mm, 0], coords[mm, 1], s=7, c=c, alpha=0.85,
                   edgecolors="none", rasterized=True)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color="#5a9e6f",
                                  label="CXCL12+", ms=5),
                       plt.Line2D([], [], marker="o", ls="", color="#c2352b",
                                  label="infiltrated T", ms=5),
                       plt.Line2D([], [], marker="o", ls="", color="#3b6ea5",
                                  label="stromal T", ms=5)],
              fontsize=7.5, frameon=False, loc="lower left")
    ax.set_title("d  T-cell states vs CXCL12 sources", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal"); ax.invert_yaxis()

    # e: psi map
    ax = axes[1, 1]
    ax.scatter(coords[:, 0], coords[:, 1], s=4, c="#eeeeea", edgecolors="none",
               rasterized=True)
    vmax = np.percentile(np.abs(obs), 90)
    sc = ax.scatter(coords[use, 0], coords[use, 1], s=8, c=obs, cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, edgecolors="none", rasterized=True)
    cb = fig.colorbar(sc, ax=ax, shrink=0.75)
    cb.set_label("CXCL12 psi", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("e  Per-T-cell CXCL12 effect (psi)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal"); ax.invert_yaxis()

    # f: pathway bars
    ax = axes[1, 2]
    names = [p[0] for p in PATHWAYS]
    vals = [p[1] for p in PATHWAYS]
    ts = [p[2] for p in PATHWAYS]
    cols = ["#c2352b" if abs(t) > 3 else "#9a9a9a" for t in ts]
    ax.barh(range(len(names)), vals, color=cols, alpha=0.85, height=0.6)
    for i, (v, t) in enumerate(zip(vals, ts)):
        ax.text(v / 2, i, f"t={t:+.1f}", va="center", ha="center", fontsize=9,
                color="white", fontweight="bold")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, c="black", lw=0.8)
    ax.set_title("f  Pathway effects on T-cell location", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    # g: sender decomposition
    ax = axes[2, 0]
    names = [s[0] for s in SENDER_FX]
    vals = [s[1] for s in SENDER_FX]
    ts = [s[2] for s in SENDER_FX]
    ax.bar(range(len(names)), vals,
           color=["#e8a13c", "#c2352b", "#5a9e6f", "#b087c7"], alpha=0.85)
    for i, (v, t) in enumerate(zip(vals, ts)):
        ax.text(i, v / 2, f"t={t:+.1f}", ha="center", va="center", fontsize=8,
                color="white", fontweight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8.5)
    ax.axhline(0, c="black", lw=0.8)
    ax.set_title("g  CXCL12 effect by sender type", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    # h: E14 method comparison
    ax = axes[2, 1]
    x = np.arange(len(E14))
    w = 0.27
    cpdb = [-np.log10(e[1]) for e in E14]
    nn = [abs(e[2]) for e in E14]
    myst = [abs(e[4]) for e in E14]
    ax.bar(x - w, myst, w, color="#c2352b", alpha=0.85, label="MyerST |t|")
    ax.bar(x, cpdb, w, color="#3b6ea5", alpha=0.85,
           label="CPDB -log10(p)")
    ax.bar(x + w, nn, w, color="#e8a13c", alpha=0.85, label="NicheNet |r|")
    ax.set_yscale("log")
    ax.axhline(3, ls=":", c="#9a9a9a", lw=1)
    ax.text(len(E14) - 0.5, 3.3, "significance (t=3)", fontsize=7.5,
            color="#777", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([e[0].replace("->", "\n->") for e in E14], fontsize=7.5)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("h  Same pairs, three methods (E14)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    # i: PTN cross-cohort
    ax = axes[2, 2]
    names = [c[0] for c in PTN_COHORTS]
    ts = [c[1] for c in PTN_COHORTS]
    ax.bar(range(len(names)), ts,
           color=["#c2352b" if abs(t) > 3 else "#9a9a9a" for t in ts],
           alpha=0.85)
    for i, t in enumerate(ts):
        ax.text(i, t + 0.5 * np.sign(t), f"t={t:+.1f}", ha="center", fontsize=9)
    ax.axhline(0, c="black", lw=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_title("i  PTN->SDC4 across cohorts (CPDB: p=0.83, missed)",
                 fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("MyerST decomposes tumor-immune communication in Xenium "
                 "breast cancer", fontsize=13.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{OUT}/fig4_v3_he.png", dpi=200)
    fig.savefig(f"{OUT}/fig4_v3_he.pdf")
    print(f"saved {OUT}/fig4_v3_he.png / .pdf")


if __name__ == "__main__":
    main()
