"""Fig 4 v4: flagship Xenium CCC figure, Nature-style redesign.

Layout 3x3:
a full-section H&E + ROI | b H&E window | c cell types
d T-cell states + CXCL12 | e psi map | f pathway effects
g sender decomposition | h method comparison | i PTN cross-cohort
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, "scripts")
from figstyle import use_style, panel_label, clean_spatial, scalebar

OUT = "outputs"
CT_COLORS = {"Cancer Epithelial": "#c2352b", "T-cells": "#3b6ea5",
             "CAFs": "#e8a13c", "Myeloid": "#5a9e6f",
             "Normal Epithelial": "#b087c7", "Endothelial": "#7fb3d5",
             "B-cells": "#f4d03f", "Plasmablasts": "#d5a6bd", "PVL": "#808b96"}
PATHWAYS = [("CXCL12→CXCR4", -1.068, -38.9), ("CD80→CTLA4", 0.208, 3.3),
            ("CD86→CTLA4", -0.226, -8.8), ("PTN→SDC4", -0.216, -4.4)]
SENDER_FX = [("CAFs", -0.149, -22.8), ("Cancer Ep.", -0.129, -16.1),
             ("Myeloid", -0.134, -28.3), ("Normal Ep.", -0.066, -12.5)]
E14 = [("CXCL12→CXCR4", 0.010, 0.463, 38.9), ("CD80→CTLA4", 0.050, 0.090, 3.3),
       ("CD86→CTLA4", 0.010, 0.047, 8.8), ("PTN→SDC4", 0.832, 0.219, 4.4)]
PTN_COHORTS = [("Xenium\nRep1", -4.4), ("Xenium\nRep2", 6.8), ("Visium", 12.6)]


def main():
    use_style()
    os.makedirs(OUT, exist_ok=True)
    d = np.load("data/processed/e10e_ccxcl12.npz", allow_pickle=True)
    labs, coords = d["labs"], d["coords"]
    is_t, t_act, cx = d["is_t"], d["t_act"], d["x_cxcl12"]
    use, obs = d["use"], d["obs"]

    fig, axes = plt.subplots(3, 3, figsize=(8.6, 8.4))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.06,
                        wspace=0.42, hspace=0.52)

    # a: full H&E + ROI
    ax = axes[0, 0]
    ax.imshow(mpimg.imread("data/processed/he_full_roi.png"))
    ax.axis("off")
    panel_label(ax, "a", -0.02, 1.06)
    ax.set_title("H&E, full section (ROI boxed)", fontsize=8.5, pad=6)

    # b: H&E window
    ax = axes[0, 1]
    ax.imshow(mpimg.imread("data/processed/he_window.png"))
    ax.axis("off")
    panel_label(ax, "b", -0.05, 1.05)
    ax.set_title("H&E, analysis window (1.5 mm)", fontsize=8.5, pad=3)

    # c: cell types
    ax = axes[0, 2]
    for l, c in CT_COLORS.items():
        m = labs == l
        if m.sum():
            ax.scatter(coords[m, 0], coords[m, 1], s=1.2, c=c,
                       edgecolors="none", rasterized=True)
    clean_spatial(ax)
    panel_label(ax, "c", -0.05, 1.05)
    ax.set_title("Cell types (12,706 cells)", fontsize=8.5, pad=3)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l, ms=3)
               for l, c in CT_COLORS.items() if (labs == l).sum()]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.05, -0.04),
              ncol=3, markerscale=1.2, handletextpad=0.15,
              columnspacing=0.6, borderaxespad=0)
    scalebar(ax, coords[:, 0].min() + 80, coords[:, 1].min() + 250, 300)

    # d: T-cell states + CXCL12
    ax = axes[1, 0]
    ax.scatter(coords[:, 0], coords[:, 1], s=0.8, c="#e6e6e0",
               edgecolors="none", rasterized=True)
    m = cx > 0
    ax.scatter(coords[m, 0], coords[m, 1], s=1.8, c="#5a9e6f", alpha=0.55,
               edgecolors="none", rasterized=True)
    for st, c, lab in [(0, "#3b6ea5", "stromal T"), (1, "#c2352b", "infiltrated T")]:
        mm = is_t & (t_act == st)
        ax.scatter(coords[mm, 0], coords[mm, 1], s=2.2, c=c, alpha=0.85,
                   edgecolors="none", rasterized=True)
    clean_spatial(ax)
    panel_label(ax, "d", -0.05, 1.05)
    ax.set_title("T-cell states vs CXCL12 sources", fontsize=8.5, pad=3)
    handles = [plt.Line2D([], [], marker="o", ls="", color="#5a9e6f",
                          label="CXCL12+", ms=3),
               plt.Line2D([], [], marker="o", ls="", color="#c2352b",
                          label="infiltrated T", ms=3),
               plt.Line2D([], [], marker="o", ls="", color="#3b6ea5",
                          label="stromal T", ms=3)]
    ax.legend(handles=handles, loc="upper left", markerscale=1.2,
              handletextpad=0.2, borderaxespad=0, bbox_to_anchor=(-0.05, -0.03), ncol=3)

    # e: psi map
    ax = axes[1, 1]
    ax.scatter(coords[:, 0], coords[:, 1], s=0.8, c="#efefea",
               edgecolors="none", rasterized=True)
    vmax = np.percentile(np.abs(obs), 95)
    sc = ax.scatter(coords[use, 0], coords[use, 1], s=2.5, c=obs, cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, edgecolors="none", rasterized=True)
    clean_spatial(ax)
    panel_label(ax, "e", -0.05, 1.05)
    ax.set_title("Per-T-cell CXCL12 effect (ψ)", fontsize=8.5, pad=3)
    cb = fig.colorbar(sc, ax=ax, fraction=0.038, pad=0.03)
    cb.set_label("ψ (logit)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_linewidth(0.5)

    # f: pathway effects
    ax = axes[1, 2]
    names = [p[0] for p in PATHWAYS]
    vals = [p[1] for p in PATHWAYS]
    ts = [p[2] for p in PATHWAYS]
    cols = ["#c2352b" if abs(t) > 3 else "#b9b9b2" for t in ts]
    ax.barh(range(len(names)), vals, color=cols, height=0.62)
    for i, (v, t) in enumerate(zip(vals, ts)):
        if abs(v) > 0.5:
            ax.text(v / 2, i, f"t={t:+.1f}", va="center", ha="center",
                    fontsize=7.5, color="white", fontweight="bold")
        else:
            ax.text(v - 0.08 if v < 0 else v + 0.08, i, f"t={t:+.1f}",
                    va="center", ha="right" if v < 0 else "left",
                    fontsize=7.5, color="#444")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7.5)
    ax.axvline(0, c="black", lw=0.6)
    ax.set_xlim(-1.6, 0.75)
    panel_label(ax, "f", -0.35, 1.08)
    ax.set_title("Pathway effects on T-cell location", fontsize=8.5, pad=3)
    ax.set_xlabel("mean ψ (paired t in bars)", fontsize=7.5)

    # g: sender decomposition
    ax = axes[2, 0]
    names = [s[0] for s in SENDER_FX]
    vals = [s[1] for s in SENDER_FX]
    ts = [s[2] for s in SENDER_FX]
    ax.bar(range(len(names)), vals,
           color=["#e8a13c", "#c2352b", "#5a9e6f", "#b087c7"], width=0.62)
    for i, (v, t) in enumerate(zip(vals, ts)):
        ax.text(i, v + 0.006, f"t={t:+.1f}", ha="center", va="bottom", fontsize=6.5,
                color="black" if i == 0 else "white", fontweight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7)
    ax.axhline(0, c="black", lw=0.6)
    panel_label(ax, "g", -0.16, 1.08)
    ax.set_title("CXCL12 effect by sender type", fontsize=8.5, pad=3)
    ax.set_ylabel("mean ψ", fontsize=7.5)

    # h: method comparison (log scale)
    ax = axes[2, 1]
    x = np.arange(len(E14))
    w = 0.26
    myst = [e[3] for e in E14]
    cpdb = [-np.log10(e[1]) for e in E14]
    nn = [e[2] for e in E14]
    ax.bar(x - w, myst, w, color="#c2352b", label="MyerST |t|")
    ax.bar(x, cpdb, w, color="#3b6ea5", label="CPDB -log10(p)")
    ax.bar(x + w, nn, w, color="#e8a13c", label="NicheNet |r|")
    ax.set_yscale("log")
    ax.set_ylim(0.01, 100)
    ax.axhline(3, ls=":", c="#888", lw=0.8)
    ax.text(3.4, 3.5, "t = 3", fontsize=6.5, color="#666", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([e[0].split("→")[0] for e in E14], fontsize=7)
    ax.legend(loc="upper right", handlelength=1.2)
    panel_label(ax, "h", -0.16, 1.08)
    ax.set_title("Same pairs, three methods", fontsize=8.5, pad=3)

    # i: PTN cross-cohort
    ax = axes[2, 2]
    names = [c[0] for c in PTN_COHORTS]
    ts = [c[1] for c in PTN_COHORTS]
    ax.bar(range(len(names)), ts, color="#c2352b", width=0.55)
    for i, t in enumerate(ts):
        ax.text(i, t + 0.6 * np.sign(t), f"t={t:+.1f}", ha="center",
                va="bottom" if t > 0 else "top", fontsize=7)
    ax.axhline(0, c="black", lw=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7)
    panel_label(ax, "i", -0.16, 1.08)
    ax.set_title("PTN→SDC4 across cohorts (CPDB: missed)", fontsize=8.5, pad=6)
    ax.set_ylabel("paired t", fontsize=7.5)

    fig.savefig(f"{OUT}/fig4_v4_nature.png")
    fig.savefig(f"{OUT}/fig4_v4_nature.pdf")
    print(f"saved {OUT}/fig4_v4_nature.png / .pdf")


if __name__ == "__main__":
    main()
