"""Fig 4 v5: Nature-style + forest plot + violin distributions.

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
                        wspace=0.55, hspace=0.52)

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

    # f: forest plot (effect - null mean, 95% CI, paired)
    ax = axes[1, 2]
    tags = {"CXCL12→CXCR4": "pw_CXCL12", "CD80→CTLA4": "pw_CD80",
            "CD86→CTLA4": "pw_CD86", "PTN→SDC4": "pw_PTN"}
    for i, (name, _, t) in enumerate(PATHWAYS):
        o = d[tags[name] + "_obs"]
        nu = d[tags[name] + "_null"]
        diff = o - nu
        m = diff.mean()
        ci = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))
        col = "#c2352b" if abs(t) > 3 else "#b9b9b2"
        ax.errorbar(m, i, xerr=ci, fmt="o", color=col, ecolor=col,
                    elinewidth=1.4, capsize=2.5, ms=5)
        tx, ha = (0.44, "right") if m < -0.8 else (-1.78, "left")
        ax.text(tx, i, f"t={t:+.1f}, n={len(diff)}", va="center",
                ha=ha, fontsize=6.8, color="#444")
    ax.axvline(0, c="black", lw=0.6, ls="--")
    ax.set_yticks(range(len(PATHWAYS)))
    ax.set_yticklabels([p[0] for p in PATHWAYS], fontsize=7.5)
    ax.set_xlim(-1.9, 0.5)
    ax.set_ylim(-0.6, 3.6)
    panel_label(ax, "f", -0.35, 1.08)
    ax.set_title("Pathway effects (paired, 95% CI)", fontsize=8.5, pad=3)
    ax.set_xlabel("ψ_gene − null (logit)", fontsize=7.5)

    # g: sender decomposition
    ax = axes[2, 0]
    names = [s[0] for s in SENDER_FX]
    vals = [s[1] for s in SENDER_FX]
    ts = [s[2] for s in SENDER_FX]
    ns = [1668, 765, 1732, 790]
    ax.bar(range(len(names)), vals, yerr=[1.96 * abs(v / t) for v, t in zip(vals, ts)],
           color=["#e8a13c", "#c2352b", "#5a9e6f", "#b087c7"], width=0.62,
           error_kw=dict(elinewidth=1.0, capsize=3, ecolor="#444"))
    for i, (v, t) in enumerate(zip(vals, ts)):
        ax.text(i, v + 0.006, f"t={t:+.1f}", ha="center", va="bottom", fontsize=6.5,
                color="black" if i == 0 else "white", fontweight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7)
    ax.axhline(0, c="black", lw=0.6)
    panel_label(ax, "g", -0.16, 1.08)
    ax.set_title("CXCL12 effect by sender type", fontsize=8.5, pad=3)
    ax.set_ylabel("mean ψ (95% CI)", fontsize=7.5)

    # h: violin — per-receiver CXCL12 effect vs frequency-matched null
    ax = axes[2, 1]
    o = d["pw_CXCL12_obs"]
    nu = d["pw_CXCL12_null"]
    rng = np.random.default_rng(1)
    parts = ax.violinplot([nu, o], positions=[0, 1], widths=0.55,
                          showmeans=False, showextrema=False)
    for pc, col in zip(parts["bodies"], ["#b9b9b2", "#c2352b"]):
        pc.set_facecolor(col); pc.set_alpha(0.45); pc.set_edgecolor("none")
    for pos, arr, col in [(0, nu, "#666"), (1, o, "#7a1f1f")]:
        jit = rng.normal(0, 0.05, min(len(arr), 300))
        sel = rng.choice(len(arr), min(len(arr), 300), replace=False)
        ax.scatter(np.zeros(len(jit)) + pos + jit, arr[sel], s=3, c=col,
                   alpha=0.35, edgecolors="none", rasterized=True)
        ax.plot([pos - 0.18, pos + 0.18], [arr.mean()] * 2, color="black",
                lw=1.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["frequency-matched\nnull genes", "CXCL12"], fontsize=7.5)
    ax.text(0.5, max(o.max(), 0.2), "paired t = −38.9", ha="center", fontsize=8,
            color="#c2352b", fontweight="bold")
    panel_label(ax, "h", -0.16, 1.08)
    ax.set_title("Per-receiver effect vs null (n = 1,806)", fontsize=8.5, pad=3)
    ax.set_ylabel("ψ (logit)", fontsize=7.5)

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

    fig.savefig(f"{OUT}/fig4_v5_stats.png")
    fig.savefig(f"{OUT}/fig4_v5_stats.pdf")
    print(f"saved {OUT}/fig4_v5_stats.png / .pdf")


if __name__ == "__main__":
    main()
