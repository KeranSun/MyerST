"""Fig 4: flagship Xenium CCC figure (2x3).

a  cell-type map of the tumor-immune interface subwindow
b  T-cell infiltration states + CXCL12-expressing cells
c  per-receiver CXCL12 psi map (geography of the exclusion effect)
d  pathway-level ligand effects (psi_gene, paired t annotated)
e  CXCL12 effect by sender cell type
f  target-semantics lesson: cytotoxicity vs location target
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
             "Plasmablasts": "#d5a6bd", "PVL": "#808b96"}

# pathway effects from E10e (paired t)
PATHWAYS = [("CXCL12->CXCR4", -1.0676, -38.9), ("CD80->CTLA4", +0.2075, +3.3),
            ("CD86->CTLA4", -0.2255, -8.8), ("PTN->SDC4", -0.2158, -4.4)]
# sender-type breakdown from E10e rerun (filled from stdout)
SENDER_FX = [("CAFs", -0.1493, -22.8), ("Cancer Ep.", -0.1293, -16.1),
             ("Myeloid", -0.1342, -28.3), ("Normal Ep.", -0.0658, -12.5)]


def scatter_cells(ax, coords, colors, s=3, alpha=0.85):
    ax.scatter(coords[:, 0], coords[:, 1], s=s, c=colors, alpha=alpha,
               edgecolors="none", rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.invert_yaxis()


def main():
    d = np.load("data/processed/e10e_ccxcl12.npz", allow_pickle=True)
    labs, coords = d["labs"], d["coords"]
    is_t, t_act = d["is_t"], d["t_act"]
    use, obs, null = d["use"], d["obs"], d["null"]
    cxcl12 = d["x_cxcl12"]
    os.makedirs(OUT, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))

    # ---- a: cell types
    ax = axes[0, 0]
    colors = [CT_COLORS.get(l, "#cccccc") for l in labs]
    order = np.argsort([l != "Cancer Epithelial" for l in labs])  # cancer on top
    scatter_cells(ax, coords[order], [colors[i] for i in order])
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l, ms=5)
               for l, c in CT_COLORS.items() if l in set(labs)]
    ax.legend(handles=handles, fontsize=6.5, frameon=False, loc="lower left",
              ncol=2, markerscale=1.2)
    ax.set_title("a  Tumor-immune interface (Xenium breast cancer)", fontsize=11)

    # ---- b: T-cell states + CXCL12
    ax = axes[0, 1]
    scatter_cells(ax, coords, ["#e8e8e8"] * len(labs), s=3)
    cx = cxcl12 > 0
    ax.scatter(coords[cx, 0], coords[cx, 1], s=6, c="#5a9e6f", alpha=0.6,
               edgecolors="none", label="CXCL12+ cells", rasterized=True)
    for st, c, lab in [(1, "#c2352b", "infiltrated T (<30um)"),
                       (0, "#3b6ea5", "stromal T")]:
        m = is_t & (t_act == st)
        ax.scatter(coords[m, 0], coords[m, 1], s=6, c=c, alpha=0.85,
                   edgecolors="none", label=lab, rasterized=True)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.set_title("b  T-cell infiltration states vs CXCL12 sources", fontsize=11)

    # ---- c: psi map
    ax = axes[0, 2]
    scatter_cells(ax, coords, ["#ececec"] * len(labs), s=3)
    vmax = np.percentile(np.abs(obs), 95)
    sc = ax.scatter(coords[use, 0], coords[use, 1], s=8, c=obs, cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, edgecolors="none", rasterized=True)
    fig.colorbar(sc, ax=ax, shrink=0.75, label="CXCL12 psi (logit)")
    ax.set_title("c  Per-T-cell CXCL12 effect (psi_gene)", fontsize=11)

    # ---- d: pathway effects
    ax = axes[1, 0]
    names = [p[0] for p in PATHWAYS]
    vals = [p[1] for p in PATHWAYS]
    ts = [p[2] for p in PATHWAYS]
    cols = ["#c2352b" if abs(t) > 3 else "#9a9a9a" for _, t in zip(vals, ts)]
    ax.barh(range(len(names)), vals, color=cols, alpha=0.85)
    for i, (v, t) in enumerate(zip(vals, ts)):
        ax.text(v / 2, i, f"t={t:+.1f}", va="center", ha="center",
                fontsize=9, color="white", fontweight="bold")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, c="black", lw=0.8)
    ax.set_title("d  Pathway-level ligand effects on T-cell location", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    # ---- e: sender-type breakdown (from rerun stdout, filled below)
    ax = axes[1, 1]
    sender_data = SENDER_FX or []
    if sender_data:
        names_e = [s[0] for s in sender_data]
        vals_e = [s[1] for s in sender_data]
        ts_e = [s[2] for s in sender_data]
        ax.bar(range(len(names_e)), vals_e,
               color=["#e8a13c", "#c2352b", "#5a9e6f", "#b087c7"][:len(names_e)],
               alpha=0.85)
        for i, (v, t) in enumerate(zip(vals_e, ts_e)):
            ax.text(i, v + 0.02 * np.sign(v), f"t={t:+.1f}", ha="center",
                    fontsize=9)
        ax.set_xticks(range(len(names_e)))
        ax.set_xticklabels(names_e, fontsize=8, rotation=15)
    ax.axhline(0, c="black", lw=0.8)
    ax.set_title("e  CXCL12 effect by sender cell type", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    # ---- f: target-semantics lesson
    ax = axes[1, 2]
    cats = ["cytotoxicity target\n(E10, wrong semantics)",
            "location target\n(E10e, matched)"]
    obs_f = [0.064, -1.068]
    null_f = [0.063, 0.308]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, obs_f, w, color="#c2352b", alpha=0.85, label="CXCL12 effect")
    ax.bar(x + w / 2, null_f, w, color="#9a9a9a", alpha=0.85,
           label="frequency-matched null")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=8)
    ax.axhline(0, c="black", lw=0.8)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("f  Same axis, different target semantics", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("MyerST explains tumor-immune communication: CXCL12-mediated "
                 "T-cell exclusion recovered with host- and target-matched design",
                 fontsize=12.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{OUT}/fig4_xenium_ccc.png", dpi=200)
    fig.savefig(f"{OUT}/fig4_xenium_ccc.pdf")
    print(f"saved {OUT}/fig4_xenium_ccc.png / .pdf")


if __name__ == "__main__":
    main()
