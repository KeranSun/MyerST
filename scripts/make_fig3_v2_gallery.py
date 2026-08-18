"""Fig 3 v2: 12-slice DLPFC gallery (the wall of tissues) + stats.

Top: 3x4 grid of all 12 slices colored by cortical layer (grouped by donor).
Bottom: b marker-hit dot plot | c cross-slice Spearman histogram |
        d host quality gates attribution.
"""

import os
import sys

import numpy as np
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, "scripts")
from figstyle import use_style, panel_label, clean_spatial

OUT = "outputs"
SLICES = ["151507", "151508", "151509", "151510", "151669", "151670",
          "151671", "151672", "151673", "151674", "151675", "151676"]
DONORS = ["Br2720"] * 4 + ["Br3871"] * 4 + ["Br8667"] * 4
LAYER_COLORS = {"L1": "#e8c547", "L2": "#e8963c", "L3": "#c2352b",
                "L4": "#8e44ad", "L5": "#3b6ea5", "L6": "#2c6e5f", "WM": "#9a9a9a"}
ORDER = ["L1", "L2", "L3", "L4", "L5", "L6", "WM"]
MARKERS = ["PCP4", "KRT17", "RORB", "CCK"]


def main():
    use_style()
    os.makedirs(OUT, exist_ok=True)

    fig = plt.figure(figsize=(9.6, 9.2))
    gs = fig.add_gridspec(3, 4, left=0.04, right=0.99, top=0.94, bottom=0.33,
                          wspace=0.08, hspace=0.28)

    # ---- top: 3x4 gallery (12 slices), rows grouped by donor
    for k, sid in enumerate(SLICES):
        r, c = k // 4, k % 4
        ax = fig.add_subplot(gs[r, c])
        a = ad.read_h5ad(f"data/raw/DLPFC_{sid}.h5ad")
        layer = a.obs["layer"].astype(str).to_numpy()
        coords = np.asarray(a.obsm["spatial"], dtype=float)
        for l in ORDER:
            m = layer == l
            if m.sum():
                ax.scatter(coords[m, 0], coords[m, 1], s=1.5,
                           c=LAYER_COLORS[l], edgecolors="none",
                           rasterized=True)
        clean_spatial(ax)
        if k == 0:
            panel_label(ax, "a", -0.10, 1.10)
        ax.set_title(sid, fontsize=7.5, pad=1)
        if c == 0:
            ax.text(-0.16, 0.5, DONORS[k], transform=ax.transAxes,
                    rotation=90, va="center", ha="center", fontsize=8,
                    color="#555")
    # shared legend
    handles = [plt.Line2D([], [], marker="o", ls="", color=LAYER_COLORS[l],
                          label=l, ms=4) for l in ORDER]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.285),
               ncol=7, markerscale=1.4, handletextpad=0.15, columnspacing=1.0,
               frameon=False)

    # ---- bottom row: three stat panels
    gs2 = fig.add_gridspec(1, 3, left=0.06, right=0.97, top=0.225,
                           bottom=0.05, wspace=0.38)

    # b: marker hits dot plot (slices x markers)
    axb = fig.add_subplot(gs2[0, 0])
    hits = {}
    for sid in SLICES:
        c_ = np.load(f"data/processed/e12_{sid}.npz", allow_pickle=True)
        genes, ig = c_["genes"], c_["ig"]
        top20 = set(genes[np.argsort(ig)[::-1][:20]])
        hits[sid] = [m in top20 for m in MARKERS]
    for j, sid in enumerate(SLICES):
        for i, m in enumerate(MARKERS):
            axb.scatter(j, i, s=42, c="#c2352b" if hits[sid][i] else "#e3e3dc",
                        edgecolors="#999" if hits[sid][i] else "#ddd",
                        linewidths=0.4)
    axb.set_xticks(range(len(SLICES)))
    axb.set_xticklabels(SLICES, rotation=90, fontsize=6)
    axb.set_yticks(range(len(MARKERS)))
    axb.set_yticklabels(MARKERS, fontsize=7.5)
    panel_label(axb, "b", -0.28, 1.12)
    axb.set_title("Known markers in top-20 (per slice)", fontsize=8.5, pad=6)
    axb.set_xlim(-0.7, 11.7)
    axb.set_ylim(-0.6, 3.6)

    # c: cross-slice Spearman histogram
    axc = fig.add_subplot(gs2[0, 1])
    igs = {}
    for sid in SLICES:
        c_ = np.load(f"data/processed/e12_{sid}.npz", allow_pickle=True)
        igs[sid] = (c_["genes"], c_["ig"])
    common = set(igs[SLICES[0]][0])
    for sid in SLICES[1:]:
        common &= set(igs[sid][0])
    common = sorted(common)
    vecs = {}
    for sid, (g, s) in igs.items():
        idx = {gg: i for i, gg in enumerate(g)}
        vecs[sid] = np.array([s[idx[x]] for x in common])
    sps = [spearmanr(vecs[SLICES[i]], vecs[SLICES[j]]).statistic
           for i in range(len(SLICES)) for j in range(i + 1, len(SLICES))]
    axc.hist(sps, bins=14, color="#3b6ea5", alpha=0.85, edgecolor="white",
             linewidth=0.4)
    axc.axvline(np.mean(sps), color="#c2352b", lw=1.2, ls="--")
    axc.text(np.mean(sps) + 0.01, axc.get_ylim()[1] * 0.9,
             f"mean = {np.mean(sps):.2f}", fontsize=7.5, color="#c2352b")
    panel_label(axc, "c", -0.20, 1.12)
    axc.set_title("Cross-slice attribution agreement", fontsize=8.5, pad=6)
    axc.set_xlabel("Spearman ρ (66 pairs)", fontsize=7.5)
    axc.set_ylabel("pairs", fontsize=7.5)

    # d: host quality gates attribution
    axd = fig.add_subplot(gs2[0, 2])
    hosts = ["STAGATE-lite\n(ARI ~0.30)", "STAGATE-official\n(ARI 0.52)"]
    agree = [0.087, 0.807]
    axd.bar(hosts, agree, color=["#b9b9b2", "#c2352b"], width=0.5)
    for i, v in enumerate(agree):
        axd.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=8)
    axd.set_ylim(0, 1.0)
    panel_label(axd, "d", -0.20, 1.12)
    axd.set_title("Host quality gates attribution", fontsize=8.5, pad=6)
    axd.set_ylabel("Spearman with GCN IG", fontsize=7.5)

    fig.savefig(f"{OUT}/fig3_v2_gallery.png")
    fig.savefig(f"{OUT}/fig3_v2_gallery.pdf")
    print(f"saved {OUT}/fig3_v2_gallery.png / .pdf")


if __name__ == "__main__":
    main()
