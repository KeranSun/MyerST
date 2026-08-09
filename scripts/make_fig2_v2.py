"""Fig 2 v2: 9-panel high-density version.

Row 1 — benchmark core:    a P1 node bars | b P2 node bars | c rank-flip slopegraph
Row 2 — mechanism:         d gene heatmap | e Myerson MC convergence | f CCC spatial map
Row 3 — CCC validation:    g edge synergy distributions | h IG gene recovery | i key findings
"""

import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from myerst.data.graph import build_knn_graph
from myerst.attribution.myerson import exact_myerson, mc_myerson
from myerst.benchmark.ccc_simulator import CCCSimulator

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
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    with open("data/processed/e8_matrix.pkl", "rb") as f:
        res = pickle.load(f)
    e9 = np.load("data/processed/e9_ccc_edges.npz")
    e9b = np.load("data/processed/e9b_local_ccc.npz")
    os.makedirs(OUT, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))

    # ---------- Row 1
    grouped_bars(axes[0, 0], res, "P1_node", NODE_METHODS,
                 "a  Recovery AUROC (node)", "AUROC")
    grouped_bars(axes[0, 1], res, "P2_node", NODE_METHODS,
                 "b  Masking fidelity (node)", "decay AUC")

    ax = axes[0, 2]
    p1 = {m: np.mean([res[r]["P1_node"][m] for r in REGIMES]) for m in NODE_METHODS}
    p2 = {m: np.mean([res[r]["P2_node"][m] for r in REGIMES]) for m in NODE_METHODS}
    rank_p1 = {m: r for r, m in enumerate(sorted(NODE_METHODS, key=lambda z: -p1[z]))}
    rank_p2 = {m: r for r, m in enumerate(sorted(NODE_METHODS, key=lambda z: -p2[z]))}
    for m in NODE_METHODS:
        ax.plot([0, 1], [rank_p1[m], rank_p2[m]], "-o", color=COLORS[m], lw=2.5, ms=6)
        ax.text(-0.05, rank_p1[m], f"{m} ({p1[m]:.2f})", ha="right", va="center",
                fontsize=8, color=COLORS[m])
        ax.text(1.05, rank_p2[m], f"{m} ({p2[m]:.2f})", ha="left", va="center",
                fontsize=8, color=COLORS[m])
    ax.set_xlim(-0.85, 1.85)
    ax.set_ylim(3.6, -0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["P1 recovery", "P2 fidelity"], fontsize=9)
    ax.set_yticks([])
    ax.set_title("c  Rank flip across protocols", fontsize=10)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)

    # ---------- Row 2
    ax = axes[1, 0]
    protos = ["P1_gene", "P2_gene", "P3_gene"]
    M = np.array([[np.mean([res[r][p][m] for r in REGIMES]) for p in protos]
                  for m in GENE_METHODS])
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0.4, vmax=1.1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["P1", "P2", "P3"], fontsize=9)
    ax.set_yticks(range(len(GENE_METHODS)))
    ax.set_yticklabels(GENE_METHODS, fontsize=8)
    for i in range(len(GENE_METHODS)):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] < 0.85 else "black")
    ax.set_title("d  Gene level: method x protocol", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.75)

    # e: Myerson MC convergence (quick exact-vs-MC on kNN graph)
    ax = axes[1, 1]
    rng = np.random.default_rng(2026)
    coords = rng.uniform(0, 20, size=(10, 2))
    edges10 = build_knn_graph(coords, k=4)
    v = lambda s: float(len(s)) ** 2
    phi_exact = exact_myerson(10, edges10, v)
    ns = [64, 256, 1024, 4096, 16384, 65536]
    errs = []
    for n_s in ns:
        phi_mc = mc_myerson(10, edges10, v, n_samples=n_s, seed=0)
        errs.append(np.max(np.abs(phi_mc - phi_exact)))
    ax.loglog(ns, errs, "o-", color="#c2352b", lw=2)
    ax.loglog(ns, errs[2] * (np.array(ns) / ns[2]) ** -0.5, "--", color="#9a9a9a",
              label="1/sqrt(M)")
    ax.set_xlabel("MC samples", fontsize=8)
    ax.set_ylabel("max |error| vs exact", fontsize=8)
    ax.set_title("e  Myerson sampler converges as 1/sqrt(M)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # f: CCC spatial map
    ax = axes[1, 2]
    labels3 = e9["labels3"]
    cc = e9["coords"]
    colors3 = {0: "#4a7fb5", 1: "#c9d6e3", 2: "#c2352b"}
    for c, name in [(0, "sender"), (1, "quiet receiver"), (2, "activating receiver")]:
        m = labels3 == c
        ax.scatter(cc[m, 0], cc[m, 1], s=18, c=colors3[c], label=name,
                   edgecolors="none")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_title("f  CCC simulation: cooperative signaling zone", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()

    # ---------- Row 3
    # g: edge synergy distributions (E9b local CCC game)
    ax = axes[2, 0]
    syn, truth_e = e9b["psi"], e9b["truth"]
    parts = [syn[truth_e], syn[~truth_e]]
    bp = ax.boxplot(parts, tick_labels=["communicating\n(ground truth)", "other edges"],
                    showfliers=False, widths=0.5, patch_artist=True)
    for patch, c in zip(bp["boxes"], ["#c2352b", "#9a9a9a"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    from scripts.e1_driver_gene_recovery import auroc
    au = auroc(syn, truth_e)
    ax.set_title(f"g  Local LR synergy: true vs other (AUROC={au:.2f})",
                 fontsize=10)
    ax.set_ylabel("pairwise synergy", fontsize=8)
    ax.axhline(0, ls=":", c="#9a9a9a", lw=1)
    ax.spines[["top", "right"]].set_visible(False)

    # h: IG gene recovery on CCC
    ax = axes[2, 1]
    ig_g, truth_g = e9["ig_gene"], e9["truth_gene"]
    res_ccc = CCCSimulator(grid_size=40, n_genes=300, seed=0).simulate()
    names = res_ccc.data.gene_names
    top = np.argsort(ig_g)[::-1][:15]
    ax.barh(range(15), ig_g[top], color=["#c2352b" if truth_g[i] else "#9a9a9a"
                                         for i in top], alpha=0.8)
    ax.set_yticks(range(15))
    ax.set_yticklabels([names[i] for i in top], fontsize=7)
    ax.invert_yaxis()
    au_g = auroc(ig_g, truth_g)
    ax.set_title(f"h  IG gene attribution on CCC (AUROC={au_g:.2f})", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    # i: key findings text
    ax = axes[2, 2]
    ax.axis("off")
    findings = [
        "i  Key findings",
        "",
        "1. No free lunch: the best explainer",
        "   depends on the evaluation protocol",
        "   (rank flip: IG-node <-> Myerson).",
        "",
        "2. Attention as explanation scores",
        "   BELOW RANDOM in masking fidelity.",
        "",
        "3. Myerson values are self-auditing:",
        "   efficiency sum(phi)=v(N)-v(0)",
        "   holds exactly (MC-independent).",
        "",
        "4. Edge synergy recovers true",
        "   communicating edges under",
        "   cooperative (CCC) targets.",
    ]
    ax.text(0.02, 0.98, "\n".join(findings), va="top", fontsize=9.5,
            family="monospace")
    ax.set_title("summary", fontsize=10)

    fig.suptitle("MyerST: topology-constrained attribution with verifiable fidelity",
                 fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(f"{OUT}/fig2_v2.png", dpi=200)
    fig.savefig(f"{OUT}/fig2_v2.pdf")
    print(f"saved {OUT}/fig2_v2.png / .pdf")


if __name__ == "__main__":
    main()
