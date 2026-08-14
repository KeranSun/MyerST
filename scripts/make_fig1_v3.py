"""Fig 1 v3: NC-standard pure schematic (no data insets).

Four stages: tissue -> host GNN -> Myerson engine -> verified outputs.
Clean vector, consistent palette, curved arrows. Bottom: audit + benchmark.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = "outputs"
INK = "#2c2c2a"
MUTE = "#6b6b66"
PAL = ["#c2352b", "#3b6ea5", "#e8a13c", "#5a9e6f", "#8e44ad", "#7fb3d5"]


def stage(ax, x, y, w, h, fc="#ffffff", ec="#d8d8d2"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.6",
                                fc=fc, ec=ec, lw=1.2, mutation_aspect=1))


def carrow(ax, x1, y, x2, color=INK):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                                 mutation_scale=22, lw=2.0, color=color,
                                 shrinkA=2, shrinkB=2))


def dots_blob(ax, cx, cy, r, n, colors, seed=0, s=26):
    rng = np.random.default_rng(seed)
    # organic blob boundary
    t = np.linspace(0, 2 * np.pi, 200)
    rb = r * (1 + 0.18 * np.sin(3 * t + 1) + 0.1 * np.cos(5 * t))
    ax.fill(cx + rb * np.cos(t), cy + rb * np.sin(t), color="#f4f4f0",
            ec="#d8d8d2", lw=1, zorder=1)
    pts = rng.uniform(-1, 1, (n, 2))
    pts = pts[(pts ** 2).sum(1) < 1]
    cluster = rng.integers(0, len(colors), len(pts))
    ax.scatter(cx + pts[:, 0] * r * 0.92, cy + pts[:, 1] * r * 0.92,
               s=s, c=[colors[i] for i in cluster], edgecolors="none", zorder=2)


def mini_graph(ax, cx, cy, scale=1.0, halos=True):
    nodes = [(-1.4, 0.3), (-0.5, 0.9), (0.4, 0.5), (1.3, 0.8),
             (-0.9, -0.7), (0.1, -0.4), (1.0, -0.7)]
    edges = [(0, 1), (1, 2), (2, 3), (0, 4), (1, 4), (1, 5), (2, 5), (2, 6),
             (3, 6), (4, 5), (5, 6)]
    P = [(cx + a * scale, cy + b * scale) for a, b in nodes]
    for i, j in edges:
        ax.plot([P[i][0], P[j][0]], [P[i][1], P[j][1]], color="#b9b9b2",
                lw=1.1, zorder=1)
    if halos:
        for grp, col in [([0, 1, 4], "#c2352b"), ([2, 3, 6], "#3b6ea5")]:
            xs = [P[i][0] for i in grp]
            ys = [P[i][1] for i in grp]
            ax.add_patch(Circle((np.mean(xs), np.mean(ys)), 0.62 * scale,
                                fc=col, alpha=0.14, ec=col, lw=1.2, zorder=2))
    for i, (px, py) in enumerate(P):
        ax.add_patch(Circle((px, py), 0.13 * scale, fc="white",
                            ec=INK, lw=1.4, zorder=3))


def main():
    os.makedirs(OUT, exist_ok=True)
    fig = plt.figure(figsize=(13.2, 5.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    y0, hh = 12, 20

    # ---- stage 1: tissue
    stage(ax, 2, y0, 21, hh)
    dots_blob(ax, 12.5, y0 + hh / 2 + 1, 6.6, 260, PAL, seed=3)
    ax.text(12.5, y0 + 1.6, "Spatial omics tissue", ha="center",
            fontsize=11, fontweight="bold", color=INK)
    ax.text(12.5, y0 - 1.4, "Visium / Xenium, multi-cohort", ha="center",
            fontsize=8.5, color=MUTE)

    # ---- stage 2: host GNN
    stage(ax, 27, y0, 21, hh)
    mini_graph(ax, 37.5, y0 + hh / 2 + 2.6, scale=1.25, halos=False)
    ax.text(37.5, y0 + 1.6, "host GNN", ha="center", fontsize=11,
            fontweight="bold", color=INK)
    ax.text(37.5, y0 - 1.4, "any architecture", ha="center", fontsize=8.5,
            color=MUTE)

    # ---- stage 3: Myerson engine
    stage(ax, 52, y0, 21, hh, fc="#fdf6f5", ec="#c2352b")
    mini_graph(ax, 59.4, y0 + hh / 2 + 2.2, scale=1.05, halos=True)
    ax.text(62.5, y0 + 4.4, "connected-coalition games", ha="center",
            fontsize=9.5, fontweight="bold", color="#c2352b")
    ax.text(62.5, y0 + 2.0, "Myerson value: unique under", ha="center",
            fontsize=8.5, color=INK)
    ax.text(62.5, y0 + 0.2, "component efficiency + fairness", ha="center",
            fontsize=8.5, color=INK)
    ax.text(62.5, y0 - 1.4, "MC sampler + coalition cache", ha="center",
            fontsize=8.5, color=MUTE)

    # ---- stage 4: outputs
    stage(ax, 77, y0, 21, hh)
    rng = np.random.default_rng(7)
    # node map
    gx, gy = np.meshgrid(np.linspace(0, 1, 8), np.linspace(0, 1, 8))
    ax.scatter(81.5 + gx * 5.4, y0 + 12.4 + gy * 3.4, s=10,
               c=rng.uniform(-1, 1, 64), cmap="RdBu_r", edgecolors="none")
    ax.text(84.2, y0 + 11.5, "nodes", ha="center", fontsize=8, color=MUTE)
    # edge arcs
    ax.add_patch(Circle((89.3, y0 + 14.2), 0.5, fc="#c2352b", alpha=0.8))
    ax.add_patch(Circle((93.3, y0 + 14.2), 0.5, fc="#3b6ea5", alpha=0.8))
    th = np.linspace(0, np.pi, 60)
    ax.plot(89.3 + 2 * np.cos(th) + 2, y0 + 14.2 + 2.0 * np.sin(th),
            color="#5a9e6f", lw=2.2)
    ax.text(91.3, y0 + 11.5, "edges (CCC)", ha="center", fontsize=8, color=MUTE)
    # gene bars
    for k, (hgt, col) in enumerate([(2.6, "#c2352b"), (1.9, "#c2352b"),
                                    (1.2, "#9a9a9a"), (0.8, "#9a9a9a"),
                                    (0.5, "#9a9a9a")]):
        ax.add_patch(plt.Rectangle((82 + 1.15 * k, y0 + 3.2), 0.9, hgt,
                                   fc=col, alpha=0.85))
    ax.text(84.8, y0 + 2.2, "genes", ha="center", fontsize=8, color=MUTE)
    ax.text(87.5, y0 - 1.4, "verified explanations", ha="center", fontsize=8.5,
            color=MUTE)

    for x1, x2 in [(23.4, 26.6), (48.4, 51.6), (73.4, 76.6)]:
        carrow(ax, x1, y0 + hh / 2, x2)

    # ---- bottom strip: audit + benchmark
    ax.add_patch(FancyBboxPatch((2, 2.5), 46, 6.4, boxstyle="round,pad=0.02,rounding_size=0.5",
                                fc="#f6f6f2", ec="#d8d8d2", lw=1))
    ax.text(25, 7.0, "self-auditing by construction", ha="center", fontsize=10,
            fontweight="bold", color=INK)
    ax.text(25, 4.2, r"$\sum_i \phi_i \;=\; v(N) - v(\varnothing)$"
            "   exact on every run  (DLPFC: 1.2787 = 1.2787)",
            ha="center", fontsize=9.5, color="#c2352b")

    ax.add_patch(FancyBboxPatch((52, 2.5), 46, 6.4, boxstyle="round,pad=0.02,rounding_size=0.5",
                                fc="#f6f6f2", ec="#d8d8d2", lw=1))
    ax.text(75, 7.0, "verification-first benchmark", ha="center", fontsize=10,
            fontweight="bold", color=INK)
    ax.text(75, 4.2, "recovery · masking fidelity · ROAR · cross-cohort — "
            "semantics decide what 'importance' means",
            ha="center", fontsize=9.5, color=INK)

    fig.savefig(f"{OUT}/fig1_v3_schematic.png", dpi=220)
    fig.savefig(f"{OUT}/fig1_v3_schematic.pdf")
    print(f"saved {OUT}/fig1_v3_schematic.png / .pdf")


if __name__ == "__main__":
    main()
