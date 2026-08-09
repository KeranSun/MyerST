"""E10d: LR-restricted host — force the model to use signaling genes.

E10/E10b found ~zero ligand-specific signal on a CYTOTOXICITY target. But
CXCL12-CXCR4 is a retention/trafficking axis (CAFs trap T-cells in stroma),
not an activation axis. Here the target is T-cell LOCATION: infiltrated
(within 30um of a Cancer Epithelial cell) vs stromal.
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import anndata as ad

from myerst.models.gcn import GCN, build_norm_adj
from myerst.data.graph import build_knn_graph, adjacency_list
from scripts.e1_driver_gene_recovery import auroc
from scipy.spatial import cKDTree

LR_PAIRS = [("CXCL12", "CXCR4"), ("CD274", "PDCD1"), ("CD80", "CTLA4"),
            ("CD86", "CTLA4"), ("PTN", "SDC4")]
SUBWINDOW = 1500.0
K_GRAPH = 8
N_PERM = 100


@torch.no_grad()
def group_marginal(model, xz, adj_sub, zone, j, sender_groups: dict[str, np.ndarray],
                   x0z, target_cls):
    """v({S, j}) - v({j}) for each named sender group; evaluated on subgraph."""
    zidx = {int(n): k for k, n in enumerate(zone)}
    n_v = len(sender_groups) + 1
    xb = xz.unsqueeze(0).repeat(n_v, 1, 1)
    xb[:, :, :] = x0z
    for k, (name, members) in enumerate(sender_groups.items()):
        keep = [zidx[int(m)] for m in members] + [zidx[int(j)]]
        xb[k, keep] = xz[keep]
    xb[-1, zidx[int(j)]] = xz[zidx[int(j)]]      # {j} only
    logits = model(xb, adj_sub)
    tgt = logits[:, zidx[int(j)], target_cls].numpy()
    v_j = tgt[-1]
    return {name: float(t - v_j) for name, t in zip(sender_groups, tgt[:-1])}


def main():
    t0 = time.perf_counter()
    print("=" * 72)
    print("E10d: LR-restricted host on Xenium CCC")
    print("=" * 72)

    a = ad.read_h5ad("data/processed/xenium_crop.h5ad")
    xy = a.obsm["spatial"]
    labs_full = a.obs["celltype"].to_numpy()
    best = (-1.0, None)
    for x0 in np.arange(xy[:, 0].min(), xy[:, 0].max() - SUBWINDOW, 500):
        for y0 in np.arange(xy[:, 1].min(), xy[:, 1].max() - SUBWINDOW, 500):
            m = ((xy[:, 0] >= x0) & (xy[:, 0] < x0 + SUBWINDOW)
                 & (xy[:, 1] >= y0) & (xy[:, 1] < y0 + SUBWINDOW))
            if m.sum() < 4000:
                continue
            vc = pd.Series(labs_full[m]).value_counts(normalize=True)
            trio = min(vc.get("Cancer Epithelial", 0), vc.get("T-cells", 0),
                       vc.get("CAFs", 0))
            if trio > best[0]:
                best = (trio, (x0, y0))
    x0, y0 = best[1]
    m = ((xy[:, 0] >= x0) & (xy[:, 0] < x0 + SUBWINDOW)
         & (xy[:, 1] >= y0) & (xy[:, 1] < y0 + SUBWINDOW))
    sub = a[m].copy()

    X = sub.X.toarray().astype(np.float32)
    lib = X.sum(1, keepdims=True)
    Xn = np.log1p(X / np.maximum(lib, 1) * np.median(lib))
    mu, sd = Xn.mean(0), Xn.std(0) + 1e-6
    Xz = ((Xn - mu) / sd).astype(np.float32)
    genes = np.array(sub.var_names)
    labs = sub.obs["celltype"].to_numpy()
    coords = sub.obsm["spatial"]
    gidx_full = {g: i for i, g in enumerate(genes)}
    SIGNAL = ["CXCL12", "CXCR4", "CD274", "PDCD1", "PDCD1LG2", "CD80", "CD86",
              "CTLA4", "TIGIT", "LAG3", "HAVCR2", "CCL5", "CCR7", "CXCL16",
              "CCL20", "CCL8", "CXCL5", "LTB", "PTN", "SDC4", "IL7R", "IL2RA",
              "IL2RG", "IL3RA", "KIT", "IGF1", "EGFR", "ERBB2", "KDR",
              "TNFRSF17", "SLAMF1", "SLAMF7", "GPR183", "CX3CR1", "FCER1G", "TYROBP"]
    sig_idx = [gidx_full[g] for g in SIGNAL if g in gidx_full]
    X = X[:, sig_idx]
    Xn = Xn[:, sig_idx]
    Xz = Xz[:, sig_idx]
    genes = genes[sig_idx]
    print(f"LR-restricted host input: {len(sig_idx)} signaling genes")
    gidx = {g: i for i, g in enumerate(genes)}

    is_t = labs == "T-cells"
    is_cancer = labs == "Cancer Epithelial"
    tree = cKDTree(coords[is_cancer])
    d_near, _ = tree.query(coords)
    t_act = np.zeros(len(labs), dtype=int)
    t_act[is_t] = (d_near[is_t] < 30.0).astype(int)
    print(f"T-cells infiltrated(<30um): {t_act[is_t].sum()}/{is_t.sum()}")

    edges = build_knn_graph(coords, k=K_GRAPH)
    adj_norm = build_norm_adj(edges, len(labs))
    x = torch.from_numpy(Xz)
    y = torch.from_numpy(t_act.astype(np.int64))
    tmask = torch.from_numpy(is_t)
    torch.manual_seed(0)
    gcn = GCN(n_feat=len(genes), n_hidden=64, n_classes=2)
    opt = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)
    gcn.train()
    for ep in range(200):
        opt.zero_grad()
        loss = F.cross_entropy(gcn(x, adj_norm)[tmask], y[tmask])
        loss.backward()
        opt.step()
    gcn.eval()
    print(f"host ready (T acc = {(gcn(x, adj_norm).argmax(1)[is_t] == y[is_t]).float().mean().item():.4f})")

    padj = adjacency_list(edges, len(labs))
    receivers = np.array([i for i in np.where(is_t)[0]
                          if any(labs[j] != "T-cells" for j in padj[i])])
    x0 = torch.from_numpy(np.tile(Xz[is_t].mean(0), (len(labs), 1)).astype(np.float32))

    # ligand presence
    lig_genes = sorted({l for l, _ in LR_PAIRS if l in gidx})
    lig_expr = {g: X[:, gidx[g]] > 0 for g in lig_genes}
    # random null genes: sample non-LR genes with similar overall frequency
    rng = np.random.default_rng(0)
    null_genes = {}
    for g in lig_genes:
        freq = lig_expr[g].mean()
        cand = [c for c in range(len(genes))
                if genes[c] not in lig_genes
                and abs((X[:, c] > 0).mean() - freq) < 0.05]
        null_genes[g] = rng.choice(cand, size=min(10, len(cand)), replace=False) if cand else np.array([], dtype=int)

    print(f"{'pathway':<20} {'n_recv':>7} {'psi_group':>10} {'null_mean':>10} {'z':>6}")
    results = []
    for lig, rec in LR_PAIRS:
        if lig not in gidx or rec not in gidx:
            continue
        rec_expr = X[:, gidx[rec]] > 0
        psi_obs, psi_nulls = [], {c: [] for c in null_genes[lig]}
        n_used = 0
        for j in receivers:
            j = int(j)
            if not rec_expr[j]:
                continue
            one = set(padj[j]) | {j}
            two = set(one)
            for u in one:
                two |= padj[u]
            zone = np.array(sorted(two), dtype=np.int64)
            senders = np.array([i for i in padj[j] if labs[i] != "T-cells"])
            s_lig = senders[lig_expr[lig][senders]]
            if len(s_lig) == 0:
                continue
            groups = {"obs": s_lig}
            for c in null_genes[lig]:
                groups[f"n{c}"] = senders[X[senders, c] > 0]
            groups = {k: v for k, v in groups.items() if len(v) > 0}
            if "obs" not in groups or len(groups) < 2:
                continue
            xz = x[zone]
            adj_sub = adj_norm[zone][:, zone]
            out = group_marginal(gcn, xz, adj_sub, zone, j, groups, x0[zone], 1)
            psi_obs.append(out["obs"])
            for c in null_genes[lig]:
                if f"n{c}" in out:
                    psi_nulls[c].append(out[f"n{c}"])
            n_used += 1
        if n_used < 20:
            print(f"{lig+'->'+rec:<20} (only {n_used} receivers, skipped)")
            continue
        obs = float(np.mean(psi_obs))
        null_flat = np.concatenate([np.array(v) for v in psi_nulls.values() if v])
        z = (obs - null_flat.mean()) / (null_flat.std() + 1e-12)
        results.append((lig, rec, n_used, obs, float(null_flat.mean()), z))
        print(f"{lig+'->'+rec:<20} {n_used:>7} {obs:>+10.4f} {null_flat.mean():>+10.4f} {z:>6.2f}")

    print(f"\ntotal {(time.perf_counter()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
