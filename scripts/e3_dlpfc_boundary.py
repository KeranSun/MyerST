"""E3 prototype: first real-data boundary explanation on DLPFC 151673.

Pipeline: h5ad -> library-size norm + log1p + top-3000 HVG + z-score ->
SpaData -> kNN graph -> two host models (supervised GCN; STAGATE-lite +
frozen-embedding head) -> explain the L5|L6 boundary with IG (domain-mean
baseline) and SpatialOcclusion -> top genes vs known layer markers.

Also reports STAGATE embedding quality (KMeans ARI vs manual annotation) and
cross-host attribution agreement (Spearman) — seeds of paper R4/R6.
"""

import time

import numpy as np
import torch
import anndata as ad
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.stats import spearmanr

from myerst.data.spadata import SpaData
from myerst.models.gcn import GCN, build_norm_adj, train_gcn
from myerst.models.stagate_lite import STAGATELite, HeadOnEmbedding, train_stagate, adj_binary
from myerst.adapters.torch_adapter import TorchModelAdapter
from myerst.adapters.base import ExplanationTarget
from myerst.explainers.ig import IGExplainer
from myerst.explainers.occlusion import SpatialOcclusion
from scripts.e1_driver_gene_recovery import boundary_spots

H5AD = "data/raw/DLPFC_151673.h5ad"
N_HVG = 3000
BOUNDARY = ("L5", "L6")
MARKERS = {"RELN", "AQP4", "NDNF", "CUX2", "LAMP5", "CUX1", "RORB", "PCP4",
           "FEZF2", "BCL11B", "KRT17", "SYT6", "MBP", "MOBP", "PLP1", "GFAP",
           "OPALIN", "FOXP2", "SATB2", "NR4A2", "COL5A2", "HTR2C", "ENC1", "HS3ST4"}


def load_dlpfc() -> SpaData:
    adata = ad.read_h5ad(H5AD)
    X = adata.X.toarray().astype(np.float32)
    genes = np.array(adata.var_names)
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    layer_names = np.array(adata.obs["layer"].astype(str))
    order = ["L1", "L2", "L3", "L4", "L5", "L6", "WM"]
    name2int = {n: i for i, n in enumerate(order)}
    labels = np.array([name2int[v] for v in layer_names], dtype=int)

    lib = X.sum(1, keepdims=True)
    Xn = np.log1p(X / np.maximum(lib, 1) * 1e4)
    var = Xn.var(0)
    hvg = np.argsort(var)[::-1][:N_HVG]
    hvg = np.sort(hvg)
    Xn = Xn[:, hvg]
    genes = genes[hvg]
    Xn = (Xn - Xn.mean(0)) / (Xn.std(0) + 1e-6)

    print(f"loaded: {Xn.shape[0]} spots x {Xn.shape[1]} HVGs; layers={order}")
    print(f"layer sizes: {dict(zip(order, np.bincount(labels)))}")
    return SpaData(X=Xn.astype(np.float32), coords=coords, labels=labels,
                   gene_names=genes.tolist()), order


def top_genes(scores, gene_names, k=20):
    idx = np.argsort(scores)[::-1][:k]
    return [(gene_names[i], float(scores[i]), gene_names[i] in MARKERS) for i in idx]


def report(title, genes):
    hits = sum(m for _, _, m in genes)
    pretty = ", ".join(f"{g}{'*' if m else ''}" for g, _, m in genes)
    print(f"  {title} [markers {hits}/{len(genes)}]\n    {pretty}")


def main():
    t0 = time.perf_counter()
    print("=" * 70)
    print("E3 prototype: DLPFC 151673 real-data boundary explanation (L5|L6)")
    print("=" * 70)

    data, order = load_dlpfc()
    data.build_graph(k=6)
    x = torch.from_numpy(data.X)
    y = torch.from_numpy(data.labels.astype(np.int64))
    A, B = order.index(BOUNDARY[0]), order.index(BOUNDARY[1])
    bspots = boundary_spots(data.labels, data.adj, A, B)
    target = ExplanationTarget(kind="domain_boundary", payload=(A, B))
    print(f"graph edges: {len(data.edges)}; boundary spots ({BOUNDARY[0]}|{BOUNDARY[1]}): {len(bspots)}")

    baseline_x = torch.from_numpy(data.domain_mean().astype(np.float32))
    ig = IGExplainer(n_steps=40)
    occ = SpatialOcclusion(batch_size=8)

    # ---- Host A: supervised GCN
    print("-" * 70)
    adj_norm = build_norm_adj(data.edges, data.n_spots)
    gcn = GCN(n_feat=data.n_genes, n_hidden=64, n_classes=7)
    train_gcn(gcn, x, adj_norm, data.labels, epochs=200, seed=0)
    with torch.no_grad():
        acc = (gcn(x, adj_norm).argmax(1) == y).float().mean().item()
    print(f"[Host A: GCN supervised] accuracy = {acc:.4f}")
    adA = TorchModelAdapter(gcn, adj_norm, boundary_spots={(A, B): bspots})
    report("GCN  IG(domain-mean) top20", top_genes(ig.explain(adA, x, target, baseline=baseline_x).node_scores, data.gene_names))
    report("GCN  Occlusion top20      ", top_genes(occ.explain(adA, x, target, baseline=baseline_x).node_scores, data.gene_names))
    ig_A = ig.explain(adA, x, target, baseline=baseline_x).node_scores

    # ---- Host B: STAGATE-lite + frozen-embedding head
    print("-" * 70)
    adj_bin = adj_binary(data.edges, data.n_spots)
    st = STAGATELite(n_feat=data.n_genes, hidden=64, latent=32, heads=2)
    train_stagate(st, x, adj_bin, epochs=200, seed=0, verbose=True)
    with torch.no_grad():
        z = st.embed(x, adj_bin).numpy()
    km = KMeans(n_clusters=7, n_init=10, random_state=0).fit(z)
    ari = adjusted_rand_score(data.labels, km.labels_)
    head = HeadOnEmbedding(st, n_classes=7)
    opt = torch.optim.Adam(head.head.parameters(), lr=0.01)
    for ep in range(60):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(x, adj_bin), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        accB = (head(x, adj_bin).argmax(1) == y).float().mean().item()
    print(f"[Host B: STAGATE-lite] KMeans ARI = {ari:.3f}; head accuracy = {accB:.4f}")
    adB = TorchModelAdapter(head, adj_bin, boundary_spots={(A, B): bspots})
    ig_B = ig.explain(adB, x, target, baseline=baseline_x).node_scores
    report("STAGATE  IG(domain-mean) top20", top_genes(ig_B, data.gene_names))

    rho, _ = spearmanr(ig_A, ig_B)
    print("-" * 70)
    print(f"cross-host IG attribution agreement (Spearman) = {rho:.3f}")
    print(f"(* = known layer marker)  total time {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
