"""E3b: DLPFC L5|L6 boundary explanation on the paper-grade host.

Host B upgraded: STAGATEOfficialPT (faithful port) with the official
seurat_v3 HVG pipeline (GMM ARI ~0.48), vs Host A supervised GCN.
IG (domain-mean baseline) + batched SpatialOcclusion on both hosts.
"""

import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from scipy.stats import spearmanr

from myerst.data.spadata import SpaData
from myerst.models.gcn import GCN, build_norm_adj, train_gcn
from myerst.models.stagate_official_pt import STAGATEOfficialPT, train_stagate_official
from myerst.models.stagate_lite import adj_binary
from myerst.adapters.torch_adapter import TorchModelAdapter
from myerst.adapters.base import ExplanationTarget
from myerst.explainers.ig import IGExplainer
from myerst.explainers.occlusion import SpatialOcclusion
from scripts.e1_driver_gene_recovery import boundary_spots
from scripts.e3_dlpfc_boundary import top_genes, report
from scripts.tune_stagate_seurat_hvg import load_seurat_hvg, ORDER

BOUNDARY = ("L5", "L6")


class HeadOnOfficialEmbedding(nn.Module):
    """Linear head on frozen STAGATEOfficialPT embeddings (adapter contract)."""

    def __init__(self, backbone: STAGATEOfficialPT, n_classes: int):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.head = nn.Linear(backbone.W1.shape[1], n_classes)

    def forward(self, x: torch.Tensor, adj_bin: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone.embed(x, adj_bin))


def main():
    t0 = time.perf_counter()
    print("=" * 70)
    print("E3b: DLPFC 151673 L5|L6 boundary — paper-grade host comparison")
    print("=" * 70)

    x_t, coords, labels = load_seurat_hvg()
    n_spots, n_genes = x_t.shape
    # gene names for reporting: reload via anndata through the same pipeline
    import anndata as ad, scanpy as sc
    adata = ad.read_h5ad("data/raw/DLPFC_151673.h5ad")
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    gene_names = np.array(adata.var_names[adata.var["highly_variable"]])

    data = SpaData(X=x_t.numpy(), coords=coords, labels=labels, gene_names=gene_names.tolist())
    data.build_graph(k=6)
    A, B = ORDER.index(BOUNDARY[0]), ORDER.index(BOUNDARY[1])
    bspots = boundary_spots(data.labels, data.adj, A, B)
    target = ExplanationTarget(kind="domain_boundary", payload=(A, B))
    baseline_x = torch.from_numpy(data.domain_mean().astype(np.float32))
    y = torch.from_numpy(data.labels.astype(np.int64))
    ig = IGExplainer(n_steps=40)
    occ = SpatialOcclusion(batch_size=8)
    print(f"{n_spots} spots x {n_genes} HVGs; boundary spots: {len(bspots)}")

    # ---- Host A: supervised GCN
    print("-" * 70)
    adj_norm = build_norm_adj(data.edges, n_spots)
    gcn = GCN(n_feat=n_genes, n_hidden=64, n_classes=7)
    train_gcn(gcn, x_t, adj_norm, data.labels, epochs=200, seed=0)
    with torch.no_grad():
        accA = (gcn(x_t, adj_norm).argmax(1) == y).float().mean().item()
    print(f"[Host A: GCN] accuracy = {accA:.4f}")
    adA = TorchModelAdapter(gcn, adj_norm, boundary_spots={(A, B): bspots})
    igA = ig.explain(adA, x_t, target, baseline=baseline_x).node_scores
    report("GCN  IG top20", top_genes(igA, data.gene_names))
    report("GCN  Occ top20", top_genes(occ.explain(adA, x_t, target, baseline=baseline_x).node_scores, data.gene_names))

    # ---- Host B: STAGATEOfficialPT + frozen head
    print("-" * 70)
    adj_bin = adj_binary(data.edges, n_spots)
    st = STAGATEOfficialPT(n_feat=n_genes, hidden=512, latent=30)
    train_stagate_official(st, x_t, adj_bin, epochs=500, verbose=False)
    with torch.no_grad():
        z = st.embed(x_t, adj_bin).numpy()
    ari_gm = adjusted_rand_score(labels, GaussianMixture(7, reg_covar=1e-4, n_init=5,
                                 random_state=0).fit(z.astype(np.float64)).predict(z.astype(np.float64)))
    head = HeadOnOfficialEmbedding(st, n_classes=7)
    opt = torch.optim.Adam(head.head.parameters(), lr=0.01)
    for _ in range(60):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(head(x_t, adj_bin), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        accB = (head(x_t, adj_bin).argmax(1) == y).float().mean().item()
    print(f"[Host B: STAGATE-official] GMM ARI = {ari_gm:.3f}; head accuracy = {accB:.4f}")
    adB = TorchModelAdapter(head, adj_bin, boundary_spots={(A, B): bspots})
    igB = ig.explain(adB, x_t, target, baseline=baseline_x).node_scores
    report("STAGATE-official  IG top20", top_genes(igB, data.gene_names))
    report("STAGATE-official  Occ top20", top_genes(occ.explain(adB, x_t, target, baseline=baseline_x).node_scores, data.gene_names))

    rho, _ = spearmanr(igA, igB)
    print("-" * 70)
    print(f"cross-host IG agreement (Spearman) = {rho:.3f}")
    print(f"total time {(time.perf_counter() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
