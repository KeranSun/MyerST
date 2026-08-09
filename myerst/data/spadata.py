"""SpaData: the core data container (numpy-first, AnnData-optional).

Design: the core path stays torch-free; conversion to torch tensors happens
only at the explainer/model boundary. AnnData interop is lazy — the package
never hard-depends on anndata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from myerst.data.graph import build_knn_graph, adjacency_list


@dataclass
class SpaData:
    """Spatial transcriptomics data bundle.

    X : (n_spots, n_genes) expression matrix (normalized/log1p recommended).
    coords : (n_spots, 2) spatial coordinates.
    labels : optional (n_spots,) array of domain/cell-type labels.
    gene_names : optional list of gene symbols.
    """

    X: np.ndarray
    coords: np.ndarray
    labels: np.ndarray | None = None
    gene_names: list[str] | None = None
    edges: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float32)
        self.coords = np.asarray(self.coords, dtype=float)
        assert self.X.shape[0] == self.coords.shape[0], "X/coords row mismatch"
        if self.labels is not None:
            self.labels = np.asarray(self.labels)
            assert len(self.labels) == self.X.shape[0]

    @property
    def n_spots(self) -> int:
        return self.X.shape[0]

    @property
    def n_genes(self) -> int:
        return self.X.shape[1]

    def build_graph(self, k: int = 6, mutual: bool = False) -> np.ndarray:
        """Build (and cache) the spatial kNN graph."""
        self.edges = build_knn_graph(self.coords, k=k, mutual=mutual)
        return self.edges

    @property
    def adj(self) -> list[set[int]]:
        assert self.edges is not None, "call build_graph() first"
        return adjacency_list(self.edges, self.n_spots)

    @classmethod
    def from_anndata(cls, adata: Any, labels_key: str | None = None) -> "SpaData":
        """Bridge from AnnData (coords in obsm['spatial'])."""
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        labels = adata.obs[labels_key].to_numpy() if labels_key else None
        return cls(
            X=X,
            coords=np.asarray(adata.obsm["spatial"]),
            labels=labels,
            gene_names=list(adata.var_names),
        )

    def domain_mean(self, gene_idx: np.ndarray | None = None) -> np.ndarray:
        """Per-domain gene means — used as the IG/occlusion baseline.

        Returns (n_spots, n_genes) matrix where each row is the mean
        expression profile of that spot's domain (spatially coherent baseline
        instead of the naive zero baseline, which fails under dropout).
        """
        assert self.labels is not None, "domain_mean requires labels"
        out = np.empty_like(self.X)
        for lab in np.unique(self.labels):
            mask = self.labels == lab
            out[mask] = self.X[mask].mean(axis=0)
        if gene_idx is not None:
            out[:, gene_idx] = self.X[:, gene_idx]
        return out
