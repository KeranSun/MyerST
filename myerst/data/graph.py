"""Spatial graph construction and traversal utilities.

Graphs are represented as undirected edge arrays of shape (E, 2) with
node ids in [0, n). All helpers are numpy/scipy only so the core layer
stays torch-free and unit-testable.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def build_knn_graph(coords: np.ndarray, k: int = 6, mutual: bool = False) -> np.ndarray:
    """Build a symmetric k-nearest-neighbor graph from spatial coordinates.

    Parameters
    ----------
    coords : (n, d) array of spot/cell coordinates.
    k : number of neighbors (excluding self).
    mutual : if True keep only mutual-kNN edges, else union (default).

    Returns
    -------
    edges : (E, 2) int array, each row (u, v) with u < v, no duplicates.
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)
    k = min(k, n - 1)
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)  # includes self
    idx = np.atleast_2d(idx)[:, 1:]

    src = np.repeat(np.arange(n), idx.shape[1])
    dst = idx.ravel()
    edges = set()
    for u, v in zip(src.tolist(), dst.tolist()):
        if u == v:
            continue
        edges.add((min(u, v), max(u, v)))
    if mutual:
        fwd = set(zip(src.tolist(), dst.tolist()))
        edges = {e for e in edges if (e[1], e[0]) in fwd}
    return np.array(sorted(edges), dtype=np.int64)


def adjacency_list(edges: np.ndarray, n_nodes: int) -> list[set[int]]:
    """Edge array -> adjacency list of sets."""
    adj = [set() for _ in range(n_nodes)]
    for u, v in np.asarray(edges, dtype=np.int64):
        adj[u].add(v)
        adj[v].add(u)
    return adj


def connected_components(nodes: set[int] | frozenset[int], adj: list[set[int]]) -> list[frozenset[int]]:
    """Connected components of the subgraph induced by `nodes` (BFS)."""
    remaining = set(nodes)
    comps: list[frozenset[int]] = []
    while remaining:
        seed = remaining.pop()
        comp = {seed}
        stack = [seed]
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if w in remaining:
                    remaining.discard(w)
                    comp.add(w)
                    stack.append(w)
        comps.append(frozenset(comp))
    return comps
