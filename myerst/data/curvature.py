"""Discrete Ricci curvature on spatial graphs.

Forman-Ricci curvature is used as a cheap O(E) geometric edge prior for
attribution masks (negative curvature ~ domain-boundary / bottleneck edges).

For an unweighted edge e = (u, v):
    F(e) = 4 - deg(u) - deg(v) + 3 * |triangles through e|
"""

from __future__ import annotations

import numpy as np

from myerst.data.graph import adjacency_list


def forman_ricci(edges: np.ndarray, n_nodes: int) -> np.ndarray:
    """Forman-Ricci curvature for every edge.

    Parameters
    ----------
    edges : (E, 2) int array.
    n_nodes : number of nodes in the graph.

    Returns
    -------
    (E,) float array of curvature values aligned with `edges` rows.
    """
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size == 0:
        return np.empty(0, dtype=float)
    adj = adjacency_list(edges, n_nodes)
    deg = np.array([len(a) for a in adj], dtype=float)

    out = np.empty(edges.shape[0], dtype=float)
    for i, (u, v) in enumerate(edges):
        triangles = len(adj[u] & adj[v])
        out[i] = 4.0 - deg[u] - deg[v] + 3.0 * triangles
    return out
