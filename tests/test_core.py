"""Core unit tests for MyerST (runnable with pytest or plain python)."""

import numpy as np

from myerst.data.graph import build_knn_graph, adjacency_list, connected_components
from myerst.data.curvature import forman_ricci
from myerst.attribution.myerson import exact_myerson, mc_myerson


def test_knn_graph_basic():
    coords = np.array([[0, 0], [1, 0], [2, 0], [10, 10]], dtype=float)
    edges = build_knn_graph(coords, k=1)
    assert edges.shape[1] == 2
    assert all(u < v for u, v in edges)          # normalized, deduplicated
    assert all(u != v for u, v in edges)          # no self loops
    assert (0, 1) in map(tuple, edges) or (1, 2) in map(tuple, edges)


def test_connected_components_path():
    adj = adjacency_list(np.array([[0, 1], [1, 2]]), 3)
    comps = connected_components({0, 2}, adj)
    assert sorted(len(c) for c in comps) == [1, 1]   # {0},{2} disconnected
    comps = connected_components({0, 1, 2}, adj)
    assert len(comps) == 1


def test_forman_ricci_known():
    # triangle: F = 4 - 2 - 2 + 3*1 = 3
    tri = np.array([[0, 1], [1, 2], [0, 2]])
    assert np.allclose(forman_ricci(tri, 3), [3, 3, 3])
    # path 0-1-2: F(0,1) = 4 - 1 - 2 = 1
    path = np.array([[0, 1], [1, 2]])
    assert np.allclose(forman_ricci(path, 3), [1, 1])


def test_exact_myerson_analytic():
    """Path graph 0-1-2, v(S) = |S|^2 -> phi = (8/3, 11/3, 8/3) by hand."""
    edges = np.array([[0, 1], [1, 2]])
    v = lambda s: float(len(s)) ** 2
    phi = exact_myerson(3, edges, v)
    expected = np.array([8 / 3, 11 / 3, 8 / 3])
    assert np.allclose(phi, expected, atol=1e-10)
    # connected graph -> efficient: sum(phi) == v(N)
    assert np.isclose(phi.sum(), 9.0)


def test_mc_myerson_converges():
    edges = np.array([[0, 1], [1, 2]])
    v = lambda s: float(len(s)) ** 2
    phi_exact = exact_myerson(3, edges, v)
    phi_mc, sem = mc_myerson(3, edges, v, n_samples=20000, seed=42, return_std=True)
    assert np.allclose(phi_mc, phi_exact, atol=0.05)
    assert np.all(sem < 0.05)


def test_mc_vs_exact_spatial_knn():
    """Random spatial kNN graph (n=8): MC matches exact within tolerance."""
    rng = np.random.default_rng(7)
    coords = rng.uniform(0, 10, size=(8, 2))
    edges = build_knn_graph(coords, k=3)
    v = lambda s: float(len(s)) ** 2
    phi_exact = exact_myerson(8, edges, v)
    phi_mc = mc_myerson(8, edges, v, n_samples=20000, seed=1)
    corr = np.corrcoef(phi_exact, phi_mc)[0, 1]
    assert corr > 0.99
    assert np.max(np.abs(phi_exact - phi_mc)) < 0.15


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
