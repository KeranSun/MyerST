"""Myerson sampler validation — prototype of the paper's R2 approximation-error experiment.

1) Path graph 0-1-2 with v(S)=|S|^2: MC estimates vs hand-derived exact values.
2) Random spatial kNN graph (n=10): exact enumeration vs MC at increasing
   sample sizes -> approximation-error curve (Extended Data figure material).
"""

import time

import numpy as np

from myerst.data.graph import build_knn_graph
from myerst.attribution.myerson import exact_myerson, mc_myerson

print("=" * 64)
print("1) Analytic check: path graph 0-1-2, v(S) = |S|^2")
print("=" * 64)
edges = np.array([[0, 1], [1, 2]])
v = lambda s: float(len(s)) ** 2
phi_exact = exact_myerson(3, edges, v)
phi_mc, sem = mc_myerson(3, edges, v, n_samples=20000, seed=42, return_std=True)
print(f"  exact (hand = 8/3, 11/3, 8/3): {np.round(phi_exact, 4)}")
print(f"  MC (n=20000):                  {np.round(phi_mc, 4)}  SEM={np.round(sem, 4)}")

print()
print("=" * 64)
print("2) Approximation-error curve: random spatial kNN graph (n=10, k=4)")
print("=" * 64)
rng = np.random.default_rng(2026)
coords = rng.uniform(0, 20, size=(10, 2))
edges = build_knn_graph(coords, k=4)
print(f"  graph: 10 nodes, {len(edges)} edges")

phi_exact = exact_myerson(10, edges, v)
print(f"  exact phi: {np.round(phi_exact, 3)}")
print(f"  {'n_samples':>10} {'max_abs_err':>12} {'mean_SEM':>10} {'time_s':>8}")
for n_samples in [256, 1024, 4096, 16384, 65536]:
    t0 = time.perf_counter()
    phi_mc, sem = mc_myerson(10, edges, v, n_samples=n_samples, seed=0, return_std=True)
    dt = time.perf_counter() - t0
    err = np.max(np.abs(phi_mc - phi_exact))
    print(f"  {n_samples:>10} {err:>12.4f} {sem.mean():>10.4f} {dt:>8.2f}")
