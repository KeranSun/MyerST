"""Myerson value attribution on spatial graphs.

The Myerson value (Myerson, 1977) is the Shapley value of the graph-restricted
game: only spatially connected coalitions can cooperate. Given a characteristic
function v(S) (e.g. a host model's score when only coalition S is "active"),
the induced game is

    v_g(S) = sum over connected components C of S in graph g of v(C)

and the Myerson value is Shapley(v_g). It is the unique allocation satisfying
component-efficiency and fairness (equal gains from bilateral agreement).

Two estimators are provided:
- exact_myerson : enumeration, O(2^n) — ground truth for small subgraphs and
  for the approximation-error experiment (paper R2).
- mc_myerson    : Monte Carlo sampling over random permutations, evaluating
  marginal contributions on the induced game v_g. Converges to exact values.
"""

from __future__ import annotations

from itertools import combinations
from math import comb, factorial
from typing import Callable

import numpy as np

from myerst.data.graph import adjacency_list, connected_components

CharacteristicFn = Callable[[frozenset[int]], float]


def induced_game(nodes_subset: frozenset[int], adj: list[set[int]], v: CharacteristicFn) -> float:
    """v_g(S): worth of a coalition under the graph restriction."""
    if not nodes_subset:
        return 0.0
    return sum(v(c) for c in connected_components(nodes_subset, adj))


def exact_myerson(n_nodes: int, edges: np.ndarray, v: CharacteristicFn) -> np.ndarray:
    """Exact Myerson values by enumeration over all coalitions.

    phi_i = sum_{S subseteq N\\{i}} |S|! (n-|S|-1)! / n! * [v_g(S+{i}) - v_g(S)]
    """
    adj = adjacency_list(edges, n_nodes)
    nodes = list(range(n_nodes))
    vg_cache: dict[frozenset[int], float] = {}

    def vg(s: frozenset[int]) -> float:
        if s not in vg_cache:
            vg_cache[s] = induced_game(s, adj, v)
        return vg_cache[s]

    phi = np.zeros(n_nodes)
    nf = factorial(n_nodes)
    for i in nodes:
        others = [j for j in nodes if j != i]
        acc = 0.0
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(n_nodes - r - 1) / nf
            for s_tuple in combinations(others, r):
                s = frozenset(s_tuple)
                acc += w * (vg(s | {i}) - vg(s))
        phi[i] = acc
    return phi


def mc_myerson(
    n_nodes: int,
    edges: np.ndarray,
    v: CharacteristicFn,
    n_samples: int = 4096,
    seed: int | None = 0,
    return_std: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Monte Carlo Myerson values via permutation sampling on the induced game.

    Each sample draws a random permutation pi; player i receives its marginal
    contribution v_g(P_i(pi)+{i}) - v_g(P_i(pi)) where P_i(pi) is the set of
    predecessors. The mean over samples converges to the exact Myerson value;
    the standard error of the mean quantifies the approximation error.
    """
    rng = np.random.default_rng(seed)
    adj = adjacency_list(edges, n_nodes)
    accum = np.zeros((n_samples, n_nodes))

    for m in range(n_samples):
        perm = rng.permutation(n_nodes)
        prefix: set[int] = set()
        prev_vg = 0.0
        for i in perm:
            prefix.add(i)
            cur = induced_game(frozenset(prefix), adj, v)
            accum[m, i] = cur - prev_vg
            prev_vg = cur

    phi = accum.mean(axis=0)
    if return_std:
        # standard error of the mean per player
        sem = accum.std(axis=0, ddof=1) / np.sqrt(n_samples)
        return phi, sem
    return phi
