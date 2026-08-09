"""Smoke tests for the simulators — lock ground-truth sanity (reproducibility)."""

import numpy as np

from myerst.benchmark.simulator import LayeredTissueSimulator
from myerst.benchmark.ccc_simulator import CCCSimulator


def test_domain_simulator_ground_truth():
    res = LayeredTissueSimulator(grid_size=20, n_layers=3, n_genes=100,
                                 n_driver_per_layer=4, seed=0).simulate()
    d = res.data
    assert d.X.shape == (400, 100)
    assert set(res.driver_genes) == {0, 1, 2}
    for ell, genes in res.driver_genes.items():
        m = d.labels == ell
        assert d.X[m][:, genes].mean() > d.X[~m][:, genes].mean()  # drivers upregulated


def test_ccc_simulator_ground_truth():
    res = CCCSimulator(grid_size=20, n_genes=100, seed=0).simulate()
    d = res.data
    assert d.edges is not None
    lig = [l for l, _ in res.lr_pairs]
    rec = [r for _, r in res.lr_pairs]
    sm = res.sender_mask
    # ligands higher in senders, receptors higher in receivers
    assert d.X[sm][:, lig].mean() > d.X[~sm][:, lig].mean()
    assert d.X[sm][:, rec].mean() < d.X[~sm][:, rec].mean()
    # targets only in activating receivers
    tgt = np.concatenate(list(res.targets.values()))
    act = res.activation.sum(1) > 0
    if act.sum() > 0 and (~sm & ~act).sum() > 0:
        assert d.X[act][:, tgt].mean() > d.X[~sm & ~act][:, tgt].mean()
    # comm edges are cross-type
    if len(res.comm_edges):
        u, v = res.comm_edges[:, 0], res.comm_edges[:, 1]
        assert np.all(res.sender_mask[u] != res.sender_mask[v])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
