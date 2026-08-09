"""DomainSimulator v0: layered tissue with known driver genes.

Generates a synthetic spatial transcriptomics slice where the ground-truth
explanation is known by construction:

- Spots on a grid, split into horizontal layers (domains).
- Each layer has its own set of upregulated "driver" genes — the ground truth
  for boundary attribution.
- Expression ~ Negative Binomial with layer-specific means, spatial smoothing
  for autocorrelation, and zero-inflation for dropout realism.

The simulator underpins experiment E1 (driver-gene recovery) and the
faithfulness benchmark (paper R2/R3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from myerst.data.spadata import SpaData


@dataclass
class SimResult:
    data: SpaData
    driver_genes: dict[int, np.ndarray]   # layer id -> gene indices driving that layer
    boundary_genes: dict[tuple[int, int], np.ndarray]  # (layerA, layerB) -> union of drivers
    passenger_genes: dict[int, np.ndarray] | None = None  # layer id -> correlated decoys


class LayeredTissueSimulator:
    def __init__(
        self,
        grid_size: int = 60,
        n_layers: int = 3,
        n_genes: int = 200,
        n_driver_per_layer: int = 10,
        base_mean: float = 0.5,
        driver_fold: float = 6.0,
        nb_dispersion: float = 2.0,
        dropout_rate: float = 0.25,
        smooth_sigma: float = 1.0,
        n_passenger_per_driver: int = 0,
        passenger_noise: float = 0.3,
        seed: int | None = 0,
    ) -> None:
        self.p = dict(
            grid_size=grid_size, n_layers=n_layers, n_genes=n_genes,
            n_driver_per_layer=n_driver_per_layer, base_mean=base_mean,
            driver_fold=driver_fold, nb_dispersion=nb_dispersion,
            dropout_rate=dropout_rate, smooth_sigma=smooth_sigma,
            n_passenger_per_driver=n_passenger_per_driver,
            passenger_noise=passenger_noise,
        )
        self.rng = np.random.default_rng(seed)

    def simulate(self) -> SimResult:
        p = self.p
        g, L, G = p["grid_size"], p["n_layers"], p["n_genes"]
        rng = self.rng

        # --- geometry: grid spots, horizontal layers
        xs, ys = np.meshgrid(np.arange(g), np.arange(g))
        coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
        n = g * g
        layer_bands = np.linspace(0, g, L + 1)
        labels = np.digitize(coords[:, 1], layer_bands[1:-1]).astype(int)

        # --- driver genes per layer (disjoint sets)
        perm = rng.permutation(G)
        dpl = p["n_driver_per_layer"]
        driver_genes = {ell: np.sort(perm[ell * dpl:(ell + 1) * dpl]) for ell in range(L)}

        # --- mean structure: base + driver upregulation
        mean = np.full((n, G), p["base_mean"])
        for ell in range(L):
            mask = labels == ell
            mean[np.ix_(mask, driver_genes[ell])] *= p["driver_fold"]

        # --- Negative Binomial sampling (Gamma-Poisson mixture)
        r = p["nb_dispersion"]
        rate = rng.gamma(shape=r, scale=mean / r)          # (n, G)
        X = rng.poisson(rate).astype(np.float32)

        # --- spatial autocorrelation: box smoothing over the grid
        sig = p["smooth_sigma"]
        if sig > 0:
            width = int(max(1, round(sig)))
            Xg = X.reshape(g, g, G)
            pad = np.pad(Xg, ((width, width), (width, width), (0, 0)), mode="reflect")
            sm = np.zeros_like(Xg)
            for dy in range(2 * width + 1):
                for dx in range(2 * width + 1):
                    sm += pad[dy:dy + g, dx:dx + g, :]
            X = (sm / (2 * width + 1) ** 2).reshape(n, G).astype(np.float32)

        # --- dropout (zero inflation)
        drop = rng.random(X.shape) < p["dropout_rate"]
        X[drop] = 0.0

        # --- passenger genes: noisy copies of drivers with NO direct label effect.
        # Univariate DE cannot tell passengers from drivers (both correlate with
        # the layer); only model-faithful attribution + fidelity tests can.
        n_pass = p["n_passenger_per_driver"]
        passenger_genes: dict[int, np.ndarray] = {}
        if n_pass > 0:
            used = set(np.concatenate(list(driver_genes.values())).tolist())
            free = [g_i for g_i in range(G) if g_i not in used]
            rng.shuffle(free)
            noise_sd = p["passenger_noise"]
            for ell in range(L):
                pgs = []
                for d in driver_genes[ell]:
                    for _ in range(n_pass):
                        if not free:
                            break
                        pg = free.pop()
                        X[:, pg] = X[:, d] + rng.normal(0, noise_sd * (X[:, d].std() + 1e-6), size=n)
                        X[:, pg] = np.clip(X[:, pg], 0, None)
                        pgs.append(pg)
                passenger_genes[ell] = np.sort(np.array(pgs, dtype=int))

        data = SpaData(
            X=X, coords=coords, labels=labels,
            gene_names=[f"GENE{i:03d}" for i in range(G)],
        )

        boundary_genes: dict[tuple[int, int], np.ndarray] = {}
        for a in range(L):
            for b in range(a + 1, L):
                boundary_genes[(a, b)] = np.union1d(driver_genes[a], driver_genes[b])

        return SimResult(data=data, driver_genes=driver_genes,
                         boundary_genes=boundary_genes,
                         passenger_genes=passenger_genes if p["n_passenger_per_driver"] > 0 else None)
