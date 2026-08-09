"""Debug: why does boundary accuracy survive removal of ALL boundary drivers?

Checks on the SPARSE sim:
1. the 12 boundary drivers actually rank in IG top-k
2. after removing top-128 IG genes, does any L0/L1 signal remain (max |DE|)?
3. confusion of the retrained model on interface spots
"""

import numpy as np
import torch

from myerst.benchmark.simulator import LayeredTissueSimulator
from myerst.models.gcn import GCN, build_norm_adj, train_gcn
from myerst.data.graph import build_knn_graph
from myerst.adapters.torch_adapter import TorchModelAdapter
from myerst.adapters.base import ExplanationTarget
from myerst.explainers.ig import IGExplainer
from scripts.e1_driver_gene_recovery import boundary_spots

sim = LayeredTissueSimulator(grid_size=40, n_layers=3, n_genes=300, seed=0,
                             n_driver_per_layer=6, driver_fold=3.0,
                             dropout_rate=0.3, n_passenger_per_driver=0)
res = sim.simulate()
data = res.data
data.build_graph(k=6)

drivers01 = set(res.driver_genes[0].tolist()) | set(res.driver_genes[1].tolist())
print("boundary drivers (layers 0|1):", sorted(drivers01))

X = np.log1p(data.X)
X = ((X - X.mean(0)) / (X.std(0) + 1e-6)).astype(np.float32)
x = torch.from_numpy(X)
baseline_x = torch.from_numpy(
    ((np.log1p(data.domain_mean()) - X.mean(0)) / (X.std(0) + 1e-6)).astype(np.float32))
adj_norm = build_norm_adj(data.edges, data.n_spots)
ref = GCN(n_feat=data.n_genes, n_hidden=32, n_classes=3)
train_gcn(ref, x, adj_norm, data.labels, epochs=150, seed=0)
interface = boundary_spots(data.labels, data.adj, 0, 1)
adapter = TorchModelAdapter(ref, adj_norm, boundary_spots={(0, 1): interface})
target = ExplanationTarget(kind="domain_boundary", payload=(0, 1))
ig = IGExplainer(40).explain(adapter, x, target, baseline=baseline_x).node_scores

order = np.argsort(ig)[::-1]
top128 = set(order[:128].tolist())
print("drivers in IG top-12:", len(drivers01 & set(order[:12].tolist())), "/ 12")
print("drivers in IG top-128:", len(drivers01 & top128), "/ 12")

# remove top-128, check residual L0/L1 signal on RAW normalized counts
keep = np.ones(300, bool)
keep[order[:128]] = False
Xr = data.X[:, keep]  # raw counts
m0 = Xr[data.labels == 0].mean(0)
m1 = Xr[data.labels == 1].mean(0)
de = np.abs(m0 - m1)
print(f"residual genes: {keep.sum()}; max |DE| L0 vs L1 = {de.max():.4f} "
      f"(driver-level DE was ~{abs(data.X[data.labels==0][:, list(drivers01)].mean() - 0):.2f} scale)")
top_residual = np.argsort(de)[::-1][:5]
print("top residual DE genes (kept idx):", top_residual, de[top_residual])

# retrain on reduced data, inspect predictions on interface spots
edges = build_knn_graph(data.coords, k=6)
adj2 = build_norm_adj(edges, data.n_spots)
x_red = torch.from_numpy(X[:, keep].astype(np.float32))
m2 = GCN(n_feat=int(keep.sum()), n_hidden=32, n_classes=3)
train_gcn(m2, x_red, adj2, data.labels, epochs=150, seed=1)
with torch.no_grad():
    pred = m2(x_red, adj2).argmax(1).numpy()
print("interface true label dist:", np.bincount(data.labels[interface], minlength=3))
print("interface pred label dist:", np.bincount(pred[interface], minlength=3))
acc = (pred[interface] == data.labels[interface]).mean()
print(f"interface acc after removing top-128: {acc:.3f}")
