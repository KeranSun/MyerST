# MyerST quickstart

End-to-end: simulate tissue → train host → explain → verify. Runs in ~2 min on CPU.

```python
import numpy as np
import torch

from myerst import (LayeredTissueSimulator, MyersonExplainer, IGExplainer,
                    TorchModelAdapter, ExplanationTarget)
from myerst.models.gcn import GCN, build_norm_adj, train_gcn

# 1. simulate a 3-layer tissue (ground truth known)
res = LayeredTissueSimulator(grid_size=40, n_layers=3, n_genes=300, seed=0).simulate()
data = res.data
data.build_graph(k=6)

# 2. preprocess + train a GCN host
X = np.log1p(data.X)
mu, sd = X.mean(0), X.std(0) + 1e-6
x = torch.from_numpy(((X - mu) / sd).astype(np.float32))
gm = np.log1p(data.X).mean(0)
baseline_x = torch.from_numpy(np.tile((gm - mu) / sd, (data.n_spots, 1)).astype(np.float32))
adj_norm = build_norm_adj(data.edges, data.n_spots)
gcn = GCN(n_feat=data.n_genes, n_hidden=32, n_classes=3)
train_gcn(gcn, x, adj_norm, data.labels, epochs=150, seed=0)

# 3. define the explanation target: the L0|L1 boundary
from scripts.e1_driver_gene_recovery import boundary_spots
interface = boundary_spots(data.labels, data.adj, 0, 1)
signs = np.where(data.labels[interface] == 0, 1.0, -1.0)
adapter = TorchModelAdapter(gcn, adj_norm, boundary_spots={(0, 1): interface})
target = ExplanationTarget(kind="domain_boundary_margin", payload=(0, 1, signs))

# 4. Myerson attribution over boundary + 1-hop players
neigh = set()
for i in interface:
    neigh |= data.adj[i]
players = np.array(sorted(neigh | set(interface.tolist())))
exp = MyersonExplainer(n_samples=256, perm_batch=32, fwd_chunk=16, seed=0).explain(
    adapter, x, target, players=players, edges=data.edges,
    n_spots=data.n_spots, baseline=baseline_x, boundary_idx=interface)

# 5. the self-audit: efficiency holds exactly
print(exp.node_scores.sum())          # == v(N) - v(empty), to machine precision

# 6. compare with IG
ig = IGExplainer(40).explain(adapter, x, target, baseline=baseline_x)
```

## Choosing the right target (hard-won lessons)

| Question | Target | Why |
|---|---|---|
| Which spots define a domain boundary? | per-side `class_score_at` | matches the semantics of the ground truth |
| Which ligand drives receiver state? | gene-level occlusion, location/state target | cell-level masking confounds with cell type |
| boundary contrast (legacy) | `domain_boundary` | raw logit diff — degenerates when ref ≈ 0 |

## Rules of thumb

- **Always use global-mean baselines** (class-mean baselines inject prototype signal).
- Report host quality (task accuracy / clustering ARI) next to every explanation.
- Run the efficiency audit; if `Σφ ≠ v(N) − v(∅)`, something is wrong with the run, not the math.
