# MyerST

**Topology-constrained game-theoretic attribution for spatial omics graph neural networks — with verifiable fidelity.**

MyerST operationalizes the Myerson value (communication games, Myerson 1977) for spatial transcriptomics: attribution of any GNN host's predictions to **genes**, **spots/cells**, and **spatial edges**, where coalitions are restricted to connected subgraphs of the tissue graph — because in a tissue, only neighbours can cooperate.

Key properties:

- **Self-auditing**: the efficiency identity `Σφ = v(N) − v(∅)` holds exactly for every Monte-Carlo permutation. Broken attribution code fails loudly, not silently.
- **Multi-level, one engine**: node values (Myerson), edge synergies (fairness axiom), gene scores (masking), plus IG / occlusion / GraphLIME / GNNExplainer-style baselines for comparison.
- **Verification-first**: masking-fidelity curves, remove-and-retrain (ROAR), and ground-truth recovery built in — because our benchmark shows *evaluation semantics decide what "importance" means*.

## Installation

```bash
pip install -e .            # from this repository
pip install -e ".[data]"    # + anndata/scanpy for real-data loaders
pip install -e ".[plot]"    # + matplotlib
```

## Quickstart (5 lines to a self-audited attribution)

```python
import numpy as np, torch
from myerst import (LayeredTissueSimulator, MyersonExplainer,
                    TorchModelAdapter, ExplanationTarget, build_knn_graph)

# simulate a 3-layer tissue with known driver genes
res = LayeredTissueSimulator(grid_size=30, n_genes=100, seed=0).simulate()
data = res.data; data.build_graph(k=6)
# ... train any (x, adj) -> logits host, then:
adapter = TorchModelAdapter(model, adj_norm, boundary_spots={(0, 1): iface})
target = ExplanationTarget(kind="domain_boundary", payload=(0, 1))
exp = MyersonExplainer(n_samples=256, seed=0).explain(
    adapter, x, target, players=players, edges=data.edges,
    n_spots=data.n_spots, baseline=baseline_x)
# efficiency self-audit: sum(exp.node_scores) == v(N) - v(empty), exactly
```

Full worked examples: `scripts/` (E1–E13, each reproduces a paper experiment) and `tutorials/quickstart.md`.

## Repository layout

| Path | Content |
|---|---|
| `myerst/` | the package (adapters / attribution / explainers / benchmark / models / data) |
| `scripts/` | experiment scripts E1–E13 + figure scripts + `audit_results.py` |
| `tests/` | 11 unit/regression tests (run `python tests/test_core.py` etc.) |
| `data/README.md` | dataset sources, URLs, citations, processing steps |
| `paper/` | manuscript (NC-format), Box 1, drafts |
| `outputs/` | paper figures (Fig 1–5) |

## Reproducing the paper

Every headline number can be recomputed from archived artifacts:

```bash
python scripts/audit_results.py     # 18-point reconciliation vs claimed values
```

## Citation

Archived release DOI: 10.5281/zenodo.21974540. Manuscript under review; citation TBD.

## License

MIT
