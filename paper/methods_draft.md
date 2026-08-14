# MyerST — Methods (draft v0.1)

## Framework overview

MyerST attributes the predictions of any spatial graph neural network (host model) to genes, spots (cells), and spatial edges, using topology-constrained cooperative game theory. The framework has three layers: (i) an adapter layer exposing a minimal contract (`forward(x) -> logits`, scalar target construction); (ii) an attribution engine (Myerson Monte-Carlo sampler with coalition caching; integrated gradients; spatial occlusion; GraphLIME and GNNExplainer-style baselines); (iii) a verification layer (masking fidelity curves, remove-and-retrain (ROAR), ground-truth recovery on simulators, and the efficiency self-audit).

## Communication games and the Myerson value

Given players N (a chosen set of spots), a spatial graph g (kNN, k=6 unless stated), and a characteristic function v(S) (scalar model target evaluated when players outside S are replaced by a baseline), the induced communication game is v^g(S) = Σ_C v(C) over connected components C of S in g. The Myerson value φ^M is the Shapley value of v^g, the unique allocation satisfying component efficiency and fairness (Box 1). We estimate it by sampling permutations of players (M = 128–1024), evaluating only connected subcoalitions with a graph-traversal incremental scheme, batching model forwards (chunks of 16–32) and caching coalition evaluations across permutations. Monte-Carlo error decays as M^(-1/2) (verified against exact computation on small graphs; max |error| tracks the reference line, Fig. 2e). The efficiency identity Σφ = v(N) − v(∅) holds exactly for every sampled permutation and serves as a per-run self-audit.

## Explanation targets

- `domain_boundary(A, B)`: mean raw logit difference over spots straddling the A/B interface. Known degenerate when the boundary set mixes classes (class cancellation, ref ≈ 0) — retained for backward compatibility, superseded by:
- `domain_boundary_margin(A, B, signs)`: class-signed probability margin, mean over boundary spots of sign·(p_A − p_B), bounded in [−1,1].
- `class_score_at(cls, spots)`: mean logit of class cls over a chosen spot set (per-side games, CCC receivers).

## Baselines and masking semantics

Masking replaces a spot/gene with a **global-mean baseline** (mean profile over all spots). Class-conditional (domain-mean) baselines were shown to inject prototype signal and invert credit (E8d lesson) and are not used. Integrated gradients use 40 steps from the same baseline. Attention scores use a separately trained 4-head GAT host (attention received from boundary spots, mean over heads).

## Fidelity evaluation

- **Recovery (P1)**: AUROC against simulator ground truth (driver genes; interface spots). Node games are formulated per side (target = own-class score at same-side interface spots), which is the semantics matching the question "which spots define the boundary".
- **Masking fidelity (P2)**: cumulative top-k masking curve; raw margin-decay AUC (no normalization — normalizing by |ref − v_end| is scale-unstable).
- **ROAR (P3)**: remove top-k, retrain, evaluate held-out boundary accuracy (evaluation set = held-out same-class spots, avoiding eval-set shrinkage contamination).

## Simulators

`LayeredTissueSimulator`: layered tissues on a square grid; per-layer driver genes with fold-change, NB-dispersed counts, dropout, optional passenger genes correlated with drivers (redundancy regimes: sparse/medium/high). `CCCSimulator`: sender/receiver cell types with an interface band; cooperative activation requires ligand from a sender neighbor AND receptor in the receiver AND pair-specific signaling competence (50%) and zone; ligand repertoires are heterogeneous across senders (50% subsets), making senders non-exchangeable (E9 lesson). Ground truth: LR pairs, downstream targets, communicating edges, per-pair activation.

## CCC analysis on real data (Xenium/Visium)

Hosts are LR-restricted (36 signaling genes) two-layer GCNs trained to predict T-cell location state (infiltrated: within 30 µm of a Cancer-Epithelial cell — units normalized to nearest-neighbor spacing on Visium). Attribution uses gene-level ligand occlusion within each receiver's 2-hop receptive field, evaluated on the exact receptive-field subgraph (equivalent for 2-layer GCNs; orders of magnitude faster). Pathway effects are per-receiver paired against frequency-matched random-gene occlusions; reported as paired t-statistics. Cell types: official annotations for Xenium breast Rep1 (EMBL) / Rep2 (10x xlsx, supervised sheet); marker argmax annotation for lung and Visium (no hard threshold — spot-level mixtures).

## Data

DLPFC (12 slices, spatialLIBD/HumanPilot official S3 + GitHub; layer annotations verified 3611/3611 concordant with prior assembly); Xenium breast Rep1/Rep2 and lung NSCLC (10x CDN); Visium breast Block A Section 1 (10x CDN). Full URLs, citations, and processing steps in `data/README.md`.

## Statistics and reproducibility

All key effects verified across ≥2 independent host seeds (CXCL12: −1.07/−1.02 logits). Multi-seed simulator benchmarks use 3 seeds (mean ± sd reported). 11 unit/regression tests cover graph algorithms, exact-vs-MC Myerson, explainer regression, and simulator ground truth. An audit script (`scripts/audit_results.py`) recomputes every headline number from archived artifacts. Code: github.com/KeranSun/MyerST; package: `pip install myerst` [TBD].
