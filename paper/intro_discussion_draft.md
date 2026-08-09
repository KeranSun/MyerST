# MyerST — Introduction & Discussion (draft v0.1)

## Introduction (4 paragraphs)

**P1 (field + gap).** Spatial omics now resolves gene expression at cellular resolution, and graph neural networks have become the default analytical engine for domain identification, cell-state prediction, and interaction analysis. Yet these models are black boxes: which genes, cells, and intercellular contacts drive a prediction is routinely interrogated with generic tools — attention weights, gradient-based saliency, occlusion — imported from other domains. Their faithfulness in spatial settings has never been systematically tested, and known pathologies (attention-explanation disagreement, instability under dropout noise) suggest the default practice is fragile.

**P2 (root cause).** We trace the fragility to a mathematical mismatch. Every Shapley-family attribution assumes free coalition formation: any subset of features or cells may jointly contribute. Tissues violate this — molecular cooperation is constrained by spatial adjacency, the very adjacency along which GNNs propagate information. Myerson (1977) resolved exactly this problem for cooperative games: restrict coalitions to connected subgraphs of a communication graph, and a unique allocation (the Myerson value) emerges from two axioms, component efficiency and fairness. Despite its foundational status in game theory, it has never been operationalized for spatial omics.

**P3 (framework).** We present MyerST: topology-constrained attribution for any spatial GNN, with three distinctive properties. (i) Self-auditing: the efficiency identity Σφ = v(N) − v(∅) holds exactly for every Monte-Carlo permutation, providing a built-in correctness check no gradient or perturbation method offers. (ii) Principled multi-level attribution: node values, edge synergies (via the fairness axiom), and gene scores from one engine. (iii) A verification-first design: every explanation ships with masking-fidelity, retraining, and recovery evaluations, because our own benchmark shows evaluation semantics decide what "importance" means.

**P4 (results summary).** A redundancy-controlled benchmark demonstrates that explanation rankings depend on operator and evaluation semantics; attention weights score below random on masking fidelity; and ROAR is structurally insensitive on spatial GNNs. On human DLPFC (12 slices, 3 donors), boundary attributions recover known layer markers (PCP4 12/12, KRT17 11/12 slices). On Xenium breast cancer, MyerST quantifies CXCL12-mediated T-cell exclusion (t = −38.9), decomposes it across sender types, replicates across four datasets and two platforms, and nominates PTN–SDC4 as a novel candidate for T-cell spatial positioning.

## Discussion (points to develop)

1. **Operator semantics as first-class objects.** The strongest lesson of our benchmark is that "importance" is not defined until the perturbation operator and evaluation protocol are fixed. We recommend per-side own-class games for boundary questions, gene-level occlusion for ligand questions, and location/state targets matched to pathway biology — a practical decision guide (Fig. 2f) distilled from failures we encountered and fixed.

2. **What the Myerson value buys — and what it does not.** MyerST is not a ranking tool competing with IG on node recovery (IG dominates there, stably). Its distinct contributions are: exact self-audit (efficiency), axiomatic edge decomposition (enabling edge/pathway-level communication attribution), and a coherent semantics for coalition-restricted credit. Framing it honestly as complementary to IG is both accurate and defensible.

3. **Host quality gates explanation quality.** Cross-host agreement jumps from 0.087 (weak host) to 0.807 (paper-grade host). Explanation pipelines should report host quality metrics (clustering ARI / task accuracy) as a standard prerequisite.

4. **Biological findings.** CXCL12 dual biology quantified spatially across cohorts; PTN–SDC4 as a testable hypothesis (with explicit limitations: transcriptomic LR inference cannot capture glycan-mediated binding; SDC4 is not PTN's canonical receptor; orthogonal validation needed).

5. **Limitations.** Single-edge attribution has an intrinsic noise floor under neighborhood aggregation; correlated features share credit by design; Monte-Carlo cost scales linearly with players × samples (though batched and cached); simulation benchmarks cannot capture all real-data confounders.

6. **Outlook.** GPU-sparse scaling to whole-slide games; integration with foundation-model hosts; prospectively designed perturbation experiments (e.g., CXCR4/PTN perturbation in organotypic cultures) as the ultimate validation loop.
