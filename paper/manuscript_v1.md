# Topology-constrained attribution reveals verifiable explanations for spatial omics graph neural networks

[Authors TBD]

---

## Abstract

Graph neural networks have become the default engine for analysing spatial omics, but their predictions remain opaque, and the generic explanation tools used to interrogate them have never been systematically validated in spatial settings. Here we show that the fragility has a mathematical root: Shapley-family attributions assume free coalition formation, which tissues violate. We introduce MyerST, which operationalizes Myerson's communication-game values for any spatial GNN, providing self-auditing node, edge and gene attributions. A redundancy-controlled benchmark shows that explanation rankings depend on operator and evaluation semantics, that attention weights score below random in masking fidelity, and that retraining-based metrics are structurally insensitive. Validated across twelve human brain slices and four tumour datasets spanning two platforms, MyerST quantifies CXCL12-mediated T-cell exclusion and nominates PTN–SDC4 as a candidate axis for T-cell positioning.

(147 words)

---

## [Main text — untitled introduction]

Spatial omics now resolves gene expression at cellular resolution, and graph neural networks (GNNs) have become the default analytical engine for tissue domain identification, cell-state prediction and interaction analysis<sup>1-5</sup>. Yet these models are black boxes. Which genes, cells and intercellular contacts drive a prediction is routinely interrogated with generic tools imported from other domains — attention weights, gradient-based saliency, or occlusion<sup>6-9</sup> — whose faithfulness in spatial settings has never been systematically tested. Warning signs exist elsewhere: attention weights can disagree with every other faithfulness measure<sup>6</sup>, and explanation benchmarks in deep learning have repeatedly shown that popular methods fail basic sanity checks<sup>10,11</sup>.

We argue the fragility has a mathematical root. Every Shapley-family attribution assumes *free coalition formation*: any subset of players may jointly contribute to the prediction<sup>12,13</sup>. Tissues violate this assumption — molecular cooperation is physically constrained by spatial adjacency, the same adjacency along which GNNs propagate information. Myerson resolved exactly this problem in 1977: restrict coalitions to connected subgraphs of a communication graph, and a unique allocation — the Myerson value — emerges from two axioms, component efficiency and fairness<sup>14</sup>. Despite its foundational status in game theory, it has never been operationalized for spatial omics.

Here we present MyerST, topology-constrained attribution for any spatial GNN host, with three distinctive properties. First, *self-auditing*: the efficiency identity Σφ = v(N) − v(∅) holds exactly for every Monte-Carlo permutation, giving a built-in correctness check that gradient- and perturbation-based explainers lack. Second, *principled multi-level attribution*: node values, edge synergies (via the fairness axiom) and gene scores from one engine. Third, *verification-first design*: every explanation ships with masking-fidelity, retraining and recovery evaluations, because our benchmark shows evaluation semantics decide what "importance" means. We validate the framework on the human dorsolateral prefrontal cortex (twelve slices, three donors) and on four tumour datasets spanning the Xenium and Visium platforms, where MyerST quantifies CXCL12-mediated T-cell exclusion and nominates PTN–SDC4 as a candidate axis for T-cell spatial positioning.

## Results

### MyerST: topology-constrained attribution with self-auditing

MyerST wraps any GNN host (GCN, STAGATE, GAT) behind a minimal adapter contract and estimates Myerson values by Monte-Carlo sampling over connected coalitions of the spatial neighbourhood graph, with coalition caching and batched evaluation (Fig. 1a; Methods). The framework's mathematical content is summarized in Box 1: on a three-node path graph with coalition worth v(S)=|S|², the Shapley value assigns identical credit (3.00, 3.00, 3.00) to all nodes — position is invisible — whereas the Myerson value assigns (2.67, 3.67, 2.67), rewarding the bridge. Two properties matter in practice. The component-efficiency axiom yields the exact identity Σφ = v(N) − v(∅), which we verify to machine precision in every experiment (e.g. 1.2787 on DLPFC): any implementation or convergence failure surfaces immediately as a violation. And the fairness axiom equates bilateral link gains, yielding a well-defined edge decomposition ψᵢⱼ = v({i,j}) − v({i}) − v({j}) + v(∅) that we exploit for communication analysis. The sampler converges at the standard MC rate M^(−1/2) (Fig. 2e), and a ~350-spot boundary band attributes in ~25 minutes on one CPU.

### Operator semantics decide what importance means

We built a redundancy-controlled simulator (known driver genes and interface spots; tunable correlated passengers) and compared five explainers under three protocols — ground-truth recovery, cumulative masking fidelity, and remove-and-retrain (ROAR<sup>11</sup>) — across three redundancy regimes and three simulation seeds. Three findings emerged (Fig. 2).

First, **rankings are operator- and semantics-dependent**. Integrated gradients (IG) dominate node-level recovery (AUROC 0.97 ± 0.01, stable across seeds) under per-side own-class games — the formulation whose semantics match the question "which spots define the boundary". Spot-masking coalitional credit (Myerson) measures a different quantity — irreplaceability under masking — and under smoothing GCNs the interface is replaceable by interior neighbours; we report this honestly (Myerson node AUROC 0.31 ± 0.03) and position Myerson where its guarantees are unique: self-audit and edge attribution (Fig. 2e). Getting to a clean benchmark required fixing four failure modes that we believe are widespread (Fig. 2f): raw logit-difference targets cancel by class (ref ≈ 0); mixed signed margins cause friendly fire between classes; class-mean baselines inject prototype signal (global-mean baselines are mandatory); and sign conventions must match across explainers.

Second, **attention weights score below random** on masking fidelity (decay AUC 0.17 ± 0.01 vs 0.22 ± 0.01 for random rankings), experimentally falsifying a still-common interpretation practice in spatial GNN analysis.

Third, **ROAR with accuracy is structurally insensitive on spatial GNNs**: even under extreme sparsification (400 spots, fold-1.5 signal), held-out accuracy stays ≈1.0 after removing all ground-truth drivers, because finite-sample layer-mean offsets plus neighbourhood aggregation guarantee separability (Fig. 2d). We recommend masking-based and recovery-based metrics for spatial GNNs and report ROAR's insensitivity as a cautionary null.

### Boundary attribution validated across twelve DLPFC slices

On the DLPFC reference dataset<sup>15</sup>, a GCN host (accuracy 0.991) attributed the L5|L6 boundary of slice 151673 (Fig. 3). Top IG gene attributions are enriched for known layer markers (PCP4, KRT17, RORB). Across all twelve slices (three donors), hosts reached 0.990–0.997 accuracy; PCP4 ranked top-20 in 12/12 slices, KRT17 in 11/12, and ≥2 known markers in every slice (Fig. 3d). Cross-slice attribution correlation averages 0.31 — we characterize this as *core-consistent, tail-heterogeneous*: the marker core replicates across donors while the tail reflects genuine donor/sectioning heterogeneity, not method noise, because same-slice cross-host agreement is far higher (0.807 between GCN and a faithful STAGATE port; Fig. 3e). Host quality gates explanation quality: agreement collapses to 0.087 for a weak host (clustering ARI ~0.30 vs 0.52; Fig. 3f). Explanation pipelines should report host quality as a standard prerequisite.

### CXCL12-mediated T-cell exclusion in breast cancer

Our flagship case decomposes the CXCL12–CXCR4 axis on Xenium breast cancer Rep1<sup>2</sup> (167,782 cells; a 12,706-cell tumour–immune interface crop). Five iterations of methodological refinement — each triggered by a diagnosed failure (single-edge dilution, cell-type confounding under whole-cell masking, target–biology mismatch, cell-type-marker swamping, null-model mispairing) — converged on a recipe now encoded in MyerST's CCC module: an LR-restricted host (36 signalling genes), a location-matched target (T-cell infiltration within 30 µm of cancer), gene-level ligand occlusion, and per-receiver paired nulls (Methods).

With this design, the CXCL12→CXCR4 effect on T-cell location is −1.07 logits (paired t = −38.9, n = 1,806 receivers), the negative sign matching the documented stromal-retention mechanism<sup>16-18</sup>: CAF-derived CXCL12 traps T cells in stroma, and CXCR4 inhibitors release them into tumour nests. The effect is spatially coherent (Fig. 4c), replicates across host seeds (−1.02, t = −32.5), and decomposes across sender types: CAFs contribute the largest per-receiver effect (−0.149, t = −22.8), myeloid cells the highest significance (t = −28.3), with sub-additive redundancy across sources (Fig. 4e) — a more nuanced picture than the canonical CAF-only narrative. The same pipeline yields among the first spatial evidence for CD80/CD86→CTLA4 in breast and lung tumours<sup>19-22</sup>, and nominates **PTN→SDC4** as a novel candidate (t = −4.4/−3.1 across seeds): PTN immunomodulation has precedent via neutrophils<sup>23</sup>, but a PTN–SDC4 role in T-cell spatial positioning is, to our knowledge, unreported (SDC4 is a documented T-cell inhibitory co-receptor<sup>24,25</sup>; PTN's syndecan binding is glycan-mediated<sup>26</sup>). We present this as a hypothesis, with explicit limitations, for experimental follow-up.

### Replication across cohorts and platforms

The identical pipeline — no per-cohort code changes — reproduces across four datasets and two platforms (Fig. 5): Xenium breast Rep1 (−1.07, t = −38.9), Xenium breast Rep2 (+0.29, t = +18.6), Xenium lung NSCLC (+0.68, t = +5.0), and Visium breast (+1.30, t = +28.5). CXCL12 is the strongest pathway in all four (|t| ≥ 5), while its sign tracks cohort infiltration stage (43% infiltrated T cells in Rep1, where retention dominates; 6.7% in Rep2, where recruitment dominates) — consistent with the axis's documented dual biology<sup>16,17,27</sup> and with TLS-level CXCL12–CXCR4 activity reported on the same dataset<sup>28</sup>. We therefore claim detectability replication with context-dependent direction. PTN→SDC4 is significant in three of three testable datasets (t = −4.4, +6.8, +12.6). Visium results carry the caveat of multi-cellular spots and coarse marker annotation and are presented as supporting evidence.

## Discussion

Our benchmark's central lesson is that "importance" is undefined until the perturbation operator and evaluation protocol are fixed. We recommend per-side own-class games for boundary questions, gene-level occlusion for ligand questions, and targets matched to pathway biology (Fig. 2f) — a decision guide distilled from failures we encountered and fixed. Within this discipline, MyerST and IG are complementary: IG is the stronger node ranker; MyerST uniquely offers exact self-audit (efficiency) and axiomatic edge decomposition, which underlies the communication analysis. Host quality gates explanation quality (0.087 vs 0.807 cross-host agreement) and should be reported as standard. Biologically, we spatially quantify CXCL12's context-dependent dual role across cohorts and nominate PTN–SDC4 for T-cell positioning — a hypothesis bounded by transcriptomic inference's blindness to glycan-mediated binding and SDC4's status as a non-canonical PTN receptor. Limitations include the single-edge noise floor under neighbourhood aggregation, credit sharing among correlated features (correct behaviour of the value, but requiring benchmark awareness), and CPU-bound MC cost (linear in players × samples). GPU-sparse scaling, foundation-model hosts, and prospectively designed perturbation experiments (CXCR4/PTN perturbation in organotypic culture) are natural next steps.

## Methods

*(condensed from methods_draft.md v0.1 — full text there; NC Methods ≤3,000 words)*

**Framework.** MyerST comprises an adapter layer (`forward(x)→logits`, scalar targets), an attribution engine (Myerson MC sampler with coalition caching; IG; spatial occlusion; GraphLIME/GNNExplainer-style baselines) and a verification layer (masking curves, ROAR, recovery, efficiency audit).

**Communication games.** v^g(S) = Σ_C v(C) over connected components of S in the kNN graph g (k=6). Myerson value = Shapley value of v^g, estimated over M = 128–1024 permutations with incremental connected-subcoalition evaluation, batched forwards (chunks 16–32) and cross-permutation caching. Error decays as M^(−1/2) (verified against exact values). The identity Σφ = v(N) − v(∅) holds exactly per permutation and is asserted every run.

**Targets.** `domain_boundary(A,B)` (raw logit difference; degenerate under class cancellation, retained for compatibility); `domain_boundary_margin(A,B,signs)` (class-signed probability margin, bounded); `class_score_at(cls, spots)` (per-side games; CCC receivers).

**Masking.** Global-mean baselines throughout (class-conditional baselines inject prototype signal and invert credit). IG: 40 steps. Attention: 4-head GAT host, attention received from boundary spots, mean over heads.

**Evaluation.** P1 recovery AUROC vs simulator ground truth (per-side games for nodes). P2 cumulative top-k masking, raw margin-decay AUC (normalization by |ref − v_end| is scale-unstable and unused). P3 ROAR: remove top-k, retrain, held-out boundary accuracy on held-out same-class spots.

**Simulators.** LayeredTissueSimulator: layered grid tissues, per-layer drivers (fold-change, NB counts, dropout), optional correlated passengers (sparse/medium/high redundancy). CCCSimulator: sender/receiver types with interface band; cooperative activation requires neighbour ligand AND own receptor AND pair-specific competence (50%) and zone; sender ligand repertoires heterogeneous (50% subsets). Ground truth: LR pairs, targets, communicating edges.

**CCC on real data.** LR-restricted hosts (36 signalling genes; list in Supplementary Table 1) predict T-cell infiltration (within 30 µm of Cancer-Epithelial cells; coordinates NN-normalized on Visium). Attribution: gene-level ligand occlusion within each receiver's 2-hop receptive field, evaluated on the exact receptive-field subgraph. Pathway effects: per-receiver paired vs frequency-matched random-gene occlusions, paired t-statistics. Annotations: EMBL (Rep1), 10x supervised (Rep2), marker argmax (lung, Visium — no hard threshold for spot mixtures).

**Data.** DLPFC 12 slices (spatialLIBD/HumanPilot official sources; annotations 3611/3611 concordant); Xenium breast Rep1/Rep2, lung NSCLC, Visium breast Block A Section 1 (10x CDN). URLs and processing in data/README.md and the reproducibility repository.

**Statistics and reproducibility.** Key effects verified across ≥2 host seeds. Simulator benchmarks: 3 seeds, mean ± s.d. 11 unit/regression tests cover graph algorithms, exact-vs-MC Myerson, explainer regression and simulator ground truth. `scripts/audit_results.py` recomputes every headline number from archived artifacts.

## Data availability

All datasets are public: DLPFC (spatialLIBD; HumanPilot), Xenium breast Rep1/Rep2 and lung (10x Genomics), Visium breast (10x Genomics). Accession URLs and checksums are listed in the reproducibility repository [DOI TBD].

## Code availability

MyerST source code, experiment scripts, and the audit script: github.com/[TBD]; PyPI package `myerst` [TBD]; archival DOI [TBD]. Reviewer access will be provided at submission.

## References

1. Ståhl PL et al. Visualization and analysis of gene expression in tissue sections by spatial transcriptomics. *Science* 353, 78–82 (2016). DOI: 10.1126/science.aaf2403
2. Janesick A et al. High resolution mapping of the tumor microenvironment using integrated single-cell, spatial and in situ analysis. *Nat Commun* 14, 8353 (2023). DOI: 10.1038/s41467-023-43458-x
3. He S et al. High-plex imaging of RNA and proteins at subcellular resolution in fixed tissue by spatial molecular imaging. *Nat Biotechnol* 40, 1794–1806 (2022). DOI: 10.1038/s41587-022-01483-z
4. Dong K, Zhang S. Deciphering spatial domains from spatially resolved transcriptomics with an adaptive graph attention auto-encoder. *Nat Commun* 13, 1739 (2022). DOI: 10.1038/s41467-022-29439-6
5. Long Y et al. Spatially informed clustering, integration, and deconvolution of spatial transcriptomics with GraphST. *Nat Commun* 14, 1155 (2023). DOI: 10.1038/s41467-023-36796-3
6. Jain S, Wallace BC. Attention is not explanation. *NAACL-HLT*, 3543–3556 (2019). DOI: 10.18653/v1/N19-1357
7. Lundberg SM, Lee SI. A unified approach to interpreting model predictions. *NeurIPS 30* (2017). arXiv:1705.07874
8. Sundararajan M, Taly A, Yan Q. Axiomatic attribution for deep networks. *ICML, PMLR 70*, 3319–3328 (2017).
9. Ying R et al. GNNExplainer: Generating explanations for graph neural networks. *NeurIPS 32* (2019). arXiv:1903.03894
10. Yuan H et al. Explainability in graph neural networks: A taxonomic survey. *IEEE TPAMI* 45, 5782–5799 (2023). DOI: 10.1109/TPAMI.2022.3204236
11. Hooker S et al. A benchmark for interpretability methods in deep neural networks. *NeurIPS 32* (2019). arXiv:1806.10758
12. Shapley LS. A value for n-person games. *Contrib. Theory Games II*, 307–317 (1953). DOI: 10.1515/9781400881970-018
13. Agarwal C et al. Evaluating explainability for graph neural networks. *Sci Data* 10, 144 (2023). DOI: 10.1038/s41597-023-01974-x
14. Myerson RB. Graphs and cooperation in games. *Math Oper Res* 2, 225–229 (1977). DOI: 10.1287/moor.2.3.225
15. Maynard KR et al. Transcriptome-scale spatial gene expression in the human dorsolateral prefrontal cortex. *Nat Neurosci* 24, 425–436 (2021). DOI: 10.1038/s41593-020-00787-0
16. Feig C et al. Targeting CXCL12 from FAP-expressing carcinoma-associated fibroblasts synergizes with anti-PD-L1 immunotherapy in pancreatic cancer. *PNAS* 110, 20212–20217 (2013). DOI: 10.1073/pnas.1320318110
17. Chen IX et al. Blocking CXCR4 alleviates desmoplasia, increases T-lymphocyte infiltration, and improves immunotherapy in metastatic breast cancer. *PNAS* 116, 4558–4566 (2019). DOI: 10.1073/pnas.1815515116
18. Bockorny B et al. BL-8040, a CXCR4 antagonist, in combination with pembrolizumab and chemotherapy for pancreatic cancer: the COMBAT trial. *Nat Med* 26, 878–885 (2020). DOI: 10.1038/s41591-020-0880-x
19. Sui Q et al. Inflammation promotes resistance to immune checkpoint inhibitors in high microsatellite instability colorectal cancer. *Nat Commun* 13, 7316 (2022). DOI: 10.1038/s41467-022-35096-6
20. Sakai SA et al. Mathematical modeling predicts optimal immune checkpoint inhibitor and radiotherapy combinations and timing of administration. *Cancer Immunol Res* 13, 353–364 (2025). DOI: 10.1158/2326-6066.CIR-24-0610
21. You S et al. Lymphatic-localized Treg–mregDC crosstalk limits antigen trafficking and restrains anti-tumor immunity. *Cancer Cell* 42, 1415–1433.e12 (2024). DOI: 10.1016/j.ccell.2024.06.014
22. Wang N et al. Spatial single-cell transcriptomic analysis in breast cancer reveals potential biomarkers for PD1 blockade therapy. *Research Square* (2024). DOI: 10.21203/rs.3.rs-4376986/v2
23. Ganguly D et al. Pleiotrophin drives a prometastatic immune niche in breast cancer. *J Exp Med* 220, e20220610 (2023). DOI: 10.1084/jem.20220610
24. Chung JS et al. DC-HIL is a T-cell inhibitory receptor that binds syndecan-4. *J Immunol* 179, 5778–5784 (2007). DOI: 10.4049/jimmunol.179.9.5778
25. Chung JS et al. Sézary syndrome cells overexpress syndecan-4 bearing distinct heparan sulfate moieties that suppress T-cell activation. *Blood* 117, 3382–3390 (2011). DOI: 10.1182/blood-2010-08-302034
26. Pleiotrophin: Activity and mechanism. *Adv Clin Chem* 98, 51–89 (2020). DOI: 10.1016/bs.acc.2020.02.003
27. Bleul CC et al. The lymphocyte chemoattractant SDF-1 is a ligand for LESTR/fusin and blocks HIV-1 entry. *Nature* 382, 829–833 (1996). DOI: 10.1038/382829a0
28. Fan X et al. Characterizing tertiary lymphoid structures associated single-cell atlas in breast cancer patients. *Cancer Cell Int* 25 (2025). DOI: 10.1186/s12935-025-03635-y
29. Huang Q et al. GraphLIME: Local interpretable model explanations for graph neural networks. *IEEE TKDE* 35, 6968–6972 (2023). DOI: 10.1109/TKDE.2022.3187455
30. Wolf FA, Angerer P, Theis FJ. SCANPY: large-scale single-cell gene expression data analysis. *Genome Biol* 19, 15 (2018). DOI: 10.1186/s13059-017-1382-0
31. Palla G et al. Squidpy: a scalable framework for spatial omics analysis. *Nat Methods* 19, 171–178 (2022). DOI: 10.1038/s41592-021-01358-2
32. Efremova M et al. CellPhoneDB: inferring cell–cell communication from combined expression of multi-subunit ligand–receptor complexes. *Nat Protoc* 15, 1484–1506 (2020). DOI: 10.1038/s41596-020-0292-x
33. Browaeys R, Saelens W, Saeys Y. NicheNet: modeling intercellular communication by linking ligands to target genes. *Nat Methods* 17, 159–162 (2020). DOI: 10.1038/s41592-019-0667-5
34. Birk S et al. Quantitative characterization of cell niches in spatially resolved omics data (NicheCompass). *Nat Genet* 57 (2025). DOI: 10.1038/s41588-025-02120-6
35. Li B et al. Benchmarking spatial and single-cell transcriptomics integration methods for transcript distribution prediction and cell type deconvolution. *Nat Methods* 19, 662–670 (2022). DOI: 10.1038/s41592-022-01480-9
36. Zhang C et al. STAMarker: determining spatial domain-specific variable genes with saliency maps in deep learning. *Nucleic Acids Res* 51, e105 (2023). DOI: 10.1093/nar/gkad801
37. Cang Z, Nie Q, et al. Screening cell–cell communication in spatial transcriptomics via collective optimal transport. *Nat Methods* 20, 218–228 (2023). DOI: 10.1038/s41592-022-01728-4

## Author contributions
[TBD]

## Competing interests
The authors declare no competing interests.

---

## Figure legends

**Fig. 1 | MyerST framework.** a, Pipeline: spatial omics data (Visium/Xenium) and any GNN host are wrapped by an adapter; the Myerson engine samples connected coalitions (MC with caching) and emits node, edge and gene attributions, all verified by the efficiency self-audit (Σφ = v(N) − v(∅), exact; DLPFC example shown) and the fidelity benchmark. b, Box-1 example: on a three-node path with v(S)=|S|², Shapley credit is position-blind (3,3,3) while Myerson rewards the bridge (2.67, 3.67, 2.67). c, Real attribution outputs: Myerson φ on the DLPFC L5|L6 boundary; per-T-cell CXCL12 effect (ψ) in Xenium breast cancer; gene attributions with known markers highlighted.

**Fig. 2 | Benchmark: operator semantics decide what importance means.** a, Node recovery AUROC under per-side games (mean ± s.d., 3 seeds): IG dominates; Myerson's coalitional credit measures irreplaceability, not boundary membership. b, Masking fidelity (raw margin-decay AUC): IG best; attention scores below random. c, Gene-level recovery across redundancy regimes. d, ROAR insensitivity: held-out accuracy after removing top-k genes (all methods ≈ flat). e, Myerson sampler convergence (M^(−1/2) reference) and exact efficiency audit. f, Evaluation-hygiene checklist: four failure modes (class-cancelling targets, friendly fire, prototype-injecting baselines, sign-convention mismatch) and their fixes.

**Fig. 3 | DLPFC boundary attribution.** a, Cortical layers of slice 151673. b, Myerson φ on the L5|L6 boundary band. c, Node recovery comparison with the efficiency audit annotation. d, IG gene attributions (red: known layer markers). e, Cross-host attribution agreement (weak vs paper-grade host). f, Host quality (clustering ARI) gates attribution agreement.

**Fig. 4 | CXCL12-mediated T-cell exclusion in Xenium breast cancer.** a, Tumour–immune interface (12,706 cells). b, T-cell infiltration states vs CXCL12-expressing cells. c, Per-T-cell CXCL12 effect (ψ_gene) geography. d, Pathway-level effects (paired t annotated). e, Sender-type decomposition of the CXCL12 effect. f, Target-semantics control: cytotoxicity target (mismatched) vs location target (matched).

**Fig. 5 | Cross-cohort, cross-platform replication.** Rows: Xenium breast Rep1, Xenium breast Rep2, Xenium lung NSCLC, Visium breast. Columns: cell-type map; T-cell states vs CXCL12 sources; pathway effects (paired t). CXCL12 is the strongest pathway in all four datasets (|t| ≥ 5); its sign tracks cohort infiltration stage.

**Box 1 | Why spatial attribution needs a new axiom.** Shapley free-coalition assumption; Myerson communication games (component efficiency + fairness); the three-node worked example; practical consequences (topology-respecting credit, self-audit, edge decomposition, computability); honest caveats (exchangeable players share credit; attribution ≠ causation; local exact games vs global sampling).
