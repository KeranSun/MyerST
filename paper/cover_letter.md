# Cover letter (draft) — Nature Communications

Dear Editors,

We submit our manuscript "Topology-constrained attribution reveals verifiable explanations for spatial omics graph neural networks" for consideration as an Article in Nature Communications, in response to the "AI in spatial omics" call.

**The problem.** Graph neural networks are now the default engine for spatial omics analysis, and every downstream biological conclusion rests on interrogating these black boxes. The field's default practice — reading attention weights or gradient saliency as biological explanations — has never been systematically validated in spatial settings. Our benchmark shows the situation is worse than suspected: attention weights score *below random* in masking fidelity, and the community's gold-standard retraining metric (ROAR) is structurally insensitive on spatial GNNs.

**The root cause and the solution.** We trace the fragility to a mathematical assumption hidden in all Shapley-family attributions: free coalition formation, which tissues violate. We introduce MyerST, the first operationalization of Myerson's communication-game values (1977) for spatial omics. Beyond topology-correct credit assignment, MyerST is self-auditing: the efficiency identity holds exactly on every run, so broken or unconverged explanations fail loudly — a property no gradient or perturbation method offers. The framework is host-agnostic and ships with the benchmark that exposed these issues.

**Why Nature Communications.** The manuscript combines (i) a methodological first (communication games in spatial omics), (ii) a benchmark with practical consequences for how the field validates explanations (four failure modes we identify are, in our reading of the literature, unreported and likely widespread), and (iii) biology: spatial quantification of CXCL12-mediated T-cell exclusion across four datasets and two platforms, decomposition of its sender sources beyond the canonical CAF-only narrative, and a novel testable hypothesis (PTN–SDC4 in T-cell positioning). We believe this combination of method, rigor infrastructure, and biological insight fits the journal's broad readership across computational biology, spatial omics, and tumor immunology.

**Reproducibility.** All data are public; every headline number is recomputable from archived artifacts via a single audit script; the package (pip-installable, MIT) and experiment scripts are available at [GitHub TBD — will be filled upon repository publication]. This manuscript is not under consideration elsewhere, and no part has been published.

**Suggested reviewers** (no conflicts): [TBD — 3–5 experts in spatial omics methods / XAI / tumor microenvironment].

**Excluded reviewers**: [TBD, if any].

Sincerely,
[Corresponding author]
