# Reporting Summary (draft answers for the Nature Portfolio template)

Nature Portfolio 要求生命科学投稿随附 Reporting Summary（官方 Word 模板，投稿系统下载）。以下是我们各条的拟答，届时誊入模板。

## Statistics

- **Statistical tests used**: paired two-sided t-statistics for per-receiver pathway effects (ligand occlusion vs frequency-matched random-gene null); AUROC for ground-truth recovery; Spearman correlation for cross-slice / cross-host attribution agreement; decay AUC for masking fidelity.
- **Sample sizes**: defined per experiment (n = number of receiver cells / spots / slices; e.g., n = 1,806 T-cell receivers in Xenium Rep1; n = 12 DLPFC slices). No animal/human subjects were recruited; all data are secondary analyses of public datasets.
- **Data exclusions**: spots with unassigned layer labels (DLPFC, official unannotated spots); cells failing annotation merge (<0.01%); criteria pre-specified by data provenance, not outcome-driven.
- **Replication**: key effects verified across ≥2 independent host seeds (direction and significance stable); simulator benchmarks across 3 simulation seeds (mean ± s.d.); cross-cohort replication on 4 datasets / 2 platforms; cross-slice replication on 12 DLPFC slices. All replication attempts successful.
- **Randomization / blinding**: not applicable (computational study; no treatment allocation). Randomness enters via fixed random seeds, all reported.

## Software and code

- **Custom code**: MyerST package (MIT license), available at [GitHub TBD] with installable package, experiment scripts (E1–E13), figure scripts, and an audit script that recomputes every headline number from archived artifacts.
- **Versions**: Python 3.13; torch 2.x; numpy/scipy/pandas/scikit-learn (pinned in repository); scanpy/anndata for data loading.
- **Third-party**: official STAGATE port (vendored, with attribution); scikit-learn; SciPy.

## Data

- All datasets are public: DLPFC (spatialLIBD / HumanPilot, 12 slices), 10x Xenium breast Rep1/Rep2, 10x Xenium lung NSCLC, 10x Visium breast (Block A Section 1). Direct URLs, citations and processing steps in `data/README.md` of the repository.
- Simulated data: two in silico simulators included in the package with fixed seeds (fully reproducible, no download needed).

## Materials / human / animal subjects

- None (no wet-lab work; no human participants; no animals). Ethics statement: not applicable.

## Figures

- Figure legends define error bars (s.d. across seeds), sample sizes, and statistical tests per panel (to be finalized against final figure versions).
