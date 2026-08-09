"""Full audit: recompute every headline number from saved artifacts and
reconcile against the claimed values. No trust, only verification.
"""

import pickle

import numpy as np
from scipy.stats import spearmanr

from scripts.e1_driver_gene_recovery import auroc

CHECKS = []


def check(name, recomputed, claimed, tol=0.01):
    ok = abs(recomputed - claimed) <= tol
    CHECKS.append((name, recomputed, claimed, ok))
    flag = "OK " if ok else "FAIL"
    print(f"  [{flag}] {name:<58} recomputed={recomputed:+.4f} claimed={claimed:+.4f}")


print("=" * 78)
print("AUDIT: recomputing headline numbers from saved artifacts")
print("=" * 78)

# ---------- E4: DLPFC Myerson node AUROC
print("\n[E4] DLPFC L5|L6 Myerson")
e4 = np.load("data/processed/e4_myerson_dlpfc_L5L6.npz", allow_pickle=True)
check("Myerson node AUROC", auroc(e4["phi"], e4["truth"]), 0.883, 0.01)
# ig_node was archived in the E5 artifact (same player set & ground truth)
e5 = np.load("data/processed/e5_fidelity_edges.npz", allow_pickle=True)
assert np.array_equal(e5["players"], e4["players"]), "player sets differ!"
check("IG node AUROC (from E5 artifact)", auroc(e5["ig_node"], e4["truth"]),
      0.993, 0.02)

# ---------- E8: multi-protocol matrix
print("\n[E8] benchmark matrix (medium regime spot checks)")
with open("data/processed/e8_matrix.pkl", "rb") as f:
    e8 = pickle.load(f)
check("P1 node IG-node (medium)", e8["medium"]["P1_node"]["IG-node"], 1.000, 0.01)
check("P2 node Myerson (medium)", e8["medium"]["P2_node"]["Myerson"], 1.671, 0.02)
check("P2 node attention < random (medium)",
      e8["medium"]["P2_node"]["attention"] - e8["medium"]["P2_node"]["random"],
      0.690 - 0.927, 0.02)

# ---------- E9b: local CCC synergy
print("\n[E9b] local CCC synergy (synthetic)")
e9b = np.load("data/processed/e9b_local_ccc.npz", allow_pickle=True)
check("local synergy AUROC", auroc(e9b["psi"], e9b["truth"]), 0.566, 0.02)

# ---------- E10e: Xenium CXCL12 flagship
print("\n[E10e] Xenium breast Rep1 CXCL12")
e10 = np.load("data/processed/e10e_ccxcl12.npz", allow_pickle=True)
obs, null = e10["obs"], e10["null"]
diff = obs - null
t_stat = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
check("CXCL12 mean psi", float(obs.mean()), -1.0676, 0.01)
check("CXCL12 paired t", float(t_stat), -38.9, 2.0)
check("n receivers", float(len(obs)), 1806, 1)

# ---------- E11: Rep2 + Lung
print("\n[E11] multi-cohort")
for name, exp_psi, exp_t in [("xenium_breast_rep2", 0.2859, 18.6),
                             ("xenium_lung", 0.6787, 5.0)]:
    d = np.load(f"data/processed/e11_{name}.npz", allow_pickle=True)
    rows = {str(r[0]): (float(r[1]), float(r[2])) for r in d["pathways"]}
    psi, t = rows["CXCL12->CXCR4"]
    check(f"{name} CXCL12 psi", psi, exp_psi, 0.02)
    check(f"{name} CXCL12 t", t, exp_t, 1.0)

# ---------- E13: Visium
print("\n[E13] Visium breast")
d = np.load("data/processed/e11_visium_breast.npz", allow_pickle=True)
rows = {str(r[0]): (float(r[1]), float(r[2])) for r in d["pathways"]}
psi, t = rows["CXCL12->CXCR4"]
check("visium CXCL12 psi", psi, 1.2957, 0.02)
check("visium CXCL12 t", t, 28.5, 1.0)
psi2, t2 = rows["PTN->SDC4"]
check("visium PTN psi", psi2, 0.4126, 0.02)

# ---------- E12: 12-slice DLPFC
print("\n[E12] 12-slice DLPFC")
SLICES = ["151507", "151508", "151509", "151510", "151669", "151670",
          "151671", "151672", "151673", "151674", "151675", "151676"]
igs = {}
n_marker_full = 0
for sid in SLICES:
    c = np.load(f"data/processed/e12_{sid}.npz", allow_pickle=True)
    genes, ig = c["genes"], c["ig"]
    igs[sid] = (genes, ig)
    top20 = set(genes[np.argsort(ig)[::-1][:20]])
    if "PCP4" in top20 and "KRT17" in top20:
        n_marker_full += 1
check("slices with PCP4+KRT17 in top20", float(n_marker_full), 12.0, 0.1)
common = set(igs[SLICES[0]][0])
for sid in SLICES[1:]:
    common &= set(igs[sid][0])
common = sorted(common)
sps = []
for i in range(len(SLICES)):
    for j in range(i + 1, len(SLICES)):
        g1, v1 = igs[SLICES[i]]
        g2, v2 = igs[SLICES[j]]
        m1 = np.array([np.where(g1 == g)[0][0] for g in common])
        m2 = np.array([np.where(g2 == g)[0][0] for g in common])
        sps.append(spearmanr(v1[m1], v2[m2]).statistic)
check("cross-slice Spearman mean", float(np.mean(sps)), 0.314, 0.03)

# ---------- summary
print("\n" + "=" * 78)
n_ok = sum(1 for *_, ok in CHECKS if ok)
print(f"AUDIT SUMMARY: {n_ok}/{len(CHECKS)} checks passed")
if n_ok < len(CHECKS):
    print("FAILED checks:")
    for name, rec, cl, ok in CHECKS:
        if not ok:
            print(f"  - {name}: recomputed {rec:+.4f} vs claimed {cl:+.4f}")
