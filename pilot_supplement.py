"""Supplementary audit tables for the pilot evaluation (reviewer requests R1.3, R2.4).
python pilot_supplement.py outputs.csv rater1.xlsx rater2.xlsx out_dir"""
import sys, os, json
import numpy as np, pandas as pd
from scipy import stats
from recal.stats_human import weighted_kappa
D = ['Concept Alignment (1-5)', 'Cognitive Alignment (1-5)', 'Knowledge-Type Alignment (1-5)',
     'Pedagogical Appropriateness (1-5)', 'Correctness (1-5)']
ORDER = ["Generic-Concept", "Bloom-Aware", "Context-Aware", "RePatch", "RePatch+Validator"]
o = pd.read_csv(sys.argv[1]); out = sys.argv[4]; os.makedirs(out, exist_ok=True)
R = {}
for k, f in [("R1", sys.argv[2]), ("R2", sys.argv[3])]:
    d = pd.read_excel(f).merge(o[["case_id", "method", "prompt"]], left_on=["Case ID", "Prompt / Learner Context"],
                               right_on=["case_id", "prompt"], how="left")
    d[D] = d[D].astype(float); d["Q"] = d[D].mean(axis=1); R[k] = d
res = {}
both = pd.concat([R["R1"].assign(rater="R1"), R["R2"].assign(rater="R2")])
# paired case x condition table (supplementary file)
Qbar = both.groupby(["Case ID", "method"]).Q.mean().unstack()[ORDER]
Qbar.round(3).to_csv(f"{out}/paired_case_condition_scores.csv")
both[["Blind ID", "Case ID", "method", "rater"] + D + ["Answer Leakage (0/1)", "Action Consistency (1-5 or N/A)"]].to_csv(f"{out}/independent_ratings_long.csv", index=False)
# dimension-level means by condition
res["dimension_means"] = both.groupby("method")[D].mean().loc[ORDER].round(3)
# score distributions
dist = {}
for r in ("R1", "R2"):
    for dd in D:
        dist[(r, dd)] = R[r][dd].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
res["score_distributions"] = pd.DataFrame(dist).T
# C3-C4 detail
x, yv = Qbar["Context-Aware"].to_numpy(), Qbar["RePatch"].to_numpy(); d = yv - x
nz = d[d != 0]; ranks = stats.rankdata(np.abs(nz))
w = stats.wilcoxon(x, yv, zero_method="wilcox", correction=False, method="approx")
rng = np.random.default_rng(42); bm, br = [], []
for _ in range(5000):
    s = rng.integers(0, len(d), len(d)); ds = d[s]; dn = ds[ds != 0]; rk = stats.rankdata(np.abs(dn))
    bm.append(ds.mean()); br.append((rk[dn > 0].sum() - rk[dn < 0].sum()) / rk.sum())
res["C3_C4"] = {"n_cases": len(d), "n_zero_differences": int((d == 0).sum()), "n_nonzero": len(nz),
                "n_tied_abs_ranks": int(len(nz) - len(np.unique(np.abs(nz)))),
                "T_plus": float(ranks[nz > 0].sum()), "T_minus": float(ranks[nz < 0].sum()),
                "W_statistic": float(w.statistic), "z": float(getattr(w, "zstatistic", np.nan)), "p_two_sided": float(w.pvalue),
                "method": "scipy.stats.wilcoxon, zero_method='wilcox', normal approximation with tie correction, no continuity correction",
                "r_rb": float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()),
                "r_rb_formula": "(T+ - T-) / (T+ + T-) on non-zero differences",
                "mean_diff": float(d.mean()), "mean_diff_ci95": np.percentile(bm, [2.5, 97.5]).round(4).tolist(),
                "r_rb_ci95": np.percentile(br, [2.5, 97.5]).round(4).tolist()}
fr = stats.friedmanchisquare(*[Qbar[c] for c in ORDER])
res["friedman"] = {"chi2": float(fr.statistic), "df": len(ORDER) - 1, "p": float(fr.pvalue),
                   "kendall_W": float(fr.statistic / (len(Qbar) * (len(ORDER) - 1)))}
kap = {dd: weighted_kappa(R["R1"][dd].astype(int), R["R2"][dd].astype(int)) for dd in D}
res["kappa"] = {"per_dimension": kap, "mean_of_dimensions": float(np.mean(list(kap.values()))),
                "pooled_over_dimensions": weighted_kappa(R["R1"][D].astype(int).values.ravel(), R["R2"][D].astype(int).values.ravel())}
ac = {}
for r in ("R1", "R2"):
    v = pd.to_numeric(R[r]["Action Consistency (1-5 or N/A)"], errors="coerce")
    ac[r] = R[r].assign(ac=v).groupby("method").ac.agg(["count", "mean"]).round(3).to_dict()
res["action_consistency"] = ac
res["leakage_counts"] = {r: R[r].groupby("method")["Answer Leakage (0/1)"].sum().to_dict() for r in R}
pr = o[o.method == "RePatch"].prompt
res["mandated_actions"] = pr.str.extract(r"Mandated repair action: (\w+)")[0].value_counts().to_dict()
res["support_regions"] = pr.str.extract(r"support intensity: (\w+)")[0].value_counts().to_dict()
for k, v in res.items():
    print(f"\n=== {k} ===\n{v.to_string() if isinstance(v, pd.DataFrame) else json.dumps(v, indent=1, default=float)}")
    if isinstance(v, pd.DataFrame): v.to_csv(f"{out}/{k}.csv")
json.dump({k: v for k, v in res.items() if not isinstance(v, pd.DataFrame)}, open(f"{out}/pilot_summary.json", "w"), indent=1, default=float)
