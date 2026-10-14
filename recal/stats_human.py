"""Statistical analysis of the human evaluation (paired repeated-measures design).

Expected long-format CSV, one row per (rater, case, condition):
  rater, case_id, condition, concept_alignment, cognitive_alignment,
  knowledge_type_alignment, pedagogical_appropriateness, correctness,   (1-5)
  leakage (0/1, this rater), action_consistency (1-5 or empty),
  intensity (1-5, realised assistance intensity; optional but recommended),
  p_t, rci_region                                                      (case metadata)
Optional: leakage_adjudicated (0/1, identical for both raters of a case x condition).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

DIMS = ["concept_alignment", "cognitive_alignment", "knowledge_type_alignment",
        "pedagogical_appropriateness", "correctness"]


def holm(pvals):
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(run, 1.0)
    return adj


def wilcoxon_rb(x, y):
    """Two-sided Wilcoxon signed-rank + matched-pairs rank-biserial r = (T+ - T-)/(T+ + T-)."""
    d = np.asarray(y) - np.asarray(x)
    d = d[d != 0]
    ranks = stats.rankdata(np.abs(d))
    tp, tm = ranks[d > 0].sum(), ranks[d < 0].sum()
    res = stats.wilcoxon(x, y, zero_method="wilcox", correction=False, method="auto")
    return {"n_nonzero": int(len(d)), "W": float(res.statistic), "p": float(res.pvalue),
            "r_rb": float((tp - tm) / (tp + tm)) if len(d) else 0.0,
            "mean_diff": float(np.mean(np.asarray(y) - np.asarray(x)))}


def cochran_q(M):
    M = np.asarray(M, int); k = M.shape[1]
    cj, ri, N = M.sum(0), M.sum(1), M.sum()
    den = k * N - (ri ** 2).sum()
    Q = (k - 1) * (k * (cj ** 2).sum() - N ** 2) / den if den else 0.0
    return Q, stats.chi2.sf(Q, k - 1)


def mcnemar_exact(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    n01, n10 = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    return n01, n10, stats.binomtest(n10, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0


def weighted_kappa(a, b, labels=(1, 2, 3, 4, 5)):
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(a, b, weights="quadratic", labels=list(labels))


def analyse(df: pd.DataFrame, conds=None, B=5000, seed=42):
    rng = np.random.default_rng(seed)
    df = df.copy()
    if conds is None:   # main progression = the C1..C5 conditions that were rated, in order
        conds = tuple(c for c in ("C1", "C2", "C3", "C4", "C5") if c in set(df["condition"]))
    allc = [c for c in ("C1", "C2", "C3", "C4", "C4M", "C4H", "C5", "C4X") if c in set(df["condition"])]
    df["Q"] = df[DIMS].mean(axis=1)
    out = {}

    # inter-rater agreement per dimension and pooled
    r1, r2 = sorted(df["rater"].unique())[:2]
    a = df[df.rater == r1].set_index(["case_id", "condition"]).sort_index()
    b = df[df.rater == r2].set_index(["case_id", "condition"]).sort_index()
    out["kappa"] = {d: weighted_kappa(a[d], b[d]) for d in DIMS}
    out["kappa"]["pooled"] = weighted_kappa(np.concatenate([a[d] for d in DIMS]),
                                            np.concatenate([b[d] for d in DIMS]))

    # evaluator-averaged case x condition score
    Qall = df.groupby(["case_id", "condition"])["Q"].mean().unstack()[allc]
    Qbar = Qall[list(conds)]
    desc = []
    for c in allc:
        v = Qall[c].to_numpy()
        bs = [rng.choice(v, len(v)).mean() for _ in range(B)]
        desc.append({"condition": c, "mean": v.mean(), "sd": v.std(ddof=1), "median": np.median(v),
                     "ci_low": np.percentile(bs, 2.5), "ci_high": np.percentile(bs, 97.5)})
    out["descriptives"] = pd.DataFrame(desc)

    fr = stats.friedmanchisquare(*[Qbar[c] for c in conds])
    n, k = Qbar.shape
    out["friedman"] = {"chi2": fr.statistic, "p": fr.pvalue, "kendall_W": fr.statistic / (n * (k - 1))}

    # adjacent (incremental) contrasts, Holm-adjusted as one family
    adj_pairs = list(zip(conds[:-1], conds[1:]))
    rows = [{"contrast": f"{x} vs {y}", **wilcoxon_rb(Qbar[x], Qbar[y])} for x, y in adj_pairs]
    for r, pa in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = pa
    # second planned confirmatory contrast: calibrated vs mismatched decision
    qm = df.groupby(["case_id", "condition"])["Q"].mean().unstack()
    planned = [("C4M", "C4", "shuffled vs calibrated decision"), ("C4H", "C4", "history-rule vs recovery-model decision"),
               ("C4X", "C5", "matched extra pass vs self-check")]
    prow = [{"contrast": f"{a} vs {b} ({lab})", **wilcoxon_rb(qm[a], qm[b])} for a, b, lab in planned if a in qm and b in qm]
    for r, pa in zip(prow, holm([r["p"] for r in prow]) if prow else []):
        r["p_holm"] = pa
    rows += prow
    out["adjacent_contrasts"] = pd.DataFrame(rows)

    # linear mixed model on rater-level scores: Q ~ condition + rater + (1 | case)
    try:
        import statsmodels.formula.api as smf
        m = smf.mixedlm("Q ~ C(condition, Treatment('C3')) + C(rater)", df, groups=df["case_id"]).fit(reml=True)
        out["lmm"] = m.summary().tables[1]
    except Exception as e:  # pragma: no cover
        out["lmm"] = f"LMM failed: {e}"

    # region x condition (does the C3->C4 gain depend on the assistance region?)
    if "rci_region" in df:
        reg = df.groupby("case_id")["rci_region"].first()
        g = (Qbar["C4"] - Qbar["C3"]).rename("gain").to_frame().join(reg)
        out["gain_by_region"] = g.groupby("rci_region")["gain"].agg(["count", "mean", "std"])
        out["kruskal_gain_by_region"] = stats.kruskal(*[v["gain"] for _, v in g.groupby("rci_region")])

    # leakage
    lk_col = "leakage_adjudicated" if "leakage_adjudicated" in df else "leakage"
    L = df.groupby(["case_id", "condition"])[lk_col].max().unstack()[list(conds)]
    out["leakage_rate"] = L.mean()
    Qc, pq = cochran_q(L.to_numpy()); out["cochran_q"] = {"Q": Qc, "p": pq}
    mc = [{"contrast": f"{x} vs {y}", **dict(zip(["n01", "n10", "p"], mcnemar_exact(L[x], L[y])))}
          for x, y in adj_pairs]
    for r, pa in zip(mc, holm([r["p"] for r in mc])):
        r["p_holm"] = pa
    out["leakage_contrasts"] = pd.DataFrame(mc)

    # mechanism: is realised intensity aligned with predicted recovery?
    if "intensity" in df and df["intensity"].notna().any():
        I = df.groupby(["case_id", "condition"])["intensity"].mean().unstack()
        p_t = df.groupby("case_id")["p_t"].first()
        rho = {c: stats.spearmanr(p_t.loc[I.index], I[c]).statistic for c in allc if c in I}
        cases = I.index.to_numpy(); diffs = []
        for _ in range(B):
            s = rng.choice(cases, len(cases))
            r3 = stats.spearmanr(p_t.loc[s], I.loc[s, "C3"]).statistic
            r4 = stats.spearmanr(p_t.loc[s], I.loc[s, "C4"]).statistic
            diffs.append(r4 - r3)
        out["intensity_alignment"] = {"spearman_by_condition": rho,
                                      "delta_C4_minus_C3": rho.get("C4", np.nan) - rho.get("C3", np.nan),
                                      "ci": tuple(np.nanpercentile(diffs, [2.5, 97.5]))}
    return out
