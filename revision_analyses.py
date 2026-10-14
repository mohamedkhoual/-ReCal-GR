"""Additional analyses requested in the first review round (IJIES 20266579).

python revision_analyses.py --interactions interactions.pkl --out results_revision
(interactions.pkl = output of run_all.load_mooc saved with pandas.to_pickle)
"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupKFold
from recal import cohort, predict, rci

C_STAR = ((0.0, 0.25, 1.25), (2.0, 1.0, 0.0))


def policy_stats(lv, y, O, U):
    r = rci.empirical_risk(lv, y, O, U)
    return {"share_HIGH": np.mean(lv == "HIGH"), "share_MED": np.mean(lv == "MEDIUM"), "share_LOW": np.mean(lv == "LOW"),
            "P(HIGH|R=1)": np.mean(lv[y == 1] == "HIGH"), "P(LOW|R=0)": np.mean(lv[y == 0] == "LOW"),
            "P(HIGH|R=0)": np.mean(lv[y == 0] == "HIGH"),
            "rec_HIGH": y[lv == "HIGH"].mean() if (lv == "HIGH").any() else np.nan,
            "rec_MED": y[lv == "MEDIUM"].mean() if (lv == "MEDIUM").any() else np.nan,
            "rec_LOW": y[lv == "LOW"].mean() if (lv == "LOW").any() else np.nan, "risk": r.mean()}


def rank_regions(score, shares):
    """Assign HIGH/MEDIUM/LOW to the lowest/middle/highest scores with given shares (HIGH, MEDIUM)."""
    n = len(score); order = np.argsort(score, kind="mergesort"); lv = np.empty(n, dtype=object)
    nh, nm = int(round(shares[0] * n)), int(round(shares[1] * n))
    lv[order[:nh]] = "HIGH"; lv[order[nh:nh + nm]] = "MEDIUM"; lv[order[nh + nm:]] = "LOW"
    return lv.astype(str)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--interactions", required=True); ap.add_argument("--out", default="results_revision")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); out = {}
    df = pd.read_pickle(a.interactions)
    co, funnel, fails = cohort.build_cohort(df, k=3, mode="shared")
    y = co["y"].to_numpy(); O, U = C_STAR

    # ---- folds, OOF, fold-derived thresholds (thresholds learned on training learners only)
    gkf = GroupKFold(n_splits=5); folds = np.zeros(len(co), int)
    p = np.zeros(len(co)); terc = []; lv_fold_q = np.empty(len(co), dtype=object)
    for f, (tr, te) in enumerate(gkf.split(co, y, co["learner_id"])):
        folds[te] = f
        sub = co.iloc[tr].reset_index(drop=True)
        p_in = predict.oof_predict(sub, "logreg")                 # inner OOF on training learners
        t1, t2 = rci.quantile_thresholds(p_in); terc.append((t1, t2))
        est = predict.make_model("logreg", cohort.FEATURES).fit(co.iloc[tr][cohort.FEATURES], y[tr])
        p[te] = est.predict_proba(co.iloc[te][cohort.FEATURES])[:, 1]
        lv_fold_q[te] = rci.region(p[te], t1, t2)
    lv_fold_q = lv_fold_q.astype(str)
    terc = np.array(terc)
    out["fold_tercile_thresholds"] = {"tau1_mean": terc[:, 0].mean(), "tau1_sd": terc[:, 0].std(ddof=1),
                                      "tau2_mean": terc[:, 1].mean(), "tau2_sd": terc[:, 1].std(ddof=1),
                                      "per_fold": terc.round(4).tolist()}
    out["oof_auc"] = roc_auc_score(y, p); out["oof_brier"] = brier_score_loss(y, p)
    yhat = (p >= 0.5).astype(int); out["confusion_0.5"] = confusion_matrix(y, yhat).tolist()   # [[TN,FP],[FN,TP]]
    t1c, t2c = rci.cost_thresholds(O, U)
    lv_c = rci.region(p, t1c, t2c)
    pol = {"RCI-C (cost thresholds)": lv_c, "RCI-Q fold-derived terciles": lv_fold_q}

    # ---- action-controlled comparators with identical region shares as RCI-C
    shares = (np.mean(lv_c == "HIGH"), np.mean(lv_c == "MEDIUM"))
    hist = co["prior_same_level_accuracy"].fillna(co["prior_concept_accuracy"]).fillna(co["prior_overall_accuracy"]).fillna(0.5).to_numpy()
    rng = np.random.default_rng(42)
    pol["History rule, same shares"] = rank_regions(hist + rng.uniform(0, 1e-9, len(hist)), shares)
    pol["Shuffled RCI-C (same shares)"] = lv_c[rng.permutation(len(lv_c))]
    out["policies"] = pd.DataFrame({k: policy_stats(v, y, O, U) for k, v in pol.items()}).T
    # learner-bootstrap CI for risk differences vs RCI-C
    learners = co["learner_id"].to_numpy(); ul = np.unique(learners)
    idx = pd.Series(np.arange(len(y))).groupby(learners).apply(np.array).to_dict()
    boots = [np.concatenate([idx[l] for l in rng.choice(ul, len(ul))]) for _ in range(1000)]
    rc = rci.empirical_risk(lv_c, y, O, U); diffs = {}
    for k_, v in pol.items():
        if k_ == "RCI-C (cost thresholds)": continue
        r2 = rci.empirical_risk(v, y, O, U); d = [(r2[s] - rc[s]).mean() for s in boots]
        diffs[k_] = {"risk_minus_RCI-C": (r2 - rc).mean(), "ci": np.percentile(d, [2.5, 97.5]).round(4).tolist()}
    out["risk_differences"] = diffs

    # ---- nearby thresholds
    near = []
    for a1 in (0.45, 0.50, 0.55):
        for a2 in (0.75, 0.80, 0.85):
            near.append({"tau1": a1, "tau2": a2, **policy_stats(rci.region(p, a1, a2), y, O, U)})
    out["nearby_thresholds"] = pd.DataFrame(near)

    # ---- selection: retained (>=3 eligible) vs excluded failures
    fails = fails.copy(); fails["retained"] = fails["n_eligible"] >= 3
    fails["n_prior"] = fails["order"]
    cols = ["prior_overall_accuracy", "prior_concept_accuracy", "prior_same_level_accuracy", "recent_failure_ratio",
            "consecutive_failures", "n_prior", "cognitive_dimension", "knowledge_type"]
    smd = {}
    for c in cols:
        a_, b_ = fails.loc[fails.retained, c].astype(float), fails.loc[~fails.retained, c].astype(float)
        smd[c] = {"retained_mean": a_.mean(), "excluded_mean": b_.mean(),
                  "SMD": (a_.mean() - b_.mean()) / np.sqrt((a_.var() + b_.var()) / 2)}
    out["selection_smd"] = pd.DataFrame(smd).T
    out["selection_level_dist"] = pd.crosstab(fails["cognitive_dimension"], fails["retained"], normalize="columns").round(3)
    out["selection_counts"] = {"failures": len(fails), "retained": int(fails.retained.sum()),
                               "learners_retained": int(fails.loc[fails.retained, "user_id"].nunique()),
                               "learners_all_failures": int(fails["user_id"].nunique()),
                               "courses_retained": int(fails.loc[fails.retained, "course_id"].nunique()),
                               "courses_all": int(fails["course_id"].nunique())}
    # ---- elapsed time
    co_t = co.copy()
    dt_k = (pd.to_datetime(co_t["t_kth_opportunity"]) - pd.to_datetime(co_t["submit_time"])).dt.total_seconds() / 3600
    dt_r = (pd.to_datetime(co_t["t_first_recovery"]) - pd.to_datetime(co_t["submit_time"])).dt.total_seconds() / 3600
    q = lambda s: {"median_h": s.median(), "q25_h": s.quantile(.25), "q75_h": s.quantile(.75),
                   "share_<=1h": (s <= 1).mean(), "share_<=24h": (s <= 24).mean(), "share_>7d": (s > 168).mean()}
    out["elapsed_to_3rd_opportunity"] = q(dt_k.dropna()); out["elapsed_to_recovery"] = q(dt_r.dropna())
    # recovery within the horizon AND within 24 h: sensitivity of the label to elapsed time
    y24 = ((dt_r <= 24) & (co_t["y"] == 1)).astype(int)
    out["label_within_24h"] = {"rate": y24.mean(), "auc_of_same_p": roc_auc_score(y24, p)}
    out["same_session_episodes_share(3rd opp <=1h)"] = (dt_k <= 1).mean()

    pd.DataFrame({"episode_row": co["row"], "learner_id": co["learner_id"], "problem_id": co["problem_id"],
                  "fold": folds, "y": y, "p_logreg_oof": p, "region_RCI_C": lv_c}).to_csv(f"{a.out}/episodes_folds_oof.csv", index=False)
    for k_, v in out.items():
        print(f"\n=== {k_} ===\n{v.round(4).to_string() if isinstance(v, pd.DataFrame) else json.dumps(v, indent=1, default=float)}")
        if isinstance(v, pd.DataFrame): v.to_csv(f"{a.out}/{k_}.csv")
    json.dump({k_: v for k_, v in out.items() if not isinstance(v, pd.DataFrame)}, open(f"{a.out}/revision_summary.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
