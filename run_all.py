"""Reproduce every predictive / RCI table and figure of the revised ReCal-GR manuscript.

  python run_all.py --fine student-problem-fine.json --problems problem.json --out results
  python run_all.py --ratings ratings_long.csv --out results          # human-evaluation statistics
"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from recal import cohort, predict, rci, stats_human

COST_STAR = ((0.0, 0.25, 1.25), (2.0, 1.0, 0.0))     # C*: dU = 1 per step; dO = 0.25 (LOW->MED), 1.0 (MED->HIGH)


def load_mooc(fine, problems):
    seqs = json.load(open(fine, encoding="utf8"))
    df = pd.DataFrame([dict(r, seqpos=i) for s in seqs for i, r in enumerate(s["seq"])])
    df["submit_time"] = pd.to_datetime(df["submit_time"])
    pr = pd.read_json(problems, lines=True)[["problem_id", "course_id", "knowledge_type", "cognitive_dimension", "concepts"]]
    df = df.merge(pr, on="problem_id", how="left").sort_values(["user_id", "submit_time", "seqpos"], kind="mergesort")
    df["order"] = df.groupby("user_id").cumcount()
    return df


def learner_boot(learners, B=1000, seed=42):
    rng = np.random.default_rng(seed); ul = np.unique(learners)
    idx = pd.Series(np.arange(len(learners))).groupby(learners).apply(np.array).to_dict()
    return [np.concatenate([idx[l] for l in rng.choice(ul, len(ul))]) for _ in range(B)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fine"); ap.add_argument("--problems"); ap.add_argument("--ratings")
    ap.add_argument("--mode", default="shared"); ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default="results")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); pd.set_option("display.width", 250)

    if a.fine:
        df = load_mooc(a.fine, a.problems)
        co, funnel, _ = cohort.build_cohort(df, k=a.k, mode=a.mode)
        print(json.dumps(funnel, indent=1)); json.dump(funnel, open(f"{a.out}/funnel.json", "w"), indent=1)

        bench, comps, oof = predict.run_benchmark(co)
        y = co["y"].to_numpy()
        bench["pr_auc_nonrecovery"] = [average_precision_score(1 - y, 1 - oof[m]) for m in bench["model"]]
        print(bench.round(4).to_string()); print(comps.round(4).to_string())
        bench.to_csv(f"{a.out}/table_prediction.csv", index=False); comps.to_csv(f"{a.out}/table_model_comparisons.csv", index=False)

        p = oof["logreg"]                                  # selected estimator (best AUC, well calibrated)
        t1q, t2q = rci.quantile_thresholds(p); O, U = COST_STAR; t1c, t2c = rci.cost_thresholds(O, U)
        lvq, lvc = rci.region(p, t1q, t2q), rci.region(p, t1c, t2c)
        perm = np.random.default_rng(7).permutation(len(p))
        pols = {"Constant MEDIUM": np.repeat("MEDIUM", len(p)), "Random (tercile shares)": lvq[perm],
                "RCI-Q (tercile thresholds)": lvq, "RCI-C (cost thresholds, C*)": lvc,
                "Oracle (observed label)": np.where(y == 1, "LOW", "HIGH")}
        boots = learner_boot(co["learner_id"].to_numpy()); rows = []
        for n, lv in pols.items():
            r = rci.empirical_risk(lv, y, O, U); bs = [r[s].mean() for s in boots]
            rows.append({"policy": n, **{f"share_{l}": np.mean(lv == l) for l in ["HIGH", "MEDIUM", "LOW"]},
                         "P(HIGH|R=1)": np.mean(lv[y == 1] == "HIGH"), "P(>LOW|R=1)": np.mean(lv[y == 1] != "LOW"),
                         "P(LOW|R=0)": np.mean(lv[y == 0] == "LOW"), "P(HIGH|R=0)": np.mean(lv[y == 0] == "HIGH"),
                         "risk": r.mean(), "ci_low": np.percentile(bs, 2.5), "ci_high": np.percentile(bs, 97.5)})
        pol = pd.DataFrame(rows); print(pol.round(4).to_string()); pol.to_csv(f"{a.out}/table_policies.csv", index=False)
        print("implied costs of tercile thresholds:", rci.implied_costs(t1q, t2q))
        for n, lv in [("RCI-Q", lvq), ("RCI-C", lvc)]:
            print(n, pd.DataFrame({"r": lv, "y": y}).groupby("r")["y"].agg(["count", "mean"]))
        worse = []
        for lam in np.geomspace(0.3, 60, 40):
            for g in (1.5, 2, 4, 8):
                O2, U2 = (0, 1 / lam, (1 + g) / lam), (2, 1, 0); t1, t2 = rci.cost_thresholds(O2, U2)
                worse.append(rci.empirical_risk(rci.region(p, t1, t2), y, O2, U2).mean()
                             <= rci.empirical_risk(lvq, y, O2, U2).mean() + 1e-12)
        print(f"cost-derived policy has risk <= tercile policy under {np.mean(worse):.0%} of {len(worse)} cost matrices")
        co.assign(p_t=p, region_C=lvc, region_Q=lvq).to_pickle(f"{a.out}/cohort_scored.pkl")

        sens = []
        for mode, k in [("shared", 1), ("shared", 2), ("shared", 5), ("primary", 3), ("exact", 3)]:
            c2, _, _ = cohort.build_cohort(df, k=k, mode=mode); r = {"definition": mode, "k": k, "episodes": len(c2),
                                                                     "learners": c2["user_id"].nunique(), "recovery_rate": c2["y"].mean()}
            for m in ["logreg", "lightgbm"]:
                pp = predict.oof_predict(c2, m); r[f"auc_{m}"] = roc_auc_score(c2["y"], pp); r[f"brier_{m}"] = brier_score_loss(c2["y"], pp)
            sens.append(r)
        sens = pd.DataFrame(sens); print(sens.round(4).to_string()); sens.to_csv(f"{a.out}/table_sensitivity.csv", index=False)

    if a.ratings:
        res = stats_human.analyse(pd.read_csv(a.ratings))
        for key, v in res.items():
            print(f"\n=== {key} ===\n{v}")


if __name__ == "__main__":
    main()
