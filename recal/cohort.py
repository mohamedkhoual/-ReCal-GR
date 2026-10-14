"""Recovery@k cohort construction for MOOC-Radar (multi-concept items).

Input: interactions DataFrame with columns
    user_id, order (global chronological index within learner), problem_id, is_correct,
    attempts, concepts (list[str]), cognitive_dimension, knowledge_type, course_id
Definitions
  failure episode t       : interaction with is_correct == 0
  eligible opportunity    : a LATER interaction of the same learner whose item has the same
                            cognitive level and shares >= 1 concept with the failed item
                            (mode="shared"); alternatives: mode="primary" (same first-listed
                            concept) and mode="exact" (identical concept set)
  Recovery@k              : 1 if any of the first k eligible opportunities is correct;
                            episodes with < k eligible opportunities are excluded
History features use strictly earlier interactions of the same learner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["prior_overall_accuracy", "prior_concept_accuracy", "prior_same_level_accuracy",
            "recent_failure_ratio", "consecutive_failures", "attempts",
            "cognitive_level", "knowledge_type", "concept"]
NUMERIC = FEATURES[:6]
CATEGORICAL = FEATURES[6:]
MAXK = 5


def _keys(df, mode):
    if mode == "shared":
        ex = df[["row", "user_id", "order", "is_correct", "cognitive_dimension", "concepts"]].explode("concepts")
        ex = ex.rename(columns={"concepts": "ckey"})
    elif mode == "primary":
        ex = df[["row", "user_id", "order", "is_correct", "cognitive_dimension"]].assign(ckey=df["concepts"].str[0])
    elif mode == "exact":
        ex = df[["row", "user_id", "order", "is_correct", "cognitive_dimension"]].assign(
            ckey=df["concepts"].apply(lambda c: "|".join(sorted(c))))
    else:
        raise ValueError(mode)
    return ex.drop_duplicates(["row", "ckey"])


def _prior_acc(ex, by):
    """Pooled prior accuracy over chains `by`, aggregated back to item rows."""
    ex = ex.sort_values(["user_id", "order"], kind="mergesort")
    g = ex.groupby(by, sort=False)["is_correct"]
    ex = ex.assign(pc=g.cumsum() - ex["is_correct"], pn=g.cumcount())
    a = ex.groupby("row")[["pc", "pn"]].sum()
    return (a["pc"] / a["pn"].replace(0, np.nan))


def build_cohort(df: pd.DataFrame, k: int = 3, mode: str = "shared", min_gap_h: float = 0.0):
    df = df.sort_values(["user_id", "order"], kind="mergesort").reset_index(drop=True)
    df["row"] = np.arange(len(df))
    c = df["is_correct"].astype(float)
    g = df.groupby("user_id", sort=False)
    n_prior = g.cumcount()
    df["prior_overall_accuracy"] = (g["is_correct"].cumsum() - df["is_correct"]) / n_prior.replace(0, np.nan)
    df["recent_failure_ratio"] = (1 - c).groupby(df["user_id"]).transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    # consecutive failures before t (reset on correct)
    wrong = (df["is_correct"] == 0).astype(int)
    blk = (df["is_correct"] == 1).astype(int).groupby(df["user_id"]).cumsum()
    run = wrong.groupby([df["user_id"], blk]).cumsum()
    df["consecutive_failures"] = run.groupby(df["user_id"]).shift(1).fillna(0).astype(int)

    exc = df[["row", "user_id", "order", "is_correct", "concepts"]].explode("concepts").drop_duplicates(["row", "concepts"])
    df["prior_concept_accuracy"] = _prior_acc(exc, ["user_id", "concepts"]).reindex(df["row"]).to_numpy()
    exl = exc.merge(df[["row", "cognitive_dimension"]], on="row")
    df["prior_same_level_accuracy"] = _prior_acc(exl, ["user_id", "concepts", "cognitive_dimension"]).reindex(df["row"]).to_numpy()

    # ----- eligible opportunities -----
    ex = _keys(df, mode).sort_values(["user_id", "order"], kind="mergesort")
    gk = ex.groupby(["user_id", "ckey", "cognitive_dimension"], sort=False)
    cand = []
    for j in range(1, MAXK + 1):
        nxt = pd.DataFrame({"row": ex["row"].to_numpy(),
                            "nrow": gk["row"].shift(-j).to_numpy(),
                            "norder": gk["order"].shift(-j).to_numpy(),
                            "ncorr": gk["is_correct"].shift(-j).to_numpy()})
        cand.append(nxt.dropna())
    cand = pd.concat(cand)
    fail_rows = set(df.loc[df["is_correct"] == 0, "row"])
    cand = cand[cand["row"].isin(fail_rows)].drop_duplicates(["row", "nrow"])
    if min_gap_h > 0:   # only opportunities at least min_gap_h hours after the failure (cross-session recovery)
        tt = df.set_index("row")["submit_time"]
        gap = (tt.reindex(cand["nrow"].astype(int)).to_numpy() - tt.reindex(cand["row"]).to_numpy()) / np.timedelta64(1, "h")
        cand = cand[gap >= min_gap_h]
    cand = cand.sort_values(["row", "norder"]).assign(j=lambda d: d.groupby("row").cumcount() + 1)
    cand = cand[cand["j"] <= MAXK]
    n_el = cand.groupby("row").size()
    lab = {}
    for kk in range(1, MAXK + 1):
        lab[f"R@{kk}"] = cand[cand["j"] <= kk].groupby("row")["ncorr"].max()
    fails = df[df["is_correct"] == 0].copy()
    fails["n_eligible"] = n_el.reindex(fails["row"]).fillna(0).astype(int).to_numpy()
    for kk in range(1, MAXK + 1):
        fails[f"R@{kk}"] = lab[f"R@{kk}"].reindex(fails["row"]).to_numpy()
        fails.loc[fails["n_eligible"] < kk, f"R@{kk}"] = np.nan

    if "submit_time" in df.columns:
        tmap = df.set_index("row")["submit_time"]
        c3 = cand[cand["j"] <= k].assign(nt=lambda d: tmap.reindex(d["nrow"].astype(int)).to_numpy())
        t_k = c3[c3["j"] == k].set_index("row")["nt"]
        t_rec = c3[c3["ncorr"] == 1].groupby("row")["nt"].min()
        fails["t_kth_opportunity"] = t_k.reindex(fails["row"]).to_numpy()
        fails["t_first_recovery"] = t_rec.reindex(fails["row"]).to_numpy()
    fails["cognitive_level"] = fails["cognitive_dimension"]
    fails["concept"] = fails["concepts"].str[0]
    funnel = {"definition": mode, "min_gap_h": min_gap_h, "interactions": int(len(df)), "learners": int(df["user_id"].nunique()),
              "problems": int(df["problem_id"].nunique()),
              "incorrect_interactions": int(len(fails))}
    for kk in range(1, MAXK + 1):
        funnel[f"failures_with_>={kk}_eligible"] = int((fails["n_eligible"] >= kk).sum())
    co = fails[fails["n_eligible"] >= k].copy()
    co["y"] = co[f"R@{k}"].astype(int)
    co["learner_id"] = co["user_id"]
    funnel.update({"k": k, "cohort_episodes": int(len(co)), "cohort_learners": int(co["user_id"].nunique()),
                   "recovered": int(co["y"].sum()), "not_recovered": int((1 - co["y"]).sum()),
                   "recovery_rate": float(co["y"].mean()),
                   "episodes_per_learner_median": float(co.groupby("user_id").size().median())})
    return co.reset_index(drop=True), funnel, fails
