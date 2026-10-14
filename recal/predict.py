"""Recovery@k estimation under learner-grouped cross-validation.

* Models: majority, prior-accuracy heuristic, logistic regression (one-hot),
  LightGBM, XGBoost, optional TabPFN.
* Optional post-hoc calibration (Platt / isotonic) fitted INSIDE each training fold.
* Metrics: ROC-AUC, accuracy, balanced accuracy, macro-F1, Brier, log-loss, ECE.
* Paired learner-level bootstrap for model differences (CI + two-sided p).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, log_loss, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .cohort import CATEGORICAL, FEATURES, NUMERIC

warnings.filterwarnings("ignore")
SEED = 42


def _encode_int(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in CATEGORICAL:
        if c in X:
            X[c] = X[c].astype("category").cat.codes.replace(-1, np.nan)
    return X


def make_model(name: str, features: list[str]):
    num = [f for f in features if f in NUMERIC]
    cat = [f for f in features if f in CATEGORICAL]
    if name == "logreg":
        pre = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=3), cat),
        ])
        return make_pipeline(pre, LogisticRegression(C=1.0, max_iter=2000))
    if name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=15, max_depth=4,
                                  min_child_samples=20, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, reg_alpha=0, reg_lambda=1,
                                  random_state=SEED, verbose=-1)
    if name == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, subsample=0.8,
                                 colsample_bytree=0.8, min_child_weight=1, gamma=0, reg_alpha=0,
                                 reg_lambda=1, random_state=SEED, eval_metric="logloss")
    if name == "tabpfn":
        from tabpfn import TabPFNClassifier  # optional dependency
        return TabPFNClassifier(random_state=SEED)
    raise ValueError(name)


def ece(y, p, n_bins=10):
    bins = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    bins[0], bins[-1] = -np.inf, np.inf
    idx = np.digitize(p, bins[1:-1])
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "roc_auc": roc_auc_score(y, p), "accuracy": accuracy_score(y, yhat),
        "balanced_accuracy": balanced_accuracy_score(y, yhat),
        "macro_f1": f1_score(y, yhat, average="macro"),
        "brier": brier_score_loss(y, p), "log_loss": log_loss(y, pc), "ece_10": ece(y, p),
        "tp": int(((yhat == 1) & (y == 1)).sum()), "tn": int(((yhat == 0) & (y == 0)).sum()),
        "fp": int(((yhat == 1) & (y == 0)).sum()), "fn": int(((yhat == 0) & (y == 1)).sum()),
    }


def oof_predict(df: pd.DataFrame, model: str, features=FEATURES, calibration: str | None = None,
                n_splits=5) -> np.ndarray:
    """Out-of-fold probabilities with learner-grouped folds (GroupKFold, deterministic)."""
    y = df["y"].to_numpy()
    groups = df["learner_id"].to_numpy()
    p = np.zeros(len(df))
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(df, y, groups):
        if model == "majority":
            p[te] = y[tr].mean(); continue
        if model == "prior_accuracy":   # transparent single-feature heuristic
            v = df["prior_same_level_accuracy"].fillna(df["prior_concept_accuracy"]).fillna(y[tr].mean())
            p[te] = v.to_numpy()[te]; continue
        X = df[features] if model == "logreg" else _encode_int(df[features])
        est = make_model(model, features)
        if calibration:
            est = CalibratedClassifierCV(est, method=calibration, cv=3)
        est.fit(X.iloc[tr], y[tr])
        p[te] = est.predict_proba(X.iloc[te])[:, 1]
    return p


def paired_bootstrap(df, pa, pb, metric=roc_auc_score, B=2000, seed=SEED):
    """Learner-level paired bootstrap of metric(pa) - metric(pb)."""
    rng = np.random.default_rng(seed)
    y = df["y"].to_numpy()
    learners = df["learner_id"].unique()
    idx_by = {l: np.where(df["learner_id"].to_numpy() == l)[0] for l in learners}
    obs = metric(y, pa) - metric(y, pb)
    d = []
    for _ in range(B):
        samp = np.concatenate([idx_by[l] for l in rng.choice(learners, len(learners), replace=True)])
        if len(np.unique(y[samp])) < 2:
            continue
        d.append(metric(y[samp], pa[samp]) - metric(y[samp], pb[samp]))
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"delta": obs, "ci_low": lo, "ci_high": hi, "p_boot": min(p, 1.0), "B_valid": len(d)}


def bootstrap_ci(df, p, metric=roc_auc_score, B=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    y = df["y"].to_numpy(); learners = df["learner_id"].unique()
    idx_by = {l: np.where(df["learner_id"].to_numpy() == l)[0] for l in learners}
    v = []
    for _ in range(B):
        s = np.concatenate([idx_by[l] for l in rng.choice(learners, len(learners), replace=True)])
        if len(np.unique(y[s])) > 1:
            v.append(metric(y[s], p[s]))
    return np.percentile(v, [2.5, 97.5])


def run_benchmark(df, models=("majority", "prior_accuracy", "logreg", "lightgbm", "xgboost"),
                  reference="lightgbm"):
    oof, rows = {}, []
    for m in models:
        try:
            oof[m] = oof_predict(df, m)
        except ImportError as e:
            print(f"[skip] {m}: {e}"); continue
        r = {"model": m, **metrics(df["y"].to_numpy(), oof[m])}
        if m != "majority":
            lo, hi = bootstrap_ci(df, oof[m]); r["auc_ci_low"], r["auc_ci_high"] = lo, hi
        rows.append(r)
    # calibrated reference + ablation without the high-cardinality concept identifier
    oof[f"{reference}+platt"] = oof_predict(df, reference, calibration="sigmoid")
    rows.append({"model": f"{reference}+platt", **metrics(df["y"].to_numpy(), oof[f"{reference}+platt"])})
    feats_nc = [f for f in FEATURES if f != "concept"]
    oof[f"{reference}-no_concept"] = oof_predict(df, reference, features=feats_nc)
    rows.append({"model": f"{reference}-no_concept",
                 **metrics(df["y"].to_numpy(), oof[f"{reference}-no_concept"])})
    comps = []
    for m in oof:
        if m != reference and m != "majority":
            comps.append({"comparison": f"{reference} vs {m}", **paired_bootstrap(df, oof[reference], oof[m])})
    return pd.DataFrame(rows), pd.DataFrame(comps), oof
