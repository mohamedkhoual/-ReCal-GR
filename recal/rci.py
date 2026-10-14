"""Recovery Calibration Index (RCI) as a cost-sensitive assistance policy.

Assistance levels (ordered by intensity):  k = 0 LOW, 1 MEDIUM, 2 HIGH.
Let y = R_t^k (1 = the learner recovers within the horizon).

  O_k : over-assistance cost of level k when the learner WOULD have recovered (y = 1)
        (lost productive struggle / expertise-reversal); O_0 = 0, increasing in k.
  U_k : under-assistance cost of level k when the learner would NOT recover (y = 0)
        (unproductive persistence / wheel-spinning);     U_2 = 0, decreasing in k.

Conditional risk of level k given p = P(y = 1 | x):
        rho_k(p) = p * O_k + (1 - p) * U_k
Bayes policy  k*(p) = argmin_k rho_k(p)  is a monotone threshold rule:
        escalate from k to k+1  iff  p < tau_{k,k+1},
        tau_{k,k+1} = dU_k / (dU_k + dO_k),  dU_k = U_k - U_{k+1},  dO_k = O_{k+1} - O_k.
Three non-empty regions require tau_{0,1} > tau_{1,2}.
Conversely, any thresholds imply cost ratios  dU_k / dO_k = tau / (1 - tau).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEVELS = np.array(["LOW", "MEDIUM", "HIGH"])
ACTIONS = {"LOW": ["WAIT", "HINT"], "MEDIUM": ["EXPLAIN", "EXAMPLE"],
           "HIGH": ["GUIDED_PRACTICE", "BACKOFF"]}


# ---------- thresholds ----------
def quantile_thresholds(p_oof: np.ndarray) -> tuple[float, float]:
    """Tercile operating points used in the study: (tau1 = HIGH|MEDIUM, tau2 = MEDIUM|LOW)."""
    return float(np.quantile(p_oof, 1 / 3)), float(np.quantile(p_oof, 2 / 3))


def implied_costs(tau1: float, tau2: float) -> dict:
    """Cost ratios implied by thresholds (normalising each under-assistance gain to 1)."""
    r01 = tau2 / (1 - tau2)          # dU_0/dO_0  (LOW -> MEDIUM step)
    r12 = tau1 / (1 - tau1)          # dU_1/dO_1  (MEDIUM -> HIGH step)
    dO01, dO12 = 1 / r01, 1 / r12
    return {"dU/dO LOW->MED": r01, "dU/dO MED->HIGH": r12,
            "dO LOW->MED (dU=1)": dO01, "dO MED->HIGH (dU=1)": dO12,
            "relative over-assistance cost of 2nd step": dO12 / dO01,
            # canonical cost vectors with dU = 1 per step
            "O": (0.0, dO01, dO01 + dO12), "U": (2.0, 1.0, 0.0)}


def cost_thresholds(O, U) -> tuple[float, float]:
    O, U = np.asarray(O, float), np.asarray(U, float)
    t01 = (U[0] - U[1]) / ((U[0] - U[1]) + (O[1] - O[0]))
    t12 = (U[1] - U[2]) / ((U[1] - U[2]) + (O[2] - O[1]))
    if not t01 > t12:
        raise ValueError("MEDIUM region empty: need tau01 > tau12")
    return float(t12), float(t01)      # (tau1, tau2) in manuscript notation


# ---------- policy ----------
def region(p, tau1, tau2):
    p = np.asarray(p)
    return np.where(p <= tau1, "HIGH", np.where(p <= tau2, "MEDIUM", "LOW"))


def select_action(row: pd.Series, r: str, kappa: int = 2, theta: float = 0.5,
                  example_levels=("apply", "analyze", "evaluate", "create"),
                  example_ktypes=("procedural",)) -> str:
    """Deterministic within-region rule a_t = pi(r_t, h_t).
    !! Authors: set this to EXACTLY the rule used in the executed prompts. !!"""
    if r == "LOW":
        return "HINT"                                   # WAIT = no generation (deployment only)
    if r == "MEDIUM":
        lvl = str(row.get("cognitive_level_name", row.get("cognitive_level", ""))).lower()
        kt = str(row.get("knowledge_type_name", row.get("knowledge_type", ""))).lower()
        return "EXAMPLE" if (lvl in example_levels or kt in example_ktypes) else "EXPLAIN"
    pca = row.get("prior_concept_accuracy", np.nan)
    persistent = row.get("consecutive_failures", 0) >= kappa and (np.isnan(pca) or pca < theta)
    return "BACKOFF" if persistent else "GUIDED_PRACTICE"


# ---------- offline policy evaluation on observed labels ----------
def empirical_risk(levels, y, O, U):
    k = pd.Series(levels).map({"LOW": 0, "MEDIUM": 1, "HIGH": 2}).to_numpy()
    O, U = np.asarray(O), np.asarray(U)
    return np.where(y == 1, O[k], U[k])


def evaluate_policies(df: pd.DataFrame, p: np.ndarray, tau1, tau2, O, U, B=2000, seed=42):
    """Compare assistance policies on the cohort (learner-level bootstrap CIs).
    over_assist  = P(HIGH | y=1);  under_assist = P(LOW | y=0)."""
    rng = np.random.default_rng(seed)
    y = df["y"].to_numpy()
    n = len(y)
    perm = rng.permutation(n)
    pol = {
        "always MEDIUM": np.repeat("MEDIUM", n),
        "random (same marginals)": region(p, tau1, tau2)[perm],
        "RCI tercile (study)": region(p, tau1, tau2),
        "oracle (observed label)": np.where(y == 1, "LOW", "HIGH"),
    }
    learners = df["learner_id"].to_numpy(); ul = np.unique(learners)
    idx_by = {l: np.where(learners == l)[0] for l in ul}
    rows = []
    for name, lv in pol.items():
        risk = empirical_risk(lv, y, O, U)
        bs = []
        for _ in range(B):
            s = np.concatenate([idx_by[l] for l in rng.choice(ul, len(ul), replace=True)])
            bs.append(risk[s].mean())
        rows.append({"policy": name, "mean_risk": risk.mean(),
                     "ci_low": np.percentile(bs, 2.5), "ci_high": np.percentile(bs, 97.5),
                     "over_assist P(HIGH|y=1)": np.mean(lv[y == 1] == "HIGH"),
                     "under_assist P(LOW|y=0)": np.mean(lv[y == 0] == "LOW"),
                     "share HIGH/MED/LOW": "/".join(f"{np.mean(lv == l):.2f}" for l in ["HIGH", "MEDIUM", "LOW"])})
    return pd.DataFrame(rows)


def recovery_by_region(df, p, tau1, tau2):
    """Observed Recovery@k rate in each RCI region (monotonicity check)."""
    r = region(p, tau1, tau2)
    return (pd.DataFrame({"region": r, "y": df["y"].to_numpy()})
            .groupby("region")["y"].agg(["count", "mean"]).reindex(["HIGH", "MEDIUM", "LOW"]))


def cost_sweep(df, p, grid=np.linspace(0.25, 4, 16)):
    """Sensitivity: thresholds and region shares as the MED->HIGH over-assistance cost varies."""
    out = []
    for c in grid:
        O, U = (0, 1, 1 + c), (2, 1, 0)
        try:
            t1, t2 = cost_thresholds(O, U)
        except ValueError:
            continue
        lv = region(p, t1, t2)
        out.append({"dO_MED->HIGH": c, "tau1": t1, "tau2": t2,
                    **{f"share_{l}": np.mean(lv == l) for l in ["HIGH", "MEDIUM", "LOW"]}})
    return pd.DataFrame(out)
