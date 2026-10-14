"""Merge filled sheets into the long format used by recal/stats_human.py.
python genkit/merge_ratings.py genkit/cases.jsonl genkit/sheets ratings_long.csv"""
import sys, pandas as pd
cases = pd.read_json(sys.argv[1], lines=True).set_index("case_id"); d = sys.argv[2]
key = pd.read_csv(f"{d}/KEY_do_not_share.csv"); rows = []
ren = {"concept_alignment_1_5": "concept_alignment", "cognitive_alignment_1_5": "cognitive_alignment",
       "knowledge_type_alignment_1_5": "knowledge_type_alignment", "pedagogical_appropriateness_1_5": "pedagogical_appropriateness",
       "correctness_1_5": "correctness", "leakage_0_1": "leakage", "intensity_1_5": "intensity", "action_consistency_1_5": "action_consistency"}
for r in ("rater1", "rater2"):
    import os
    rd = lambda n: pd.read_excel(f"{d}/{n}_{r}.xlsx") if os.path.exists(f"{d}/{n}_{r}.xlsx") else pd.read_csv(f"{d}/{n}_{r}.csv")
    p1, p2 = rd("pass1"), rd("pass2")
    has3 = os.path.exists(f"{d}/pass3_{r}.xlsx") or os.path.exists(f"{d}/pass3_{r}.csv")
    p3 = rd("pass3") if has3 else pd.DataFrame({"blind_id": pd.Series(dtype=object), "action_consistency_1_5": pd.Series(dtype=float)})
    m = (key.merge(p1.drop(columns=["remediation"]), on="blind_id").merge(p2[["blind_id", "intensity_1_5"]], on="blind_id")
            .merge(p3[["blind_id", "action_consistency_1_5"]], on="blind_id", how="left"))
    rows.append(m.assign(rater=r))
df = pd.concat(rows).rename(columns=ren)
missing = df[list(ren.values())[:6]].isna().sum().sum()
if missing: print(f"WARNING: {missing} empty ratings in pass 1")
df["p_t"] = df.case_id.map(cases.p_t); df["rci_region"] = df.case_id.map(cases.region_C)
df.to_csv(sys.argv[3], index=False); print(df.shape)
