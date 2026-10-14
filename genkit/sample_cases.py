"""Draw the 60 generative-evaluation cases (20 per RCI region, cost thresholds) and write prompts."""
import sys, json, ast, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd, numpy as np
from recal import rci, prompts
BLOOM = {1: "remember", 2: "understand", 3: "apply", 4: "analyze", 5: "evaluate", 6: "create"}
KTYPE = {1: "factual", 2: "conceptual", 3: "procedural", 4: "metacognitive"}
co = pd.read_pickle(sys.argv[1]); prob = pd.read_json(sys.argv[2], lines=True).set_index("problem_id")
out = sys.argv[3]; os.makedirs(out, exist_ok=True)
# history-rule regions with the same region shares as RCI-C (action-controlled comparator)
hist = co["prior_same_level_accuracy"].fillna(co["prior_concept_accuracy"]).fillna(co["prior_overall_accuracy"]).fillna(0.5)
shares = (co.region_C == "HIGH").mean(), (co.region_C == "MEDIUM").mean()
rk = hist.rank(method="first") / len(hist)
co["history_region"] = np.where(rk <= shares[0], "HIGH", np.where(rk <= shares[0] + shares[1], "MEDIUM", "LOW"))
# one case per learner: sample 20 per region without repeating a learner anywhere in the sample
rng = np.random.default_rng(42); used = set(); parts = []
for r in ["HIGH", "MEDIUM", "LOW"]:
    pool = co[(co.region_C == r) & ~co.user_id.isin(used)].sample(frac=1, random_state=42).drop_duplicates("user_id").head(20)
    used |= set(pool.user_id); parts.append(pool)
cases = pd.concat(parts).reset_index(drop=True)
assert cases.user_id.nunique() == len(cases)
cases["case_id"] = [f"Case_{i+1:03d}" for i in range(len(cases))]
rows = []
for _, r in cases.iterrows():
    d = ast.literal_eval(prob.loc[r.problem_id, "detail"])
    row = r.to_dict()
    row.update(cognitive_level_name=BLOOM[int(r.cognitive_level)], knowledge_type_name=KTYPE[int(r.knowledge_type)],
               concept_list=list(r.concepts), item_stem=d.get("content", ""), item_options=d.get("option", {}),
               item_answer=d.get("answer", ""))
    row["action"] = rci.select_action(pd.Series(row), r.region_C)
    row["history_action"] = rci.select_action(pd.Series(row), r.history_region)
    rows.append(row)
mm = prompts.assign_mismatch([x["region_C"] for x in rows], {"HIGH": ["GUIDED_PRACTICE"], "MEDIUM": ["EXPLAIN"], "LOW": ["HINT"]})
for x, (mr, ma) in zip(rows, mm):
    x["mismatch_region"], x["mismatch_action"] = mr, ma
keep = ["case_id", "user_id", "problem_id", "course_id", "concept_list", "cognitive_level_name", "knowledge_type_name",
        "prior_overall_accuracy", "prior_concept_accuracy", "prior_same_level_accuracy", "recent_failure_ratio",
        "consecutive_failures", "p_t", "region_C", "action", "mismatch_region", "mismatch_action", "history_region", "history_action", "y",
        "item_stem", "item_options", "item_answer"]
cdf = pd.DataFrame(rows)[keep]
cdf.to_json(f"{out}/cases.jsonl", orient="records", lines=True, force_ascii=False)
print(cdf.groupby("region_C").agg(n=("case_id", "size"), p=("p_t", "mean"), recovered=("y", "mean")))
print(cdf.groupby(["region_C", "action"]).size()); print("distinct learners:", cdf.user_id.nunique())
print("history region == RCI region:", (cdf.history_region == cdf.region_C).mean())
