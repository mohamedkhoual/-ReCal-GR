"""Blind rating sheets (three passes, done in this order by each rater, independently).
Pass 1: five quality dimensions + leakage. Raters see the item, its correct answer, the target annotations and the
        learner context, but NOT the prompt, p_t, the region, the action or the condition.
Pass 2: realised assistance intensity (1-5) for every output, still without any action field.
Pass 3: action consistency (1-5), only for outputs generated under a mandated action.
python genkit/make_rating_sheets.py genkit/cases.jsonl genkit/outputs.jsonl genkit/sheets [--rate C3,C4,C4M,C4H,C5,C4X]"""
import sys, os, json, random, csv
args = [a for a in sys.argv[1:] if not a.startswith("--rate")]
rate = next((a.split("=", 1)[1] if "=" in a else None for a in sys.argv[1:] if a.startswith("--rate")), None)
RATE = (rate or "C3,C4,C4M,C4H,C5,C4X").split(",")
cases = {c["case_id"]: c for c in map(json.loads, open(args[0], encoding="utf8"))}
outs = [o for o in map(json.loads, open(args[1], encoding="utf8")) if o["condition"] in RATE]
od = args[2]; os.makedirs(od, exist_ok=True)
tr_file = os.path.join(os.path.dirname(args[1]) or ".", "items_en.jsonl")
ITEM_EN = {r["case_id"]: r["item_en"] for r in map(json.loads, open(tr_file, encoding="utf8"))} if os.path.exists(tr_file) else {}
random.Random(42).shuffle(outs)
for i, o in enumerate(outs): o["blind_id"] = f"B{i+1:03d}"
with open(f"{od}/KEY_do_not_share.csv", "w", newline="", encoding="utf8") as f:
    w = csv.writer(f); w.writerow(["blind_id", "case_id", "condition"])
    for o in outs: w.writerow([o["blind_id"], o["case_id"], o["condition"]])
f2 = lambda v: "unknown" if v is None or v != v else f"{v:.2f}"
for rater in ("rater1", "rater2"):
    order = outs[:] if rater == "rater1" else outs[::-1]          # different presentation order per rater
    with open(f"{od}/pass1_{rater}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["blind_id", "item", "options", "item_english_translation", "correct_answer", "concepts", "cognitive_level", "knowledge_type",
                    "learner_history", "remediation", "concept_alignment_1_5", "cognitive_alignment_1_5",
                    "knowledge_type_alignment_1_5", "pedagogical_appropriateness_1_5", "correctness_1_5", "leakage_0_1"])
        for o in order:
            c = cases[o["case_id"]]
            hist = (f"prior accuracy on this concept={f2(c['prior_concept_accuracy'])}; at this level={f2(c['prior_same_level_accuracy'])}; "
                    f"failures in last 3 answers={f2(c['recent_failure_ratio'])}; consecutive failures={c['consecutive_failures']}")
            w.writerow([o["blind_id"], c["item_stem"], json.dumps(c["item_options"], ensure_ascii=False), ITEM_EN.get(o["case_id"], ""), c["item_answer"],
                        ", ".join(c["concept_list"]), c["cognitive_level_name"], c["knowledge_type_name"], hist, o["output"]] + [""] * 6)
    with open(f"{od}/pass2_{rater}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["blind_id", "remediation", "intensity_1_5"])
        for o in order: w.writerow([o["blind_id"], o["output"], ""])
    with open(f"{od}/pass3_{rater}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["blind_id", "remediation", "mandated_action", "action_consistency_1_5"])
        for o in order:
            c = cases[o["case_id"]]
            a = {"C4": c["action"], "C5": c["action"], "C4X": c["action"], "C4M": c["mismatch_action"],
                 "C4H": c["history_action"]}.get(o["condition"])
            if a: w.writerow([o["blind_id"], o["output"], a, ""])
print("rated conditions:", RATE, "| outputs:", len(outs))

# Excel versions with 1-5 / 0-1 drop-down lists (raters fill these; merge_ratings reads .xlsx first)
try:
    import pandas as pd
    from openpyxl import load_workbook
    from openpyxl.worksheet.datavalidation import DataValidation
    from openpyxl.styles import Alignment
    for rater in ("rater1", "rater2"):
        for ps in ("pass1", "pass2", "pass3"):
            src = f"{od}/{ps}_{rater}.csv"; dst = src[:-4] + ".xlsx"
            pd.read_csv(src, encoding="utf-8-sig").to_excel(dst, index=False)
            wb = load_workbook(dst); ws = wb.active
            v15 = DataValidation(type="list", formula1='"1,2,3,4,5"', allow_blank=True)
            v01 = DataValidation(type="list", formula1='"0,1"', allow_blank=True)
            ws.add_data_validation(v15); ws.add_data_validation(v01)
            for col in ws.iter_cols(min_row=1, max_row=1):
                h = col[0].value; L = col[0].column_letter
                ws.column_dimensions[L].width = 60 if h in ("remediation", "item_english_translation", "item") else 18
                rng = f"{L}2:{L}{ws.max_row}"
                if h.endswith("_1_5"): v15.add(rng)
                if h.endswith("_0_1"): v01.add(rng)
            for row in ws.iter_rows(min_row=2):
                for cell in row: cell.alignment = Alignment(wrap_text=True, vertical="top")
            ws.freeze_panes = "B2"; wb.save(dst)
    print("Excel sheets written")
except ImportError:
    pass
