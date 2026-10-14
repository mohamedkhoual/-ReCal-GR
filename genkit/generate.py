"""Generate remediation for C1-C5, C4M, C4H and C4X with Qwen2.5-3B-Instruct (run on a GPU, e.g. Colab).
pip install transformers accelerate torch ; python genkit/generate.py genkit/cases.jsonl genkit/outputs.jsonl"""
import sys, os, json; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from recal.prompts import build, SYSTEM, SELF_CHECK, ACTION_SPEC, EXTRA_PASS
MODEL = "Qwen/Qwen2.5-3B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16   # T4 -> float16
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="auto")
except TypeError:   # older transformers
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE, device_map="auto")
def gen(user):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**enc, do_sample=False, max_new_tokens=192, repetition_penalty=1.05)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

cases = [json.loads(l) for l in open(sys.argv[1], encoding="utf8")]
# English translation of each item, for the human raters only (never shown to the generator)
TR = ("Translate this multiple-choice item into English. Keep the option letters. Do not answer it.\n\n"
      "Item: {stem}\nOptions: {opts}")
tr_path = os.path.join(os.path.dirname(sys.argv[2]) or ".", "items_en.jsonl")
tr_done = {json.loads(l)["case_id"] for l in open(tr_path, encoding="utf8")} if os.path.exists(tr_path) else set()
with open(tr_path, "a", encoding="utf8") as ft:
    for c in cases:
        if c["case_id"] in tr_done: continue
        opts = "; ".join(f"{k}. {v}" for k, v in (c.get("item_options") or {}).items())
        ft.write(json.dumps({"case_id": c["case_id"], "item_en": gen(TR.format(stem=c["item_stem"], opts=opts))}, ensure_ascii=False) + "\n")
print("item translations written to", tr_path, flush=True)
done = set()
if os.path.exists(sys.argv[2]):      # resume after a Colab disconnection: keep finished cases (C4X is written last)
    done = {o["case_id"] for o in map(json.loads, open(sys.argv[2], encoding="utf8")) if o["condition"] == "C4X"}
    keep = [l for l in open(sys.argv[2], encoding="utf8") if json.loads(l)["case_id"] in done]
    open(sys.argv[2], "w", encoding="utf8").writelines(keep)
    print(f"resuming: {len(done)} cases already done", flush=True)
with open(sys.argv[2], "a", encoding="utf8") as f:
    for c in cases:
        if c["case_id"] in done: continue
        res = {}
        for cond in ["C1", "C2", "C3", "C4", "C4M", "C4H"]:
            p = build(cond, c); res[cond] = gen(p)
            f.write(json.dumps({"case_id": c["case_id"], "condition": cond, "prompt": p, "output": res[cond]}, ensure_ascii=False) + "\n")
        # C5: self-check of the C4 draft, at most one revision, then re-check
        txt, revised = res["C4"], 0
        def is_revision(r):   # a usable revision: not a bare verdict ("VALID", "INVALID", ...) and at least 8 words
            head = r.strip().upper()
            return not head.startswith(("VALID", "INVALID")) and len(r.split()) >= 8
        chk = gen(SELF_CHECK.format(action=c["action"], spec=ACTION_SPEC[c["action"]], stem=c["item_stem"], text=txt))
        if is_revision(chk):
            txt, revised = chk, 1
        f.write(json.dumps({"case_id": c["case_id"], "condition": "C5", "prompt": "self-check of C4", "output": txt,
                            "revised": revised}, ensure_ascii=False) + "\n"); f.flush()
        # C4X: matched extra generation pass without any checking instruction (control for C5)
        f.write(json.dumps({"case_id": c["case_id"], "condition": "C4X", "prompt": "neutral extra pass of C4",
                            "output": gen(EXTRA_PASS.format(text=res["C4"]))}, ensure_ascii=False) + "\n"); f.flush()
        print(c["case_id"], "done", flush=True)
