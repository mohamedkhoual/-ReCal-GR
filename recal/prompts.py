"""Prompt builders for the generative conditions (cumulative information).
C1 concept | C2 +Bloom level | C3 +knowledge type, learner context, p_t | C4 +RCI region & action
C4M as C4 but region/action from a different region (shuffled decision)
C4H as C4 but region/action from a learner-history rule with the same region shares
C5 = C4 + self-check pass (one revision max) | C4X = C4 + matched neutral extra pass (control for C5)"""
from __future__ import annotations
import numpy as np

ACTION_SPEC = {
    "HINT": "Give ONE short hint (at most 2 sentences) that points to the key idea. Do not explain the full reasoning.",
    "EXPLAIN": "Explain the underlying concept concisely (3-5 sentences) without solving the item.",
    "EXAMPLE": "Give one worked example on a DIFFERENT but analogous item, then invite the learner to retry.",
    "GUIDED_PRACTICE": "Break the task into 3-4 guided steps, each ending with a question to the learner; do not give the final answer.",
    "BACKOFF": "Step back to the prerequisite idea needed for this concept: a short explanation plus one easier check question.",
}
SYSTEM = ("You are a tutor writing remediation for a learner who has just answered a multiple-choice item incorrectly. "
          "Never reveal or point to the correct option. Write in English.")

def _ctx(c):
    f = lambda v: "unknown" if v is None or v != v else f"{v:.2f}"
    return (f"Learner context: prior accuracy on this concept = {f(c['prior_concept_accuracy'])}, prior accuracy on this "
            f"concept at this cognitive level = {f(c['prior_same_level_accuracy'])}, failure ratio over the last 3 answers = "
            f"{f(c['recent_failure_ratio'])}, consecutive failures = {c['consecutive_failures']}.")

def build(cond: str, c: dict) -> str:
    opts = "; ".join(f"{k}. {v}" for k, v in (c.get("item_options") or {}).items())
    s = [f"Item (original language): {c['item_stem']}", f"Options: {opts}",
         f"Target concept(s): {', '.join(c['concept_list'])}."]
    if cond != "C1":
        s.append(f"Target cognitive level (revised Bloom): {c['cognitive_level_name']}.")
    if cond in ("C3", "C4", "C4M", "C4H", "C5"):
        s += [f"Knowledge type: {c['knowledge_type_name']}.", _ctx(c),
              f"Estimated probability that the learner recovers within the next 3 comparable items: {c['p_t']:.2f}."]
    if cond in ("C4", "C5"):
        r, a = c["region_C"], c["action"]
    elif cond == "C4M":
        r, a = c["mismatch_region"], c["mismatch_action"]
    elif cond == "C4H":
        r, a = c["history_region"], c["history_action"]
    else:
        r = a = None
    if r:
        s.append(f"Assistance intensity decided by the tutoring policy: {r}. Mandated remediation action: {a}. {ACTION_SPEC[a]}")
    s.append("Write the remediation now.")
    return "\n".join(s)

SELF_CHECK = ("Check the remediation below against six criteria: concept alignment, cognitive-level alignment, knowledge-type "
              "alignment, correctness, answer leakage (it must not reveal the correct option), and consistency with the "
              "mandated action ({action}: {spec}). If all six hold, reply exactly VALID. Otherwise reply with one revised "
              "remediation only.\n\nItem: {stem}\nRemediation:\n{text}")

EXTRA_PASS = ("Improve the clarity and formatting of the remediation below without changing its content, its length category "
              "or its pedagogical action. Return only the remediation.\n\nRemediation:\n{text}")


def assign_mismatch(regions, actions_by_region, seed=42):
    """C4M: each case receives the decision of a different region (balanced pseudo-random choice)."""
    rng = np.random.default_rng(seed); order = ["LOW", "MEDIUM", "HIGH"]; out = []
    for r in regions:
        alt = [x for x in order if x != r]; rr = alt[rng.integers(2)]
        out.append((rr, actions_by_region[rr][0]))
    return out
