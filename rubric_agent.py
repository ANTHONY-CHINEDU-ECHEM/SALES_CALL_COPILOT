"""Deterministic Rubric scoring engine: converts call-level signals into a
machine-readable, itemized quality score — the same "Rubric engine that
sits above raw LLM outputs to remove score drift" pattern described in the
briefing's production platform (this one is a genuine 1:1 match, not a
simplified stand-in, since it's rule-based by design in both places).
"""

RUBRIC_ITEMS = [
    ("objection_correctly_identified", 25),
    ("grounded_response_used", 30),
    ("citation_provided", 20),
    ("response_latency_acceptable", 15),
    ("no_hallucinated_claim_detected", 10),
]


def score_call(objection_detected: str, objection_true: str, coaching_response: str,
               retrieved_docs: list, latency_seconds: float) -> dict:
    breakdown = {}
    breakdown["objection_correctly_identified"] = (RUBRIC_ITEMS[0][1]
        if objection_detected == objection_true else 0)
    breakdown["grounded_response_used"] = (RUBRIC_ITEMS[1][1]
        if retrieved_docs and any(d["title"] in coaching_response or d["content"][:30] in coaching_response for d in retrieved_docs) else 0)
    breakdown["citation_provided"] = RUBRIC_ITEMS[2][1] if ("source:" in coaching_response or "doc_id" in coaching_response) else 0
    breakdown["response_latency_acceptable"] = RUBRIC_ITEMS[3][1] if latency_seconds <= 3.0 else 0
    # heuristic hallucination check: flag if response mentions a number not present in any retrieved doc
    import re
    response_numbers = set(re.findall(r"\d+%|\$\d+", coaching_response))
    doc_numbers = set()
    for d in retrieved_docs:
        doc_numbers |= set(re.findall(r"\d+%|\$\d+", d["content"]))
    hallucinated = bool(response_numbers - doc_numbers)
    breakdown["no_hallucinated_claim_detected"] = 0 if hallucinated else RUBRIC_ITEMS[4][1]

    total = sum(breakdown.values())
    max_total = sum(w for _, w in RUBRIC_ITEMS)
    return {"breakdown": breakdown, "total_score": total, "max_score": max_total,
            "pct": round(total / max_total * 100, 1)}
