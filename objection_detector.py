"""Rule-based objection-type classifier: keyword/pattern matching over a
live transcript snippet. Stands in for a fine-tuned classifier — same
output contract (an objection_type string), swappable later.
"""
import re

PATTERNS = {
    "price_too_high": [r"\btoo expensive\b", r"\bprice\b.*\bhigh\b", r"\bcan'?t afford\b", r"\bbudget\b.*\btight\b"],
    "no_budget": [r"\bno budget\b", r"\bnot budgeted\b", r"\bdon'?t have (the )?budget\b"],
    "need_more_features": [r"\bdoesn'?t have\b", r"\bmissing (a )?feature\b", r"\bneed(s)? .* capability\b"],
    "already_using_competitor": [r"\balready use\b", r"\bcurrently (using|with)\b", r"\bcompetitor\b", r"\bhappy with our current\b"],
    "not_a_priority_now": [r"\bnot (a )?priority\b", r"\bmaybe next quarter\b", r"\bcircle back\b", r"\bnot right now\b"],
    "need_stakeholder_buyin": [r"\bneed to check with\b", r"\bboss\b", r"\bteam needs to agree\b", r"\bstakeholders?\b"],
    "security_compliance_concern": [r"\bsecurity\b", r"\bcompliance\b", r"\bsoc ?2\b", r"\baudit\b"],
    "implementation_time_concern": [r"\btoo long\b.*\b(implement|roll ?out|set ?up)\b", r"\bhow long.*implement\b", r"\bmigration\b.*\b(slow|risky)\b"],
}


def detect(transcript_snippet: str) -> str:
    text = transcript_snippet.lower()
    for obj_type, patterns in PATTERNS.items():
        if any(re.search(p, text) for p in patterns):
            return obj_type
    return "unclassified"


if __name__ == "__main__":
    print(detect("The prospect said it's too expensive for their current budget."))
    print(detect("They mentioned they already use a competitor and are happy with it."))
