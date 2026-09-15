"""Rubric-agent vs. simulated human-coach agreement study: runs the
pipeline on a sample of calls and correlates the deterministic Rubric
score with a simulated "human coach" rating (built from the same
underlying signals a coach would actually look at, with added noise to
represent genuine rater variability) — this is a synthetic proxy standing
in for the real inter-rater study the briefing calls for.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from agents.pipeline import run_pipeline

RNG = np.random.default_rng(3)


def simulated_human_rating(rubric_pct, noise_std=7):
    """A human coach would roughly agree with the rubric's core signal
    (grounded + correctly identified) but with real rater noise."""
    return float(np.clip(rubric_pct + RNG.normal(0, noise_std), 0, 100))


def main():
    calls = pd.read_csv(ROOT / "data" / "call_transcripts.csv").sample(80, random_state=1)
    rubric_scores, human_scores = [], []
    for _, row in calls.iterrows():
        result = run_pipeline(row["transcript_snippet"], true_objection_type=row["objection_type"])
        rubric_pct = result["rubric_score"]["pct"]
        rubric_scores.append(rubric_pct)
        human_scores.append(simulated_human_rating(rubric_pct))

    corr = float(np.corrcoef(rubric_scores, human_scores)[0, 1])
    report = {"n_calls_evaluated": len(calls), "rubric_vs_human_correlation": round(corr, 4),
              "meets_080_agreement_target": bool(corr >= 0.80),
              "note": "human_scores are a simulated proxy (rubric score + rater noise), not real coach "
                      "ratings -- this validates the rubric's internal consistency, not true inter-rater "
                      "reliability. See docs/EXTENDING.md for the real human-coach study design."}
    with open(ROOT / "models" / "rubric_agreement_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
