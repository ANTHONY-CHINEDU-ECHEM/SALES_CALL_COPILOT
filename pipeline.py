"""Orchestrates the full 4-agent pipeline: detector -> retrieval ->
coaching -> rubric scoring, on a sample call."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from agents.objection_detector import detect
from agents.llm_client import LLMClient
from agents.rubric_agent import score_call
from knowledge_base.retriever import retrieve


def run_pipeline(transcript_snippet: str, true_objection_type: str = None):
    t0 = time.time()
    detected = detect(transcript_snippet)
    docs = retrieve(transcript_snippet, objection_type=detected, top_k=3)
    client = LLMClient()
    suggestion = client.generate_coaching_suggestion(detected, docs, complex_objection=False)
    latency = time.time() - t0

    result = {"detected_objection": detected, "retrieved_docs": [d["title"] for d in docs],
              "coaching_suggestion": suggestion, "latency_seconds": round(latency, 4)}

    if true_objection_type:
        rubric = score_call(detected, true_objection_type, suggestion, docs, latency)
        result["rubric_score"] = rubric

    return result


if __name__ == "__main__":
    sample = "Prospect: honestly this feels too expensive for what our budget allows right now."
    result = run_pipeline(sample, true_objection_type="price_too_high")
    import json
    print(json.dumps(result, indent=2))
