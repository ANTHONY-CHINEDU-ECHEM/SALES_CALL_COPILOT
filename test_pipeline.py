import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

INDEX_PATH = ROOT / "models" / "tfidf_index.joblib"


def test_data_generated():
    kb = pd.read_csv(ROOT / "data" / "knowledge_base.csv")
    calls = pd.read_csv(ROOT / "data" / "call_transcripts.csv")
    assert len(kb) >= 250
    assert len(calls) >= 500


def test_objection_detector_matches_known_phrases():
    from agents.objection_detector import detect
    assert detect("it's too expensive for our budget") == "price_too_high"
    assert detect("we already use a competitor and are happy") == "already_using_competitor"


@pytest.mark.skipif(not INDEX_PATH.exists(), reason="Run knowledge_base/ingestion.py first")
def test_retrieval_returns_relevant_docs():
    from knowledge_base.retriever import retrieve
    results = retrieve("price is too high", objection_type="price_too_high", top_k=3)
    assert len(results) == 3
    assert any(r["objection_type"] == "price_too_high" for r in results)


@pytest.mark.skipif(not INDEX_PATH.exists(), reason="Run knowledge_base/ingestion.py first")
def test_full_pipeline_runs_and_cites_source():
    from agents.pipeline import run_pipeline
    result = run_pipeline("This feels too expensive right now.", true_objection_type="price_too_high")
    assert result["detected_objection"] == "price_too_high"
    assert "source:" in result["coaching_suggestion"] or "doc_id" in result["coaching_suggestion"]
    assert result["latency_seconds"] < 3.0
    assert result["rubric_score"]["pct"] > 0


@pytest.mark.skipif(not INDEX_PATH.exists(), reason="Run knowledge_base/ingestion.py first")
def test_api_coach_endpoint():
    from fastapi.testclient import TestClient
    from api.server import app
    client = TestClient(app)
    r = client.post("/coach", json={"transcript_snippet": "Their budget is too tight for this.",
                                     "true_objection_type": "price_too_high"})
    assert r.status_code == 200
    assert r.json()["latency_seconds"] < 3.0
