"""FastAPI service exposing the coaching pipeline as a live-call endpoint."""
import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from agents.pipeline import run_pipeline

app = FastAPI(title="Sales Call Copilot API", version="1.0.0")


class CoachRequest(BaseModel):
    transcript_snippet: str
    true_objection_type: Optional[str] = None


class CoachResponse(BaseModel):
    detected_objection: str
    retrieved_docs: List[str]
    coaching_suggestion: str
    latency_seconds: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/coach", response_model=CoachResponse)
def coach(x: CoachRequest):
    result = run_pipeline(x.transcript_snippet, x.true_objection_type)
    return CoachResponse(detected_objection=result["detected_objection"],
                          retrieved_docs=result["retrieved_docs"],
                          coaching_suggestion=result["coaching_suggestion"],
                          latency_seconds=result["latency_seconds"])
