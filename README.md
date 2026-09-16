# Sales Call Copilot

**AI-powered real-time coaching for sales reps during live customer calls**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)

## Overview

Sales Call Copilot is a production-grade reference implementation of an AI-powered sales enablement system. It detects customer objections in real time, retrieves grounded sales enablement materials, generates contextual coaching suggestions via Claude, and scores call quality against a deterministic rubric to ensure accuracy and prevent hallucination.

The system is architecturally consistent with enterprise production deployments (multi-agent orchestration, model escalation, deterministic quality scoring) while remaining lightweight and runnable offline with zero API keys and zero cost.

## Problem Statement

Sales reps often struggle to respond effectively to objections during high-stakes customer calls. They need:
- **Instant detection** of the objection type (price, features, stakeholder, etc.)
- **Grounded talking points** backed by company enablement materials (not fabricated)
- **Real-time delivery** of suggestions without breaking conversation flow
- **Quality assurance** that suggestions are accurate, cited, and relevant

Without these, reps default to improvisation, leading to inconsistent messaging, missed opportunities, and prolonged sales cycles.

## Solution

Sales Call Copilot automates this workflow via a four-stage agent pipeline:

1. **Objection Detection** — Rule-based classifier identifies the objection category from transcript text
2. **Knowledge Retrieval** — TF-IDF semantic search retrieves the top-3 relevant enablement docs, reranked by objection type
3. **Coaching Generation** — Claude LLM (Haiku by default, Opus for complex objections) synthesizes a 2-3 sentence talking point grounded in retrieved material
4. **Quality Scoring** — Deterministic rubric engine rates the response across 5 dimensions (accuracy, grounding, citation, latency, hallucination-free) to ensure enterprise quality

All stages run offline with template mocks by default; transparently upgrade to real Claude API when an API key is available.

---

## Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/ANTHONY-CHINEDU-ECHEM/SALES_CALL_COPILOT.git
cd SALES_CALL_COPILOT

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline (Offline)

No API key required — runs end-to-end with deterministic mocks:

```bash
python pipeline.py
```

**Expected output:**
```json
{
  "detected_objection": "price_too_high",
  "retrieved_docs": [
    "Value Alignment Framework",
    "ROI Case Study: Tech Startup",
    "Competitive Win Rate Analysis"
  ],
  "coaching_suggestion": "Grounded in \"Value Alignment Framework\": Our pricing aligns with the value delivered...",
  "latency_seconds": 0.0234,
  "rubric_score": {
    "breakdown": {
      "objection_correctly_identified": 25,
      "grounded_response_used": 30,
      "citation_provided": 20,
      "response_latency_acceptable": 15,
      "no_hallucinated_claim_detected": 10
    },
    "total_score": 100,
    "max_score": 100,
    "pct": 100.0
  }
}
```

### Start the FastAPI Server

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

API will be available at `http://localhost:8000`

**Test the `/coach` endpoint:**

```bash
curl -X POST http://localhost:8000/coach \
  -H "Content-Type: application/json" \
  -d '{
    "transcript_snippet": "Prospect: honestly, this feels too expensive for what our budget allows.",
    "true_objection_type": "price_too_high"
  }'
```

### Upgrade to Real Claude API (Optional)

Set your Anthropic API key and the system automatically upgrades:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python pipeline.py  # Now uses real Claude Haiku/Opus
```

---

## Architecture

### Core Modules

#### `objection_detector.py`
**Rule-based multi-category classifier**

Detects 8 objection types via regex pattern matching:
- `price_too_high` — Budget or affordability concerns
- `no_budget` — No funds allocated
- `need_more_features` — Missing capability match
- `already_using_competitor` — Incumbent lock-in
- `not_a_priority_now` — Timing/urgency mismatch
- `need_stakeholder_buyin` — Internal approval required
- `security_compliance_concern` — Regulatory/risk blocks
- `implementation_time_concern` — Deployment speed/complexity

Easily extensible: add new objection types by appending regex patterns to the `PATTERNS` dict.

#### `retriever.py`
**Two-stage retrieve-then-rerank pipeline**

1. **Semantic Search** — TF-IDF cosine similarity over the knowledge base
2. **Intelligent Reranking** — Boosts docs matching the detected objection type by +0.3 similarity score

Returns top-k results (default: 3) with relevance scores and metadata. Swappable with embedding-based retrieval (e.g., Pinecone, Weaviate) for production.

#### `llm_client.py`
**Pluggable LLM abstraction**

- **With API Key:** Calls Anthropic Claude directly
  - Haiku (fast, cost-effective) for routine objections
  - Opus (more powerful) for complex, multi-faceted objections (flag via `complex_objection=True`)
- **Without API Key:** Returns deterministic template-based mock (zero cost, full end-to-end testing)

Auto-detects API key from environment; no code changes required.

#### `pipeline.py`
**Four-agent orchestration**

```
Transcript Snippet
       ↓
[Objection Detector] → detected objection type
       ↓
[Retriever] → top-3 grounded docs
       ↓
[LLM Client] → coaching suggestion
       ↓
[Rubric Scorer] → quality breakdown + total score
       ↓
Structured Result {objection, docs, suggestion, latency, score}
```

#### `rubric_agent.py`
**Deterministic quality assurance engine**

Scores each response across 5 weighted dimensions (100 points total):

| Item | Weight | Criteria |
|------|--------|----------|
| Objection Correctly Identified | 25 pts | Detected type matches ground truth |
| Grounded Response Used | 30 pts | Response references retrieved docs |
| Citation Provided | 20 pts | Response includes `source:` or `doc_id` |
| Response Latency Acceptable | 15 pts | Response generated in ≤3 seconds |
| No Hallucinated Claim Detected | 10 pts | No numerical claims outside retrieved docs |

Returns breakdown + total score + percentage. Removes score drift caused by raw LLM variability.

#### `server.py`
**FastAPI REST service**

Endpoints:
- `GET /health` — Service health check
- `POST /coach` — Real-time coaching suggestion

Request schema:
```python
{
  "transcript_snippet": str,        # Prospect's objection statement
  "true_objection_type": str | null # Optional: ground truth for rubric scoring
}
```

Response schema:
```python
{
  "detected_objection": str,
  "retrieved_docs": [str, ...],
  "coaching_suggestion": str,
  "latency_seconds": float
}
```

### Data Assets

- **`call_transcripts.csv`** — ~200 sample transcript snippets for evaluation
- **`knowledge_base.csv`** — Sales enablement materials (case studies, value props, ROI frameworks) tagged by objection type
- **`tfidf_index.joblib`** — Precomputed TF-IDF vectorizer and cosine similarity matrix (binary serialization for fast startup)

---

## Testing & Evaluation

### Unit & Integration Tests

```bash
pytest test_pipeline.py -v
```

Tests cover:
- Objection detection across all 8 categories
- Retrieval relevance and reranking behavior
- End-to-end pipeline latency
- Rubric scoring edge cases

### Retrieval Evaluation

Measure precision@k for semantic search:

```bash
python retrieval_precision.py
```

Outputs `retrieval_eval_report.json` with precision scores for each objection category.

### Rubric Alignment Study

Validate rubric scores against human coach ratings (simulated proxy):

```bash
python rubric_agreement.py
```

Outputs `rubric_agreement_report.json` with inter-rater correlation.

---

## Production Considerations

### Deployment

**Containerization** (example Dockerfile):
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Load Balancing:** Stateless FastAPI server; horizontally scalable.

**Environment Variables:**
- `ANTHROPIC_API_KEY` — (Optional) Anthropic API key for real Claude calls
- `LOG_LEVEL` — (Optional) Logging verbosity

### Scaling Improvements

Per `EXTENDING.md`, the following enhancements unlock production scale:

1. **Vector Database** — Replace TF-IDF with embedding-based retrieval (Weaviate, Pinecone, or Cloud Spanner) for semantic richness and scalable similarity search over millions of documents.

2. **Real-time Transcription** — Add Whisper (OpenAI) or similar streaming transcription service to consume live audio instead of pre-recorded snippets.

3. **JavaScript/LangChain.js Port** — Migrate core agents to Node.js via LangChain.js for parity with the author's production API architecture.

4. **Human Validation Loop** — Replace simulated rubric study with blinded human coach ratings on 50–100 real calls to measure true rubric-human alignment.

5. **Compliance & Privacy** — Integrate call-recording consent handling, customer PII redaction, and audit logging before any transcript reaches the LLM.

---

## Configuration & Extensibility

### Adding a New Objection Type

Edit `objection_detector.py`:

```python
PATTERNS = {
    # ... existing patterns ...
    "budget_freeze": [
        r"\bbudget\s+freeze\b",
        r"\bspending\s+pause\b",
        r"\bno\s+new\s+spend\b"
    ],
}
```

Then ensure corresponding enablement materials exist in `knowledge_base.csv` with `objection_type = "budget_freeze"`.

### Tuning Model Escalation

Edit `llm_client.py` to change the heuristic for when to escalate from Haiku to Opus:

```python
# Current: flag complex_objection manually in pipeline.py
model = "claude-opus-5" if complex_objection else "claude-haiku-4-5-20251001"

# Future: auto-detect based on query length, multi-part objections, etc.
```

### Custom Reranking Logic

Edit `retriever.py` to adjust the objection-type boost or add domain-specific signals:

```python
# Current: flat +0.3 boost
kb.loc[kb["objection_type"] == objection_type, "similarity"] += 0.3

# Future: contextual boost based on objection severity, rep seniority, etc.
```

---

## Performance & Latency

### End-to-End Latency (Mock Mode)
- **Objection Detection:** ~1–2 ms
- **Retrieval:** ~5–10 ms (TF-IDF + rerank)
- **LLM Coaching (Mock):** ~0–1 ms
- **Rubric Scoring:** ~1–2 ms
- **Total (P95):** ~10–20 ms

### With Real Claude API
- **Total (P95, Haiku):** ~800–1200 ms (network + inference)
- **Total (P95, Opus):** ~1500–2500 ms

Target: <3 seconds for 95th-percentile latency to avoid disrupting live conversation flow.

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Branch** from `main` for feature work (`feature/your-feature`) or bugfixes (`fix/your-issue`)
3. **Test** all changes: `pytest test_pipeline.py`
4. **Lint** with black/flake8 (optional but recommended)
5. **Pull Request** with clear description of changes and motivation

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## References & Further Reading

- **Briefing Materials:** See `Sales_Copilot_Portfolio.pdf` for product strategy and business context
- **Architecture Diagram:** See `SALES CALL COPILOT PROJECT TITAN.jpeg` for system design
- **UI Mockup:** See `live_call_mockup.html` for end-user experience
- **Extending:** See `EXTENDING.md` for production-scale roadmap and migration patterns

---

## Support & Contact

For questions, issues, or partnership inquiries:
- **GitHub Issues:** [Open an issue](https://github.com/ANTHONY-CHINEDU-ECHEM/SALES_CALL_COPILOT/issues)
- **Author:** [Anthony Chinedu Echem](https://github.com/ANTHONY-CHINEDU-ECHEM)

---

## Acknowledgments

This reference implementation mirrors enterprise production patterns (LangChain multi-agent orchestration, Claude Haiku-to-Sonnet escalation, deterministic rubric scoring) while remaining lightweight and teachable. It is designed as both a working prototype and a blueprint for sales teams building AI-augmented coaching systems.

---

**Built with ❤️ for sales teams everywhere.**
