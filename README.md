# Sales Call Copilot

**AI powered real time coaching for sales reps during live customer calls**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)

## Table of Contents

1. [Overview](#overview)
2. [Rationale and Business Value](#rationale-and-business-value)
3. [Problem Statement](#problem-statement)
4. [Solution](#solution)
5. [Quick Start](#quick-start)
6. [Architecture](#architecture)
7. [Testing and Evaluation](#testing-and-evaluation)
8. [Production Considerations](#production-considerations)
9. [Configuration and Extensibility](#configuration-and-extensibility)
10. [Performance and Latency](#performance-and-latency)
11. [Contributing](#contributing)
12. [License](#license)
13. [References and Further Reading](#references-and-further-reading)
14. [Support and Contact](#support-and-contact)
15. [Acknowledgments](#acknowledgments)

## Overview

Sales Call Copilot is a production grade reference implementation of an AI powered sales enablement system. It detects customer objections in real time, retrieves grounded sales enablement materials, generates contextual coaching suggestions through Claude, and scores call quality against a deterministic rubric to ensure accuracy and prevent hallucination.

The system is architecturally consistent with enterprise production deployments, including multi agent orchestration, model escalation, and deterministic quality scoring, while remaining lightweight and runnable offline with zero API keys and zero cost.

---

## Rationale and Business Value

### Why a Real Time Coaching Layer, Rather Than Post Call Training

Most sales organizations invest heavily in objection handling training, battle cards, and call review sessions, and almost all of that investment lands after the call that actually mattered. A rep who freezes on a pricing objection with a prospect on the line does not benefit from a battle card sitting in a wiki they cannot open mid conversation, or from a coaching note their manager writes three days later after listening to the recording. The commercial cost of that gap is concentrated exactly where it is most expensive: in the live moment when a deal is won, stalled, or lost. Sales Call Copilot exists to move a narrow, well defined slice of that coaching, objection handling, into the only moment it can actually change the outcome of the call.

This is a deliberately narrow scope, and that narrowness is itself part of the rationale. A general purpose sales coaching AI that tries to guide an entire conversation is both a harder product to build reliably and a much easier product to distrust, because a rep has no way to spot check dozens of open ended suggestions in real time. Objection detection and grounded response retrieval is a bounded problem: there is a finite, enumerable set of objection categories a sales team is likely to face, a finite set of enablement materials the company has already approved, and a clear, checkable question for each suggestion, namely whether it is actually grounded in something the company has published. Bounding the problem this tightly is what makes it possible to build a system a rep can trust enough to actually use live, rather than one more tool that gets switched off after the novelty wears off.

### Why Grounding and a Deterministic Rubric Are the Core Product, Not a Safety Add On

It would be straightforward to build a version of this system that simply asks a large language model to generate a helpful sounding response to any objection. It would also be actively dangerous to put in front of a live sales call, because a fluent, confident, and entirely fabricated claim about pricing, a competitor's weakness, or a compliance certification is exactly the kind of error that damages trust with a prospect and creates real legal or reputational exposure for the company. This is why the retrieval step and the rubric scoring step are treated as first class parts of the architecture rather than optional guardrails bolted on afterward.

The retrieval step ensures every coaching suggestion is generated with specific, approved company material already in context, so the model is asked to synthesize from real content rather than invent content from general knowledge. The rubric step then independently checks the output against that same material, specifically flagging any numerical claim that cannot be traced back to a retrieved document. The result is a system whose failure mode, when it does fail, is a low rubric score and a suggestion a rep can choose not to use, rather than a confidently delivered fabrication in front of a paying customer. This is the same design instinct that governs high stakes AI deployments generally: it is not enough for a system to usually be right, it has to be possible to independently verify each individual output, cheaply and automatically, every time.

### Why the System Runs Fully Offline by Default

A recurring failure mode for AI proof of concepts inside a sales organization is that the interesting version of the demo requires an API key, a cloud budget line, and a procurement conversation before anyone outside the original engineering team can actually try it. Sales Call Copilot is deliberately built so that the entire four stage pipeline, objection detection, retrieval, coaching generation, and rubric scoring, runs end to end with zero external API calls and zero cost, using deterministic template based mocks in place of a live model call. This is not merely a testing convenience. It means a sales enablement leader, a solutions engineer, or an engineering manager evaluating this pattern can clone the repository and see the complete, realistic shape of the system's behavior, including its data flow, its scoring logic, and its latency profile, without needing anyone's approval to spend money first. The moment a real Anthropic API key is set in the environment, the system transparently upgrades to live Claude calls, with no code changes required, which keeps the path from evaluation to a live pilot as short as possible.

### Why Two Models, Not One

The pipeline defaults to a fast, inexpensive model for routine objections and escalates to a more capable model only for objections explicitly flagged as complex or multi faceted. This tiered approach reflects a straightforward economic argument: the large majority of objections a sales team encounters are routine and well covered by existing enablement material, where a fast, inexpensive model produces a perfectly serviceable, grounded suggestion. Reserving a more expensive, more capable model for the harder, less common cases keeps the average cost per coached objection low without sacrificing quality on the objections that actually need deeper reasoning, which matters considerably at the call volumes a real sales organization generates.

### Where the Business Value Actually Shows Up

Three quality problems repeat across most manually run sales organizations: message inconsistency, where different reps improvise different, sometimes contradictory answers to the same objection; slow ramp time, where a new rep takes months to internalize the objection handling playbook that senior reps carry in their heads; and coaching that arrives too late to change the outcome of the specific call being reviewed. A real time, grounded coaching layer addresses all three simultaneously by surfacing the same, approved answer to the same objection for every rep, at the moment the objection is raised, rather than relying on memory, tenure, or a coaching session that happens after the deal is already decided. None of the figures in this document are audited outcomes from a production deployment; this repository is a reference implementation intended to demonstrate the mechanism, the architecture, and the quality guardrails that a real pilot would need before any of those benefits could be credibly measured and attributed.

---

## Problem Statement

Sales reps often struggle to respond effectively to objections during high stakes customer calls. They need:

- **Instant detection** of the objection type, such as price, features, or a stakeholder concern.
- **Grounded talking points**, backed by company enablement materials rather than fabricated on the spot.
- **Real time delivery** of suggestions, without breaking the flow of the conversation.
- **Quality assurance** that suggestions are accurate, cited, and relevant.

Without these, reps default to improvisation, which leads to inconsistent messaging, missed opportunities, and prolonged sales cycles.

---

## Solution

Sales Call Copilot automates this workflow through a four stage agent pipeline:

1. **Objection detection.** A rule based classifier identifies the objection category from the transcript text.
2. **Knowledge retrieval.** TF IDF semantic search retrieves the top three relevant enablement documents, reranked by objection type.
3. **Coaching generation.** A Claude model (Haiku by default, Opus for complex objections) synthesizes a two to three sentence talking point, grounded in the retrieved material.
4. **Quality scoring.** A deterministic rubric engine rates the response across five dimensions, including accuracy, grounding, citation, latency, and freedom from hallucination, to ensure enterprise quality.

All stages run offline with template mocks by default, and transparently upgrade to the real Claude API when an API key is available.

---

## Quick Start

### Prerequisites

- Python 3.8 or later
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

No API key is required; this runs end to end with deterministic mocks:

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

The API will be available at `http://localhost:8000`.

**Test the `/coach` endpoint:**

```bash
curl -X POST http://localhost:8000/coach \
  -H "Content-Type: application/json" \
  -d '{
    "transcript_snippet": "Prospect: honestly, this feels too expensive for what our budget allows.",
    "true_objection_type": "price_too_high"
  }'
```

### Upgrade to the Real Claude API (Optional)

Set your Anthropic API key, and the system automatically upgrades:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python pipeline.py  # now uses real Claude Haiku or Opus
```

---

## Architecture

### Core Modules

#### `objection_detector.py`

**A rule based, multi category classifier.**

Detects 8 objection types through regex pattern matching:

- `price_too_high`: budget or affordability concerns.
- `no_budget`: no funds allocated.
- `need_more_features`: a missing capability match.
- `already_using_competitor`: incumbent lock in.
- `not_a_priority_now`: a timing or urgency mismatch.
- `need_stakeholder_buyin`: internal approval required.
- `security_compliance_concern`: regulatory or risk blockers.
- `implementation_time_concern`: deployment speed or complexity concerns.

Easily extensible: new objection types can be added by appending regex patterns to the `PATTERNS` dictionary.

#### `retriever.py`

**A two stage retrieve then rerank pipeline.**

1. **Semantic search:** TF IDF cosine similarity, computed over the knowledge base.
2. **Intelligent reranking:** boosts documents that match the detected objection type by adding 0.3 to their similarity score.

Returns the top k results (3 by default), with relevance scores and metadata. This can be swapped for embedding based retrieval, for example Pinecone or Weaviate, in a production deployment.

#### `llm_client.py`

**A pluggable LLM abstraction.**

- **With an API key:** calls Anthropic's Claude directly. Haiku, which is fast and cost effective, handles routine objections; Opus, which is more powerful, handles complex, multi faceted objections, flagged through `complex_objection=True`.
- **Without an API key:** returns a deterministic, template based mock, at zero cost, supporting full end to end testing.

The system automatically detects an API key from the environment; no code changes are required.

#### `pipeline.py`

**Four agent orchestration.**

```
Transcript Snippet
       |
       v
[Objection Detector] -> detected objection type
       |
       v
[Retriever] -> top 3 grounded documents
       |
       v
[LLM Client] -> coaching suggestion
       |
       v
[Rubric Scorer] -> quality breakdown plus total score
       |
       v
Structured Result {objection, docs, suggestion, latency, score}
```

#### `rubric_agent.py`

**A deterministic quality assurance engine.**

Scores each response across five weighted dimensions, totaling 100 points:

| Item | Weight | Criteria |
|---|---|---|
| Objection correctly identified | 25 points | The detected type matches the ground truth |
| Grounded response used | 30 points | The response references the retrieved documents |
| Citation provided | 20 points | The response includes `source:` or `doc_id` |
| Response latency acceptable | 15 points | The response was generated in 3 seconds or less |
| No hallucinated claim detected | 10 points | No numerical claims appear outside the retrieved documents |

Returns the breakdown, the total score, and a percentage. This removes score drift caused by raw model variability.

#### `server.py`

**A FastAPI REST service.**

Endpoints:

- `GET /health`: a service health check.
- `POST /coach`: a real time coaching suggestion.

Request schema:

```python
{
  "transcript_snippet": str,        # the prospect's objection statement
  "true_objection_type": str | null # optional: ground truth, for rubric scoring
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

- **`call_transcripts.csv`**: approximately 200 sample transcript snippets, used for evaluation.
- **`knowledge_base.csv`**: sales enablement materials, including case studies, value propositions, and ROI frameworks, tagged by objection type.
- **`tfidf_index.joblib`**: a precomputed TF IDF vectorizer and cosine similarity matrix, serialized in binary form for a fast startup.

---

## Testing and Evaluation

### Unit and Integration Tests

```bash
pytest test_pipeline.py -v
```

Tests cover:

- Objection detection, across all 8 categories.
- Retrieval relevance and reranking behavior.
- End to end pipeline latency.
- Rubric scoring edge cases.

### Retrieval Evaluation

Measure precision at k for semantic search:

```bash
python retrieval_precision.py
```

This produces `retrieval_eval_report.json`, with precision scores for each objection category.

### Rubric Alignment Study

Validate rubric scores against human coach ratings (a simulated proxy):

```bash
python rubric_agreement.py
```

This produces `rubric_agreement_report.json`, with inter rater correlation.

---

## Production Considerations

### Deployment

**Containerization** (an example Dockerfile):

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Load balancing:** the FastAPI server is stateless, and is horizontally scalable.

**Environment variables:**

- `ANTHROPIC_API_KEY`: optional; an Anthropic API key, for real Claude calls.
- `LOG_LEVEL`: optional; controls logging verbosity.

### Scaling Improvements

Per `EXTENDING.md`, the following enhancements unlock production scale:

1. **A vector database.** Replace TF IDF with embedding based retrieval, such as Weaviate, Pinecone, or Cloud Spanner, for greater semantic richness and scalable similarity search over millions of documents.

2. **Real time transcription.** Add Whisper (from OpenAI) or a similar streaming transcription service, to consume live audio rather than pre recorded snippets.

3. **A JavaScript or LangChain.js port.** Migrate the core agents to Node.js through LangChain.js, for parity with the author's production API architecture.

4. **A human validation loop.** Replace the simulated rubric study with blinded human coach ratings on 50 to 100 real calls, to measure the true alignment between the rubric and human judgment.

5. **Compliance and privacy.** Integrate call recording consent handling, customer personal information redaction, and audit logging, before any transcript reaches the LLM.

---

## Configuration and Extensibility

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

Then ensure corresponding enablement materials exist in `knowledge_base.csv`, with `objection_type = "budget_freeze"`.

### Tuning Model Escalation

Edit `llm_client.py` to change the heuristic for when to escalate from Haiku to Opus:

```python
# Current: flag complex_objection manually in pipeline.py
model = "claude-opus-5" if complex_objection else "claude-haiku-4-5-20251001"

# Future: auto-detect based on query length, multi-part objections, etc.
```

### Custom Reranking Logic

Edit `retriever.py` to adjust the objection type boost, or to add domain specific signals:

```python
# Current: flat +0.3 boost
kb.loc[kb["objection_type"] == objection_type, "similarity"] += 0.3

# Future: contextual boost based on objection severity, rep seniority, etc.
```

---

## Performance and Latency

### End to End Latency (Mock Mode)

- **Objection detection:** approximately 1 to 2 milliseconds.
- **Retrieval:** approximately 5 to 10 milliseconds (TF IDF plus reranking).
- **LLM coaching (mock):** approximately 0 to 1 millisecond.
- **Rubric scoring:** approximately 1 to 2 milliseconds.
- **Total (95th percentile):** approximately 10 to 20 milliseconds.

### With the Real Claude API

- **Total (95th percentile, Haiku):** approximately 800 to 1,200 milliseconds (network plus inference).
- **Total (95th percentile, Opus):** approximately 1,500 to 2,500 milliseconds.

**Target:** a 95th percentile latency under 3 seconds, to avoid disrupting the flow of a live conversation.

---

## Contributing

Contributions are welcome. Please follow these guidelines:

1. **Fork** the repository.
2. **Branch** from `main` for feature work (`feature/your-feature`) or bug fixes (`fix/your-issue`).
3. **Test** all changes: `pytest test_pipeline.py`.
4. **Lint** with black or flake8 (optional, but recommended).
5. **Submit a pull request**, with a clear description of the changes and the motivation behind them.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## References and Further Reading

- **Briefing materials:** see `Sales_Copilot_Portfolio.pdf` for product strategy and business context.
- **Architecture diagram:** see `SALES CALL COPILOT PROJECT TITAN.jpeg` for the system design.
- **UI mockup:** see `live_call_mockup.html` for the end user experience.
- **Extending:** see `EXTENDING.md` for the production scale roadmap and migration patterns.

---

## Support and Contact

For questions, issues, or partnership inquiries:

- **GitHub Issues:** [open an issue](https://github.com/ANTHONY-CHINEDU-ECHEM/SALES_CALL_COPILOT/issues).
- **Author:** [Anthony Chinedu Echem](https://github.com/ANTHONY-CHINEDU-ECHEM).

---

## Acknowledgments

This reference implementation mirrors enterprise production patterns, including multi agent orchestration, tiered model escalation between Claude models, and deterministic rubric scoring, while remaining lightweight and teachable. It is designed as both a working prototype and a blueprint for sales teams building AI augmented coaching systems.

---

**Built with care for sales teams everywhere.**
