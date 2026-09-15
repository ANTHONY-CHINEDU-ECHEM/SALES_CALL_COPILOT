# Extending this reference implementation

This repo is architecturally consistent with the author's actual production
work (LangChain.js multi-agent orchestration, Claude Haiku-to-Sonnet
escalation, deterministic Rubric scoring) — the simplifications below are
about infrastructure weight, not architecture pattern.

1. **Real vector store / knowledge graph.** Replace
   `knowledge_base/retriever.py`'s TF-IDF index with a Cloud Spanner-backed
   embedding store + knowledge graph for structured reasoning across
   related objections/case studies.
2. **Real-time transcription.** Add a Whisper streaming service ahead of
   `agents/objection_detector.py` for live audio instead of text snippets.
3. **Node.js/LangChain.js production port.** Port `agents/pipeline.py` to
   LangChain.js agents for parity with the production Node.js API pattern
   described in the briefing.
4. **Real human-coach study.** Replace `eval/rubric_agreement.py`'s
   simulated proxy with an actual blinded study: have 2-3 sales coaches
   rate the same 50-100 real calls, then correlate against the Rubric
   agent's score.
5. **Consent & PII redaction.** Add call-recording consent-notice handling
   and PII redaction ahead of any real transcript reaching the LLM.
