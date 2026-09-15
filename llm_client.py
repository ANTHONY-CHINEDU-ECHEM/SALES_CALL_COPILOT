"""Pluggable LLM client: uses the real Anthropic API if ANTHROPIC_API_KEY
is set in the environment, otherwise falls back to a deterministic
template-based mock — so the entire pipeline runs end-to-end with zero API
keys and zero cost, and upgrades transparently to a real model when a key
is available. Mirrors the "Claude Haiku with escalation to Sonnet on
complex objections" pattern from the briefing via the `complex` flag.
"""
import os


class LLMClient:
    def __init__(self):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.use_real_api = bool(self.api_key)
        if self.use_real_api:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)

    def generate_coaching_suggestion(self, objection_type: str, retrieved_docs: list, complex_objection: bool = False) -> str:
        context = "\n\n".join(f"[{d['doc_type']}] {d['title']}: {d['content']}" for d in retrieved_docs)
        if self.use_real_api:
            model = "claude-opus-5" if complex_objection else "claude-haiku-4-5-20251001"
            prompt = (
                f"A sales prospect just raised this objection type: {objection_type}.\n\n"
                f"Here is our team's relevant enablement material:\n{context}\n\n"
                f"In 2-3 sentences, give the rep a specific, grounded talking point for right now. "
                f"Cite which piece of enablement material you're drawing from."
            )
            resp = self._client.messages.create(
                model=model, max_tokens=250,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        else:
            return self._mock_generate(objection_type, retrieved_docs)

    def _mock_generate(self, objection_type: str, retrieved_docs: list) -> str:
        if not retrieved_docs:
            return f"No grounded material found for '{objection_type}' — escalate to a coach before responding."
        top = retrieved_docs[0]
        return (f"Grounded in \"{top['title']}\": {top['content']} "
                f"(source: {top['doc_type']}, doc_id={top['doc_id']})")


if __name__ == "__main__":
    client = LLMClient()
    print(f"Using real API: {client.use_real_api}")
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from knowledge_base.retriever import retrieve
    docs = retrieve("price is too high", objection_type="price_too_high")
    print(client.generate_coaching_suggestion("price_too_high", docs))
