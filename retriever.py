"""Dense-ish retrieval (TF-IDF cosine similarity) + a simple rerank step
that boosts documents matching the detected objection_type — mirrors a
two-stage retrieve-then-rerank RAG pipeline."""
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

MODEL_DIR = Path(__file__).parent.parent / "models"
_index = None


def _load():
    global _index
    if _index is None:
        _index = joblib.load(MODEL_DIR / "tfidf_index.joblib")
    return _index


def retrieve(query: str, objection_type: str = None, top_k: int = 3):
    idx = _load()
    q_vec = idx["vectorizer"].transform([query])
    sims = cosine_similarity(q_vec, idx["matrix"])[0]
    kb = idx["kb"].copy()
    kb["similarity"] = sims
    if objection_type:
        kb.loc[kb["objection_type"] == objection_type, "similarity"] += 0.3  # rerank boost
    top = kb.sort_values("similarity", ascending=False).head(top_k)
    return top[["doc_id", "title", "content", "objection_type", "doc_type", "similarity"]].to_dict("records")
