"""Builds a TF-IDF retrieval index over the enablement knowledge base —
the lightweight stand-in for a vector DB / knowledge-graph-backed store."""
import joblib
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def main():
    kb = pd.read_csv(ROOT / "data" / "knowledge_base.csv")
    corpus = (kb["title"] + " " + kb["content"]).tolist()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    matrix = vectorizer.fit_transform(corpus)
    joblib.dump({"vectorizer": vectorizer, "matrix": matrix, "kb": kb}, MODEL_DIR / "tfidf_index.joblib")
    print(f"Indexed {len(kb)} documents, vocabulary size {len(vectorizer.vocabulary_)}")


if __name__ == "__main__":
    main()
