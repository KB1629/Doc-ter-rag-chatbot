import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sentence_transformers import SentenceTransformer
from config.config import EMBEDDING_MODEL

_model = None

def get_embedding_model():
    """Load and cache the embedding model (downloads once, reuses after)."""
    global _model
    try:
        if _model is None:
            _model = SentenceTransformer(EMBEDDING_MODEL)
        return _model
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model: {e}")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert a list of text strings into embedding vectors."""
    try:
        model = get_embedding_model()
        return model.encode(texts, show_progress_bar=False).tolist()
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}")
