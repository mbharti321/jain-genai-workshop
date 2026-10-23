"""Session 1 - Embeddings: turn text into vectors so we can compare *meaning*."""
import numpy as np
from chromadb.utils import embedding_functions

from . import config

_ef = None


def get_embedding_function():
    """Local sentence-transformers model (free, no API key; downloads ~90 MB once)."""
    global _ef
    if _ef is None:
        _ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL
        )
    return _ef


def embed(texts):
    return np.array(get_embedding_function()(list(texts)))


def cosine_similarity(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0
