"""Session 1 - Retrieval: keyword search (BM25-lite), semantic search, and hybrid."""
import math
import re
from collections import Counter
from typing import Dict, List

from .embeddings import cosine_similarity, embed
from .vector_store import VectorStore

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "to", "in", "on", "for", "and",
    "or", "what", "who", "how", "do", "does", "i", "my", "me", "it", "be", "at", "by",
    "with", "can", "this", "that", "from", "as", "there", "any",
}


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9%.]+", text.lower()) if t not in STOPWORDS]


def semantic_search(store: VectorStore, query: str, k: int = 4) -> List[Dict]:
    """Meaning-based: compares embedding vectors with cosine similarity."""
    return store.search(query, k)


def keyword_search(store: VectorStore, query: str, k: int = 4,
                   k1: float = 1.5, b: float = 0.75) -> List[Dict]:
    """Word-matching: BM25 scoring. Great for exact terms, blind to synonyms."""
    docs = store.all_chunks()
    if not docs:
        return []
    tokenized = [tokenize(d["text"]) for d in docs]
    n = len(docs)
    avgdl = sum(len(t) for t in tokenized) / n or 1
    df = Counter()
    for toks in tokenized:
        df.update(set(toks))

    q_tokens = tokenize(query)
    scored = []
    for doc, toks in zip(docs, tokenized):
        tf = Counter(toks)
        score = 0.0
        for q in q_tokens:
            if q not in tf:
                continue
            idf = math.log(1 + (n - df[q] + 0.5) / (df[q] + 0.5))
            score += idf * tf[q] * (k1 + 1) / (tf[q] + k1 * (1 - b + b * len(toks) / avgdl))
        if score > 0:
            scored.append({**doc, "score": score})
    return sorted(scored, key=lambda d: d["score"], reverse=True)[:k]


def hybrid_search(store: VectorStore, query: str, k: int = 4, rrf_k: int = 60) -> List[Dict]:
    """Combine both rankings with Reciprocal Rank Fusion; report cosine as the score."""
    sem = semantic_search(store, query, k * 2)
    kw = keyword_search(store, query, k * 2)
    fused, by_id = {}, {}
    for results in (sem, kw):
        for rank, r in enumerate(results):
            fused[r["id"]] = fused.get(r["id"], 0.0) + 1.0 / (rrf_k + rank + 1)
            by_id.setdefault(r["id"], r)
    ranked = sorted(fused, key=fused.get, reverse=True)[:k]
    if not ranked:
        return []
    vecs = embed([query] + [by_id[i]["text"] for i in ranked])
    out = []
    for i, cid in enumerate(ranked):
        out.append({**by_id[cid], "score": cosine_similarity(vecs[0], vecs[i + 1])})
    return out


def search(store: VectorStore, query: str, k: int = 4, mode: str = "semantic") -> List[Dict]:
    if mode == "keyword":
        return keyword_search(store, query, k)
    if mode == "hybrid":
        return hybrid_search(store, query, k)
    return semantic_search(store, query, k)
