"""Sessions 1-3 - The RAG pipeline: ingest -> retrieve -> generate (with fallback)."""
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional

from . import config
from .chunking import chunk_text
from .llm import generate, generate_with_usage
from .loader import load_file
from .prompts import FALLBACK, build_rag_prompt
from .retriever import search
from .vector_store import VectorStore


def _semantic_llm(prompt: str) -> str:
    return generate(prompt, max_tokens=4000, temperature=0.0)


class RAGAssistant:
    def __init__(self, collection: str = "knowledge_base"):
        self.store = VectorStore(collection)

    # ------------------------------------------------------------ ingestion
    def ingest(self, paths: List, strategy: str = "hybrid",
               chunk_size: int = config.DEFAULT_CHUNK_SIZE,
               overlap: int = config.DEFAULT_OVERLAP, reset: bool = False) -> Dict:
        if reset:
            self.store.reset()
        chunks = []
        for path in paths:
            for page in load_file(Path(path)):
                pieces = chunk_text(page["text"], strategy, chunk_size, overlap,
                                    llm_fn=_semantic_llm)
                for i, piece in enumerate(pieces):
                    key = f"{page['source']}|{page['page']}|{strategy}|{chunk_size}|{i}|{piece[:60]}"
                    chunks.append({
                        "id": hashlib.md5(key.encode()).hexdigest(),
                        "text": piece,
                        "metadata": {"source": page["source"], "page": page["page"],
                                     "chunk_index": i, "strategy": strategy},
                    })
        self.store.add_chunks(chunks)
        return {"files": len(paths), "chunks": len(chunks)}

    # ------------------------------------------------------------ retrieval
    def retrieve(self, question: str, top_k: int = config.DEFAULT_TOP_K,
                 mode: str = "semantic") -> List[Dict]:
        return search(self.store, question, top_k, mode)

    # ------------------------------------------------------------ generation
    def ask(self, question: str, top_k: int = config.DEFAULT_TOP_K,
            min_score: float = config.DEFAULT_MIN_SCORE, mode: str = "semantic",
            history: Optional[List[Dict]] = None) -> Dict:
        t0 = time.time()
        hits = self.retrieve(question, top_k, mode)
        t_retrieve = time.time() - t0

        # Fallback BEFORE calling the LLM: saves cost and prevents hallucination.
        # (Keyword scores are BM25, not cosine, so the threshold only applies otherwise.)
        best = max((h["score"] for h in hits), default=0.0)
        if not hits or (mode != "keyword" and best < min_score):
            return {"answer": FALLBACK, "sources": hits, "grounded": False,
                    "best_score": best, "retrieval_s": t_retrieve, "generation_s": 0.0,
                    "usage": {"input": 0, "output": 0, "source": "not called"}}

        system, user = build_rag_prompt(question, hits, history)
        t1 = time.time()
        answer, usage = generate_with_usage(user, system_prompt=system)
        t_generate = time.time() - t1

        if usage is None:
            try:
                import tiktoken

                tokenizer = tiktoken.get_encoding("cl100k_base")
                input_tokens = len(tokenizer.encode(f"{system}\n\n{user}"))
                output_tokens = len(tokenizer.encode(answer))
            except ImportError:
                input_tokens = max(1, round(len(f"{system}\n\n{user}") / 4))
                output_tokens = max(1, round(len(answer) / 4))
            usage = {
                "input": input_tokens,
                "output": output_tokens,
                "source": "estimated",
            }
        else:
            usage["source"] = "provider"

        grounded = FALLBACK.lower() not in answer.lower()
        return {"answer": answer, "sources": hits, "grounded": grounded,
                "best_score": best, "retrieval_s": t_retrieve, "generation_s": t_generate,
                "usage": usage}
