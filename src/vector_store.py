"""Session 1 - ChromaDB vector store (persistent, cosine distance)."""
from typing import Dict, List

import chromadb

from . import config
from .embeddings import get_embedding_function


class VectorStore:
    def __init__(self, name: str = "knowledge_base"):
        self.name = name
        self.client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.collection = self._open()

    def _open(self):
        return self.client.get_or_create_collection(
            name=self.name,
            embedding_function=get_embedding_function(),
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[Dict]) -> None:
        """chunks: [{'id', 'text', 'metadata'}]. Upsert = safe to re-run."""
        if not chunks:
            return
        for i in range(0, len(chunks), 128):
            batch = chunks[i:i + 128]
            self.collection.upsert(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
            )

    def search(self, query: str, k: int = 4) -> List[Dict]:
        n = self.count()
        if n == 0:
            return []
        res = self.collection.query(query_texts=[query], n_results=min(k, n))
        hits = []
        for i, cid in enumerate(res["ids"][0]):
            meta = res["metadatas"][0][i]
            hits.append({
                "id": cid,
                "text": res["documents"][0][i],
                "source": meta.get("source", "?"),
                "page": meta.get("page", 1),
                "score": 1.0 - res["distances"][0][i],  # cosine similarity
            })
        return hits

    def all_chunks(self) -> List[Dict]:
        res = self.collection.get(include=["documents", "metadatas"])
        return [
            {"id": res["ids"][i], "text": res["documents"][i],
             "source": res["metadatas"][i].get("source", "?"),
             "page": res["metadatas"][i].get("page", 1)}
            for i in range(len(res["ids"]))
        ]

    def count(self) -> int:
        return self.collection.count()

    def sources(self) -> List[str]:
        return sorted({c["source"] for c in self.all_chunks()})

    def reset(self) -> None:
        self.client.delete_collection(self.name)
        self.collection = self._open()
