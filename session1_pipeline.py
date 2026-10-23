"""Session 1 deliverable: document processing + retrieval pipeline (no LLM needed).

Examples:
  python session1_pipeline.py
  python session1_pipeline.py --strategy sliding --size 300 --overlap 60
  python session1_pipeline.py --query "How late can I arrive for an exam?"
  python session1_pipeline.py --compare-strategies
"""
import argparse

from src import config
from src.chunking import STRATEGIES, chunk_text
from src.embeddings import cosine_similarity, embed
from src.loader import list_documents, load_file
from src.rag_pipeline import RAGAssistant
from src.retriever import keyword_search, semantic_search

DEFAULT_QUERIES = [
    "What is the minimum attendance to write exams?",
    "Can I carry my phone into the exam hall?",       # semantic: 'phone' vs 'mobile'
    "How much do I pay if I return a book late?",
    "Who is the vice chancellor?",                    # not in the docs!
]


def show(title, hits):
    print(f"\n  {title}")
    if not hits:
        print("    (no results)")
    for h in hits:
        snippet = h["text"].replace("\n", " ")[:110]
        print(f"    {h['score']:.3f} | {h['source']} p{h['page']} | {snippet}...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=str(config.SAMPLE_DOCS_DIR))
    ap.add_argument("--strategy", default="hybrid", choices=[s for s in STRATEGIES if s != "semantic"])
    ap.add_argument("--size", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--query", default=None)
    ap.add_argument("--compare-strategies", action="store_true")
    args = ap.parse_args()

    files = list_documents(args.docs)
    print(f"Loaded {len(files)} document(s): {[f.name for f in files]}")

    if args.compare_strategies:
        print("\nChunk counts per strategy (size/overlap as given):")
        text = "\n\n".join(p["text"] for f in files for p in load_file(f))
        for s in STRATEGIES:
            if s == "semantic":
                continue
            chunks = chunk_text(text, s, args.size, args.overlap)
            avg = sum(map(len, chunks)) / max(1, len(chunks))
            print(f"  {s:10s} -> {len(chunks):3d} chunks, avg {avg:5.0f} chars")
        return

    print("\nEmbedding demo (cosine similarity):")
    a, b, c = embed(["phones are banned in exams", "mobile devices not allowed in exam hall",
                     "the library opens at 8 AM"])
    print(f"  'phones banned' vs 'mobile devices not allowed' = {cosine_similarity(a, b):.3f}")
    print(f"  'phones banned' vs 'library opens at 8 AM'      = {cosine_similarity(a, c):.3f}")

    assistant = RAGAssistant("session1_demo")
    stats = assistant.ingest(files, args.strategy, args.size, args.overlap, reset=True)
    print(f"\nIndexed {stats['chunks']} chunks using '{args.strategy}' (size={args.size}).")

    for q in ([args.query] if args.query else DEFAULT_QUERIES):
        print(f"\n=== Q: {q}")
        show("Semantic (cosine similarity)", semantic_search(assistant.store, q, args.k))
        show("Keyword (BM25)", keyword_search(assistant.store, q, args.k))


if __name__ == "__main__":
    main()
