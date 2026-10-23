"""Session 3 - Evaluate retrieval relevance and answer quality; tune settings.

  python evaluate.py                          # full answers on the sample handbook
  python evaluate.py --sweep                  # retrieval hit-rate for chunk sizes x top-k (no LLM cost)
  python evaluate.py --docs data/new_collection --questions data/eval_questions_technova.json
"""
import argparse
import csv
import json
from pathlib import Path

from src import config
from src.loader import list_documents
from src.rag_pipeline import RAGAssistant


def load_questions(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_eval(assistant, questions, top_k, min_score, mode):
    rows = []
    for q in questions:
        res = assistant.ask(q["question"], top_k=top_k, min_score=min_score, mode=mode)
        ans = res["answer"].lower()
        if q["type"] == "answerable":
            passed = res["grounded"] and any(k.lower() in ans for k in q["expected_keywords"])
        elif q["type"] == "unanswerable":
            passed = not res["grounded"]
        else:  # ambiguous -> needs human review
            passed = None
        rows.append({"type": q["type"], "question": q["question"],
                     "result": {True: "PASS", False: "FAIL", None: "REVIEW"}[passed],
                     "best_score": round(res["best_score"], 3),
                     "retrieval_s": round(res["retrieval_s"], 3),
                     "generation_s": round(res["generation_s"], 3),
                     "answer": res["answer"].replace("\n", " ")})
    return rows


def retrieval_hit_rate(assistant, questions, k, mode):
    """Fraction of answerable questions whose top-k passages contain an expected keyword."""
    qs = [q for q in questions if q["type"] == "answerable"]
    hits = 0
    for q in qs:
        text = " ".join(h["text"].lower() for h in assistant.retrieve(q["question"], k, mode))
        hits += any(kw.lower() in text for kw in q["expected_keywords"])
    return hits / max(1, len(qs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=str(config.SAMPLE_DOCS_DIR))
    ap.add_argument("--questions", default=str(config.DATA_DIR / "eval_questions.json"))
    ap.add_argument("--collection", default="eval_kb")
    ap.add_argument("--strategy", default="hybrid")
    ap.add_argument("--size", type=int, default=500)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--min-score", type=float, default=config.DEFAULT_MIN_SCORE)
    ap.add_argument("--mode", default="semantic", choices=["semantic", "keyword", "hybrid"])
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    files = list_documents(args.docs)
    questions = load_questions(args.questions)
    assistant = RAGAssistant(args.collection)

    if args.sweep:
        sizes, ks = [150, 300, 500, 1000], [1, 2, 4, 6]
        print(f"Retrieval hit-rate ({args.strategy}, {args.mode}) - rows: chunk size, cols: top-k")
        print("size  " + "  ".join(f"k={k:<3}" for k in ks))
        for size in sizes:
            assistant.ingest(files, args.strategy, size, size // 5, reset=True)
            rates = [retrieval_hit_rate(assistant, questions, k, args.mode) for k in ks]
            print(f"{size:<6}" + "  ".join(f"{r:>5.0%}" for r in rates))
        return

    assistant.ingest(files, args.strategy, args.size, args.size // 5, reset=True)
    rows = run_eval(assistant, questions, args.k, args.min_score, args.mode)

    for r in rows:
        print(f"[{r['result']:6}] ({r['type']:12}) {r['question']}\n"
              f"         score={r['best_score']}  {r['answer'][:140]}")
    scored = [r for r in rows if r["result"] != "REVIEW"]
    passed = sum(r["result"] == "PASS" for r in scored)
    print(f"\nScore: {passed}/{len(scored)} auto-graded passed "
          f"({sum(r['result'] == 'REVIEW' for r in rows)} ambiguous need manual review)")

    out = config.BASE_DIR / "results"
    out.mkdir(exist_ok=True)
    out_file = out / f"eval_{args.strategy}_{args.size}_k{args.k}_{args.mode}.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
