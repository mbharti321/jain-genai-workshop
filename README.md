# GenAI Knowledge Assistant (RAG) - Workshop Starter

Build a document Q&A assistant: **ingest → chunk → embed → retrieve → generate → cite sources**.

## Setup (5 minutes)

```bash
py -m venv .venv
source .venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt # py -m pip install -r requirements.txt
cp .env.example .env              # then add your API key
```

The first run downloads the embedding model (~90 MB), so internet access is needed once.

## Project structure

```
app.py                      Streamlit UI (Session 2-3)
session1_pipeline.py        Chunk + embed + search from the terminal (Session 1)
session2_prompt_types.py    Same task, 5 prompt styles (Session 2)
evaluate.py                 Test questions, scoring, parameter sweep (Session 3)
src/
  config.py                 Settings from .env
  loader.py                 Load PDF / TXT / MD
  chunking.py               fixed, sliding, sentence, paragraph, hybrid, semantic (LLM)
  embeddings.py             sentence-transformers + cosine similarity
  vector_store.py           ChromaDB wrapper
  retriever.py              semantic, keyword (BM25), hybrid (RRF)
  llm.py                    Anthropic / OpenAI-compatible API calls
  prompts.py                System prompt, context builder, prompt-type demos
  rag_pipeline.py           RAGAssistant: ingest(), retrieve(), ask() + fallback
data/
  sample_docs/              Fictional university handbook
  new_collection/           Fictional company policy (Session 3 task)
  eval_questions*.json      Answerable / ambiguous / unanswerable questions
```

## Session 1 - Retrieval pipeline (no API key needed)

```bash
python session1_pipeline.py                              # embeddings demo + semantic vs keyword
python session1_pipeline.py --strategy sliding --size 300 --overlap 60
python session1_pipeline.py --compare-strategies
python session1_pipeline.py --query "Can I wear a smartwatch in the exam?"
```
Try: a query with exact words vs a paraphrase, and see where keyword search fails and semantic wins.

## Session 2 - Knowledge assistant

```bash
python session2_prompt_types.py
streamlit run app.py
```
In the app: **Load samples** → ask questions → open the **[1] [2]** source expanders.
Test covered questions ("How many books can I borrow?") and uncovered ones ("Who is the VC?").

The standalone teaching UI is also available with `streamlit run rag_chatbot_app.py`; it displays token usage and estimated API cost for each answer and for the conversation. In its provider selector, **OpenAI** uses `OPENAI_API_KEY` and `OPENAI_MODEL`, while **Gemini** uses `GEMINI_API_KEY` and `GEMINI_CHAT_MODEL`.

## Session 3 - Improve and evaluate

```bash
python evaluate.py                    # PASS/FAIL per question, CSV saved in results/
python evaluate.py --sweep            # chunk size x top-k hit-rate grid (free, no LLM)
python evaluate.py --mode hybrid
python evaluate.py --docs data/new_collection --questions data/eval_questions_technova.json
```
In the app, upload `data/new_collection/technova_hr_policy.txt` and tune chunk size, top-k, search mode, and the fallback threshold slider.

## Where the key ideas live

| Concept | File |
|---|---|
| Chunking strategies | `src/chunking.py` |
| Why embeddings / cosine similarity | `src/embeddings.py` |
| ChromaDB | `src/vector_store.py` |
| Keyword vs semantic retrieval | `src/retriever.py` |
| Prompt + retrieved data | `src/prompts.py` |
| Fallback / hallucination guard | `src/rag_pipeline.py` (`ask`) |
| Latency, token usage, and estimated API cost | shown under each answer and in the UI sidebar |

Both Streamlit UIs use provider-reported input/output token usage when available and label tokenizer-based counts as estimates otherwise. Cost is an estimate based on the model's configured USD-per-million-token rates; verify current pricing with your provider. For custom or compatible models, set both `LLM_INPUT_PRICE_PER_MILLION_USD` and `LLM_OUTPUT_PRICE_PER_MILLION_USD` in `.env`.

## Using other LLM providers

For the standalone teaching UI, configure OpenAI and Gemini independently using the provider-specific keys and model variables shown in `.env.example`. The separate `app.py` pipeline uses `LLM_PROVIDER` from `.env`; for an OpenAI-compatible service, set `LLM_PROVIDER=openai` and configure `OPENAI_BASE_URL` as needed.

## Optional extensions

- **Multiple documents**: already supported; upload several files.
- **Conversation history**: tick the checkbox in the sidebar; extend with query rewriting.
- **FastAPI**: wrap `RAGAssistant.ask()` in a `POST /ask` endpoint and call it from the UI.
- **Re-ranking**, **PDF tables/OCR**, **streaming responses**, **answer-quality scoring with an LLM judge**.

## Troubleshooting

- `ANTHROPIC_API_KEY is missing` → check `.env` exists and has the key.
- Changed chunking but results look the same → keep "Clear collection before indexing" ticked.
- Scanned PDFs return no text → they need OCR (e.g., pytesseract).
