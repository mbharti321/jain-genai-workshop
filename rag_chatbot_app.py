"""End-to-end grounded RAG chatbot UI.

Run from the repository root with:
    streamlit run rag_chatbot_app.py

The application intentionally mirrors the teaching notebook:
document loading -> hybrid chunking -> MiniLM embeddings -> ChromaDB ->
cosine retrieval -> grounded LLM response -> local conversation history.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

from src import config

DOCS_DIR = BASE_DIR / "data" / "sample_docs"
CHROMA_DIR = BASE_DIR / "chroma_db_teaching"
HISTORY_FILE = BASE_DIR / "conversation_history_ui.json"
FALLBACK = "I couldn't find this information in the provided documents."
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def approximate_tokens(text: str) -> int:
    """Use a rough estimate when the selected provider tokenizer is unavailable."""
    return max(1, round(len(text) / 4))


def count_tokens(text: str) -> int:
    """Count tokens with tiktoken when installed, otherwise show a useful estimate."""
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except ImportError:
        return approximate_tokens(text)


def load_document(path: Path) -> list[dict[str, Any]]:
    """Extract supported files into text records with citation metadata."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        records = []
        for page_number, page in enumerate(PdfReader(str(path)).pages, 1):
            text = (page.extract_text() or "").strip()
            if text:
                records.append({"text": text, "source": path.name, "page": page_number})
        return records
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        return [{"text": text, "source": path.name, "page": 1}] if text else []
    if suffix == ".docx":
        from docx import Document

        document = Document(str(path))
        parts = [p.text.strip() for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        text = "\n".join(parts).strip()
        return [{"text": text, "source": path.name, "page": 1}] if text else []
    return []


def split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"[ \t]+", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]


def paragraph_chunks(text: str, size: int = 500) -> list[str]:
    """Keep paragraphs together where possible, then enforce a size limit."""
    result: list[str] = []
    current = ""
    for paragraph in [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]:
        candidate = f"{current}\n\n{paragraph}".strip()
        if current and len(candidate) > size:
            result.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        result.append(current)
    return result


def make_hybrid_chunks(records: list[dict[str, Any]], size: int = 500) -> list[dict[str, Any]]:
    """Split long paragraphs at sentence boundaries while preserving provenance."""
    chunks = []
    for record in records:
        units: list[str] = []
        for paragraph in record["text"].split("\n\n"):
            units.extend(split_sentences(paragraph) if len(paragraph) > size else [paragraph.strip()])
        pieces = paragraph_chunks("\n\n".join(unit for unit in units if unit), size)
        for index, text in enumerate(pieces):
            stable_key = f"{record['source']}|{record['page']}|hybrid|{index}|{text}"
            item = dict(record)
            item.update(
                text=text,
                chunk_index=index,
                strategy="hybrid",
                id=hashlib.sha1(stable_key.encode()).hexdigest(),
            )
            chunks.append(item)
    return chunks


@st.cache_resource(show_spinner="Loading the local embedding model...")
def get_embedder():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    return model


def build_collection(chunks: list[dict[str, Any]]):
    """Create a persistent Chroma collection using cosine distance."""
    import chromadb

    embedder = get_embedder()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection("teaching_rag_ui")
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name="teaching_rag_ui", metadata={"hnsw:space": "cosine"}
    )
    vectors = embedder.encode([chunk["text"] for chunk in chunks], normalize_embeddings=True)
    metadata = [
        {key: chunk[key] for key in ("source", "page", "chunk_index", "strategy")}
        for chunk in chunks
    ]
    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=metadata,
        embeddings=np.asarray(vectors).tolist(),
    )
    return collection


@st.cache_resource
def get_collection():
    files = sorted(p for p in DOCS_DIR.glob("*") if p.suffix.lower() in {".pdf", ".docx", ".txt", ".md"})
    records = [record for path in files for record in load_document(path)]
    chunks = make_hybrid_chunks(records)
    return build_collection(chunks), len(files), len(records), len(chunks)


def retrieve(question: str, top_k: int, min_score: float) -> list[dict[str, Any]]:
    """Use hybrid retrieval so paraphrases and exact campus names both work."""
    collection, *_ = get_collection()
    query_vector = get_embedder().encode([question], normalize_embeddings=True).tolist()
    result = collection.query(
        query_embeddings=query_vector,
        # Retrieve a wider candidate set before applying the combined score.
        n_results=min(max(top_k * 3, 12), collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    stop_words = {
        "a", "an", "the", "and", "are", "can", "give", "has", "have", "how",
        "is", "of", "on", "or", "their", "what", "where", "which", "with",
    }
    wanted = {
        token for token in re.findall(r"[a-z0-9%]+", question.lower())
        if token not in stop_words and len(token) > 2
    }
    candidates = []
    for text, metadata, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        semantic_score = 1 - float(distance)
        document_tokens = set(re.findall(r"[a-z0-9%]+", text.lower()))
        keyword_matches = wanted & document_tokens
        keyword_score = len(keyword_matches) / max(1, len(wanted))
        # A keyword match can rescue an exact term such as a campus name,
        # while semantic similarity handles paraphrases and related wording.
        combined_score = max(semantic_score, keyword_score)
        if combined_score >= min_score or keyword_matches:
            candidates.append({
                "text": text,
                **metadata,
                "score": combined_score,
                "semantic_score": semantic_score,
                "keyword_matches": sorted(keyword_matches),
            })
    return sorted(candidates, key=lambda item: item["score"], reverse=True)[:top_k]


def load_history() -> list[dict[str, str]]:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    return []


def save_history(history: list[dict[str, str]]) -> None:
    HISTORY_FILE.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def compact_history(history: list[dict[str, str]], keep_last_messages: int = 4) -> list[dict[str, str]]:
    """Keep a short deterministic summary plus the newest messages."""
    if len(history) <= keep_last_messages:
        return history
    older = history[:-keep_last_messages]
    user_topics = [message["content"] for message in older if message["role"] == "user"]
    summary = "Earlier conversation topics: " + " | ".join(user_topics)
    return [{"role": "system", "content": summary}] + history[-keep_last_messages:]


def build_prompt(question: str, hits: list[dict[str, Any]], history: list[dict[str, str]]) -> str:
    history_text = "\n".join(f"{item['role'].upper()}: {item['content']}" for item in history)
    context = "\n\n".join(
        f"[{index}] {hit['source']} page {hit['page']}\n{hit['text']}"
        for index, hit in enumerate(hits, 1)
    )
    return (
        "CONVERSATION HISTORY:\n"
        f"{history_text}\n\n"
        "DOCUMENT CONTEXT:\n"
        f"{context}\n\n"
        f"QUESTION: {question}\n"
        "Answer only from DOCUMENT CONTEXT. If absent, use the exact fallback. Cite sources as [1], [2]."
    )


def call_llm(
    prompt: str,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict[str, int] | None]:
    """Call the selected provider with explicit, student-visible controls."""
    if provider == "Local (Ollama)":
        import ollama

        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        input_tokens = response.get("prompt_eval_count")
        output_tokens = response.get("eval_count")
        usage = (
            {"input": input_tokens, "output": output_tokens}
            if input_tokens is not None and output_tokens is not None
            else None
        )
        return response["message"]["content"], usage
    if provider == "OpenAI":
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")

        response = OpenAI(api_key=api_key).chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = None
        if response.usage is not None:
            usage = {
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens,
            }
        return response.choices[0].message.content or "", usage
    if provider == "Gemini":
        import google.generativeai as genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file.")
        genai.configure(api_key=api_key)
        response = genai.GenerativeModel(model).generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
        )
        usage_metadata = getattr(response, "usage_metadata", None)
        usage = None
        if usage_metadata is not None:
            input_tokens = usage_metadata.prompt_token_count
            output_tokens = usage_metadata.candidates_token_count
            if input_tokens is not None and output_tokens is not None:
                usage = {"input": input_tokens, "output": output_tokens}
        return response.text, usage
    raise ValueError("Unknown provider")


def token_usage(prompt: str, answer: str, reported_usage: dict[str, int] | None) -> dict[str, Any]:
    """Use provider counts when available; otherwise label tokenizer counts as estimates."""
    if reported_usage is not None:
        return {**reported_usage, "source": "provider"}
    return {
        "input": count_tokens(prompt),
        "output": count_tokens(answer),
        "source": "estimated",
    }


def estimate_cost(
    usage: dict[str, Any],
    provider: str,
    pricing: dict[str, float] | None,
) -> float | None:
    """Estimate API cost from configured per-million-token rates."""
    if usage["source"] == "not called" or provider == "Local (Ollama)":
        return 0.0
    if usage["source"] == "unavailable" or pricing is None:
        return None
    return (
        usage["input"] * pricing["input"] + usage["output"] * pricing["output"]
    ) / 1_000_000


def cost_label(usage: dict[str, Any]) -> str:
    """Explain why a turn has a cost estimate, no charge, or no known amount."""
    if usage.get("provider") == "Local (Ollama)":
        return "local (no API charge)"
    if usage.get("source") == "not called":
        return "not applicable (LLM not called)"
    if usage.get("source") == "unavailable":
        return "unavailable (LLM request failed)"
    cost = usage.get("cost_usd")
    if cost is None:
        return "unavailable (model rate not configured)"
    return f"${cost:.6f}"


def safe_error_message(error: Exception) -> str:
    """Keep actionable provider errors while removing any credential text."""
    detail = f"{type(error).__name__}: {error}"
    for variable in ("OPENAI_API_KEY", "GEMINI_API_KEY"):
        secret = os.getenv(variable)
        if secret:
            detail = detail.replace(secret, "[redacted API key]")
    detail = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[redacted API key]", detail)
    detail = re.sub(r"\bAIza[A-Za-z0-9_-]{20,}\b", "[redacted API key]", detail)
    return detail[:1000]


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = load_history()


def main() -> None:
    st.set_page_config(page_title="Grounded RAG Chatbot", page_icon="📚", layout="wide")
    init_state()
    st.title("📚 Grounded RAG Chatbot")
    st.caption("Answers are generated only from the documents in data/sample_docs.")

    with st.sidebar:
        st.header("Pipeline controls")
        providers = ["Local (Ollama)", "OpenAI", "Gemini"]
        # OpenAI is the default because the workspace is configured with OPENAI_API_KEY.
        provider = st.selectbox("LLM provider", providers, index=providers.index("OpenAI"))
        default_models = {
            "Local (Ollama)": os.getenv("OLLAMA_CHAT_MODEL", "llama3.2"),
            "OpenAI": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "Gemini": os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash"),
        }
        model = st.text_input("Model", value=default_models[provider])
        if provider == "Local (Ollama)":
            st.caption("Requires Ollama to be running and the selected model to be pulled locally.")
        temperature = st.slider("Temperature", 0.0, 1.5, 0.0, 0.1)
        max_tokens = st.number_input("Maximum output tokens", min_value=32, max_value=2048, value=300, step=32)
        top_k = st.slider("Retrieved chunks (top_k)", 1, 8, 4)
        min_score = st.slider("Minimum cosine similarity", 0.0, 1.0, 0.30, 0.05)
        st.divider()
        # st.cache_resource makes later runs fast, while this status explains the first-run work.
        with st.status("Loading documents and building the vector index...", expanded=False) as index_status:
            collection, file_count, record_count, chunk_count = get_collection()
            index_status.update(
                label=f"Index ready: {chunk_count} chunks from {file_count} files",
                state="complete",
            )
        st.metric("Indexed chunks", chunk_count)
        st.caption(f"{file_count} files → {record_count} records → hybrid chunks → ChromaDB cosine search")
        st.caption(f"Embedding model: {EMBEDDING_MODEL}")
        assistant_messages = [
            message for message in st.session_state.messages
            if message.get("role") == "assistant" and message.get("usage")
        ]
        conversation_tokens = sum(
            message["usage"].get("input", 0) + message["usage"].get("output", 0)
            for message in assistant_messages
        )
        conversation_costs = [
            message["usage"]["cost_usd"]
            for message in assistant_messages
            if message["usage"].get("cost_usd") is not None
        ]
        unknown_cost_turns = sum(
            message["usage"].get("cost_usd") is None for message in assistant_messages
        )
        st.metric("Conversation tokens", f"{conversation_tokens:,}")
        if unknown_cost_turns:
            cost_display = (
                "Rates needed" if not conversation_costs else f"${sum(conversation_costs):.6f}"
            )
            st.metric("Estimated API cost", cost_display)
            st.caption(f"Cost is unavailable for {unknown_cost_turns} turn(s); configure model rates in .env.")
        else:
            st.metric("Estimated API cost", f"${sum(conversation_costs):.6f}")
        try:
            pricing = config.model_pricing(model)
        except ValueError as error:
            pricing = None
            st.error(str(error))
        if provider == "Local (Ollama)":
            st.caption("Local Ollama usage is shown, but no API cost is charged.")
        elif pricing:
            st.caption(
                "Rates are estimates per 1M tokens: "
                f"${pricing['input']:g} input / ${pricing['output']:g} output. "
                "Confirm current pricing with your provider."
            )
        else:
            st.caption(
                "No rate configured for this model. Set both "
                "LLM_*_PRICE_PER_MILLION_USD values in .env."
            )
        if len(st.session_state.messages) >= 10:
            st.info("Five conversations reached. Compact history to reduce prompt tokens.")
        if st.button("Compact conversation history"):
            st.session_state.messages = compact_history(st.session_state.messages)
            save_history(st.session_state.messages)
            st.rerun()
        if st.button("Clear conversation"):
            st.session_state.messages = []
            save_history([])
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("error_detail"):
                st.error(f"LLM request failed: {message['error_detail']}")
            if message.get("usage"):
                usage = message["usage"]
                st.caption(
                    f"Input: {usage['input']:,} tokens · Output: {usage['output']:,} tokens · "
                    f"Total: {usage['input'] + usage['output']:,} ({usage.get('source', 'unknown')}) · "
                    f"Estimated cost: {cost_label(usage)}"
                )

    question = st.chat_input("Ask a question about the provided documents")
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)
    # Keep a visible status open while the slow operations happen.
    with st.chat_message("assistant"):
        with st.status("Working on your question...", expanded=True) as response_status:
            response_status.write("Searching the provided documents...")
            hits = retrieve(question, top_k=top_k, min_score=min_score)
            if not hits:
                response_status.update(label="No matching document evidence found", state="complete")
                answer = FALLBACK
                usage = {"input": 0, "output": 0, "source": "not called"}
                sources = []
            else:
                prompt = build_prompt(question, hits, st.session_state.messages)
                response_status.write(f"Found {len(hits)} relevant chunks. Sending grounded context to {model}...")
                error_detail = None
                try:
                    answer, reported_usage = call_llm(
                        prompt, provider, model, temperature, int(max_tokens)
                    )
                    response_status.update(label="Response received", state="complete")
                except Exception as error:
                    # Never display a framework traceback to students, and never invent an answer
                    # when the selected provider is unavailable or the local model is missing.
                    answer = (
                        f"I found relevant information in the documents, but the selected {provider} "
                        f"model could not be reached. Check the model/API configuration and try again."
                    )
                    response_status.update(label="LLM request failed", state="error")
                    error_detail = safe_error_message(error)
                    usage = {"input": 0, "output": 0, "source": "unavailable"}
                else:
                    usage = token_usage(prompt, answer, reported_usage)
                sources = hits
            if not hits:
                error_detail = None
            usage["provider"] = provider
            usage["model"] = model
            usage["cost_usd"] = estimate_cost(usage, provider, pricing)

    assistant_message = {"role": "assistant", "content": answer, "usage": usage}
    if error_detail:
        assistant_message["error_detail"] = error_detail
    st.session_state.messages.extend([{"role": "user", "content": question}, assistant_message])
    save_history(st.session_state.messages)
    with st.chat_message("assistant"):
        st.markdown(answer)
        if error_detail:
            st.error(f"LLM request failed: {error_detail}")
        st.caption(
            f"Input: {usage['input']:,} tokens · Output: {usage['output']:,} tokens · "
            f"Total this turn: {usage['input'] + usage['output']:,} ({usage['source']}) · "
            f"Estimated API cost: {cost_label(usage)}"
        )
        if sources:
            with st.expander("Sources used"):
                for index, source in enumerate(sources, 1):
                    st.markdown(f"**[{index}] {source['source']} — page {source['page']} — similarity {source['score']:.3f}**")
                    st.write(source["text"][:500] + ("..." if len(source["text"]) > 500 else ""))

    # Rerun so the sidebar token metric immediately reflects this completed turn.
    st.rerun()


if __name__ == "__main__":
    main()
