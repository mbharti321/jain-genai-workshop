"""Streamlit UI for the GenAI Knowledge Assistant.   Run:  streamlit run app.py"""
import streamlit as st

from src import config
from src.chunking import STRATEGIES
from src.loader import SUPPORTED, list_documents
from src.rag_pipeline import RAGAssistant

st.set_page_config(page_title="Knowledge Assistant", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model (first run downloads it)...")
def get_assistant(collection: str) -> RAGAssistant:
    return RAGAssistant(collection)


def render_sources(sources):
    for i, s in enumerate(sources, start=1):
        label = f"[{i}] {s['source']} · page {s['page']} · score {s['score']:.2f}"
        with st.expander(label):
            st.write(s["text"])


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("1 · Documents")
    collection = st.text_input("Collection name", value="knowledge_base")
    assistant = get_assistant(collection)

    uploaded = st.file_uploader("Upload PDF / TXT / MD",
                                type=[e.strip(".") for e in SUPPORTED],
                                accept_multiple_files=True)

    st.subheader("Chunking")
    strategy = st.selectbox("Strategy", STRATEGIES, index=STRATEGIES.index("hybrid"),
                            help="'semantic' uses the LLM, so it is slower and costs tokens.")
    chunk_size = st.slider("Chunk size (characters)", 100, 2000, config.DEFAULT_CHUNK_SIZE, 50)
    overlap = st.slider("Overlap (characters)", 0, 500, config.DEFAULT_OVERLAP, 20,
                        help="Used by 'sliding'. For 'sentence'/'hybrid', >0 carries 1 sentence.")
    reset_first = st.checkbox("Clear collection before indexing", value=True)

    def run_ingest(paths):
        with st.spinner(f"Chunking + embedding {len(paths)} file(s)..."):
            stats = assistant.ingest(paths, strategy, chunk_size, overlap, reset=reset_first)
        st.success(f"Indexed {stats['chunks']} chunks from {stats['files']} file(s).")

    c1, c2 = st.columns(2)
    if c1.button("Index uploads", disabled=not uploaded, use_container_width=True):
        saved = []
        config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        for f in uploaded:
            dest = config.UPLOAD_DIR / f.name
            dest.write_bytes(f.getbuffer())
            saved.append(dest)
        run_ingest(saved)
    if c2.button("Load samples", use_container_width=True):
        run_ingest(list_documents(config.SAMPLE_DOCS_DIR))

    st.metric("Chunks indexed", assistant.store.count())
    if assistant.store.count():
        st.caption("Sources: " + ", ".join(assistant.store.sources()))

    st.header("2 · Retrieval")
    mode = st.radio("Search mode", ["semantic", "keyword", "hybrid"], horizontal=True)
    top_k = st.slider("Passages to retrieve (top-k)", 1, 10, config.DEFAULT_TOP_K)
    min_score = st.slider("Min similarity (fallback threshold)", 0.0, 1.0,
                          config.DEFAULT_MIN_SCORE, 0.05,
                          help="Below this, the assistant says 'not found' without calling the LLM.")
    use_history = st.checkbox("Use conversation history", value=False)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------ main chat
st.title("📚 GenAI Knowledge Assistant")
st.caption("Answers are grounded in retrieved passages from your documents, with sources.")

if "messages" not in st.session_state:
    st.session_state.messages = []

usage_totals = [
    message.get("usage", {})
    for message in st.session_state.messages
    if message.get("role") == "assistant"
]
total_tokens = sum(
    usage.get("input", 0) + usage.get("output", 0) for usage in usage_totals
)
known_costs = [
    usage["cost_usd"] for usage in usage_totals if usage.get("cost_usd") is not None
]
unknown_cost_turns = sum(usage.get("cost_usd") is None for usage in usage_totals)
total_cost = sum(known_costs)

with st.sidebar:
    st.subheader("Usage and estimated cost")
    st.metric("LLM tokens used", f"{total_tokens:,}")
    st.caption("Counts chat-model input/output tokens; local retrieval embeddings are not included.")
    if unknown_cost_turns:
        st.metric("Estimated API cost", "Rates needed" if not known_costs else f"${total_cost:.6f}")
        st.caption(
            f"Cost is unavailable for {unknown_cost_turns} turn(s). Configure the model's "
            "per-million-token rates in .env to include them."
        )
    else:
        st.metric("Estimated API cost", f"${total_cost:.6f}")
    model_name = (
        config.ANTHROPIC_MODEL
        if config.LLM_PROVIDER == "anthropic"
        else config.OPENAI_MODEL
    )
    try:
        pricing = config.model_pricing(model_name)
    except ValueError as error:
        pricing = None
        st.error(str(error))
    if pricing:
        st.caption(
            "Rates are estimates per 1M tokens: "
            f"${pricing['input']:g} input / ${pricing['output']:g} output. "
            "Confirm current pricing with your provider."
        )

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("meta"):
            st.caption(m["meta"])
        if m.get("usage"):
            usage = m["usage"]
            cost_text = (
                f"${usage['cost_usd']:.6f}"
                if usage.get("cost_usd") is not None
                else "unavailable"
            )
            st.caption(
                f"Input: {usage['input']:,} tokens · Output: {usage['output']:,} tokens · "
                f"Total: {usage['input'] + usage['output']:,} tokens "
                f"({usage['source']}) · Estimated cost: {cost_text}"
            )
        if m.get("sources"):
            render_sources(m["sources"])

question = st.chat_input("Ask a question about your documents...")
if question:
    if assistant.store.count() == 0:
        st.warning("Index some documents first (sidebar → Load samples or Index uploads).")
        st.stop()

    history = [{"role": m["role"], "content": m["content"]}
               for m in st.session_state.messages] if use_history else None

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    response_saved = False
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents and thinking..."):
                res = assistant.ask(question, top_k=top_k, min_score=min_score,
                                    mode=mode, history=history)
                usage = res["usage"]
                cost_usd = 0.0 if usage["source"] == "not called" else None
                if pricing is not None and usage["source"] != "not called":
                    cost_usd = (
                        usage["input"] * pricing["input"]
                        + usage["output"] * pricing["output"]
                    ) / 1_000_000
                usage["cost_usd"] = cost_usd
                meta = (f"{'✅ grounded' if res['grounded'] else '⚠️ not found in documents'} · "
                        f"best score {res['best_score']:.2f} · "
                        f"retrieval {res['retrieval_s']:.2f}s · generation {res['generation_s']:.2f}s")
            st.markdown(res["answer"])
            st.caption(meta)
            cost_text = (
                f"${usage['cost_usd']:.6f}"
                if usage["cost_usd"] is not None
                else "unavailable; configure model rates in .env"
            )
            st.caption(
                f"Tokens this turn: {usage['input']:,} input + {usage['output']:,} output "
                f"= {usage['input'] + usage['output']:,} ({usage['source']}) · "
                f"Estimated API cost: {cost_text}"
            )
            render_sources(res["sources"])
            st.session_state.messages.append({"role": "assistant", "content": res["answer"],
                                              "meta": meta, "sources": res["sources"],
                                              "usage": usage})
            response_saved = True
        except Exception as e:  # e.g. missing API key
            st.error(f"Something went wrong: {e}")
    if response_saved:
        st.rerun()
