"""Session 1 - Chunking strategies.

fixed      : cut every N characters (simple, may split mid-sentence)
sliding    : fixed windows that overlap (keeps context across boundaries)
sentence   : pack whole sentences up to N characters
paragraph  : keep paragraphs together, merge small ones up to N characters
hybrid     : paragraphs first, long paragraphs split by sentences  (good default)
semantic   : an LLM marks topic boundaries (smartest, but slow and costs money)
"""
import re
from typing import Callable, List, Optional


def _clean(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def _merge(units: List[str], chunk_size: int, sep: str = "\n\n") -> List[str]:
    chunks, cur = [], ""
    for u in units:
        if cur and len(cur) + len(sep) + len(u) > chunk_size:
            chunks.append(cur)
            cur = u
        else:
            cur = f"{cur}{sep}{u}" if cur else u
    if cur:
        chunks.append(cur)
    return chunks


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def fixed_chunks(text, chunk_size=500, overlap=0, **_) -> List[str]:
    text = _clean(text)
    pieces = (text[i:i + chunk_size].strip() for i in range(0, len(text), chunk_size))
    return [p for p in pieces if p]


def sliding_chunks(text, chunk_size=500, overlap=100, **_) -> List[str]:
    text = _clean(text)
    step = max(1, chunk_size - overlap)
    pieces = (text[i:i + chunk_size].strip() for i in range(0, len(text), step))
    return [p for p in pieces if p]


def sentence_chunks(text, chunk_size=500, overlap=0, **_) -> List[str]:
    """overlap > 0 means: carry the last sentence into the next chunk."""
    sentences = split_sentences(_clean(text))
    chunks, cur = [], []
    for s in sentences:
        if cur and len(" ".join(cur)) + 1 + len(s) > chunk_size:
            chunks.append(" ".join(cur))
            cur = cur[-1:] if overlap > 0 else []
        cur.append(s)
    if cur:
        tail = " ".join(cur)
        if not chunks or tail != chunks[-1]:
            chunks.append(tail)
    return chunks


def paragraph_chunks(text, chunk_size=500, overlap=0, **_) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", _clean(text)) if p.strip()]
    return _merge(paragraphs, chunk_size)


def hybrid_chunks(text, chunk_size=500, overlap=100, **_) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", _clean(text)) if p.strip()]
    units = []
    for p in paragraphs:
        if len(p) > chunk_size:
            units.extend(sentence_chunks(p, chunk_size, overlap=1))
        else:
            units.append(p)
    return _merge(units, chunk_size)


# ---------------------------------------------------------------- semantic (LLM)
SEMANTIC_PROMPT = """You are splitting a document into topically coherent chunks.
Insert the marker ||| between passages wherever the topic clearly changes.
Rules: do NOT rewrite, summarise, add or delete any words. Output the original
text with the markers only.

TEXT:
{text}"""


def _windows(text: str, max_chars: int = 4000) -> List[str]:
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    return _merge(paragraphs, max_chars)


def _nlen(s: str) -> int:
    return len(re.sub(r"\s+", "", s))


def semantic_chunks(text, chunk_size=800, overlap=0,
                    llm_fn: Optional[Callable[[str], str]] = None, **_) -> List[str]:
    """LLM-based chunking. Falls back to paragraph chunks if the LLM misbehaves."""
    if llm_fn is None:
        return hybrid_chunks(text, chunk_size, overlap)
    chunks = []
    for window in _windows(text):
        try:
            out = llm_fn(SEMANTIC_PROMPT.format(text=window))
            parts = [p.strip() for p in out.split("|||") if p.strip()]
            ratio = _nlen("".join(parts)) / max(1, _nlen(window))
            if not parts or ratio < 0.9 or ratio > 1.1:  # LLM changed the text
                parts = paragraph_chunks(window, chunk_size)
        except Exception:
            parts = paragraph_chunks(window, chunk_size)
        chunks.extend(parts)
    return chunks


CHUNKERS = {
    "fixed": fixed_chunks,
    "sliding": sliding_chunks,
    "sentence": sentence_chunks,
    "paragraph": paragraph_chunks,
    "hybrid": hybrid_chunks,
    "semantic": semantic_chunks,
}
STRATEGIES = list(CHUNKERS)


def chunk_text(text: str, strategy: str = "hybrid", chunk_size: int = 500,
               overlap: int = 100, llm_fn=None) -> List[str]:
    if strategy not in CHUNKERS:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {STRATEGIES}")
    return CHUNKERS[strategy](text, chunk_size=chunk_size, overlap=overlap, llm_fn=llm_fn)
