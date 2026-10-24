"""Central configuration. Values come from .env (see .env.example)."""
import math
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
NEW_COLLECTION_DIR = DATA_DIR / "new_collection"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None

# USD per million tokens. These defaults are examples; confirm current provider pricing.
MODEL_PRICING_USD_PER_MILLION = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
}
INPUT_PRICE_PER_MILLION_USD = os.getenv("LLM_INPUT_PRICE_PER_MILLION_USD")
OUTPUT_PRICE_PER_MILLION_USD = os.getenv("LLM_OUTPUT_PRICE_PER_MILLION_USD")

# Retrieval defaults
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100
DEFAULT_TOP_K = 4
DEFAULT_MIN_SCORE = 0.30


def model_pricing(model: str) -> Optional[dict[str, float]]:
    """Return configured per-million-token rates, or None when rates are unknown."""
    if INPUT_PRICE_PER_MILLION_USD is not None or OUTPUT_PRICE_PER_MILLION_USD is not None:
        if INPUT_PRICE_PER_MILLION_USD is None or OUTPUT_PRICE_PER_MILLION_USD is None:
            raise ValueError(
                "Set both LLM_INPUT_PRICE_PER_MILLION_USD and "
                "LLM_OUTPUT_PRICE_PER_MILLION_USD, or neither."
            )
        try:
            rates = {
                "input": float(INPUT_PRICE_PER_MILLION_USD),
                "output": float(OUTPUT_PRICE_PER_MILLION_USD),
            }
        except ValueError as error:
            raise ValueError("LLM token prices must be numeric USD amounts per 1M tokens.") from error
        if any(not math.isfinite(rate) or rate < 0 for rate in rates.values()):
            raise ValueError("LLM token prices must be finite, non-negative USD amounts.")
        return rates
    return MODEL_PRICING_USD_PER_MILLION.get(model)
