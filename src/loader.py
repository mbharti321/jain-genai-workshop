"""Session 1 - Load PDF / text documents into a list of pages."""
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader

SUPPORTED = {".pdf", ".txt", ".md"}


def load_file(path) -> List[Dict]:
    """Return [{'text', 'source', 'page'}, ...] (one entry per PDF page)."""
    path = Path(path)
    ext = path.suffix.lower()
    pages = []

    if ext == ".pdf":
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"text": text, "source": path.name, "page": i})
    elif ext in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            pages.append({"text": text, "source": path.name, "page": 1})
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return pages


def list_documents(directory) -> List[Path]:
    return sorted(p for p in Path(directory).glob("*") if p.suffix.lower() in SUPPORTED)
