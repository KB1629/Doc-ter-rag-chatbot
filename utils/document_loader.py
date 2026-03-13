import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pymupdf4llm
import tempfile
from config.config import CHUNK_SIZE, CHUNK_OVERLAP


def parse_pdf(uploaded_file) -> list[dict]:
    """
    Parse a Streamlit uploaded PDF file.
    Returns a list of chunks: [{"text": ..., "page": N, "source": filename}, ...]
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        pages = pymupdf4llm.to_markdown(tmp_path, page_chunks=True)
        os.unlink(tmp_path)

        chunks = []
        for page in pages:
            text = page["text"].strip()
            page_num = page["metadata"]["page_number"]
            if not text:
                continue
            for chunk in _split_text(text, uploaded_file.name, page_num):
                chunks.append(chunk)

        return chunks

    except Exception as e:
        raise RuntimeError(f"Failed to parse PDF '{uploaded_file.name}': {e}")


def _split_text(text: str, source: str, page: int) -> list[dict]:
    """Split text into overlapping chunks with metadata."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append({
            "text": text[start:end],
            "page": page + 1,
            "source": source,
        })
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks
