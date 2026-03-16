import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pymupdf4llm
import tempfile
from config.config import CHUNK_SIZE, CHUNK_OVERLAP


def parse_pdf(uploaded_file, source_name: str = None) -> list[dict]:
    """Parse a PDF file (Streamlit upload or BytesIO). Returns list of chunks."""
    try:
        name = source_name or getattr(uploaded_file, "name", "document.pdf")
        data = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        pages = pymupdf4llm.to_markdown(tmp_path, page_chunks=True)
        os.unlink(tmp_path)
        chunks = []
        for page in pages:
            text = page["text"].strip()
            page_num = page["metadata"]["page_number"]
            if not text:
                continue
            for chunk in _split_text(text, name, page_num):
                chunks.append(chunk)
        if not chunks:
            raise RuntimeError(
                f"No text could be extracted from '{name}'. "
                "This appears to be a scanned/image-based PDF. "
                "Please upload a text-based PDF instead."
            )
        return chunks
    except Exception as e:
        raise RuntimeError(f"Failed to parse PDF: {e}")


def _split_text(text: str, source: str, page: int) -> list[dict]:
    """Split text into overlapping chunks with metadata."""
    try:
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
    except Exception as e:
        raise RuntimeError(f"Failed to split text: {e}")
