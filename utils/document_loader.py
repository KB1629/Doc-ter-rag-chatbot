import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pymupdf4llm
import tempfile
from config.config import CHUNK_SIZE, CHUNK_OVERLAP


def _ocr_page(tmp_path: str, page_num: int) -> str:
    """OCR a single page using pytesseract as fallback for image-based pages."""
    try:
        import pytesseract
        from pdf2image import convert_from_path
        images = convert_from_path(tmp_path, first_page=page_num, last_page=page_num, dpi=200)
        if images:
            return pytesseract.image_to_string(images[0])
    except Exception:
        pass
    return ""


def parse_pdf(uploaded_file, source_name: str = None) -> list[dict]:
    """Parse a PDF file (Streamlit upload or BytesIO). Returns list of chunks.
    Falls back to OCR for image-based/scanned pages."""
    try:
        name = source_name or getattr(uploaded_file, "name", "document.pdf")
        data = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            pages = pymupdf4llm.to_markdown(tmp_path, page_chunks=True)
            chunks = []
            for page in pages:
                text = page["text"].strip()
                page_num = page["metadata"]["page_number"]
                if not text:
                    text = _ocr_page(tmp_path, page_num).strip()
                if not text:
                    continue
                for chunk in _split_text(text, name, page_num):
                    chunks.append(chunk)
        finally:
            os.unlink(tmp_path)

        if not chunks:
            raise RuntimeError(
                f"No text could be extracted from '{name}' even after OCR. "
                "The file may be corrupted or contain only non-text content."
            )
        return chunks
    except RuntimeError:
        raise
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
