import os
from dotenv import load_dotenv

load_dotenv()

def _get(key: str) -> str:
    """Read from st.secrets (Streamlit Cloud) or os.getenv (local)."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, "")

GROQ_API_KEY     = _get("GROQ_API_KEY")
TAVILY_API_KEY   = _get("TAVILY_API_KEY")

# Warn early if keys are missing
import logging
if not GROQ_API_KEY:
    logging.warning("GROQ_API_KEY is not set")
if not TAVILY_API_KEY:
    logging.warning("TAVILY_API_KEY is not set")

GROQ_MODEL       = "llama-3.3-70b-versatile"
EMBEDDING_MODEL  = "BAAI/bge-small-en-v1.5"
CHROMA_DIR       = "chroma_db"
CHUNK_SIZE       = 1200
CHUNK_OVERLAP    = 150
TOP_K_RESULTS    = 6
