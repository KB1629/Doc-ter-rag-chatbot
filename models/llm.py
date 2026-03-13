import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_groq import ChatGroq
from config.config import GROQ_API_KEY, GROQ_MODEL


def get_llm():
    """Return configured Groq chat model."""
    try:
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0.3,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialise LLM: {e}")
