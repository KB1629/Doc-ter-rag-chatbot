import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import GEMINI_API_KEY, GEMINI_MODEL


def get_llm():
    """Return configured Gemini chat model."""
    try:
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.3,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialise LLM: {e}")
