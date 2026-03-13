import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tavily import TavilyClient
from config.config import TAVILY_API_KEY

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = TavilyClient(api_key=TAVILY_API_KEY)
    return _client


def web_search(query: str, max_results: int = 4) -> list[dict]:
    """
    Search the web using Tavily.
    Returns list of {"title": ..., "url": ..., "content": ...}
    """
    try:
        client = _get_client()
        response = client.search(query, max_results=max_results)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            }
            for r in response.get("results", [])
        ]
    except Exception as e:
        raise RuntimeError(f"Web search failed: {e}")
