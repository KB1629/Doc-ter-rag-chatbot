import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tavily import TavilyClient
from config.config import TAVILY_API_KEY

_client = None


def _get_client():
    global _client
    try:
        if _client is None:
            _client = TavilyClient(api_key=TAVILY_API_KEY)
        return _client
    except Exception as e:
        raise RuntimeError(f"Failed to initialise Tavily client: {e}")


def web_search(query: str, max_results: int = 4) -> list[dict]:
    """Search the web using Tavily with retry on connection errors."""
    last_err = None
    for attempt in range(3):
        try:
            client = _get_client()
            response = client.search(query, max_results=max_results)
            return [
                {
                    "title": r.get("title", ""),
                    "url":   r.get("url", ""),
                    "content": r.get("content", ""),
                }
                for r in response.get("results", [])
            ]
        except Exception as e:
            last_err = e
            # Only retry on connection-level errors
            err_str = str(e).lower()
            if any(k in err_str for k in ("connection", "reset", "timeout", "aborted")):
                time.sleep(1.5 * (attempt + 1))   # 1.5s, 3s back-off
                _client = None                     # force new client on retry
                continue
            break   # non-network error — don't retry
    raise RuntimeError(f"Web search failed after retries: {last_err}")
