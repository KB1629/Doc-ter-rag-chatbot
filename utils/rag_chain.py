import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from models.llm import get_llm
from utils.vector_store import search
from utils.web_search import web_search

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm


def _llm_needs_web(query: str) -> bool:
    """Ask LLM whether this query needs a live web search."""
    try:
        prompt = (
    "You are a routing assistant for a chatbot that has access to "
    "uploaded documents and live web search.\n\n"
    "Decide if this question requires a LIVE WEB SEARCH to answer "
    "accurately. Web search is needed for: current events, today's "
    "date-sensitive info, real-time data, latest news, anything not "
    "likely covered in static uploaded documents, or if the user explicitly asks to search online, refer to the internet, or check the web.\n\n"
    "Web search is NOT needed for: questions about uploaded document "
    "content, general knowledge, definitions, historical facts, or "
    "conceptual explanations.\n\n"
    f"Question: {query}\n\n"
    "Reply with ONLY one word: YES or NO."
)

        response = _get_llm().invoke([HumanMessage(content=prompt)])
        return response.content.strip().upper().startswith("Y")
    except Exception:
        return False


def _format_doc_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[📄 Page {c['page']} of {c['source']}]\n{c['text']}" for c in chunks
    )


def _format_web_context(results: list[dict]) -> str:
    return "\n\n".join(
        f"[🌐 {r['title']} — {r['url']}]\n{r['content']}" for r in results
    )


def _build_system_prompt(mode: str, doc_context: str, web_context: str) -> str:
    length_instruction = (
        "Give a short, concise answer in 2-3 sentences." if mode == "concise"
        else "Give a thorough, detailed answer with full explanations."
    )
    context_block = ""
    if doc_context:
        context_block += f"DOCUMENT CONTEXT:\n{doc_context}\n\n"
    if web_context:
        context_block += f"WEB SEARCH CONTEXT:\n{web_context}\n\n"

    return f"""You are Doc-tor AI, an intelligent research assistant.
{length_instruction}

After each paragraph, add a citation in brackets showing exactly where that information came from.
For document content use: [📄 Page N of filename]
For web content use: [🌐 Source Title — URL]
Only cite sources that were actually used. If context is insufficient, say so clearly.

{context_block}Answer based only on the context provided above."""


def ask(query: str, history: list[dict], mode: str, has_docs: bool) -> dict:
    """
    Main entry point. Runs LLM routing + ChromaDB search in parallel,
    then fetches web results if needed.
    Returns: {"answer", "source", "doc_chunks", "web_results"}
    """
    try:
        # Run LLM routing decision and ChromaDB search in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_web_needed = executor.submit(_llm_needs_web, query)
            future_chunks = executor.submit(search, query) if has_docs else None

            needs_web = future_web_needed.result()
            doc_chunks = future_chunks.result() if future_chunks else []

        # Fetch web results only if LLM decided it's needed
        web_results = web_search(query) if needs_web else []

        # Determine source label for UI badge
        if doc_chunks and web_results:
            source = "both"
        elif web_results:
            source = "web"
        elif doc_chunks:
            source = "docs"
        else:
            source = "llm"

        doc_context = _format_doc_context(doc_chunks)
        web_context = _format_web_context(web_results)
        system_prompt = _build_system_prompt(mode, doc_context, web_context)

        messages = [SystemMessage(content=system_prompt)]
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=query))

        response = _get_llm().invoke(messages)

        return {
            "answer": response.content,
            "source": source,
            "doc_chunks": doc_chunks,
            "web_results": web_results,
        }

    except Exception as e:
        return {
            "answer": f"Something went wrong: {e}",
            "source": "error",
            "doc_chunks": [],
            "web_results": [],
        }
