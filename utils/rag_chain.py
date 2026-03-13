import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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


def _refine_search_query(query: str, history: list[dict]) -> str:
    """Rewrite vague query into a precise web search query using conversation context."""
    try:
        context = "\n".join(f"{m['role']}: {m['content']}" for m in history[-4:])
        prompt = (
            f"Given this conversation:\n{context}\n\n"
            f"Rewrite this user question into a precise web search query (max 10 words, no fluff):\n{query}\n\n"
            "Reply with only the search query, nothing else."
        )
        return _get_llm().invoke([HumanMessage(content=prompt)]).content.strip()
    except Exception:
        return query


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
Preserve exact spellings of names, places, and technical terms as they appear in the source.

{context_block}Answer based only on the context provided above."""


def ask(query: str, history: list[dict], mode: str, has_docs: bool) -> dict:
    try:
        # Single parallel: only ChromaDB search (no separate routing LLM call)
        doc_chunks = search(query) if has_docs else []

        # Simple keyword-based routing to save API calls
        web_triggers = ["latest", "recent", "news", "today", "current",
                        "2024", "2025", "2026", "price", "stock", "search",
                        "online", "internet", "web"]
        needs_web = any(w in query.lower() for w in web_triggers)

        if needs_web:
            search_query = _refine_search_query(query, history)
            web_results = web_search(search_query)
        else:
            web_results = []

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
