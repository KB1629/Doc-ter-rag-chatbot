import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from models.llm import get_llm
from utils.vector_store import search
from utils.web_search import web_search
from utils.data_analyzer import query_csv, _get_schema

_llm = None


def _get_llm():
    global _llm
    try:
        if _llm is None:
            _llm = get_llm()
        return _llm
    except Exception as e:
        raise RuntimeError(f"Failed to initialise LLM: {e}")


def _route_query(query: str, has_docs: bool, has_csv: bool) -> str:
    """
    Route the query to the right sources using a strong LLM prompt.
    Returns one of: "sql", "rag", "web", "sql+rag", "sql+web", "rag+web", "all", "llm"
    """
    try:
        sources = []
        if has_csv:
            sources.append("CSV_DATA: structured tabular data (use SQL for aggregations, counts, averages, comparisons, trends, rankings, filtering rows)")
        if has_docs:
            sources.append("DOCUMENTS: unstructured text (use RAG for explanations, summaries, qualitative info, named entities, descriptions)")
        sources.append("WEB_SEARCH: live internet (use for current events, latest news, real-time data, anything not in uploaded files)")
        sources.append("LLM_ONLY: general knowledge (use only if no other source is relevant)")

        sources_str = "\n".join(f"- {s}" for s in sources)

        prompt = (
            "You are a query routing expert for a multi-source AI assistant.\n\n"
            "Available sources:\n"
            f"{sources_str}\n\n"
            "Routing rules:\n"
            "- Use CSV_DATA for: numbers, statistics, aggregations (sum/avg/max/min/count), "
            "comparisons between rows, trends over time, rankings, filtering by value\n"
            "- Use DOCUMENTS for: explanations, summaries, qualitative descriptions, "
            "named entities, skills, experience, goals, narrative content\n"
            "- Use WEB_SEARCH for: anything requiring current/real-time information, "
            "industry standards, job requirements, latest trends, or if user says 'search online'\n"
            "- Combine sources when the question needs multiple types of information\n\n"
            f"User question: {query}\n\n"
            "Reply with ONLY the source combination needed, exactly as written:\n"
            "sql / rag / web / sql+rag / sql+web / rag+web / all / llm"
        )
        response = _get_llm().invoke([HumanMessage(content=prompt)]).content.strip().lower()

        valid = {"sql", "rag", "web", "sql+rag", "sql+web", "rag+web", "all", "llm"}
        for v in valid:
            if v in response:
                return v
        return "rag" if has_docs else ("sql" if has_csv else "web")
    except Exception:
        return "rag" if has_docs else "web"


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
    try:
        return "\n\n".join(
            f"[📄 Page {c['page']} of {c['source']}]\n{c['text']}" for c in chunks
        )
    except Exception:
        return ""


def _format_web_context(results: list[dict]) -> str:
    try:
        return "\n\n".join(
            f"[🌐 {r['title']} — {r['url']}]\n{r['content']}" for r in results
        )
    except Exception:
        return ""


def _build_system_prompt(mode: str, doc_context: str, web_context: str, csv_schema: str) -> str:
    try:
        length_instruction = (
            "Give a short, concise answer in 2-3 sentences." if mode == "concise"
            else "Give a thorough, detailed answer with full explanations."
        )
        context_block = ""
        if csv_schema:
            context_block += f"CSV DATA SCHEMA:\n{csv_schema}\n\n"
        if doc_context:
            context_block += f"DOCUMENT CONTEXT:\n{doc_context}\n\n"
        if web_context:
            context_block += f"WEB SEARCH CONTEXT:\n{web_context}\n\n"
        return (
            f"You are Analyser Bot, an intelligent student career assistant.\n"
            f"{length_instruction}\n\n"
            "After each paragraph, cite the source:\n"
            "- Document content: [📄 Page N of filename]\n"
            "- Web content: [🌐 Source Title — URL]\n"
            "- CSV/data content: [📊 Data Analysis]\n"
            "Only cite sources actually used. If context is insufficient, say so clearly.\n\n"
            f"{context_block}Answer based only on the context provided above."
        )
    except Exception:
        return "You are Analyser Bot. Answer the user's question helpfully."


def ask(query: str, history: list[dict], mode: str, has_docs: bool,
        csv_df: pd.DataFrame = None) -> dict:
    """
    Main entry point. Routes query across CSV (SQL), PDF (RAG), and web sources.
    Returns: {"answer", "source", "doc_chunks", "web_results", "sql_result"}
    """
    try:
        has_csv = csv_df is not None and not csv_df.empty
        route = _route_query(query, has_docs, has_csv)

        doc_chunks, web_results, sql_result, csv_schema = [], [], None, ""

        # SQL path
        if "sql" in route and has_csv:
            sql_result = query_csv(query, csv_df, mode)

        # RAG path
        if "rag" in route and has_docs:
            doc_chunks = search(query)

        # Web path
        if "web" in route:
            search_query = _refine_search_query(query, history)
            web_results = web_search(search_query)

        # LLM only fallback
        if route == "llm" or (not doc_chunks and not web_results and not sql_result):
            route = "llm"

        # Determine source badge — only mark sql active if answer is meaningful
        active = []
        if sql_result and sql_result.get("answer") and not sql_result["answer"].startswith("Data query failed"):
            active.append("sql")
        if doc_chunks:
            active.append("docs")
        if web_results:
            active.append("web")

        # Map active sources to badge key
        if set(active) == {"sql", "docs", "web"}:
            source = "all"
        elif set(active) == {"docs", "web"}:
            source = "both"
        elif len(active) == 1:
            source = active[0]
        elif active:
            source = "+".join(active)
        else:
            source = "llm"

        # If SQL handled it fully, return directly
        if route == "sql" and sql_result:
            return {
                "answer": sql_result["answer"],
                "source": "sql",
                "doc_chunks": [],
                "web_results": [],
                "sql_result": sql_result,
            }

        # Build combined prompt for mixed sources
        doc_context = _format_doc_context(doc_chunks)
        web_context = _format_web_context(web_results)
        sql_answer = sql_result["answer"] if sql_result and sql_result.get("answer") else ""
        sql_context = _get_schema(csv_df) if has_csv and "sql" in route else ""
        system_prompt = _build_system_prompt(mode, doc_context, web_context, sql_context)

        messages = [SystemMessage(content=system_prompt)]
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        # Inject SQL findings into the question so LLM can combine with other sources
        combined_query = query
        if sql_answer:
            combined_query = f"{query}\n\n[Data Analysis Result: {sql_answer}]"
        messages.append(HumanMessage(content=combined_query))

        response = _get_llm().invoke(messages)

        return {
            "answer": response.content,
            "source": source,
            "doc_chunks": doc_chunks,
            "web_results": web_results,
            "sql_result": sql_result,
        }

    except Exception as e:
        return {
            "answer": f"Something went wrong: {e}",
            "source": "error",
            "doc_chunks": [],
            "web_results": [],
            "sql_result": None,
        }
