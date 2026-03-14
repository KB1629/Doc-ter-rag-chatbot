import io
import sys
import os
import sqlite3
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.llm import get_llm
from langchain_core.messages import HumanMessage

_llm = None


def _get_llm():
    global _llm
    try:
        if _llm is None:
            _llm = get_llm()
        return _llm
    except Exception as e:
        raise RuntimeError(f"Failed to initialise LLM: {e}")
    return _llm


def load_csv(uploaded_file) -> pd.DataFrame:
    """Load a CSV from a Streamlit uploaded file or BytesIO object."""
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV: {e}")


def _get_schema(df: pd.DataFrame, table_name: str = "data") -> str:
    """Return a compact schema description for the LLM."""
    try:
        col_info = ", ".join(f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes))
        sample = df.head(3).to_string(index=False)
        return (
            f"Table name: {table_name}\n"
            f"Columns: {col_info}\n"
            f"Row count: {len(df)}\n"
            f"Sample rows:\n{sample}"
        )
    except Exception as e:
        return f"Schema unavailable: {e}"


def _load_into_sqlite(df: pd.DataFrame, table_name: str = "data") -> sqlite3.Connection:
    """Load DataFrame into an in-memory SQLite database."""
    try:
        conn = sqlite3.connect(":memory:")
        df.to_sql(table_name, conn, index=False, if_exists="replace")
        return conn
    except Exception as e:
        raise RuntimeError(f"Failed to load data into SQLite: {e}")


def _generate_sql(query: str, schema: str) -> str:
    """Ask LLM to generate a SQL query for the user's question."""
    try:
        prompt = (
            f"You are a SQL expert. Given this table schema:\n\n{schema}\n\n"
            f"Write a single SQLite SQL query to answer: {query}\n\n"
            "Rules:\n"
            "- Return ONLY the SQL query, no explanation\n"
            "- Use the exact table name and column names from the schema\n"
            "- Keep it simple and correct\n"
            "- Do not use markdown code blocks"
        )
        sql = _get_llm().invoke([HumanMessage(content=prompt)]).content.strip()
        sql = sql.replace("```sql", "").replace("```", "").strip()
        return sql
    except Exception as e:
        raise RuntimeError(f"SQL generation failed: {e}")


def _explain_result(query: str, sql: str, result_df: pd.DataFrame, mode: str) -> str:
    """Ask LLM to explain the SQL result in plain English."""
    try:
        length_instruction = (
            "Give a short, concise answer in 2-3 sentences." if mode == "concise"
            else "Give a thorough, detailed answer."
        )
        prompt = (
            f"A user asked: {query}\n"
            f"SQL query run: {sql}\n"
            f"Result:\n{result_df.to_string(index=False)}\n\n"
            f"Explain this result in plain English. {length_instruction}\n"
            "Cite specific numbers from the result."
        )
        return _get_llm().invoke([HumanMessage(content=prompt)]).content.strip()
    except Exception as e:
        return f"Could not explain result: {e}"


def query_csv(query: str, df: pd.DataFrame, mode: str) -> dict:
    """
    Main entry: Text-to-SQL pipeline.
    Returns {"answer", "sql", "result_df", "can_visualize"}
    """
    try:
        table_name = "data"
        schema = _get_schema(df, table_name)
        sql = _generate_sql(query, schema)

        conn = _load_into_sqlite(df, table_name)
        # Split multi-statement SQL (LLM sometimes generates two SELECTs)
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        frames = []
        for stmt in statements:
            try:
                frames.append(pd.read_sql_query(stmt, conn))
            except Exception:
                pass
        conn.close()
        result_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        answer = _explain_result(query, sql, result_df, mode)
        numeric_cols = result_df.select_dtypes(include="number").columns
        can_visualize = len(numeric_cols) >= 1 and len(result_df) >= 2

        return {
            "answer": answer,
            "sql": sql,
            "result_df": result_df,
            "can_visualize": can_visualize,
        }
    except Exception as e:
        return {
            "answer": f"Data query failed: {e}",
            "sql": "",
            "result_df": pd.DataFrame(),
            "can_visualize": False,
        }


def generate_chart(result_df: pd.DataFrame, query: str) -> bytes:
    """
    Generate a matplotlib chart from a SQL result DataFrame.
    Returns PNG bytes.
    """
    try:
        numeric_cols = result_df.select_dtypes(include="number").columns.tolist()
        text_cols = result_df.select_dtypes(exclude="number").columns.tolist()

        fig, ax = plt.subplots(figsize=(8, 4))

        if text_cols and numeric_cols:
            x = result_df[text_cols[0]].astype(str)
            y = result_df[numeric_cols[0]]
            ax.bar(x, y, color="#4ade80")
            ax.set_xlabel(text_cols[0])
            ax.set_ylabel(numeric_cols[0])
        elif len(numeric_cols) >= 2:
            ax.plot(result_df[numeric_cols[0]], result_df[numeric_cols[1]], marker="o", color="#60a5fa")
            ax.set_xlabel(numeric_cols[0])
            ax.set_ylabel(numeric_cols[1])
        else:
            result_df[numeric_cols[0]].plot(ax=ax, color="#c084fc")
            ax.set_ylabel(numeric_cols[0])

        ax.set_title(query[:60])
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close("all")
        buf.seek(0)
        return buf.read()

    except Exception as e:
        raise RuntimeError(f"Chart generation failed: {e}")
