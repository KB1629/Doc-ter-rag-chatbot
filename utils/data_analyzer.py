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
    """Generate a meaningful chart from a SQL result DataFrame. Returns PNG bytes."""
    try:
        numeric_cols = result_df.select_dtypes(include="number").columns.tolist()
        text_cols = result_df.select_dtypes(exclude="number").columns.tolist()

        # Filter out near-constant numeric columns (variance < 1)
        useful_numeric = [c for c in numeric_cols if result_df[c].var() > 1.0] or numeric_cols

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#1a1f2e")
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values(): sp.set_edgecolor("#2a3a5a")

        if text_cols and useful_numeric:
            x_col, y_col = text_cols[0], useful_numeric[0]
            x = result_df[x_col].astype(str)
            y = result_df[y_col]
            # Use line if many points, bar if few
            if len(result_df) > 10:
                ax.plot(range(len(x)), y.values, marker="o", color="#38bdf8", linewidth=2)
                ax.set_xticks(range(len(x)))
                ax.set_xticklabels(x, rotation=45, ha="right", fontsize=8)
            else:
                ax.bar(x, y, color="#4ade80")
                plt.xticks(rotation=45, ha="right", fontsize=9)
            ax.set_xlabel(x_col, color="#94a3b8", fontsize=9)
            ax.set_ylabel(y_col, color="#94a3b8", fontsize=9)
        elif len(useful_numeric) >= 2:
            ax.scatter(result_df[useful_numeric[0]], result_df[useful_numeric[1]],
                       color="#f97316", alpha=0.7)
            ax.set_xlabel(useful_numeric[0], color="#94a3b8", fontsize=9)
            ax.set_ylabel(useful_numeric[1], color="#94a3b8", fontsize=9)
        else:
            result_df[useful_numeric[0]].plot(ax=ax, color="#c084fc")
            ax.set_ylabel(useful_numeric[0], color="#94a3b8", fontsize=9)

        ax.set_title(query[:70], color="#e2e8f0", pad=10)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close("all")
        buf.seek(0)
        return buf.read()

    except Exception as e:
        raise RuntimeError(f"Chart generation failed: {e}")


def generate_doc_summary(text: str) -> str:
    """Generate a 3-sentence summary of a document's text via LLM."""
    try:
        llm = _get_llm()
        prompt = f"Summarise the following document in exactly 3 concise sentences:\n\n{text[:3000]}"
        return llm.invoke([HumanMessage(content=prompt)]).content
    except Exception:
        return "Summary unavailable."


def generate_dashboard_analysis(doc_summaries: dict, csv_name: str, df: pd.DataFrame) -> tuple:
    """
    One LLM call: returns (pdf_summaries dict, csv_summary str, chart_plan list).
    chart_plan: [{"title":..., "x":col, "y":col, "type":"bar"|"line"|"pie"|"scatter",
                  "xlabel":..., "ylabel":...}]
    """
    try:
        import json, re
        llm = _get_llm()
        pdf_section = "\n".join(f"- {n}: {s[:400]}" for n, s in doc_summaries.items()) or "No PDFs."

        if df is not None:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            cat_cols = df.select_dtypes(exclude="number").columns.tolist()
            # Pre-compute variance and unique counts so LLM can make informed decisions
            col_stats = {
                col: {
                    "variance": round(float(df[col].var()), 2),
                    "unique": int(df[col].nunique()),
                    "sample_values": df[col].dropna().unique()[:5].tolist()
                }
                for col in df.columns
            }
            useful_numeric = [c for c in numeric_cols if df[c].var() > 1.0]
            csv_info = (
                f"CSV '{csv_name}': {len(df)} rows, columns: {list(df.columns)}\n"
                f"Categorical columns: {cat_cols}\n"
                f"Numeric columns with meaningful variance (var > 1): {useful_numeric}\n"
                f"Column stats (variance, unique count, sample values): {json.dumps(col_stats, default=str)}"
            )
        else:
            csv_info = "No CSV loaded."
            useful_numeric, cat_cols = [], []

        prompt = f"""You are a data visualisation expert analysing student career files.

## Uploaded Files
PDFs:
{pdf_section}

{csv_info}

## Your Task
Return ONLY a valid JSON object with these three keys:

### 1. pdf_summaries
For each PDF, write 3-4 sentences describing:
- What type of document it is
- Key content (skills listed, career goals, experience, projects, certifications etc.)
- What a recruiter or student would find useful in it

### 2. csv_summary
1-2 sentences describing what the CSV contains and what insights it offers.

### 3. charts
Suggest exactly 2-3 chart objects. Each chart must:
- Use ONLY column names from: {list(df.columns) if df is not None else []}
- Use a column with meaningful variance as the Y axis (from: {useful_numeric})
- NEVER use a column with near-zero variance (var < 1) as Y axis
- Use different x+y combinations across charts — no repeated pairs
- Choose the chart type that best fits the data relationship:
  * bar — comparing values across categories
  * line — showing trend or progression
  * pie — showing proportion/distribution of a categorical column (max 10 unique values)
  * scatter — showing correlation between two numeric columns
- Include a clear human-readable "xlabel" and "ylabel" (not just column names — add units or context if helpful)
- Include a descriptive "title" that explains what the chart shows

## Output Format (ONLY valid JSON, no explanation):
{{
  "pdf_summaries": {{"filename.pdf": "summary text"}},
  "csv_summary": "summary text",
  "charts": [
    {{
      "title": "Marks Scored per Subject",
      "type": "bar",
      "x": "Subject",
      "y": "Marks",
      "xlabel": "Subject Name",
      "ylabel": "Marks Scored (out of 100)"
    }}
  ]
}}"""

        raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            charts = data.get("charts", [])
            # Safety: filter out charts using low-variance Y columns
            if df is not None:
                charts = [
                    c for c in charts
                    if c.get("y") in df.columns
                    and c.get("x") in df.columns
                    and (c.get("y") not in df.select_dtypes(include="number").columns
                         or df[c["y"]].var() > 1.0)
                ]
            return (
                data.get("pdf_summaries", {}),
                data.get("csv_summary", ""),
                charts
            )
    except Exception:
        pass
    return {}, "", []
