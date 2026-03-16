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


def _generate_sql_and_chart(query: str, schema: str) -> tuple:
    """
    Single LLM call: returns (sql_string, chart_spec_dict).
    chart_spec: {"type": bar|line|pie|scatter, "x": col, "y": col,
                 "xlabel": str, "ylabel": str, "title": str} or None
    """
    try:
        import json, re
        prompt = f"""You are a SQL and data visualisation expert. A student is analysing their academic data.

## Table Schema
{schema}

## User Question
"{query}"

## Your Task
Return ONLY a valid JSON object with two keys: "sql" and "chart".

### sql
- A single valid SQLite SELECT statement that answers the question
- Use ONLY the exact table name "data" and column names from the schema above
- If the question needs top/bottom N results, use ORDER BY + LIMIT
- If the question needs aggregation (average, count, sum), use GROUP BY with an alias (e.g. AVG(Marks) AS avg_marks)
- Do NOT use markdown, backticks, or any explanation — just the raw SQL string

### chart
- Decide if a chart would be meaningful for this result. If yes, specify:
  - type: choose the BEST type for the data:
    * "pie" — if result has 2-8 categorical groups with a numeric value (e.g. grade distribution)
    * "bar" — if comparing values across a small number of categories (≤15 rows)
    * "line" — if showing a trend or progression over ordered categories
    * "scatter" — if showing correlation between two numeric columns
    * "barh" — if category labels are long (>10 chars average) or there are many categories
  - x: the column name AS IT WILL APPEAR IN THE RESULT (use alias if aggregated, e.g. "avg_marks")
  - y: the column name AS IT WILL APPEAR IN THE RESULT (use alias if aggregated, e.g. "avg_marks")
  - xlabel: human-readable X axis label with context (e.g. "Subject Name", "Semester Number")
  - ylabel: human-readable Y axis label with units if applicable (e.g. "Marks Scored (out of 100)")
  - title: a clear descriptive chart title (e.g. "Average Marks per Semester")
- If no chart is meaningful (e.g. single value result, text-only result), set "chart" to null

## Output Format (ONLY valid JSON, no other text):
{{
  "sql": "SELECT ...",
  "chart": {{
    "type": "bar",
    "x": "Subject",
    "y": "Marks",
    "xlabel": "Subject Name",
    "ylabel": "Marks Scored (out of 100)",
    "title": "Marks Scored per Subject"
  }}
}}"""

        raw = _get_llm().invoke([HumanMessage(content=prompt)]).content.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            raise ValueError(f"LLM returned no JSON: {raw[:100]}")
        data = json.loads(match.group())
        sql = data.get("sql", "").strip()
        chart = data.get("chart")
        return sql, chart
    except Exception as e:
        raise RuntimeError(f"SQL/chart generation failed: {e}")


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
    Returns {"answer", "sql", "result_df", "can_visualize", "chart_spec"}
    """
    try:
        table_name = "data"
        schema = _get_schema(df, table_name)
        sql, chart_spec = _generate_sql_and_chart(query, schema)

        conn = _load_into_sqlite(df, table_name)
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        frames = []
        for stmt in statements:
            try:
                frames.append(pd.read_sql_query(stmt, conn))
            except Exception:
                pass
        conn.close()
        result_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        if result_df.empty:
            answer = "The query returned no results. Try rephrasing your question."
        else:
            answer = _explain_result(query, sql, result_df, mode)
        numeric_cols = result_df.select_dtypes(include="number").columns
        can_visualize = len(numeric_cols) >= 1 and len(result_df) >= 2

        # Validate chart_spec columns exist in result — fix aliases (e.g. AVG(Marks) → avg_marks)
        if chart_spec and isinstance(chart_spec, dict):
            x, y = chart_spec.get("x"), chart_spec.get("y")
            result_numeric = result_df.select_dtypes(include="number").columns.tolist()
            result_text = result_df.select_dtypes(exclude="number").columns.tolist()
            # Fix x: if not in result, use first text column
            if x not in result_df.columns:
                chart_spec["x"] = result_text[0] if result_text else (result_df.columns[0] if len(result_df.columns) else None)
            # Fix y: if not in result, use first numeric column
            if y not in result_df.columns:
                chart_spec["y"] = result_numeric[0] if result_numeric else None
            # If still no valid x or y, discard spec
            if not chart_spec.get("x") or not chart_spec.get("y"):
                chart_spec = None

        return {
            "answer": answer,
            "sql": sql,
            "result_df": result_df,
            "can_visualize": can_visualize,
            "chart_spec": chart_spec,
        }
    except Exception as e:
        return {
            "answer": f"Data query failed: {e}",
            "sql": "",
            "result_df": pd.DataFrame(),
            "can_visualize": False,
            "chart_spec": None,
        }


def generate_chart(result_df: pd.DataFrame, query: str, chart_spec: dict = None) -> bytes:
    """Generate a chart from SQL result using chart_spec from LLM."""
    try:
        if not chart_spec or not isinstance(chart_spec, dict):
            raise ValueError("No chart spec provided.")

        ctype = chart_spec.get("type", "bar")
        x_col = chart_spec.get("x")
        y_col = chart_spec.get("y")
        xlabel = chart_spec.get("xlabel", x_col)
        ylabel = chart_spec.get("ylabel", y_col)
        title = chart_spec.get("title", query[:60])

        if not x_col or not y_col or x_col not in result_df.columns or y_col not in result_df.columns:
            raise ValueError(f"Columns '{x_col}' or '{y_col}' not found in data.")

        # Aggregate if x has repeated values (e.g. multiple subjects per semester)
        if result_df[x_col].duplicated().any():
            plot_df = result_df.groupby(x_col, sort=False)[y_col].mean().reset_index()
        else:
            plot_df = result_df[[x_col, y_col]].copy()

        COLORS = ["#38bdf8", "#4ade80", "#f97316", "#c084fc", "#fb923c", "#facc15"]
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#1a1f2e")
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values(): sp.set_edgecolor("#2a3a5a")

        if ctype == "pie":
            ax.pie(plot_df[y_col].values, labels=plot_df[x_col].astype(str),
                   autopct="%1.1f%%", colors=COLORS[:len(plot_df)],
                   textprops={"color": "#e2e8f0"})
            xlabel = ylabel = None
        elif ctype == "barh":
            ax.barh(plot_df[x_col].astype(str), plot_df[y_col], color=COLORS[0])
            ax.set_xlabel(ylabel, color="#94a3b8", fontsize=9)
            ax.set_ylabel(xlabel, color="#94a3b8", fontsize=9)
            xlabel = ylabel = None
        elif ctype == "scatter":
            ax.scatter(plot_df[x_col], plot_df[y_col], color=COLORS[2], alpha=0.7)
        elif ctype == "line":
            ax.plot(plot_df[x_col].astype(str), plot_df[y_col],
                    marker="o", color=COLORS[1], linewidth=2)
            plt.xticks(rotation=45, ha="right", fontsize=8)
        else:  # bar
            ax.bar(plot_df[x_col].astype(str), plot_df[y_col], color=COLORS[0])
            plt.xticks(rotation=45, ha="right", fontsize=9)

        ax.set_title(title, color="#e2e8f0", pad=10)
        if xlabel: ax.set_xlabel(xlabel, color="#94a3b8", fontsize=9)
        if ylabel: ax.set_ylabel(ylabel, color="#94a3b8", fontsize=9)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close("all")
        buf.seek(0)
        return buf.read()

    except Exception as e:
        raise RuntimeError(f"Chart generation failed: {e}")


def generate_dashboard_analysis(pdf_texts: dict, csv_name: str, df: pd.DataFrame) -> tuple:
    """
    Single LLM call: takes raw PDF texts + CSV DataFrame.
    Returns (pdf_summaries dict, csv_summary str, chart_plan list).
    """
    try:
        import json, re
        llm = _get_llm()

        pdf_section = "\n".join(
            f"--- {name} ---\n{text[:1500]}" for name, text in pdf_texts.items()
        ) if pdf_texts else "No PDFs uploaded."

        if df is not None:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            cat_cols = df.select_dtypes(exclude="number").columns.tolist()
            useful_numeric = [c for c in numeric_cols if df[c].var() > 1.0]
            csv_info = (
                f"CSV '{csv_name}': {len(df)} rows, columns: {list(df.columns)}\n"
                f"Categorical columns: {cat_cols}\n"
                f"Numeric columns with meaningful variance: {useful_numeric}"
            )
        else:
            csv_info = "No CSV uploaded."
            useful_numeric = []

        prompt = f"""You are analysing student career documents. Read the content below and return ONLY valid JSON (no markdown fences).

{pdf_section}

{csv_info}

Return this exact JSON structure:
{{
  "pdf_summaries": {{"filename.pdf": "3-4 sentence summary of what this document contains and what is useful in it"}},
  "csv_summary": "1-2 sentences describing the CSV data and what insights it offers",
  "charts": [
    {{
      "title": "descriptive chart title",
      "type": "bar",
      "x": "column_name",
      "y": "column_name",
      "xlabel": "human readable x label",
      "ylabel": "human readable y label with units"
    }}
  ]
}}

Chart rules:
- Use ONLY column names from: {list(df.columns) if df is not None else []}
- Y axis must be from: {useful_numeric}
- Suggest 2-3 charts with different x+y pairs
- Types: bar (compare categories), line (trend over ordered values), pie (proportions, max 10 unique values), scatter (two numeric columns)"""

        raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            raise ValueError(f"LLM returned no JSON. Response: {raw[:200]}")
        data = json.loads(match.group())
        charts = data.get("charts", [])
        if df is not None:
            charts = [
                c for c in charts
                if c.get("x") in df.columns and c.get("y") in df.columns
                and (c.get("y") not in df.select_dtypes(include="number").columns
                     or df[c["y"]].var() > 1.0)
            ]
        return (data.get("pdf_summaries", {}), data.get("csv_summary", ""), charts)
    except Exception as e:
        raise RuntimeError(f"Dashboard analysis failed: {e}")
