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
- If the question needs aggregation (average, count, sum), use GROUP BY
- Do NOT use markdown, backticks, or any explanation — just the raw SQL string

### chart
- Decide if a chart would be meaningful for this result. If yes, specify:
  - type: choose the BEST type for the data:
    * "pie" — if result has 2-8 categorical groups with a numeric value (e.g. grade distribution)
    * "bar" — if comparing values across a small number of categories (≤15 rows)
    * "line" — if showing a trend or progression over ordered categories
    * "scatter" — if showing correlation between two numeric columns
    * "barh" — if category labels are long (>10 chars average) or there are many categories
  - x: the column name to use as X axis (or category labels for pie)
  - y: the column name to use as Y axis (or values for pie) — must have meaningful variance
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
        if match:
            data = json.loads(match.group())
            sql = data.get("sql", "").strip()
            chart = data.get("chart")
            return sql, chart
    except Exception:
        pass
    return "", None


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

        answer = _explain_result(query, sql, result_df, mode)
        numeric_cols = result_df.select_dtypes(include="number").columns
        can_visualize = len(numeric_cols) >= 1 and len(result_df) >= 2

        # Validate chart_spec columns exist in result
        if chart_spec and isinstance(chart_spec, dict):
            x, y = chart_spec.get("x"), chart_spec.get("y")
            if x not in result_df.columns or (y and y not in result_df.columns):
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
    """Generate a chart from SQL result. Uses chart_spec from LLM if provided."""
    try:
        COLORS = ["#38bdf8","#4ade80","#f97316","#c084fc","#fb923c","#facc15"]
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#1a1f2e")
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values(): sp.set_edgecolor("#2a3a5a")

        numeric_cols = result_df.select_dtypes(include="number").columns.tolist()
        text_cols = result_df.select_dtypes(exclude="number").columns.tolist()
        useful_numeric = [c for c in numeric_cols if result_df[c].var() > 1.0] or numeric_cols

        # Use LLM chart spec if valid
        if chart_spec and isinstance(chart_spec, dict):
            ctype = chart_spec.get("type", "bar")
            x_col = chart_spec.get("x")
            y_col = chart_spec.get("y")
            xlabel = chart_spec.get("xlabel", x_col)
            ylabel = chart_spec.get("ylabel", y_col)
            title = chart_spec.get("title", query[:60])
        else:
            # Fallback: infer from result shape
            ctype = "bar"
            x_col = text_cols[0] if text_cols else None
            y_col = useful_numeric[0] if useful_numeric else None
            xlabel, ylabel = x_col, y_col
            title = query[:60]
            if len(result_df) > 10: ctype = "line"
            if not text_cols and len(useful_numeric) >= 2: ctype = "scatter"

        if ctype == "pie" and x_col and y_col:
            pie_data = result_df.set_index(x_col)[y_col]
            ax.pie(pie_data.values, labels=pie_data.index.astype(str),
                   autopct="%1.1f%%", colors=COLORS[:len(pie_data)],
                   textprops={"color": "#e2e8f0"})
            xlabel = ylabel = None
        elif ctype == "barh" and x_col and y_col:
            ax.barh(result_df[x_col].astype(str), result_df[y_col], color=COLORS[0])
            ax.set_xlabel(ylabel, color="#94a3b8", fontsize=9)
            ax.set_ylabel(xlabel, color="#94a3b8", fontsize=9)
            xlabel = ylabel = None
        elif ctype == "scatter" and x_col and y_col:
            ax.scatter(result_df[x_col], result_df[y_col], color=COLORS[2], alpha=0.7)
        elif ctype == "line" and x_col and y_col:
            ax.plot(result_df[x_col].astype(str), result_df[y_col],
                    marker="o", color=COLORS[1], linewidth=2)
            plt.xticks(rotation=45, ha="right", fontsize=8)
        else:  # bar default
            if x_col and y_col:
                ax.bar(result_df[x_col].astype(str), result_df[y_col], color=COLORS[0])
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
