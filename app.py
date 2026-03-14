import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
import streamlit as st
from streamlit_mic_recorder import mic_recorder

from utils.document_loader import parse_pdf
from utils.vector_store import add_chunks, clear_collection, get_stored_sources
from utils.rag_chain import ask
from utils.voice import text_to_speech, speech_to_text
from utils.data_analyzer import load_csv, query_csv, generate_chart
from models.llm import get_llm
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Analyser Bot", page_icon="🔬", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
    .source-badge {
        display: inline-block; padding: 2px 8px; border-radius: 8px;
        font-size: 11px; font-weight: 500; margin-bottom: 4px; opacity: 0.6;
    }
    .badge-docs  { background: #1a3a2a; color: #4ade80; }
    .badge-web   { background: #1a2a3a; color: #60a5fa; }
    .badge-both  { background: #2a1a3a; color: #c084fc; }
    .badge-llm   { background: #2a2a1a; color: #facc15; }
    .badge-sql   { background: #1a2a3a; color: #f97316; }
    .block-container { padding-bottom: 100px; }
    .stAudio { display: none; }
    /* hide mic recorder iframe white bar, keep button visible */
    .streamlit-mic-recorder { background: transparent !important; border: none !important; }
    iframe[title="streamlit_mic_recorder"] { 
        background: transparent !important; 
        min-height: 0 !important;
        height: 60px !important;
    }
    .intro-box {
        background: linear-gradient(135deg, #1a1f2e, #0f1117);
        border: 1px solid #2a3a5a; border-radius: 16px;
        padding: 28px 32px; margin-bottom: 24px;
    }
    .feature-grid {
        display: grid; grid-template-columns: repeat(3, 1fr);
        gap: 12px; margin-top: 16px;
    }
    .feature-card {
        background: #1a1f2e; border: 1px solid #2a3a5a;
        border-radius: 10px; padding: 14px 16px;
    }
    .feature-card h4 { margin: 0 0 6px 0; font-size: 14px; color: #e2e8f0; }
    .feature-card p  { margin: 0; font-size: 12px; color: #94a3b8; }
    .query-chip {
        display: inline-block; background: #1e293b; border: 1px solid #334155;
        border-radius: 20px; padding: 5px 14px; margin: 4px;
        font-size: 12px; color: #94a3b8; cursor: default;
    }
    .step-row { display: flex; gap: 12px; margin-top: 12px; }
    .step { background: #1a1f2e; border-radius: 8px; padding: 10px 14px; flex: 1;
            font-size: 12px; color: #94a3b8; border: 1px solid #2a3a5a; }
    .step strong { color: #60a5fa; display: block; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

BADGE = {
    "docs":    '<span class="source-badge badge-docs">📄 From Documents</span>',
    "web":     '<span class="source-badge badge-web">🌐 From Web Search</span>',
    "both":    '<span class="source-badge badge-both">📄🌐 Documents + Web</span>',
    "llm":     '<span class="source-badge badge-llm">🤖 General Knowledge</span>',
    "sql":     '<span class="source-badge badge-sql">📊 Data Analysis</span>',
    "sql+docs":'<span class="source-badge badge-both">📊📄 Data + Documents</span>',
    "sql+web": '<span class="source-badge badge-both">📊🌐 Data + Web</span>',
    "all":     '<span class="source-badge badge-both">📊📄🌐 All Sources</span>',
    "error":   '<span class="source-badge badge-llm">⚠️ Error</span>',
}


def generate_doc_summary(text: str) -> str:
    try:
        llm = get_llm()
        prompt = f"Summarise the following document in exactly 3 concise sentences:\n\n{text[:3000]}"
        return llm.invoke([HumanMessage(content=prompt)]).content
    except Exception:
        return "Summary unavailable."


def handle_upload(uploaded_files):
    for f in uploaded_files:
        if f.name in st.session_state.processed_files:
            continue
        with st.spinner(f"Processing {f.name}..."):
            try:
                if f.name.endswith(".csv"):
                    df = load_csv(f)
                    st.session_state.csv_dataframes[f.name] = df
                    # Save raw bytes so CSV survives page refresh
                    f.seek(0)
                    st.session_state.csv_bytes[f.name] = f.read()
                    st.session_state.processed_files.add(f.name)
                else:
                    chunks = parse_pdf(f)
                    add_chunks(chunks)
                    full_text = " ".join(c["text"] for c in chunks)
                    summary = generate_doc_summary(full_text)
                    st.session_state.doc_summaries[f.name] = summary
                    st.session_state.processed_files.add(f.name)
            except Exception as e:
                st.error(f"Failed to process {f.name}: {e}")


def render_sidebar():
    try:
        audio = None
        mode = "detailed"
        with st.sidebar:
            st.title("🔬 Analyser Bot")
            st.caption("Upload marks CSVs, resume PDFs, and ask anything.")

            uploaded_files = st.file_uploader(
                "Upload PDF(s) or CSV(s)", type=["pdf", "csv"],
                accept_multiple_files=True, key="uploader"
            )
            if uploaded_files:
                handle_upload(uploaded_files)

            stored = get_stored_sources()
            if stored:
                st.markdown("**📄 Loaded Documents:**")
                for src in stored:
                    st.markdown(f"- `{src}`")
                    if src in st.session_state.doc_summaries:
                        with st.expander(f"Summary: {src}"):
                            st.write(st.session_state.doc_summaries[src])

            if st.session_state.get("csv_dataframes"):
                st.markdown("**📊 Loaded CSV Files:**")
                for name, df in st.session_state.csv_dataframes.items():
                    st.markdown(f"- `{name}` ({len(df)} rows × {len(df.columns)} cols)")

            st.divider()
            st.markdown("**🎙️ Voice Input**")
            audio = mic_recorder(start_prompt="🎙️ Speak", stop_prompt="⏹️ Stop", use_container_width=True, key="mic")
            st.divider()
            mode = st.radio("Response Mode", ["Concise", "Detailed"], index=1)

            st.divider()
            if st.button("🗑️ Clear Chat & Documents", use_container_width=True):
                st.session_state.messages = []
                st.session_state.processed_files = set()
                st.session_state.doc_summaries = {}
                st.session_state.csv_dataframes = {}
                st.session_state.csv_bytes = {}
                clear_collection()
                st.rerun()

            if st.session_state.get("messages"):
                st.divider()
                chat_text = "\n\n".join(
                    f"**{m['role'].capitalize()}:** {m['content']}"
                    for m in st.session_state.messages
                )
                st.download_button(
                    "⬇️ Download Conversation", data=chat_text,
                    file_name="doctor_ai_chat.md", mime="text/markdown",
                    use_container_width=True,
                )
        return mode.lower(), audio
    except Exception as e:
        st.error(f"Sidebar error: {e}")
        return "detailed", None


def render_sql_result(sql_result: dict):
    """Render SQL query, result table, and visualize button below an assistant message."""
    if not sql_result or not sql_result.get("sql"):
        return
    with st.expander("🔍 SQL Query Used"):
        st.code(sql_result["sql"], language="sql")
    result_df = sql_result.get("result_df")
    if result_df is not None and not result_df.empty:
        st.dataframe(result_df, use_container_width=True)
    if sql_result.get("can_visualize") and result_df is not None:
        if st.button("📊 Visualize", key=f"chart_{id(sql_result)}"):
            with st.spinner("Generating chart..."):
                try:
                    chart_bytes = generate_chart(result_df, sql_result.get("sql", ""))
                    st.image(chart_bytes, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart error: {e}")


def render_message(msg: dict):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            source = msg.get("source", "llm")
            st.markdown(BADGE.get(source, BADGE["llm"]), unsafe_allow_html=True)
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("sql_result"):
                render_sql_result(msg["sql_result"])
            if st.button("🔊 Listen", key=f"tts_{msg['id']}"):
                with st.spinner("Playing..."):
                    try:
                        audio = text_to_speech(msg["content"])
                        st.audio(audio, format="audio/mp3", autoplay=True)
                    except Exception as e:
                        st.error(f"TTS error: {e}")


def main():
    for key, default in [
        ("messages", []),
        ("processed_files", set()),
        ("doc_summaries", {}),
        ("csv_dataframes", {}),
        ("csv_bytes", {}),
        ("msg_counter", 0),
        ("last_audio_id", None),
        ("heard_text", ""),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # Restore CSVs from saved bytes after page refresh
    for name, raw in st.session_state.csv_bytes.items():
        if name not in st.session_state.csv_dataframes:
            try:
                import io
                st.session_state.csv_dataframes[name] = load_csv(io.BytesIO(raw))
            except Exception:
                pass

    mode, audio = render_sidebar()
    has_docs = bool(get_stored_sources())

    # Merge all CSVs into one DataFrame if multiple uploaded
    csv_frames = list(st.session_state.csv_dataframes.values())
    if len(csv_frames) == 1:
        combined_csv = csv_frames[0]
    elif len(csv_frames) > 1:
        combined_csv = pd.concat(csv_frames, ignore_index=True)
    else:
        combined_csv = None

    # ── Intro section (hidden once chat starts) ──────────────────────────
    if not st.session_state.messages:
        st.markdown("""
<div class="intro-box">
  <h2 style="margin:0 0 6px 0; color:#e2e8f0;">🔬 Analyser Bot</h2>
  <p style="margin:0 0 18px 0; color:#94a3b8; font-size:15px;">
    Your personal career intelligence assistant — upload your marks, resume, and goals,
    then ask anything. Get data-driven answers with charts, citations, and live job market insights.
  </p>

  <div class="feature-grid">
    <div class="feature-card">
      <h4>📊 Data Analysis</h4>
      <p>Upload semester mark CSVs. Ask for averages, rankings, trends — SQL runs automatically and charts are generated.</p>
    </div>
    <div class="feature-card">
      <h4>📄 Document Q&A</h4>
      <p>Upload your resume or goal statement as PDF. Ask what skills you have, what experience is listed, or what your goals say.</p>
    </div>
    <div class="feature-card">
      <h4>🌐 Live Web Search</h4>
      <p>Ask about job requirements, company expectations, or industry trends — Analyser Bot searches the web in real time.</p>
    </div>
    <div class="feature-card">
      <h4>🤖 Smart Routing</h4>
      <p>The AI decides which source to use — data, documents, web, or all three — based on your question. No manual switching.</p>
    </div>
    <div class="feature-card">
      <h4>🎙️ Voice I/O</h4>
      <p>Speak your question using the mic in the sidebar. Click 🔊 Listen on any response to hear it read aloud.</p>
    </div>
    <div class="feature-card">
      <h4>⚡ Concise / Detailed</h4>
      <p>Toggle between a quick 2-sentence answer or a full in-depth explanation — your choice, every time.</p>
    </div>
  </div>

  <div style="margin-top:20px;">
    <p style="color:#64748b; font-size:12px; margin-bottom:8px;">💡 TRY ASKING</p>
    <span class="query-chip">What is my average CGPA across all semesters?</span>
    <span class="query-chip">Which semester did I score the highest?</span>
    <span class="query-chip">What skills are listed in my resume?</span>
    <span class="query-chip">What does Google require for a Data Engineer role?</span>
    <span class="query-chip">Am I ready for a Data Science internship?</span>
    <span class="query-chip">What should I improve to get into product companies?</span>
  </div>

  <div style="margin-top:20px;">
    <p style="color:#64748b; font-size:12px; margin-bottom:8px;">🚀 HOW TO GET STARTED</p>
    <div class="step-row">
      <div class="step"><strong>Step 1 — Upload</strong>Add your semester marks CSV and/or resume PDF using the sidebar uploader.</div>
      <div class="step"><strong>Step 2 — Ask</strong>Type or speak your question in the chat box below. Be specific for best results.</div>
      <div class="step"><strong>Step 3 — Explore</strong>View charts, expand SQL queries, click sources, and listen to answers.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("### 💬 Chat")
    st.caption("Ask about your data, documents, or anything career-related.")

    for msg in st.session_state.messages:
        render_message(msg)

    user_input = st.chat_input("Ask anything...")

    voice_query = ""
    if audio and audio.get("bytes") and audio.get("id") != st.session_state.last_audio_id:
        st.session_state.last_audio_id = audio.get("id")
        with st.spinner("Transcribing..."):
            try:
                voice_query = speech_to_text(audio["bytes"])
                if voice_query:
                    st.session_state.heard_text = voice_query
            except Exception as e:
                st.error(f"Transcription error: {e}")

    if st.session_state.get("heard_text") and not voice_query:
        st.session_state.heard_text = ""

    if voice_query:
        st.info(f"🎙️ Heard: *{voice_query}*")

    user_input = user_input or voice_query

    if user_input:
        st.session_state.msg_counter += 1
        user_msg = {"role": "user", "content": user_input, "id": st.session_state.msg_counter}
        st.session_state.messages.append(user_msg)

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask(
                    query=user_input,
                    history=st.session_state.messages[:-1],
                    mode=mode,
                    has_docs=has_docs,
                    csv_df=combined_csv,
                )
            source = result["source"]
            st.markdown(BADGE.get(source, BADGE["llm"]), unsafe_allow_html=True)
            st.markdown(result["answer"])

            if result.get("sql_result"):
                render_sql_result(result["sql_result"])

            st.session_state.msg_counter += 1
            ai_msg = {
                "role": "assistant",
                "content": result["answer"],
                "source": source,
                "id": st.session_state.msg_counter,
                "sql_result": result.get("sql_result"),
            }
            st.session_state.messages.append(ai_msg)

            if st.button("🔊 Listen", key=f"tts_{ai_msg['id']}"):
                with st.spinner("Playing..."):
                    try:
                        audio_out = text_to_speech(result["answer"])
                        st.audio(audio_out, format="audio/mp3", autoplay=True)
                    except Exception as e:
                        st.error(f"TTS error: {e}")


if __name__ == "__main__":
    main()
