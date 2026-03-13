import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
from streamlit_mic_recorder import mic_recorder

from utils.document_loader import parse_pdf
from utils.vector_store import add_chunks, clear_collection, get_stored_sources
from utils.rag_chain import ask
from utils.voice import text_to_speech, speech_to_text
from models.llm import get_llm
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Doc-tor AI", page_icon="🩺", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
    .source-badge {
        display: inline-block; padding: 2px 8px; border-radius: 8px;
        font-size: 11px; font-weight: 500; margin-bottom: 4px;
        opacity: 0.6;
    }
    .badge-docs { background: #1a3a2a; color: #4ade80; }
    .badge-web  { background: #1a2a3a; color: #60a5fa; }
    .badge-both { background: #2a1a3a; color: #c084fc; }
    .badge-llm  { background: #2a2a1a; color: #facc15; }
    .block-container { padding-bottom: 100px; }
    .stAudio { display: none; }
</style>
""", unsafe_allow_html=True)


BADGE = {
    "docs": '<span class="source-badge badge-docs">📄 From Documents</span>',
    "web":  '<span class="source-badge badge-web">🌐 From Web Search</span>',
    "both": '<span class="source-badge badge-both">📄🌐 Documents + Web</span>',
    "llm":  '<span class="source-badge badge-llm">🤖 General Knowledge</span>',
    "error":'<span class="source-badge badge-llm">⚠️ Error</span>',
}


def generate_doc_summary(text: str) -> str:
    """Generate a 3-line summary of a document using the LLM."""
    try:
        llm = get_llm()
        prompt = f"Summarise the following document in exactly 3 concise sentences:\n\n{text[:3000]}"
        return llm.invoke([HumanMessage(content=prompt)]).content
    except Exception:
        return "Summary unavailable."


def handle_upload(uploaded_files):
    """Parse, chunk, embed and store uploaded PDFs. Show summary in sidebar."""
    for f in uploaded_files:
        if f.name not in st.session_state.processed_files:
            with st.spinner(f"Processing {f.name}..."):
                try:
                    chunks = parse_pdf(f)
                    add_chunks(chunks)
                    full_text = " ".join(c["text"] for c in chunks)
                    summary = generate_doc_summary(full_text)
                    st.session_state.doc_summaries[f.name] = summary
                    st.session_state.processed_files.add(f.name)
                except Exception as e:
                    st.error(f"Failed to process {f.name}: {e}")


def render_sidebar():
    audio = None
    with st.sidebar:
        st.title("🩺 Doc-tor AI")
        st.caption("Upload PDFs and ask anything.")

        uploaded_files = st.file_uploader(
            "Upload PDF(s)", type="pdf", accept_multiple_files=True, key="uploader"
        )
        if uploaded_files:
            handle_upload(uploaded_files)

        stored = get_stored_sources()
        if stored:
            st.markdown("**Loaded Documents:**")
            for src in stored:
                st.markdown(f"- 📄 `{src}`")
                if src in st.session_state.doc_summaries:
                    with st.expander(f"Summary: {src}"):
                        st.write(st.session_state.doc_summaries[src])

        st.divider()
        st.markdown("**🎙️ Voice Input**")
        audio = mic_recorder(start_prompt="🎙️ Speak", stop_prompt="⏹️ Stop", key="mic")
        st.divider()
        mode = st.radio("Response Mode", ["Concise", "Detailed"], index=1)

        st.divider()
        if st.button("🗑️ Clear Chat & Documents", use_container_width=True):
            st.session_state.messages = []
            st.session_state.processed_files = set()
            st.session_state.doc_summaries = {}
            clear_collection()
            st.rerun()

        if st.session_state.get("messages"):
            st.divider()
            chat_text = "\n\n".join(
                f"**{m['role'].capitalize()}:** {m['content']}"
                for m in st.session_state.messages
            )
            st.download_button(
                "⬇️ Download Conversation",
                data=chat_text,
                file_name="doctor_ai_chat.md",
                mime="text/markdown",
                use_container_width=True,
            )

    return mode.lower(), audio


def render_message(msg: dict):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            source = msg.get("source", "llm")
            st.markdown(BADGE.get(source, ""), unsafe_allow_html=True)
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if st.button("🔊 Listen", key=f"tts_{msg['id']}"):
                with st.spinner("Playing..."):
                    try:
                        audio = text_to_speech(msg["content"])
                        st.audio(audio, format="audio/mp3", autoplay=True)
                    except Exception as e:
                        st.error(f"TTS error: {e}")


def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "doc_summaries" not in st.session_state:
        st.session_state.doc_summaries = {}
    if "msg_counter" not in st.session_state:
        st.session_state.msg_counter = 0
    if "last_audio_id" not in st.session_state:
        st.session_state.last_audio_id = None
    if "heard_text" not in st.session_state:
        st.session_state.heard_text = ""

    mode, audio = render_sidebar()
    has_docs = bool(get_stored_sources())

    st.title("🩺 Doc-tor AI")
    st.caption("Ask questions about your documents or anything else — I'll find the answer.")

    for msg in st.session_state.messages:
        render_message(msg)

    # Chat input pinned to bottom
    user_input = st.chat_input("Ask anything...")

    voice_query = ""
    if audio and audio.get("bytes") and audio.get("id") != st.session_state.get("last_audio_id"):
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
                )
            source = result["source"]
            st.markdown(BADGE.get(source, ""), unsafe_allow_html=True)
            st.markdown(result["answer"])

            st.session_state.msg_counter += 1
            ai_msg = {
                "role": "assistant",
                "content": result["answer"],
                "source": source,
                "id": st.session_state.msg_counter,
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
