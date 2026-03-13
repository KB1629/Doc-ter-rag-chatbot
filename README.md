# 🩺 Doc-tor AI — Intelligent Research Chatbot

> Upload any PDF. Ask anything. Get cited, sourced answers — with voice.

**Doc-tor AI** is an intelligent conversational assistant that lets you chat with your documents. Upload one or multiple PDFs and ask questions in natural language. The assistant retrieves relevant context from your documents, searches the web when needed, and responds with inline citations so you always know where the answer came from.

Built for the NeoStats AI Engineer Case Study using Streamlit, Gemini 2.0 Flash, ChromaDB, and Tavily.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 Multi-PDF Upload | Upload multiple PDFs at once — all indexed into a single searchable knowledge base |
| 🔍 RAG (Retrieval-Augmented Generation) | Answers grounded in your documents using vector similarity search |
| 🌐 Live Web Search | Automatically searches the web for real-time or date-sensitive queries |
| 🧠 Smart Query Routing | LLM decides whether to use documents, web, or both — in parallel |
| 📝 Inline Citations | Every paragraph cites its source: `[📄 Page N of file.pdf]` or `[🌐 Title — URL]` |
| 🎙️ Voice Input | Speak your question using the browser microphone |
| 🔊 Voice Output | Listen to any response with a single click using high-quality TTS |
| ⚡ Response Modes | Toggle between Concise (2-3 sentences) and Detailed (full explanation) |
| 📋 Document Summary | Auto-generates a 3-line summary for every uploaded PDF |
| ⬇️ Download Conversation | Export the full chat as a Markdown file |
| 🏷️ Source Badge | Each response is labelled: Documents / Web / Both / General Knowledge |

---

## 🏗️ Project Structure

```
doc-tor-ai/
├── config/
│   └── config.py          ← API keys and project-wide settings
├── models/
│   ├── llm.py             ← Gemini 2.0 Flash initialisation
│   └── embeddings.py      ← BAAI/bge-small-en-v1.5 embedding model
├── utils/
│   ├── document_loader.py ← PDF parsing and text chunking (pymupdf4llm)
│   ├── vector_store.py    ← ChromaDB storage and similarity search
│   ├── web_search.py      ← Tavily live web search wrapper
│   ├── rag_chain.py       ← Query routing, prompt building, LLM orchestration
│   └── voice.py           ← Speech-to-text and text-to-speech
├── app.py                 ← Main Streamlit UI
├── requirements.txt       ← Python dependencies
└── .env                   ← API keys (not commited)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Llama 3.3 70B (`langchain-groq` via Groq API) |
| Embeddings | `BAAI/bge-small-en-v1.5` via `sentence-transformers` |
| Vector Database | ChromaDB (local persistent storage) |
| PDF Parsing | `pymupdf4llm` (markdown-aware PDF extraction) |
| Web Search | Tavily API |
| Voice Input | `streamlit-mic-recorder` + `SpeechRecognition` |
| Voice Output | `edge-tts` (Microsoft Neural TTS) |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/KB1629/Doc-ter-rag-chatbot.git
cd Doc-ter-rag-chatbot
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up API keys

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Get your keys here:
- **Gemini**: [Google AI Studio](https://aistudio.google.com/app/apikey) — free, no credit card
- **Tavily**: [Tavily](https://tavily.com) — free tier, 1,000 searches/month

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deployment (Streamlit Cloud)

1. Push your code to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set the main file as `app.py`
4. Add your API keys under **Settings → Secrets**:
```
GEMINI_API_KEY = "your_key"
TAVILY_API_KEY = "your_key"
```
5. Deploy — your app will be live at a public URL

**Live App:** `[Add Streamlit Cloud link here after deployment]`

---

## 💡 How It Works

1. **Upload PDFs** → parsed page-by-page into text chunks with page metadata
2. **Chunks embedded** → converted to vectors using `BAAI/bge-small-en-v1.5` and stored in ChromaDB
3. **You ask a question** → two things happen in parallel:
   - ChromaDB retrieves the most relevant document chunks
   - LLM decides if a live web search is also needed
4. **Context assembled** → document chunks + web results combined into a structured prompt
5. **Gemini responds** → with inline citations per paragraph and a source badge
6. **Voice** → speak your question or listen to the response

---

## 📌 Notes

- The `chroma_db/` folder is created automatically on first use and persists between sessions
- Uploaded documents stay in the vector store until you click **Clear Chat & Documents**
- Voice input requires microphone permission in your browser
- The app works without any PDFs uploaded — it will use web search and general knowledge
