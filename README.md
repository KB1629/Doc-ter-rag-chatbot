# 🔬 Analyser Bot — Student Career Intelligence Assistant

> Upload your marks, resume, and goals. Ask anything. Get data-driven answers with charts, citations, and live job market insights.

---

## 🎯 Problem Statement

Final-year engineering students face a common challenge: they have their semester mark sheets, a resume, and a career goal — but no clear picture of where they stand versus what top companies actually require. Manually comparing a personal profile against job descriptions across multiple sources is time-consuming and inconsistent.

**Analyser Bot solves this** by letting students upload their data and documents, then ask natural language questions. The bot automatically decides whether to run SQL on the marks data, retrieve context from the resume PDF, search the web for live job requirements — or combine all three — and responds with cited, chart-backed answers.

---

## 💡 Example Use Case

A student uploads:
- `semester_marks.csv` — marks across 4 semesters
- `resume.pdf` — their current resume
- `career_goal.pdf` — a short document describing their target role

They then ask:

| Question | What happens |
|---|---|
| *"What is my average CGPA across all semesters?"* | SQL runs on CSV → bar chart generated |
| *"Which semester did I score the highest?"* | SQL → ranked result table |
| *"What skills are listed in my resume?"* | RAG retrieves from resume PDF |
| *"What does Google require for a Data Engineer?"* | Live Tavily web search |
| *"Am I ready for a Data Science internship? What should I improve?"* | All 3 sources combined → cited answer |

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 Text-to-SQL on CSV | Upload marks or any structured data — the bot generates SQL, runs it, and explains results in plain English |
| 📈 Auto Chart Generation | Bar charts and line graphs generated automatically from SQL results — zero manual configuration |
| 📄 RAG on PDFs | Resume, goal statements, or any PDF indexed into ChromaDB for semantic retrieval |
| 🌐 Live Web Search | Real-time Tavily search for job requirements, industry trends, company expectations |
| 🤖 Smart Auto-Routing | LLM decides which source(s) to use per query — SQL, RAG, web, or all three |
| 📝 Inline Citations | Every answer cites its source: `[📄 Page N]`, `[🌐 URL]`, `[📊 Data Analysis]` |
| 🎙️ Voice Input | Speak questions via browser mic using `streamlit-mic-recorder` |
| 🔊 Voice Output | Click Listen on any response for high-quality neural TTS via `edge-tts` |
| ⚡ Concise / Detailed | Toggle response length — 2-sentence summary or full explanation |
| 📋 Document Summaries | Auto-generated 3-line summary for every uploaded PDF |
| ⬇️ Download Chat | Export full conversation as a Markdown file |
| 🏷️ Source Badges | Each response labelled: Documents / Web / Data / General Knowledge |

---

## 🏗️ Project Structure

```
analyser-bot/
├── config/
│   ├── __init__.py
│   └── config.py          ← API keys loaded from .env, project-wide constants
├── models/
│   ├── __init__.py
│   ├── llm.py             ← Groq LLM initialisation (llama-3.3-70b-versatile)
│   └── embeddings.py      ← BAAI/bge-small-en-v1.5 embedding model
├── utils/
│   ├── __init__.py
│   ├── document_loader.py ← PDF parsing and text chunking (pymupdf4llm)
│   ├── vector_store.py    ← ChromaDB storage and similarity search
│   ├── web_search.py      ← Tavily live web search wrapper
│   ├── rag_chain.py       ← Multi-source routing, prompt building, LLM orchestration
│   ├── voice.py           ← Speech-to-text (SpeechRecognition) and TTS (edge-tts)
│   └── data_analyzer.py   ← Text-to-SQL pipeline, result explanation, chart generation
├── app.py                 ← Main Streamlit UI
├── requirements.txt       ← Python dependencies
├── .env                   ← API keys (not committed)
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Llama 3.3 70B via Groq API (`langchain-groq`) |
| Embeddings | `BAAI/bge-small-en-v1.5` via `sentence-transformers` |
| Vector Database | ChromaDB (local persistent storage) |
| PDF Parsing | `pymupdf4llm` (markdown-aware extraction) |
| Structured Data | SQLite in-memory via `pandas` + Text-to-SQL |
| Chart Generation | `matplotlib` (from SQL result DataFrames) |
| Web Search | Tavily API |
| Voice Input | `streamlit-mic-recorder` + `SpeechRecognition` + Google STT |
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
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Get your keys:
- **Groq**: [console.groq.com](https://console.groq.com) — free tier, 14,400 requests/day
- **Tavily**: [tavily.com](https://tavily.com) — free tier, 1,000 searches/month

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
4. Add secrets under **Settings → Secrets**:

```toml
GROQ_API_KEY = "your_key"
TAVILY_API_KEY = "your_key"
```

5. Deploy — your app will be live at a public URL

**Live App:** `[Add Streamlit Cloud link here after deployment]`

---

## 💡 How It Works

```
User uploads CSV + PDF
        ↓
User asks a question
        ↓
LLM routing agent decides:
  ├── SQL query?     → Text-to-SQL on CSV → result table + chart
  ├── Doc question?  → ChromaDB similarity search → relevant PDF chunks
  ├── Web question?  → Tavily search → live results
  └── Complex?       → All three combined
        ↓
LLM assembles answer with inline citations
        ↓
Response shown with source badge + TTS option
```

---

## 📌 Notes

- `chroma_db/` is created automatically on first use and persists between sessions
- Uploaded documents stay indexed until you click **Clear Chat & Documents**
- Multiple CSVs are merged automatically before SQL analysis
- Voice input requires microphone permission in your browser
- The app works without any uploads — it will use web search and general knowledge

---

## 📦 Deliverables

- ✅ Working Streamlit app with all mandatory + additional features
- ✅ GitHub repository with clean structure
- ✅ Deployed on Streamlit Cloud
- ✅ PPT presentation deck
