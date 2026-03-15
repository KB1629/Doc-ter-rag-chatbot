# 🔬 Analyser Bot — Student Career Intelligence Assistant

> Upload your marks, resume, and goals. Ask anything. Get data-driven answers with charts, citations, and live job market insights.

**Live App:** https://doc-ter--chatbot.streamlit.app

---

## 🚀 How to Use (Quick Start)

Sample test files are included in the `tests/` folder — no need to prepare anything.

1. Go to the **[live app](https://doc-ter--chatbot.streamlit.app)** — sample files are pre-loaded automatically
2. On the **📊 Dashboard** tab — click **🔍 Analyse All** to see PDF summaries and CSV charts
3. Switch to the **💬 Chat** tab — ask any question in natural language
4. Or upload your own files via the **Dashboard uploader** (PDF or CSV)

**Try these questions in the Chat tab:**
- `"What is my average marks across all semesters?"` → SQL + chart
- `"What skills are listed in my resume?"` → RAG from PDF
- `"What does Google require for a Data Engineer?"` → Live web search
- `"Am I ready for a Data Science internship? What should I improve?"` → All 3 combined

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
| 📈 Auto Chart Generation | Smart charts generated from SQL results and dashboard analysis — LLM picks the best visualisations |
| 📄 RAG on PDFs | Resume, goal statements, or any PDF indexed into ChromaDB for semantic retrieval |
| 🌐 Live Web Search | Real-time Tavily search for job requirements, industry trends, company expectations |
| 🤖 Smart Auto-Routing | LLM decides which source(s) to use per query — SQL, RAG, web, or all three |
| 📝 Inline Citations | Every answer cites its source: `[📄 Page N]`, `[🌐 URL]`, `[📊 Data Analysis]` |
| 🎙️ Voice Input | Speak questions via browser mic using `streamlit-mic-recorder` |
| 🔊 Voice Output | Click Listen on any response for high-quality neural TTS via `edge-tts` |
| ⚡ Concise / Detailed | Toggle response length — 2-sentence summary or full explanation |
| 📋 Dashboard | Upload tab with one-click Analyse All — LLM summaries per PDF + smart CSV charts |
| ⬇️ Download Chat | Export full conversation as a Markdown file |
| 🏷️ Source Badges | Each response labelled: Documents / Web / Data / General Knowledge |

---

## 🏗️ Project Structure

```
analyser-bot/
├── config/
│   ├── __init__.py
│   └── config.py          ← API keys loaded from .env / st.secrets, project constants
├── models/
│   ├── __init__.py
│   ├── llm.py             ← Groq LLM initialisation (llama-3.3-70b-versatile)
│   └── embeddings.py      ← BAAI/bge-small-en-v1.5 embedding model
├── utils/
│   ├── __init__.py
│   ├── document_loader.py ← PDF parsing and text chunking (pymupdf4llm)
│   ├── vector_store.py    ← ChromaDB storage and similarity search
│   ├── web_search.py      ← Tavily live web search wrapper (with retry)
│   ├── rag_chain.py       ← Multi-source routing, prompt building, LLM orchestration
│   ├── voice.py           ← Speech-to-text (SpeechRecognition) and TTS (edge-tts)
│   └── data_analyzer.py   ← Text-to-SQL, chart generation, dashboard LLM analysis
├── tests/
│   ├── arjun_sharma_resume.pdf       ← Sample student resume
│   ├── arjun_sharma_career_goal.pdf  ← Sample career goal statement
│   └── btech_marks.csv               ← Sample B.Tech semester marks
├── app.py                 ← Main Streamlit UI (Dashboard + Chat tabs)
├── requirements.txt       ← Python dependencies
├── packages.txt           ← System packages (ffmpeg for voice)
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

**Live App:** https://doc-ter--chatbot.streamlit.app/

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

## 🧪 Sample Test Files

A `tests/` folder is included in this repository with ready-to-use sample files:

| File | Type | Description |
|---|---|---|
| `arjun_sharma_resume.pdf` | PDF | Sample student resume with skills, projects, and experience |
| `arjun_sharma_career_goal.pdf` | PDF | Career goal statement targeting Data/AI Engineer roles |
| `btech_marks.csv` | CSV | B.Tech semester marks across 4 semesters (24 rows × 6 cols) |

**How to use:**
1. Upload all three files via the sidebar uploader on the live app
2. PDF summaries will appear instantly in the sidebar
3. CSV preview (scrollable) will appear in the sidebar
4. Start asking questions — the sample data is already loaded on the deployed app for anyone to test directly

> The live app at https://doc-ter--chatbot.streamlit.app already has these files pre-loaded so you can test all features immediately without uploading anything.

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
