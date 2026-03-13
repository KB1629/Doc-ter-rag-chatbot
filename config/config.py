import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

GROQ_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHROMA_DIR = "chroma_db"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
TOP_K_RESULTS = 6
