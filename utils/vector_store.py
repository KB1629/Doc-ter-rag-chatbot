import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chromadb
from chromadb.config import Settings
from models.embeddings import embed_texts
from config.config import CHROMA_DIR, TOP_K_RESULTS

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    try:
        if _collection is None:
            _client = chromadb.PersistentClient(
                path=CHROMA_DIR,
                settings=Settings(anonymized_telemetry=False)
            )
            _collection = _client.get_or_create_collection("documents")
        return _collection
    except Exception as e:
        raise RuntimeError(f"Failed to initialise ChromaDB: {e}")


def add_chunks(chunks: list[dict]):
    """Embed and store document chunks in ChromaDB."""
    try:
        collection = _get_collection()
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)
        ids = [f"{c['source']}_p{c['page']}_{i}" for i, c in enumerate(chunks)]
        metadatas = [{"page": c["page"], "source": c["source"]} for c in chunks]
        collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
    except Exception as e:
        raise RuntimeError(f"Failed to store chunks: {e}")


def search(query: str) -> list[dict]:
    """Return top-k relevant chunks for a query."""
    try:
        collection = _get_collection()
        if collection.count() == 0:
            return []
        query_embedding = embed_texts([query])[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(TOP_K_RESULTS, collection.count()),
            include=["documents", "metadatas"]
        )
        return [
            {"text": doc, "page": meta["page"], "source": meta["source"]}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]
    except Exception as e:
        raise RuntimeError(f"Vector search failed: {e}")


def clear_collection():
    """Delete all stored chunks (used when user clears chat/docs)."""
    try:
        global _collection, _client
        _get_collection()  # ensures _client is initialised
        _client.delete_collection("documents")
        _collection = _client.get_or_create_collection("documents")
    except Exception as e:
        raise RuntimeError(f"Failed to clear collection: {e}")


def get_stored_sources() -> list[str]:
    """Return list of unique document names currently stored."""
    try:
        collection = _get_collection()
        if collection.count() == 0:
            return []
        results = collection.get(include=["metadatas"])
        # Normalize to basename so full paths and short names don't duplicate
        sources = list({os.path.basename(m["source"]) for m in results["metadatas"]})
        return sorted(sources)
    except Exception as e:
        return []
