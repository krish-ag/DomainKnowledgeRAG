"""Utilities for storing and retrieving corpus chunks with ChromaDB."""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from threading import Lock
from typing import Any, Sequence
from uuid import uuid4

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
CHROMA_PATH = Path(
    os.getenv("CHROMA_DB_PATH", Path(__file__).resolve().parent / "chroma_db")
).expanduser()

# Expensive resources are initialized on first use, keeping imports and app startup fast.
client: Any | None = None
embedding_model: Any | None = None
_client_lock = Lock()
_model_lock = Lock()


def _get_client() -> Any:
    global client
    if client is None:
        with _client_lock:
            if client is None:
                CHROMA_PATH.mkdir(parents=True, exist_ok=True)
                try:
                    chromadb = import_module("chromadb")
                except ModuleNotFoundError as exc:
                    raise RuntimeError(
                        "ChromaDB is required; install it with 'pip install chromadb'."
                    ) from exc
                client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client


def _get_embedding_model() -> Any:
    global embedding_model
    if embedding_model is None:
        with _model_lock:
            if embedding_model is None:
                model_name = os.getenv("CHROMA_EMBEDDING_MODEL", DEFAULT_MODEL_NAME)
                try:
                    sentence_transformers = import_module("sentence_transformers")
                except ModuleNotFoundError as exc:
                    raise RuntimeError(
                        "Sentence Transformers is required; install it with "
                        "'pip install sentence-transformers'."
                    ) from exc
                embedding_model = sentence_transformers.SentenceTransformer(model_name)
    return embedding_model


def _collection_name(corpus_id: int) -> str:
    if isinstance(corpus_id, bool) or not isinstance(corpus_id, int):
        raise TypeError("corpus_id must be an integer")
    if corpus_id <= 0:
        raise ValueError("corpus_id must be greater than zero")
    return f"corpus_{corpus_id}"


def get_collection(corpus_id: int) -> Any:
    """Return the corpus collection, creating it when necessary."""
    return _get_client().get_or_create_collection(
        name=_collection_name(corpus_id),
        metadata={"hnsw:space": "cosine"},
    )


def store_chunks(corpus_id: int, chunks: Sequence[str]) -> int:
    """Embed and store chunks, returning the number stored."""
    if isinstance(chunks, (str, bytes)):
        raise TypeError("chunks must be a sequence of strings, not a string")

    documents = list(chunks)
    if any(not isinstance(chunk, str) for chunk in documents):
        raise TypeError("every chunk must be a string")
    if any(not chunk.strip() for chunk in documents):
        raise ValueError("chunks cannot contain empty strings")
    if not documents:
        return 0

    embeddings = _get_embedding_model().encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    get_collection(corpus_id).add(
        ids=[f"chunk_{uuid4().hex}" for _ in documents],
        documents=documents,
        embeddings=embeddings.tolist(),
    )
    return len(documents)


def search_chunks(corpus_id: int, question: str, top_k: int = DEFAULT_TOP_K) -> Any:
    """Return the nearest stored chunks for a natural-language question."""
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    if not question.strip():
        raise ValueError("question cannot be empty")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    collection = get_collection(corpus_id)
    count = collection.count()
    if count == 0:
        return {
            "ids": [[]],
            "embeddings": None,
            "documents": [[]],
            "uris": None,
            "data": None,
            "metadatas": [[]],
            "distances": [[]],
            "included": ["metadatas", "documents", "distances"],
        }

    question_embedding = _get_embedding_model().encode(
        question.strip(),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=min(top_k, count),
    )


def delete_collection(corpus_id: int) -> bool:
    """Delete a corpus collection and report whether it existed."""
    name = _collection_name(corpus_id)
    chroma_client = _get_client()
    collection_names = {
        item if isinstance(item, str) else item.name
        for item in chroma_client.list_collections()
    }
    if name not in collection_names:
        return False

    chroma_client.delete_collection(name=name)
    return True
