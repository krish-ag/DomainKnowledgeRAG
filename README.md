# Domain Knowledge Copilot

A Retrieval-Augmented Generation (RAG) application that lets users upload PDF documents into named knowledge bases (corpora), embed them with vector search, and ask natural language questions grounded in those documents.

## Features

- **Multi-tenant corpora** — each user manages their own isolated knowledge bases
- **PDF ingestion** — upload PDFs up to 20 MB; text is extracted, chunked, and embedded automatically
- **Semantic search** — ChromaDB + Sentence Transformers power similarity-based retrieval
- **Generative answers** — Groq (Llama 3.3 70B) synthesises cited answers; falls back to extractive mode if no API key is set
- **Chat history** — every question/answer pair is persisted and queryable per corpus
- **JWT authentication** — PBKDF2-SHA256 password hashing, 60-minute token expiry
- **Session persistence** — Streamlit frontend keeps you logged in across page reloads via localStorage
- **Pluggable LLM** — swap Groq for any other provider by replacing the generator hook in `rag.py`

## Tech Stack

| Layer | Library |
|---|---|
| Frontend | Streamlit 1.58 + streamlit-local-storage |
| API framework | FastAPI 0.138 |
| Vector store | ChromaDB 1.5 |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| LLM | Groq API (Llama 3.3 70B) |
| PDF parsing | PyPDF |
| Database | SQLite |
| Server | Uvicorn |
| Python | 3.12 |

## Project Structure

```
.
├── app.py              # Streamlit frontend — auth flow, corpus picker, chat UI
├── main.py             # App factory, startup hooks, router registration
├── auth.py             # JWT creation/verification, password hashing
├── auth_routes.py      # POST /signup, POST /login
├── corpus_routes.py    # CRUD for /corpora
├── upload_routes.py    # POST /corpora/{id}/upload — PDF ingestion pipeline
├── query_routes.py     # POST /corpora/{id}/query, GET /corpora/{id}/chat-history
├── rag.py              # Core RAG: chunking, retrieval, prompt building, answer generation
├── chroma_utils.py     # ChromaDB collection management
├── database.py         # SQLite connection factory
├── create_tables.py    # Schema initialisation
├── models.py           # Dataclasses: User, Corpus, Document, ChatMessage
├── schemas.py          # Pydantic request/response schemas
└── requirements.txt    # Pinned dependencies
```

## Getting Started

### Prerequisites

- Python 3.12+
- pip

### Installation

```bash
git clone <repo-url>
cd Domain_Knowledge_Copilot

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

Set these environment variables (or rely on the listed defaults):

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | `change_this_to_a_long_secret_key` | JWT signing secret — **change in production** |
| `GROQ_API_KEY` | *(unset)* | Groq API key — enables generative answers; omit to use extractive fallback |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `CHROMA_DB_PATH` | `./chroma_db` | Directory for the ChromaDB vector store |
| `CHROMA_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformers model name |
| `MAX_UPLOAD_BYTES` | `20971520` (20 MB) | Maximum PDF upload size |

You can place these in a `.env` file at the project root — `python-dotenv` is included and will load it automatically.

### Run

Start the backend and frontend in separate terminals:

```bash
# Terminal 1 — FastAPI backend
python -m uvicorn main:app --reload
```

```bash
# Terminal 2 — Streamlit frontend
streamlit run app.py
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI backend | http://localhost:8000 |
| Interactive API docs | http://localhost:8000/docs |

## API Reference

### Authentication

| Method | Path | Description |
|---|---|---|
| `POST` | `/signup` | Create an account |
| `POST` | `/login` | Obtain a JWT token |

Pass the token as `Authorization: Bearer <token>` on all subsequent requests.

### Corpora

| Method | Path | Description |
|---|---|---|
| `POST` | `/corpora` | Create a new corpus |
| `GET` | `/corpora` | List your corpora |
| `GET` | `/corpora/{corpus_id}` | Get corpus details |
| `DELETE` | `/corpora/{corpus_id}` | Delete corpus and its vectors |

### Documents

| Method | Path | Description |
|---|---|---|
| `POST` | `/corpora/{corpus_id}/upload` | Upload a PDF (multipart/form-data) |

### Query

| Method | Path | Description |
|---|---|---|
| `POST` | `/corpora/{corpus_id}/query` | Ask a question; returns answer + sources |
| `GET` | `/corpora/{corpus_id}/chat-history` | Retrieve past Q&A (max 100) |

### Health

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service liveness check |

## How It Works

```
User uploads PDF
  └─ Text extracted (PyPDF)
  └─ Chunked (1 000 chars, 150-char overlap at word boundaries)
  └─ Embedded → 384-dim vectors (Sentence Transformers)
  └─ Stored in ChromaDB (one collection per corpus)

User asks a question
  └─ Question embedded with the same model
  └─ Top-5 similar chunks retrieved (cosine distance ≤ 1.5)
  └─ Context fitted within 12 000-char budget
  └─ Prompt assembled (context + question + source guardrails)
  └─ Answer generated (Groq/LLM) or extracted (fallback)
  └─ Q&A saved to chat history
```

## Connecting a Custom LLM

`rag.py` exposes a `configure_generator(callable)` hook called at startup in `main.py`. Replace the Groq client with any completion API:

```python
# main.py — swap in your own LLM
from rag import configure_generator

def my_generator(prompt: str) -> str:
    response = my_llm_client.complete(prompt)
    return response.text

configure_generator(my_generator)
```

If no generator is configured, the system returns the retrieved source chunks verbatim (extractive mode).

## Database Schema

```
users          (id, username, email, password, created_at)
corpora        (id, name, user_id → users CASCADE, created_at)
documents      (id, filename, corpus_id → corpora CASCADE, created_at)
chat_messages  (id, question, answer, user_id → users CASCADE, corpus_id → corpora CASCADE, created_at)
```

Removing a user cascades to all their corpora, documents, and chat history.

## Security Notes

- Rotate `SECRET_KEY` before deploying to production.
- CORS is currently open (`allow_origins=["*"]`); restrict to your frontend domain in production.
- PDF validation uses magic-byte checking (`%PDF-`) in addition to MIME type and size checks.
- Passwords are hashed with PBKDF2-SHA256 at 260 000 iterations.
