# main.py

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer

from auth import create_access_token, hash_password, verify_password, verify_token
from create_tables import create_tables
from database import get_connection
from rag import configure_generator, process_document
from schemas import CorpusCreate, UserCreate, UserLogin
from dotenv import load_dotenv
from query_routes import (
    router as query_router
)

load_dotenv()
UPLOAD_DIR = Path(__file__).with_name("uploads")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

app = FastAPI(
    title="Domain Knowledge Co-Pilot",
    description="RAG based document question answering system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _make_groq_generator():
    import os
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from groq import Groq
    except ModuleNotFoundError:
        return None

    client = Groq(api_key=api_key)

    def generator(prompt: str) -> str:
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        return response.choices[0].message.content

    return generator


def startup() -> None:
    create_tables()
    UPLOAD_DIR.mkdir(exist_ok=True)
    gen = _make_groq_generator()
    if gen:
        configure_generator(gen)
        print("LLM: Groq generator configured.")
    else:
        print("LLM: No GROQ_API_KEY set — running in extractive (source-only) mode.")


@app.on_event("startup")
def on_startup() -> None:
    startup()


def create_user(user: UserCreate) -> dict[str, Any]:
    with get_connection() as conn:
        try:
            cursor = conn.execute(
                """
                INSERT INTO users (username, email, password)
                VALUES (?, ?, ?)
                """,
                (user.username, user.email, hash_password(user.password)),
            )
            conn.commit()
        except Exception as exc:
            if "unique" in str(exc).lower():
                raise ValueError("Username or email already exists") from exc
            raise

    return {
        "id": cursor.lastrowid,
        "username": user.username,
        "email": user.email,
    }


def authenticate_user(credentials: UserLogin) -> dict[str, Any] | None:
    with get_connection() as conn:
        user = conn.execute(
            "SELECT id, username, email, password FROM users WHERE email = ?",
            (credentials.email,),
        ).fetchone()

    if user is None or not verify_password(credentials.password, user["password"]):
        return None

    return dict(user)


def create_user_corpus(user_id: int, corpus: CorpusCreate) -> dict[str, Any]:
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO corpora (name, user_id) VALUES (?, ?)",
            (corpus.name, user_id),
        )
        conn.commit()

    return {"id": cursor.lastrowid, "name": corpus.name}


def list_user_corpora(user_id: int) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, name FROM corpora WHERE user_id = ? ORDER BY id DESC",
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def save_document(corpus_id: int, filename: str, content: bytes) -> dict[str, Any]:
    corpus_dir = UPLOAD_DIR / str(corpus_id)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    file_path = corpus_dir / Path(filename).name
    file_path.write_bytes(content)

    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO documents (filename, corpus_id) VALUES (?, ?)",
            (file_path.name, corpus_id),
        )
        conn.commit()

    return {"id": cursor.lastrowid, "filename": file_path.name}



def current_user_id(token: str = Depends(oauth2_scheme)) -> int:
    user_id = verify_token(token)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    return int(user_id)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Domain Knowledge Co-Pilot Running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/signup", status_code=status.HTTP_201_CREATED)
def signup(user: UserCreate) -> dict[str, Any]:
    try:
        created_user = create_user(user)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    return {"message": "User created", "user": created_user}


@app.post("/login")
def login(credentials: UserLogin) -> dict[str, str]:
    user = authenticate_user(credentials)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    token = create_access_token({"user_id": user["id"], "sub": user["email"]})
    return {"access_token": token, "token_type": "bearer"}


@app.post("/corpora", status_code=status.HTTP_201_CREATED)
def create_corpus(
        corpus: CorpusCreate,
        user_id: int = Depends(current_user_id),
) -> dict[str, Any]:
    return create_user_corpus(user_id, corpus)


@app.get("/corpora")
def get_corpora(user_id: int = Depends(current_user_id)) -> list[dict[str, Any]]:
    return list_user_corpora(user_id)


@app.delete("/corpora/{corpus_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_corpus(corpus_id: int, user_id: int = Depends(current_user_id)) -> None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id FROM corpora WHERE id = ? AND user_id = ?",
            (corpus_id, user_id),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Corpus not found")
        conn.execute("DELETE FROM corpora WHERE id = ?", (corpus_id,))
        conn.commit()

    from chroma_utils import delete_collection
    delete_collection(corpus_id)


@app.post("/corpora/{corpus_id}/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
        corpus_id: int,
        file: Annotated[UploadFile, File(...)],
        user_id: int = Depends(current_user_id),
) -> dict[str, Any]:
    del user_id
    filename = file.filename or "document.pdf"
    content = await file.read()
    doc = save_document(corpus_id, filename, content)

    file_path = UPLOAD_DIR / str(corpus_id) / Path(filename).name
    try:
        stats = process_document(corpus_id, file_path)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    doc.update(stats)
    return doc



if __name__ == "__main__":
    startup()
    print("Run with: python -m uvicorn main:app --reload")

app.include_router(
    query_router
)
