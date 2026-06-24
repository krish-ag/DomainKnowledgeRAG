"""Authenticated routes for corpus questions and chat history."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import OAuth2PasswordBearer


# The current project keeps its application modules in a directory with spaces.
APP_DIR = Path(__file__).resolve().parent / "Domain knowledge co pilot"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from auth import verify_token  # noqa: E402
from database import get_db  # noqa: E402
from schemas import ChatMessageResponse, QueryRequest, QueryResponse  # noqa: E402

from rag import generate_answer


router = APIRouter(prefix="/corpora", tags=["Query"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def current_user_id(token: Annotated[str, Depends(oauth2_scheme)]) -> int:
    """Resolve the authenticated user ID or return a consistent 401 response."""
    user_id = verify_token(token)
    try:
        if user_id is None:
            raise ValueError
        return int(user_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def _require_owned_corpus(
    db: sqlite3.Connection,
    corpus_id: int,
    user_id: int,
) -> sqlite3.Row:
    corpus = db.execute(
        "SELECT id, name FROM corpora WHERE id = ? AND user_id = ?",
        (corpus_id, user_id),
    ).fetchone()
    if corpus is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Corpus not found",
        )
    return corpus


@router.post("/{corpus_id}/query", response_model=QueryResponse)
def query_corpus(
    corpus_id: int,
    query: QueryRequest,
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    user_id: Annotated[int, Depends(current_user_id)],
) -> dict[str, str]:
    """Answer a question from an owned corpus and persist the exchange."""
    _require_owned_corpus(db, corpus_id, user_id)
    question = query.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question cannot be empty",
        )

    try:
        result = generate_answer(corpus_id=corpus_id, question=question)
        answer = result["answer"]
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The knowledge search service is unavailable",
        ) from exc

    if not isinstance(answer, str) or not answer.strip():
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The answer service returned an invalid response",
        )

    try:
        db.execute(
            """
            INSERT INTO chat_messages (question, answer, user_id)
            VALUES (?, ?, ?)
            """,
            (question, answer.strip(), user_id),
        )
        db.commit()
    except sqlite3.DatabaseError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save chat history",
        ) from exc

    return {"answer": answer.strip()}


@router.get(
    "/{corpus_id}/chat-history",
    response_model=list[ChatMessageResponse],
)
def get_chat_history(
    corpus_id: int,
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    user_id: Annotated[int, Depends(current_user_id)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> list[dict[str, Any]]:
    """Return the authenticated user's newest chat messages."""
    _require_owned_corpus(db, corpus_id, user_id)
    rows = db.execute(
        """
        SELECT id, question, answer, created_at
        FROM chat_messages
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (user_id, limit),
    ).fetchall()
    return [dict(row) for row in rows]
