# corpus_routes.py

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from auth import verify_token
from database import get_connection
from schemas import CorpusCreate, CorpusResponse, MessageResponse


router = APIRouter(prefix="/corpora", tags=["Corpora"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def current_user_id(token: str = Depends(oauth2_scheme)) -> int:
    user_id = verify_token(token)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    try:
        return int(user_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token user",
        ) from exc


def _corpus_by_id(corpus_id: int, user_id: int) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT id, name
            FROM corpora
            WHERE id = ? AND user_id = ?
            """,
            (corpus_id, user_id),
        ).fetchone()


@router.post("", response_model=CorpusResponse, status_code=status.HTTP_201_CREATED)
def create_corpus(
    corpus: CorpusCreate,
    user_id: int = Depends(current_user_id),
) -> dict[str, Any]:
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO corpora (name, user_id) VALUES (?, ?)",
            (corpus.name, user_id),
        )
        conn.commit()

    return {"id": cursor.lastrowid, "name": corpus.name}


@router.get("", response_model=list[CorpusResponse])
def get_corpora(user_id: int = Depends(current_user_id)) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, name
            FROM corpora
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (user_id,),
        ).fetchall()

    return [dict(row) for row in rows]


@router.get("/{corpus_id}", response_model=CorpusResponse)
def get_corpus(
    corpus_id: int,
    user_id: int = Depends(current_user_id),
) -> dict[str, Any]:
    corpus = _corpus_by_id(corpus_id, user_id)
    if corpus is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Corpus not found",
        )

    return dict(corpus)


@router.delete("/{corpus_id}", response_model=MessageResponse)
def delete_corpus(
    corpus_id: int,
    user_id: int = Depends(current_user_id),
) -> dict[str, str]:
    if _corpus_by_id(corpus_id, user_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Corpus not found",
        )

    with get_connection() as conn:
        conn.execute(
            "DELETE FROM corpora WHERE id = ? AND user_id = ?",
            (corpus_id, user_id),
        )
        conn.commit()

    return {"message": "Corpus deleted successfully"}
