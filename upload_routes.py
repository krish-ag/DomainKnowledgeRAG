"""Authenticated PDF upload and ingestion routes."""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.security import OAuth2PasswordBearer
from starlette.concurrency import run_in_threadpool

from auth import verify_token
from database import get_db


# rag.py currently lives one directory above the application modules.
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag import process_document  # noqa: E402


router = APIRouter(prefix="/corpora", tags=["Upload"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
READ_SIZE = 1024 * 1024
ALLOWED_CONTENT_TYPES = {"application/pdf", "application/octet-stream"}


def current_user_id(token: Annotated[str, Depends(oauth2_scheme)]) -> int:
    """Resolve a bearer token to a numeric user ID."""
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
) -> None:
    corpus = db.execute(
        "SELECT 1 FROM corpora WHERE id = ? AND user_id = ?",
        (corpus_id, user_id),
    ).fetchone()
    if corpus is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Corpus not found",
        )


def _validate_upload(file: UploadFile) -> str:
    original_name = Path(file.filename or "").name.strip()
    if not original_name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="A filename is required",
        )
    if Path(original_name).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF documents are supported",
        )
    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Invalid PDF content type",
        )
    return original_name


async def _save_upload(file: UploadFile, destination: Path) -> int:
    """Stream an upload to disk while enforcing the configured size limit."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    first_chunk = True
    try:
        with destination.open("wb") as output:
            while chunk := await file.read(READ_SIZE):
                if first_chunk:
                    first_chunk = False
                    if not chunk.startswith(b"%PDF-"):
                        raise HTTPException(
                            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            detail="Uploaded content is not a valid PDF",
                        )
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"PDF exceeds the {MAX_UPLOAD_BYTES}-byte limit",
                    )
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    if total == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded PDF is empty",
        )
    return total


def _delete_document_record(db: sqlite3.Connection, document_id: int) -> None:
    try:
        db.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        db.commit()
    except sqlite3.DatabaseError:
        db.rollback()


@router.post("/{corpus_id}/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    corpus_id: int,
    file: Annotated[UploadFile, File(...)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    user_id: Annotated[int, Depends(current_user_id)],
) -> dict[str, Any]:
    """Persist and index a PDF belonging to an authenticated user's corpus."""
    _require_owned_corpus(db, corpus_id, user_id)
    original_name = _validate_upload(file)

    stored_name = f"{uuid4().hex}.pdf"
    destination = UPLOAD_DIR / str(corpus_id) / stored_name
    size = await _save_upload(file, destination)

    try:
        cursor = db.execute(
            "INSERT INTO documents (filename, corpus_id) VALUES (?, ?)",
            (original_name, corpus_id),
        )
        db.commit()
        document_id = int(cursor.lastrowid)
    except sqlite3.DatabaseError as exc:
        db.rollback()
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save document metadata",
        ) from exc

    try:
        result = await run_in_threadpool(
            process_document,
            corpus_id=corpus_id,
            pdf_path=destination,
        )
    except (FileNotFoundError, ValueError) as exc:
        _delete_document_record(db, document_id)
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        _delete_document_record(db, document_id)
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document indexing is unavailable",
        ) from exc
    except Exception as exc:
        _delete_document_record(db, document_id)
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process the uploaded document",
        ) from exc

    return {
        "message": "Document uploaded successfully",
        "document_id": document_id,
        "filename": original_name,
        "size_bytes": size,
        "chunks_stored": result["chunks_stored"],
        "pages_processed": result["pages_processed"],
    }
