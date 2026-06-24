# models.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class User:
    id: int | None = None
    username: str = ""
    email: str = ""
    password: str = ""
    created_at: str | datetime | None = None


@dataclass(slots=True)
class Corpus:
    id: int | None = None
    name: str = ""
    user_id: int | None = None
    created_at: str | datetime | None = None


@dataclass(slots=True)
class Document:
    id: int | None = None
    filename: str = ""
    corpus_id: int | None = None
    created_at: str | datetime | None = None


@dataclass(slots=True)
class ChatMessage:
    id: int | None = None
    question: str = ""
    answer: str = ""
    user_id: int | None = None
    created_at: str | datetime | None = None


def model_from_row(model_class: type[Any], row: Any) -> Any:
    """Create a model instance from a sqlite3.Row or dictionary."""
    return model_class(**dict(row))
