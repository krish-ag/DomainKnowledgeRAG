# auth_routes.py

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, HTTPException, status

from auth import create_access_token, hash_password, verify_password
from database import get_connection
from schemas import MessageResponse, TokenResponse, UserCreate, UserLogin


router = APIRouter(tags=["Authentication"])


def _user_by_email(email: str) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute(
            "SELECT id, username, email, password FROM users WHERE email = ?",
            (email,),
        ).fetchone()


@router.post("/signup", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
def signup(user: UserCreate) -> dict[str, str]:
    if _user_by_email(user.email) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    with get_connection() as conn:
        try:
            conn.execute(
                """
                INSERT INTO users (username, email, password)
                VALUES (?, ?, ?)
                """,
                (user.username, user.email, hash_password(user.password)),
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username or email already registered",
            ) from exc

    return {"message": "User created successfully"}


@router.post("/login", response_model=TokenResponse)
def login(login_data: UserLogin) -> dict[str, str]:
    user = _user_by_email(login_data.email)

    if user is None or not verify_password(login_data.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    access_token = create_access_token(
        {
            "user_id": user["id"],
            "sub": user["email"],
        }
    )

    return {"access_token": access_token, "token_type": "bearer"}

