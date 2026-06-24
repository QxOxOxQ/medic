from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID

from dotenv import dotenv_values
from fastapi import HTTPException, Request, Response, status
from sqlalchemy.orm import Session, sessionmaker

from rag.config import PROJECT_ROOT
from rag.database.repositories import UserRepository


SESSION_COOKIE_NAME = "medic_dashboard_session"
SESSION_TTL_SECONDS = 12 * 60 * 60
AUTH_ENV_NAMES = {
    "username": "MEDIC_DASHBOARD_USERNAME",
    "password": "MEDIC_DASHBOARD_PASSWORD",
    "session_secret": "MEDIC_SESSION_SECRET",
    "cookie_secure": "MEDIC_DASHBOARD_COOKIE_SECURE",
}


@dataclass(frozen=True)
class AuthSettings:
    username: str
    password: str
    session_secret: str
    cookie_secure: bool


@dataclass(frozen=True)
class SessionData:
    user_id: UUID
    username: str
    csrf_token: str
    expires_at: float


@dataclass(frozen=True)
class AuthenticatedUser:
    id: UUID
    username: str
    is_admin: bool


class AuthConfigurationError(ValueError):
    pass


def load_auth_settings() -> AuthSettings:
    dotenv_settings = dotenv_values(PROJECT_ROOT / ".env")

    def lookup(name: str) -> str | None:
        return os.getenv(name) or dotenv_settings.get(name)

    required_names = list(AUTH_ENV_NAMES.values())
    missing = [name for name in required_names if not lookup(name)]
    if missing:
        names = ", ".join(missing)
        raise AuthConfigurationError(f"Missing dashboard auth settings: {names}")

    return AuthSettings(
        username=lookup(AUTH_ENV_NAMES["username"]) or "",
        password=lookup(AUTH_ENV_NAMES["password"]) or "",
        session_secret=lookup(AUTH_ENV_NAMES["session_secret"]) or "",
        cookie_secure=_env_flag(lookup(AUTH_ENV_NAMES["cookie_secure"])),
    )


def credentials_are_valid(
    *,
    username: str,
    password: str,
    settings: AuthSettings,
) -> bool:
    """Legacy env-only credential check kept for callers outside DB-backed auth."""
    return hmac.compare_digest(username, settings.username) and hmac.compare_digest(
        password,
        settings.password,
    )


def create_session_cookie(
    settings: AuthSettings,
    *,
    user_id: UUID,
    username: str,
) -> str:
    session = {
        "user_id": str(user_id),
        "username": username,
        "csrf_token": secrets.token_urlsafe(32),
        "expires_at": time.time() + SESSION_TTL_SECONDS,
    }
    payload = _encode_json(session)
    signature = _sign(payload, settings.session_secret)
    return f"{payload}.{signature}"


def set_session_cookie(
    response: Response,
    settings: AuthSettings,
    *,
    user_id: UUID,
    username: str,
) -> None:
    response.set_cookie(
        SESSION_COOKIE_NAME,
        create_session_cookie(settings, user_id=user_id, username=username),
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
    )


def clear_session_cookie(response: Response, settings: AuthSettings) -> None:
    response.delete_cookie(
        SESSION_COOKIE_NAME,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
    )


def read_session(request: Request, settings: AuthSettings) -> SessionData | None:
    cookie = request.cookies.get(SESSION_COOKIE_NAME)
    if not cookie or "." not in cookie:
        return None

    payload, signature = cookie.rsplit(".", 1)
    expected_signature = _sign(payload, settings.session_secret)
    if not hmac.compare_digest(signature, expected_signature):
        return None

    try:
        data = _decode_json(payload)
        session = SessionData(
            user_id=UUID(str(data["user_id"])),
            username=str(data["username"]),
            csrf_token=str(data["csrf_token"]),
            expires_at=float(data["expires_at"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None

    if session.expires_at < time.time():
        return None
    return session


def require_session(request: Request, settings: AuthSettings) -> SessionData:
    session = read_session(request, settings)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    return session


def require_active_user(
    request: Request,
    settings: AuthSettings,
    session_factory: sessionmaker[Session],
) -> AuthenticatedUser:
    session_data = require_session(request, settings)
    with session_factory() as db_session:
        user = UserRepository(db_session).get_by_id(session_data.user_id)
        if user is None or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )
        return AuthenticatedUser(
            id=user.id,
            username=user.username,
            is_admin=user.is_admin,
        )


def verify_csrf(
    request: Request,
    settings: AuthSettings,
    *,
    token: str | None = None,
) -> SessionData:
    session = require_session(request, settings)
    submitted_token = token or request.headers.get("x-csrf-token")
    if not submitted_token or not hmac.compare_digest(
        submitted_token,
        session.csrf_token,
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid CSRF token",
        )
    return session


def _env_flag(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _sign(payload: str, secret: str) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _encode_json(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_json(payload: str) -> dict[str, Any]:
    padding = "=" * (-len(payload) % 4)
    raw = base64.urlsafe_b64decode(f"{payload}{padding}".encode("ascii"))
    return cast(dict[str, Any], json.loads(raw.decode("utf-8")))
