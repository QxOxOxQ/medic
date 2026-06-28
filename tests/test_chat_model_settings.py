from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from dashboard.app import create_app
from dashboard.auth import AuthSettings
from rag.database.migrations import upgrade_database
from rag.database.repositories import UserRepository
from rag.database.session import create_database_engine


def _session_factory(tmp_path: Path) -> sessionmaker:
    database_url = f"sqlite:///{tmp_path / 'settings.db'}"
    upgrade_database(database_url)
    engine = create_database_engine(database_url)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with factory() as session:
        UserRepository(session).seed_admin(username="admin", password="secret")
        session.commit()
    return factory


def _client(session_factory: sessionmaker) -> TestClient:
    app = create_app(
        auth_settings=AuthSettings(
            username="admin",
            password="secret",
            session_secret="test-session-secret",
            cookie_secure=False,
        ),
        database_session_factory=session_factory,
    )
    return TestClient(app)


def _login(client: TestClient) -> None:
    response = client.post(
        "/login",
        data={"username": "admin", "password": "secret"},
        follow_redirects=False,
    )
    assert response.status_code == 303


def _csrf_token(client: TestClient) -> str:
    response = client.get("/")
    match = re.search(r'name="csrf-token" content="([^"]+)"', response.text)
    assert match is not None
    return match.group(1)


def test_chat_model_setting_defaults_to_openai(tmp_path: Path) -> None:
    client = _client(_session_factory(tmp_path))
    _login(client)

    response = client.get("/api/settings/chat-model")

    assert response.status_code == 200
    body = response.json()
    assert body["selected"] == "openai"
    assert {option["key"] for option in body["options"]} == {
        "deepseek",
        "deepseek-v4",
        "openai",
        "gemini",
        "claude-opus",
    }


def test_chat_model_setting_is_persisted_per_user(tmp_path: Path) -> None:
    session_factory = _session_factory(tmp_path)
    client = _client(session_factory)
    _login(client)
    csrf_token = _csrf_token(client)

    updated = client.put(
        "/api/settings/chat-model",
        json={"key": "deepseek"},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert updated.status_code == 200
    assert updated.json()["selected"] == "deepseek"
    assert client.get("/api/settings/chat-model").json()["selected"] == "deepseek"

    with session_factory() as session:
        user = UserRepository(session).get_by_username("admin")
        assert user is not None
        assert user.preferred_chat_model == "deepseek"


def test_chat_model_setting_rejects_unknown_model(tmp_path: Path) -> None:
    client = _client(_session_factory(tmp_path))
    _login(client)
    csrf_token = _csrf_token(client)

    response = client.put(
        "/api/settings/chat-model",
        json={"key": "claude"},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 400


def test_chat_model_setting_requires_csrf(tmp_path: Path) -> None:
    client = _client(_session_factory(tmp_path))
    _login(client)

    response = client.put("/api/settings/chat-model", json={"key": "gemini"})

    assert response.status_code == 403
