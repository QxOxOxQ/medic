from __future__ import annotations

import re
import time
import uuid
from datetime import datetime
from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sqlalchemy.orm import sessionmaker

import rag.config as settings_module
from agents.models import AgentAnswer
from backend.chat_use_cases import ChatConversationUseCase
from backend.use_cases import AnswerQuestionUseCase
from dashboard.app import create_app
from dashboard.auth import AuthConfigurationError, AuthSettings, load_auth_settings
from dashboard.documents import qdrant_index_preview_for_content_hash
from dashboard.jobs import JobStore
from rag.config import DocumentPreparationSettings
from rag.database import DocumentRepository, UserRepository
from rag.database.chat_store import SqlAlchemyChatConversationStore
from rag.database.migrations import upgrade_database
from rag.database.security import verify_password
from rag.database.session import create_database_engine
from rag.document_preparation import calculate_text_sha256, prepare_documents
from rag.full_process import FullProcess


ADMIN_ORIGIN_HEADERS = {"origin": "http://testserver"}


class _SearchServiceStub:
    def search(self, **_: object) -> list[dict[str, object]]:
        return []


class _AgentRunnerStub:
    def answer(self, request: object) -> AgentAnswer:
        return AgentAnswer(
            answer="Stub answer.",
            agents=(),
            sources=(),
            insufficient_context=False,
        )


def _settings(tmp_path: Path) -> DocumentPreparationSettings:
    return DocumentPreparationSettings(
        raw_documents_dir=tmp_path / "data" / "raw",
        parsed_markdown_dir=tmp_path / "data" / "parsed",
    )


def _client(
    tmp_path: Path,
    monkeypatch,
    *,
    job_store: JobStore | None = None,
) -> tuple[TestClient, DocumentPreparationSettings]:
    _isolate_external_env(tmp_path, monkeypatch)
    document_settings = _settings(tmp_path)
    session_factory = _database_session_factory(tmp_path)
    answer_use_case = AnswerQuestionUseCase(
        agent_runner_factory=lambda **_: _AgentRunnerStub(),
    )
    chat_use_case = ChatConversationUseCase(
        agent_runner_factory=lambda **_: _AgentRunnerStub(),
        conversation_store=SqlAlchemyChatConversationStore(session_factory),
    )
    app = create_app(
        auth_settings=AuthSettings(
            username="admin",
            password="secret",
            session_secret="test-session-secret",
            cookie_secure=False,
        ),
        document_settings=document_settings,
        job_store=job_store,
        database_session_factory=session_factory,
        answer_question_use_case=answer_use_case,
        chat_conversation_use_case=chat_use_case,
        search_service=_SearchServiceStub(),
    )
    return TestClient(app), document_settings


def test_load_auth_settings_requires_cookie_secure(
    tmp_path,
    monkeypatch,
) -> None:
    _isolate_external_env(tmp_path, monkeypatch)
    monkeypatch.setattr("dashboard.auth.PROJECT_ROOT", tmp_path)
    (tmp_path / ".env").write_text(
        "MEDIC_DASHBOARD_USERNAME=admin\n"
        "MEDIC_DASHBOARD_PASSWORD=secret\n"
        "MEDIC_SESSION_SECRET=session-secret\n",
        encoding="utf-8",
    )

    try:
        load_auth_settings()
    except AuthConfigurationError as error:
        assert "MEDIC_DASHBOARD_COOKIE_SECURE" in str(error)
    else:
        raise AssertionError("Expected missing cookie secure config to fail")


def test_healthcheck_is_public(tmp_path: Path, monkeypatch) -> None:
    client, _ = _client(tmp_path, monkeypatch)

    response = client.get("/healthz")

    assert response.status_code == 204
    assert response.content == b""


def _isolate_external_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
    for env_name in settings_module.SETTINGS["env"].values():
        monkeypatch.delenv(env_name, raising=False)


def _database_session_factory(tmp_path: Path) -> sessionmaker:
    database_url = f"sqlite:///{tmp_path / 'dashboard.db'}"
    upgrade_database(database_url)
    engine = create_database_engine(database_url)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with factory() as session:
        UserRepository(session).seed_admin(username="admin", password="secret")
        session.commit()
    return factory


def _seed_document_record(
    client: TestClient,
    *,
    relative_raw_path: str,
    original_filename: str | None = None,
    parsed_markdown_path: str | None = None,
    content_hash: str | None = None,
    byte_size: int | None = None,
    processed_at: str | None = None,
    status: str = "raw",
    processing_error: str | None = None,
) -> None:
    with client.app.state.database_session_factory() as session:
        user = UserRepository(session).get_by_username("admin")
        assert user is not None
        repository = DocumentRepository(session)
        if status == "failed":
            repository.mark_processing_failed(
                owner_user_id=user.id,
                relative_raw_path=relative_raw_path,
                original_filename=original_filename or Path(relative_raw_path).name,
                byte_size=byte_size,
                processed_at=_datetime_from_iso_z(processed_at),
                processing_error=processing_error or "Processing failed",
            )
        elif parsed_markdown_path is None and content_hash is None:
            repository.create_uploaded_document(
                owner_user_id=user.id,
                original_filename=original_filename or Path(relative_raw_path).name,
                relative_raw_path=relative_raw_path,
                byte_size=byte_size or 0,
            )
        else:
            repository.upsert_prepared_document(
                owner_user_id=user.id,
                relative_raw_path=relative_raw_path,
                original_filename=original_filename or Path(relative_raw_path).name,
                parsed_markdown_path=parsed_markdown_path,
                content_hash=content_hash,
                byte_size=byte_size,
                processed_at=_datetime_from_iso_z(processed_at),
                status=status,
            )
        session.commit()


def _datetime_from_iso_z(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _login(
    client: TestClient,
    *,
    username: str = "admin",
    password: str = "secret",
) -> None:
    response = client.post(
        "/login",
        data={"username": username, "password": password},
        follow_redirects=False,
    )
    assert response.status_code == 303


def _csrf_token(client: TestClient) -> str:
    response = client.get("/")
    assert response.status_code == 200
    match = re.search(r'name="csrf-token" content="([^"]+)"', response.text)
    assert match is not None
    return match.group(1)


def _create_pdf(pdf_path: Path, text: str) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    document.save(pdf_path)
    document.close()


def test_dashboard_redirects_to_login_without_session(tmp_path, monkeypatch) -> None:
    client, _ = _client(tmp_path, monkeypatch)

    response = client.get("/", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_dashboard_login_rejects_invalid_credentials_and_sets_session(
    tmp_path,
    monkeypatch,
) -> None:
    client, _ = _client(tmp_path, monkeypatch)

    invalid = client.post(
        "/login",
        data={"username": "admin", "password": "wrong"},
    )
    valid = client.post(
        "/login",
        data={"username": "admin", "password": "secret"},
        follow_redirects=False,
    )

    assert invalid.status_code == 401
    assert valid.status_code == 303
    assert "medic_dashboard_session=" in valid.headers["set-cookie"]
    assert "HttpOnly" in valid.headers["set-cookie"]


def test_admin_redirects_to_admin_login_without_session(tmp_path, monkeypatch) -> None:
    client, _ = _client(tmp_path, monkeypatch)

    response = client.get("/admin/", follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["location"].endswith("/admin/login")


def test_admin_rejects_logged_in_non_admin_user(tmp_path, monkeypatch) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    with client.app.state.database_session_factory() as session:
        UserRepository(session).create_user(username="viewer", password="secret")
        session.commit()
    _login(client, username="viewer", password="secret")

    response = client.get("/admin/", follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["location"].endswith("/admin/login")


def test_admin_dashboard_allows_active_admin_user(tmp_path, monkeypatch) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    _login(client)

    response = client.get("/admin/")
    rendered_dashboard = client.get("/")

    assert response.status_code == 200
    assert "Medic Admin" in response.text
    assert 'href="/admin"' in rendered_dashboard.text


def test_admin_accepts_sqladmin_login_form(tmp_path, monkeypatch) -> None:
    client, _ = _client(tmp_path, monkeypatch)

    response = client.post(
        "/admin/login",
        data={"username": "admin", "password": "secret"},
        headers=ADMIN_ORIGIN_HEADERS,
        follow_redirects=False,
    )
    admin_response = client.get("/admin/")

    assert response.status_code == 302
    assert "medic_sqladmin_session=" in response.headers["set-cookie"]
    assert admin_response.status_code == 200
    assert "Medic Admin" in admin_response.text


def test_admin_renders_user_create_and_edit_forms(tmp_path, monkeypatch) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    with client.app.state.database_session_factory() as session:
        user = UserRepository(session).create_user(username="editor", password="old")
        user_id = user.id
        session.commit()
    _login(client)

    create_response = client.get("/admin/user/create")
    edit_response = client.get(f"/admin/user/edit/{user_id}")

    assert create_response.status_code == 200
    assert edit_response.status_code == 200
    assert "Password" in create_response.text
    assert "$argon2" not in edit_response.text


def test_admin_creates_user_with_hashed_password(tmp_path, monkeypatch) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    _login(client)

    response = client.post(
        "/admin/user/create",
        data={
            "username": "NewUser",
            "password_hash": "new-secret",
            "is_active": "y",
            "is_admin": "y",
            "save": "Save",
        },
        headers=ADMIN_ORIGIN_HEADERS,
        follow_redirects=False,
    )

    assert response.status_code == 302
    with client.app.state.database_session_factory() as session:
        user = UserRepository(session).get_by_username("newuser")
        assert user is not None
        assert user.password_hash != "new-secret"
        assert verify_password("new-secret", user.password_hash)
        assert user.is_active is True
        assert user.is_admin is True


def test_admin_preserves_user_password_when_edit_password_is_empty(
    tmp_path,
    monkeypatch,
) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    with client.app.state.database_session_factory() as session:
        user = UserRepository(session).create_user(username="editor", password="old")
        user_id = user.id
        old_hash = user.password_hash
        session.commit()
    _login(client)

    response = client.post(
        f"/admin/user/edit/{user_id}",
        data={
            "username": "Editor",
            "password_hash": "",
            "is_active": "y",
            "save": "Save",
        },
        headers=ADMIN_ORIGIN_HEADERS,
        follow_redirects=False,
    )

    assert response.status_code == 302
    with client.app.state.database_session_factory() as session:
        user = UserRepository(session).get_by_id(user_id)
        assert user is not None
        assert user.username == "editor"
        assert user.password_hash == old_hash
        assert verify_password("old", user.password_hash)
        assert user.is_admin is False


def test_admin_updates_user_password_when_edit_password_is_present(
    tmp_path,
    monkeypatch,
) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    with client.app.state.database_session_factory() as session:
        user = UserRepository(session).create_user(username="editor", password="old")
        user_id = user.id
        old_hash = user.password_hash
        session.commit()
    _login(client)

    response = client.post(
        f"/admin/user/edit/{user_id}",
        data={
            "username": "editor",
            "password_hash": "new",
            "is_active": "y",
            "save": "Save",
        },
        headers=ADMIN_ORIGIN_HEADERS,
        follow_redirects=False,
    )

    assert response.status_code == 302
    with client.app.state.database_session_factory() as session:
        user = UserRepository(session).get_by_id(user_id)
        assert user is not None
        assert user.password_hash != old_hash
        assert verify_password("new", user.password_hash)
        assert UserRepository(session).authenticate(
            username="editor",
            password="new",
        ) is not None


def test_admin_rejects_mutation_without_same_origin_header(
    tmp_path,
    monkeypatch,
) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    _login(client)

    response = client.post(
        "/admin/user/create",
        data={
            "username": "blocked",
            "password_hash": "secret",
            "is_active": "y",
            "is_admin": "y",
            "save": "Save",
        },
        follow_redirects=False,
    )

    assert response.status_code == 403
    with client.app.state.database_session_factory() as session:
        assert UserRepository(session).get_by_username("blocked") is None


def test_admin_blocks_demoting_last_active_admin(tmp_path, monkeypatch) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    _login(client)
    with client.app.state.database_session_factory() as session:
        admin = UserRepository(session).get_by_username("admin")
        assert admin is not None
        admin_id = admin.id

    response = client.post(
        f"/admin/user/edit/{admin_id}",
        data={
            "username": "admin",
            "password_hash": "",
            "is_active": "y",
            "save": "Save",
        },
        headers=ADMIN_ORIGIN_HEADERS,
        follow_redirects=False,
    )

    assert response.status_code == 400
    assert "At least one active admin user is required." in response.text
    with client.app.state.database_session_factory() as session:
        admin = UserRepository(session).get_by_id(admin_id)
        assert admin is not None
        assert admin.is_admin is True
        assert admin.is_active is True


def test_admin_blocks_deactivating_last_active_admin(tmp_path, monkeypatch) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    _login(client)
    with client.app.state.database_session_factory() as session:
        admin = UserRepository(session).get_by_username("admin")
        assert admin is not None
        admin_id = admin.id

    response = client.post(
        f"/admin/user/edit/{admin_id}",
        data={
            "username": "admin",
            "password_hash": "",
            "is_admin": "y",
            "save": "Save",
        },
        headers=ADMIN_ORIGIN_HEADERS,
        follow_redirects=False,
    )

    assert response.status_code == 400
    assert "At least one active admin user is required." in response.text
    with client.app.state.database_session_factory() as session:
        admin = UserRepository(session).get_by_id(admin_id)
        assert admin is not None
        assert admin.is_admin is True
        assert admin.is_active is True


def test_admin_blocks_deleting_last_active_admin(tmp_path, monkeypatch) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    _login(client)
    with client.app.state.database_session_factory() as session:
        admin = UserRepository(session).get_by_username("admin")
        assert admin is not None
        admin_id = admin.id

    response = client.delete(
        f"/admin/user/delete?pks={admin_id}",
        headers=ADMIN_ORIGIN_HEADERS,
        follow_redirects=False,
    )

    assert response.status_code == 200
    with client.app.state.database_session_factory() as session:
        assert UserRepository(session).get_by_id(admin_id) is not None


def test_login_page_ignores_session_for_inactive_database_user(
    tmp_path,
    monkeypatch,
) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    _login(client)
    with client.app.state.database_session_factory() as session:
        user = UserRepository(session).get_by_username("admin")
        assert user is not None
        user.is_active = False
        session.commit()

    response = client.get("/login", follow_redirects=False)

    assert response.status_code == 200
    assert "Invalid username or password." not in response.text


def test_dashboard_rejects_mutation_without_csrf(tmp_path, monkeypatch) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    _login(client)

    response = client.post("/api/jobs/ingest")

    assert response.status_code == 403


def test_dashboard_has_no_language_preferences_endpoint(tmp_path, monkeypatch) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    _login(client)
    csrf_token = _csrf_token(client)

    response = client.put(
        "/api/user" + "/preferences",
        json={"removed" + "_language": "en"},
        headers={"X-CSRF-Token": csrf_token},
    )
    rendered = client.get("/")

    assert response.status_code == 404
    assert '<html lang="en">' in rendered.text
    assert "language" + "-select" not in rendered.text
    assert "preferred" + "_language" not in rendered.text


def test_dashboard_renders_process_detail_panel(tmp_path, monkeypatch) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    _login(client)

    response = client.get("/")

    assert response.status_code == 200
    assert "Process details" in response.text
    assert "Source PDFs" in response.text
    assert "Documents in database" in response.text
    assert '<th data-i18n="documents.status">Status</th>' in response.text
    assert '<th data-i18n="common.index">Index</th>' in response.text
    assert '<th data-i18n="common.actions">Actions</th>' in response.text
    assert '<th data-i18n="documents.select">Select</th>' in response.text
    assert 'id="select-all-documents"' in response.text
    assert "Mark all" in response.text
    assert 'id="delete-selected"' in response.text
    assert (
        'id="pdf-file" name="file" type="file" accept="application/pdf,.pdf" '
        "multiple required"
    ) in response.text
    assert 'data-process-tab="markdown"' in response.text
    assert 'data-process-tab="chunks"' in response.text
    assert 'data-process-tab="index"' in response.text
    assert 'id="process-error"' in response.text
    assert 'data-step="embed"' in response.text
    assert "Medical agent" in response.text
    assert "Test RAG" not in response.text
    assert 'id="chat-form"' in response.text
    assert 'id="chat-question"' in response.text
    assert 'id="chat-state"' in response.text
    assert 'id="chat-history"' in response.text
    assert 'id="chat-sources"' in response.text
    assert "Pipeline history" in response.text
    assert "/static/i18n.js" in response.text
    assert "/static/chat.js" in response.text
    assert "/static/search.js" not in response.text
    assert "/static/job-events.js" in response.text
    assert "/static/job-history.js" in response.text


def test_dashboard_refresh_dashboard_uses_global_app_reference() -> None:
    script = (
        Path(__file__).parents[1] / "dashboard" / "static" / "dashboard.js"
    ).read_text(encoding="utf-8")
    match = re.search(
        r"async refreshDashboard\(\) \{\n(?P<body>.*?)\n  \},",
        script,
        re.DOTALL,
    )

    assert match is not None
    body = match.group("body")
    assert "const app = window.MedicDashboard;" in body
    assert "this.documents" not in body
    assert "this.jobHistory" not in body


def test_dashboard_has_no_language_preferences_script() -> None:
    script = (
        Path(__file__).parents[1] / "dashboard" / "static" / "dashboard.js"
    ).read_text(encoding="utf-8")

    assert "save" + "Language" not in script
    assert "/api/user" + "/preferences" not in script
    assert "language" + "Select" not in script


def test_dashboard_uploads_pdf_and_rejects_non_pdf(tmp_path, monkeypatch) -> None:
    client, settings = _client(tmp_path, monkeypatch)
    _login(client)
    csrf_token = _csrf_token(client)

    uploaded = client.post(
        "/api/documents/upload",
        data={"csrf_token": csrf_token},
        files={"file": ("report.pdf", b"%PDF-1.4\n", "application/pdf")},
    )
    rejected = client.post(
        "/api/documents/upload",
        data={"csrf_token": csrf_token},
        files={"file": ("report.txt", b"plain text", "text/plain")},
    )
    documents = client.get("/api/documents")

    assert uploaded.status_code == 200
    upload_path = uploaded.json()["uploads"][0]["relative_raw_path"]
    assert upload_path.endswith("/report.pdf")
    assert (settings.raw_documents_dir / upload_path).read_bytes() == b"%PDF-1.4\n"
    assert rejected.status_code == 400
    assert documents.json()["documents"][0]["relative_raw_path"] == upload_path


def test_dashboard_uploads_multiple_pdfs(tmp_path, monkeypatch) -> None:
    client, settings = _client(tmp_path, monkeypatch)
    _login(client)
    csrf_token = _csrf_token(client)

    response = client.post(
        "/api/documents/upload",
        data={"csrf_token": csrf_token},
        files=[
            ("file", ("report.pdf", b"%PDF-1.4\nreport", "application/pdf")),
            ("file", ("summary.pdf", b"%PDF-1.4\nsummary", "application/pdf")),
        ],
    )
    documents = client.get("/api/documents")

    assert response.status_code == 200
    payload = response.json()
    assert payload["uploaded_count"] == 2
    uploads_by_name = {
        Path(upload["relative_raw_path"]).name: upload
        for upload in payload["uploads"]
    }
    report_path = uploads_by_name["report.pdf"]["relative_raw_path"]
    summary_path = uploads_by_name["summary.pdf"]["relative_raw_path"]
    assert (settings.raw_documents_dir / report_path).read_bytes() == (
        b"%PDF-1.4\nreport"
    )
    assert (settings.raw_documents_dir / summary_path).read_bytes() == (
        b"%PDF-1.4\nsummary"
    )
    assert {
        document["relative_raw_path"] for document in documents.json()["documents"]
    } == {report_path, summary_path}


def test_dashboard_delete_removes_raw_parsed_and_database_record(
    tmp_path,
    monkeypatch,
) -> None:
    client, settings = _client(tmp_path, monkeypatch)
    raw_path = settings.raw_documents_dir / "report.pdf"
    parsed_path = settings.parsed_markdown_dir / "report.md"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(b"%PDF-1.4\n")
    parsed_path.write_text("parsed", encoding="utf-8")
    _seed_document_record(
        client,
        relative_raw_path="report.pdf",
        parsed_markdown_path="report.md",
        content_hash="hash",
        byte_size=len(b"%PDF-1.4\n"),
        processed_at="2026-06-02T10:40:22Z",
        status="prepared",
    )
    _login(client)
    csrf_token = _csrf_token(client)

    response = client.post(
        "/api/documents/delete",
        data={"csrf_token": csrf_token, "relative_raw_path": "report.pdf"},
    )

    assert response.status_code == 200
    assert not raw_path.exists()
    assert not parsed_path.exists()
    with client.app.state.database_session_factory() as session:
        assert DocumentRepository(session).get_by_relative_raw_path("report.pdf") is None
    assert response.json()["qdrant_cleanup"]["attempted"] is True


def test_dashboard_delete_selected_removes_only_selected_documents(
    tmp_path,
    monkeypatch,
) -> None:
    client, settings = _client(tmp_path, monkeypatch)
    selected_raw = settings.raw_documents_dir / "selected.pdf"
    kept_raw = settings.raw_documents_dir / "kept.pdf"
    selected_parsed = settings.parsed_markdown_dir / "selected.md"
    kept_parsed = settings.parsed_markdown_dir / "kept.md"
    selected_raw.parent.mkdir(parents=True, exist_ok=True)
    selected_parsed.parent.mkdir(parents=True, exist_ok=True)
    selected_raw.write_bytes(b"%PDF-1.4\n")
    kept_raw.write_bytes(b"%PDF-1.4\n")
    selected_parsed.write_text("selected", encoding="utf-8")
    kept_parsed.write_text("kept", encoding="utf-8")
    _seed_document_record(
        client,
        relative_raw_path="selected.pdf",
        parsed_markdown_path="selected.md",
        content_hash="selected-hash",
        byte_size=len(b"%PDF-1.4\n"),
        processed_at="2026-06-02T10:40:22Z",
        status="prepared",
    )
    _seed_document_record(
        client,
        relative_raw_path="kept.pdf",
        parsed_markdown_path="kept.md",
        content_hash="kept-hash",
        byte_size=len(b"%PDF-1.4\n"),
        processed_at="2026-06-02T10:45:22Z",
        status="prepared",
    )
    _login(client)
    csrf_token = _csrf_token(client)

    response = client.post(
        "/api/documents/delete-selected",
        json={"relative_raw_paths": ["selected.pdf"]},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 200
    assert response.json()["deleted_count"] == 1
    assert not selected_raw.exists()
    assert not selected_parsed.exists()
    assert kept_raw.exists()
    assert kept_parsed.exists()
    with client.app.state.database_session_factory() as session:
        repository = DocumentRepository(session)
        assert repository.get_by_relative_raw_path("selected.pdf") is None
        assert repository.get_by_relative_raw_path("kept.pdf") is not None


def test_dashboard_document_process_requires_session_and_returns_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    client, settings = _client(tmp_path, monkeypatch)
    raw_path = settings.raw_documents_dir / "report.pdf"
    parsed_path = settings.parsed_markdown_dir / "report.md"
    markdown = "Clinical dashboard detail note."
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(b"%PDF-1.4\n")
    parsed_path.write_text(markdown, encoding="utf-8")
    _seed_document_record(
        client,
        relative_raw_path="report.pdf",
        parsed_markdown_path="report.md",
        content_hash=calculate_text_sha256(markdown),
        byte_size=len(b"%PDF-1.4\n"),
        processed_at="2026-06-02T10:40:22Z",
    )

    unauthenticated = client.get(
        "/api/documents/process",
        params={"relative_raw_path": "report.pdf"},
    )
    _login(client)
    response = client.get(
        "/api/documents/process",
        params={"relative_raw_path": "report.pdf"},
    )

    payload = response.json()
    assert unauthenticated.status_code == 401
    assert response.status_code == 200
    assert payload["document"]["relative_raw_path"] == "report.pdf"
    assert payload["markdown"] == markdown
    assert payload["chunk_count"] == 1
    assert payload["chunks"][0]["content"] == markdown
    assert payload["chunks"][0]["char_start"] == 0
    assert payload["chunks"][0]["char_end"] == len(markdown)
    assert payload["index"]["points"] == []
    assert payload["index"]["error"]


def test_dashboard_exposes_failed_document_error_in_list_and_detail(
    tmp_path,
    monkeypatch,
) -> None:
    client, settings = _client(tmp_path, monkeypatch)
    raw_path = settings.raw_documents_dir / "broken.pdf"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(b"%PDF-1.4\n")
    _seed_document_record(
        client,
        relative_raw_path="broken.pdf",
        byte_size=len(b"%PDF-1.4\n"),
        processed_at="2026-06-02T10:40:22Z",
        status="failed",
        processing_error="RuntimeError: parser failed",
    )
    _login(client)

    documents = client.get("/api/documents")
    detail = client.get(
        "/api/documents/process",
        params={"relative_raw_path": "broken.pdf"},
    )

    assert documents.status_code == 200
    assert documents.json()["documents"][0]["status"] == "failed"
    assert (
        documents.json()["documents"][0]["processing_error"]
        == "RuntimeError: parser failed"
    )
    assert detail.status_code == 200
    assert detail.json()["document"]["processing_error"] == "RuntimeError: parser failed"
    assert detail.json()["markdown"] is None


def test_qdrant_index_preview_reads_point_vectors_from_memory_client() -> None:
    client = QdrantClient(":memory:")
    collection_name = "dashboard_documents"
    point_id = str(uuid.uuid4())
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(size=3, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )
    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=point_id,
                vector={
                    "dense": [0.1, 0.2, 0.3],
                    "sparse": models.SparseVector(
                        indices=[3, 7],
                        values=[0.4, 0.6],
                    ),
                },
                payload={
                    "source": "report.md",
                    "content_hash": "hash",
                    "char_start": 4,
                    "char_end": 18,
                    "content": "Indexed dashboard chunk.",
                },
            )
        ],
    )

    preview = qdrant_index_preview_for_content_hash(
        client=client,
        collection_name=collection_name,
        content_hash="hash",
        vector_name="dense",
    )

    assert preview["available"] is True
    assert preview["collection_exists"] is True
    assert preview["shown_points"] == 1
    assert preview["points"][0]["id"] == point_id
    assert preview["points"][0]["char_start"] == 4
    assert preview["points"][0]["embedding"] == {
        "vector_name": "dense",
        "kind": "dense",
        "dimensions": 3,
        "rows": 1,
        "sample": [0.267261, 0.534522, 0.801784],
    }
    assert preview["points"][0]["embeddings"] == [
        {
            "vector_name": "dense",
            "kind": "dense",
            "dimensions": 3,
            "rows": 1,
            "sample": [0.267261, 0.534522, 0.801784],
        },
        {
            "vector_name": "sparse",
            "kind": "sparse",
            "dimensions": 2,
            "rows": 1,
            "sample": [0.4, 0.6],
            "indices_sample": [3, 7],
        },
    ]


def test_prepare_documents_emits_progress_events(tmp_path, monkeypatch) -> None:
    _isolate_external_env(tmp_path, monkeypatch)
    settings = _settings(tmp_path)
    _create_pdf(settings.raw_documents_dir / "report.pdf", "Dashboard progress")
    events = []

    summary = prepare_documents(settings=settings, progress_callback=events.append)

    assert summary.prepared == 1
    assert any(
        event["step"] == "discover" and event["status"] == "succeeded"
        for event in events
    )
    assert any(
        event["step"] == "prepare" and event["status"] == "succeeded"
        for event in events
    )


def test_dashboard_ingest_job_endpoint_finishes_for_empty_dataset(
    tmp_path,
    monkeypatch,
) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    unauthenticated_history = client.get("/api/jobs")
    _login(client)
    csrf_token = _csrf_token(client)

    started = client.post(
        "/api/jobs/ingest",
        headers={"X-CSRF-Token": csrf_token},
    )

    assert started.status_code == 200
    job_id = started.json()["job"]["id"]
    snapshot = None
    for _ in range(20):
        snapshot = client.get(f"/api/jobs/{job_id}").json()
        if snapshot["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)

    assert snapshot is not None
    assert snapshot["status"] == "succeeded"
    events = client.get(f"/api/jobs/{job_id}/events")
    history = client.get("/api/jobs")

    assert unauthenticated_history.status_code == 401
    assert events.status_code == 200
    assert "event: progress" in events.text
    assert "event: done" in events.text
    assert history.status_code == 200
    assert history.json()["jobs"][0]["id"] == job_id
    assert history.json()["jobs"][0]["status"] == "succeeded"


def test_dashboard_ingest_without_selection_marks_missing_raw_document_stale(
    tmp_path,
    monkeypatch,
) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    _seed_document_record(
        client,
        relative_raw_path="missing.pdf",
        byte_size=123,
    )
    _login(client)
    csrf_token = _csrf_token(client)

    started = client.post(
        "/api/jobs/ingest",
        headers={"X-CSRF-Token": csrf_token},
    )

    assert started.status_code == 200
    job_id = started.json()["job"]["id"]
    snapshot = None
    for _ in range(40):
        snapshot = client.get(f"/api/jobs/{job_id}").json()
        if snapshot["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)

    assert snapshot is not None
    assert snapshot["status"] == "succeeded", snapshot
    with client.app.state.database_session_factory() as session:
        document = DocumentRepository(session).get_by_relative_raw_path("missing.pdf")
        assert document is not None
        assert document.status == "stale"


def test_dashboard_ingest_job_endpoint_runs_selected_documents_only(
    tmp_path,
    monkeypatch,
) -> None:
    indexed_sources = []

    def record_index_text(*, text, source_metadata):
        indexed_sources.append(source_metadata["source"])
        return 1

    job_store = JobStore(
        process_factory=lambda settings: FullProcess(
            settings=settings,
            indexer=record_index_text,
        )
    )
    client, settings = _client(tmp_path, monkeypatch, job_store=job_store)
    _create_pdf(settings.raw_documents_dir / "selected.pdf", "Selected document")
    _create_pdf(settings.raw_documents_dir / "skipped.pdf", "Skipped document")
    _seed_document_record(
        client,
        relative_raw_path="selected.pdf",
        byte_size=(settings.raw_documents_dir / "selected.pdf").stat().st_size,
    )
    _login(client)
    csrf_token = _csrf_token(client)

    started = client.post(
        "/api/jobs/ingest",
        json={"relative_raw_paths": ["selected.pdf"]},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert started.status_code == 200
    job_id = started.json()["job"]["id"]
    snapshot = None
    for _ in range(40):
        snapshot = client.get(f"/api/jobs/{job_id}").json()
        if snapshot["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)

    assert snapshot is not None
    assert snapshot["status"] == "succeeded", snapshot
    assert (settings.parsed_markdown_dir / "selected.md").exists()
    assert not (settings.parsed_markdown_dir / "skipped.md").exists()
    assert indexed_sources == ["selected.md"]
