from __future__ import annotations

import subprocess
from typing import Any, cast
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import FastAPI
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker
from sqladmin import Admin, ModelView
from sqladmin.authentication import AuthenticationBackend
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse, RedirectResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send
from wtforms import BooleanField, Form, PasswordField
from wtforms.validators import InputRequired

from dashboard.auth import (
    SESSION_TTL_SECONDS,
    AuthSettings,
    clear_session_cookie,
    read_session,
)
from rag.config import PROJECT_ROOT
from rag.database.models import (
    ChatConversation,
    ChatMessage,
    ChatMessageSource,
    ChatRun,
    ChatTraceEvent,
    Document,
    DocumentChunk,
    User,
)
from rag.database.repositories import UserRepository, normalize_username
from rag.database.security import hash_password


ADMIN_SESSION_COOKIE_NAME = "medic_sqladmin_session"
ADMIN_SESSION_USER_ID = "admin_user_id"
CSRF_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
_TEMPLATES_DIR = str(PROJECT_ROOT / "templates")


def _get_git_commit_id() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return "unknown"


def configure_admin(
    app: FastAPI,
    *,
    auth_settings: AuthSettings,
    session_factory: sessionmaker[Session],
) -> None:
    admin = Admin(
        app=app,
        session_maker=session_factory,
        base_url="/admin",
        title="Medic Admin",
        middlewares=[Middleware(AdminCsrfMiddleware)],
        authentication_backend=AdminAuth(
            auth_settings=auth_settings,
            session_factory=session_factory,
        ),
        templates_dir=_TEMPLATES_DIR,
    )
    admin.templates.env.globals["git_commit_id"] = _get_git_commit_id()
    admin.add_view(UserAdmin)
    admin.add_view(DocumentAdmin)
    admin.add_view(DocumentChunkAdmin)
    admin.add_view(ChatConversationAdmin)
    admin.add_view(ChatMessageAdmin)
    admin.add_view(ChatRunAdmin)
    admin.add_view(ChatTraceEventAdmin)
    admin.add_view(ChatMessageSourceAdmin)


class AdminCsrfMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if not self._requires_origin_check(scope):
            await self._app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        if _same_origin_admin_request(request):
            await self._app(scope, receive, send)
            return

        response = PlainTextResponse("Invalid admin request origin.", status_code=403)
        await response(scope, receive, send)

    def _requires_origin_check(self, scope: Scope) -> bool:
        if scope["type"] != "http":
            return False
        return str(scope.get("method", "GET")).upper() not in CSRF_SAFE_METHODS


class AdminAuth(AuthenticationBackend):
    def __init__(
        self,
        *,
        auth_settings: AuthSettings,
        session_factory: sessionmaker[Session],
    ) -> None:
        super().__init__(
            secret_key=auth_settings.session_secret,
            session_cookie=ADMIN_SESSION_COOKIE_NAME,
            max_age=SESSION_TTL_SECONDS,
            same_site="lax",
            https_only=auth_settings.cookie_secure,
        )
        self._auth_settings = auth_settings
        self._session_factory = session_factory

    async def login(self, request: Request) -> bool:
        form = await request.form()
        username = str(form.get("username", ""))
        password = str(form.get("password", ""))

        with self._session_factory() as session:
            user = UserRepository(session).authenticate(
                username=username,
                password=password,
            )
            if user is None or not user.is_admin:
                return False
            request.session[ADMIN_SESSION_USER_ID] = str(user.id)
            return True

    async def logout(self, request: Request) -> Response:
        request.session.clear()
        response = RedirectResponse("/login", status_code=302)
        clear_session_cookie(response, self._auth_settings)
        return response

    async def authenticate(self, request: Request) -> bool:
        if self._sqladmin_session_is_allowed(request):
            return True
        return self._dashboard_session_is_allowed(request)

    def _sqladmin_session_is_allowed(self, request: Request) -> bool:
        user_id = _uuid_from_session(request.session.get(ADMIN_SESSION_USER_ID))
        if user_id is None:
            return False
        return self._user_is_active_admin(user_id)

    def _dashboard_session_is_allowed(self, request: Request) -> bool:
        session_data = read_session(request, self._auth_settings)
        if session_data is None:
            return False
        return self._user_is_active_admin(session_data.user_id)

    def _user_is_active_admin(self, user_id: UUID) -> bool:
        with self._session_factory() as session:
            user = UserRepository(session).get_by_id(user_id)
            return bool(user is not None and user.is_active and user.is_admin)


class UserAdmin(ModelView, model=User):
    name = "User"
    name_plural = "Users"
    icon = "fa-solid fa-user"
    column_list = [User.id, User.username, User.is_active, User.is_admin, User.created_at]
    column_details_exclude_list = [User.password_hash]
    column_export_exclude_list = [User.password_hash]
    column_searchable_list = [User.username]
    column_sortable_list = [User.username, User.created_at, User.updated_at]
    form_columns = ["username", "password_hash", "is_active", "is_admin"]
    form_create_rules = ["username", "password_hash", "is_active", "is_admin"]
    form_edit_rules = ["username", "password_hash", "is_active", "is_admin"]
    form_overrides = {
        "password_hash": PasswordField,
        "is_active": BooleanField,
        "is_admin": BooleanField,
    }
    form_args = {
        "password_hash": {
            "label": "Password",
            "description": "Leave empty while editing to keep the current password.",
        },
    }
    form_widget_args = {"password_hash": {"autocomplete": "new-password"}}
    column_labels = {"password_hash": "Password"}

    async def scaffold_form(self, rules: list[str] | None = None) -> type[Form]:
        form = await super().scaffold_form(rules)
        password_field = getattr(form, "password_hash", None)
        if password_field is not None:
            validators = password_field.kwargs.get("validators", [])
            password_field.kwargs["validators"] = [
                validator
                for validator in validators
                if not isinstance(validator, InputRequired)
            ]
        return form

    async def on_model_change(
        self,
        data: dict[str, Any],
        model: Any,
        is_created: bool,
        request: Request,
    ) -> None:
        if not is_created:
            self._ensure_active_admin_remains(model, data)
        if "username" in data:
            data["username"] = normalize_username(str(data["username"]))
        password = str(data.get("password_hash") or "")
        if not password:
            if is_created:
                raise ValueError("Password is required.")
            data.pop("password_hash", None)
            return
        data["password_hash"] = hash_password(password)

    async def on_model_delete(self, model: Any, request: Request) -> None:
        self._ensure_not_last_active_admin(model)

    def _ensure_active_admin_remains(
        self,
        model: Any,
        data: dict[str, Any],
    ) -> None:
        if not isinstance(model, User):
            return
        if not model.is_active or not model.is_admin:
            return
        next_is_active = bool(data.get("is_active", model.is_active))
        next_is_admin = bool(data.get("is_admin", model.is_admin))
        if next_is_active and next_is_admin:
            return
        if _other_active_admin_exists(self._session_factory(), model.id):
            return
        raise ValueError("At least one active admin user is required.")

    def _ensure_not_last_active_admin(self, model: Any) -> None:
        if not isinstance(model, User):
            return
        if not model.is_active or not model.is_admin:
            return
        if _other_active_admin_exists(self._session_factory(), model.id):
            return
        raise ValueError("At least one active admin user is required.")

    def _session_factory(self) -> sessionmaker[Session]:
        return cast(sessionmaker[Session], self.session_maker)


class DocumentAdmin(ModelView, model=Document):
    name = "Document"
    name_plural = "Documents"
    icon = "fa-solid fa-file-pdf"
    column_list = [
        Document.id,
        Document.owner_user_id,
        Document.original_filename,
        Document.status,
        Document.created_at,
        Document.updated_at,
    ]
    column_searchable_list = [
        Document.original_filename,
        Document.relative_raw_path,
        Document.content_hash,
    ]
    column_sortable_list = [
        Document.original_filename,
        Document.status,
        Document.created_at,
        Document.updated_at,
    ]
    form_columns = [
        "owner_user_id",
        "original_filename",
        "relative_raw_path",
        "parsed_markdown_path",
        "content_hash",
        "byte_size",
        "status",
        "processing_error",
        "processed_at",
        "indexed_at",
    ]


class DocumentChunkAdmin(ModelView, model=DocumentChunk):
    name = "Document chunk"
    name_plural = "Document chunks"
    icon = "fa-solid fa-file-lines"
    column_list = [
        DocumentChunk.id,
        DocumentChunk.document_id,
        DocumentChunk.chunk_index,
        DocumentChunk.qdrant_point_id,
        DocumentChunk.created_at,
    ]
    column_searchable_list = [DocumentChunk.content, DocumentChunk.qdrant_point_id]
    column_sortable_list = [DocumentChunk.chunk_index, DocumentChunk.created_at]
    form_columns = [
        "document_id",
        "chunk_index",
        "char_start",
        "char_end",
        "content",
        "qdrant_point_id",
    ]


class ChatConversationAdmin(ModelView, model=ChatConversation):
    name = "Chat conversation"
    name_plural = "Chat conversations"
    icon = "fa-solid fa-comments"
    column_list = [
        ChatConversation.id,
        ChatConversation.owner_user_id,
        ChatConversation.title,
        ChatConversation.created_at,
        ChatConversation.updated_at,
    ]
    column_searchable_list = [ChatConversation.title]
    column_sortable_list = [
        ChatConversation.title,
        ChatConversation.created_at,
        ChatConversation.updated_at,
    ]
    form_columns = ["owner_user_id", "title"]


class ChatMessageAdmin(ModelView, model=ChatMessage):
    name = "Chat message"
    name_plural = "Chat messages"
    icon = "fa-solid fa-message"
    column_list = [
        ChatMessage.id,
        ChatMessage.conversation_id,
        ChatMessage.role,
        ChatMessage.sequence,
        ChatMessage.created_at,
    ]
    column_searchable_list = [ChatMessage.content]
    column_sortable_list = [
        ChatMessage.role,
        ChatMessage.sequence,
        ChatMessage.created_at,
    ]
    form_columns = [
        "conversation_id",
        "role",
        "content",
        "sequence",
        "insufficient_context",
    ]
    form_overrides = {"insufficient_context": BooleanField}


class ChatRunAdmin(ModelView, model=ChatRun):
    name = "Chat run"
    name_plural = "Chat runs"
    icon = "fa-solid fa-person-running"
    column_list = [
        ChatRun.id,
        ChatRun.conversation_id,
        ChatRun.status,
        ChatRun.started_at,
        ChatRun.finished_at,
    ]
    column_searchable_list = [ChatRun.question, ChatRun.answer, ChatRun.error]
    column_sortable_list = [ChatRun.status, ChatRun.started_at, ChatRun.finished_at]
    form_columns = [
        "conversation_id",
        "assistant_message_id",
        "status",
        "question",
        "answer",
        "insufficient_context",
        "error",
        "started_at",
        "finished_at",
    ]
    form_overrides = {"insufficient_context": BooleanField}


class ChatTraceEventAdmin(ModelView, model=ChatTraceEvent):
    name = "Chat trace event"
    name_plural = "Chat trace events"
    icon = "fa-solid fa-list-check"
    column_list = [
        ChatTraceEvent.id,
        ChatTraceEvent.run_id,
        ChatTraceEvent.sequence,
        ChatTraceEvent.event_type,
        ChatTraceEvent.status,
        ChatTraceEvent.created_at,
    ]
    form_columns = [
        "run_id",
        "sequence",
        "event_type",
        "title",
        "status",
        "agent_name",
        "tool_name",
        "payload",
        "duration_ms",
    ]
    column_searchable_list = [
        ChatTraceEvent.event_type,
        ChatTraceEvent.title,
        ChatTraceEvent.agent_name,
        ChatTraceEvent.tool_name,
    ]
    column_sortable_list = [
        ChatTraceEvent.sequence,
        ChatTraceEvent.event_type,
        ChatTraceEvent.status,
        ChatTraceEvent.created_at,
    ]


class ChatMessageSourceAdmin(ModelView, model=ChatMessageSource):
    name = "Chat message source"
    name_plural = "Chat message sources"
    icon = "fa-solid fa-quote-left"
    column_list = [
        ChatMessageSource.id,
        ChatMessageSource.message_id,
        ChatMessageSource.run_id,
        ChatMessageSource.source_id,
        ChatMessageSource.document_name,
        ChatMessageSource.score,
        ChatMessageSource.created_at,
    ]
    form_columns = [
        "message_id",
        "run_id",
        "source_id",
        "source",
        "content_hash",
        "document_id",
        "document_name",
        "relative_raw_path",
        "qdrant_point_id",
        "chunk_index",
        "char_start",
        "char_end",
        "retrieval_query",
        "score",
        "excerpt",
    ]
    column_searchable_list = [
        ChatMessageSource.source_id,
        ChatMessageSource.source,
        ChatMessageSource.document_name,
        ChatMessageSource.excerpt,
    ]
    column_sortable_list = [
        ChatMessageSource.source_id,
        ChatMessageSource.document_name,
        ChatMessageSource.score,
        ChatMessageSource.created_at,
    ]


def _uuid_from_session(value: object) -> UUID | None:
    if value is None:
        return None
    try:
        return UUID(str(value))
    except ValueError:
        return None


def _same_origin_admin_request(request: Request) -> bool:
    origin = request.headers.get("origin")
    if origin:
        return _url_has_same_origin(origin, str(request.url))
    referer = request.headers.get("referer")
    if referer:
        return _url_has_same_origin(referer, str(request.url))
    return False


def _url_has_same_origin(candidate: str, target: str) -> bool:
    try:
        candidate_url = urlsplit(candidate)
        target_url = urlsplit(target)
    except ValueError:
        return False
    return (
        candidate_url.scheme == target_url.scheme
        and candidate_url.netloc == target_url.netloc
    )


def _other_active_admin_exists(
    session_factory: sessionmaker[Session],
    user_id: UUID,
) -> bool:
    with session_factory() as session:
        count = session.scalar(
            select(func.count())
            .select_from(User)
            .where(
                User.id != user_id,
                User.is_active.is_(True),
                User.is_admin.is_(True),
            )
        )
        return bool(count)
