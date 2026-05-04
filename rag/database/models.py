from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid


DOCUMENT_STATUSES = ("raw", "prepared", "indexed", "failed", "stale")
CHAT_MESSAGE_ROLES = ("user", "assistant")
CHAT_RUN_STATUSES = ("running", "succeeded", "failed")


class Base(DeclarativeBase):
    pass


def utc_now() -> datetime:
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    username: Mapped[str] = mapped_column(String(150), nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    documents: Mapped[list[Document]] = relationship(back_populates="owner")
    chat_conversations: Mapped[list[ChatConversation]] = relationship(
        back_populates="owner",
    )


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(
            "status in ('raw', 'prepared', 'indexed', 'failed', 'stale')",
            name="ck_documents_status",
        ),
        Index("ix_documents_owner_user_id", "owner_user_id"),
        Index("ix_documents_content_hash", "content_hash"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    owner_user_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    relative_raw_path: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
        unique=True,
    )
    parsed_markdown_path: Mapped[str | None] = mapped_column(String(1024))
    content_hash: Mapped[str | None] = mapped_column(String(64))
    byte_size: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="raw")
    processing_error: Mapped[str | None] = mapped_column(Text)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    owner: Mapped[User] = relationship(back_populates="documents")
    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentChunk.chunk_index",
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunks_index"),
        UniqueConstraint("qdrant_point_id", name="uq_document_chunks_qdrant_point_id"),
        Index("ix_document_chunks_document_id", "document_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    char_start: Mapped[int | None] = mapped_column(Integer)
    char_end: Mapped[int | None] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    qdrant_point_id: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    document: Mapped[Document] = relationship(back_populates="chunks")


class ChatConversation(Base):
    __tablename__ = "chat_conversations"
    __table_args__ = (
        Index("ix_chat_conversations_owner_user_id", "owner_user_id"),
        Index("ix_chat_conversations_updated_at", "updated_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    owner_user_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(String(160), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    owner: Mapped[User] = relationship(back_populates="chat_conversations")
    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ChatMessage.sequence",
    )
    runs: Mapped[list[ChatRun]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ChatRun.created_at",
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (
        CheckConstraint(
            "role in ('user', 'assistant')",
            name="ck_chat_messages_role",
        ),
        UniqueConstraint(
            "conversation_id",
            "sequence",
            name="uq_chat_messages_conversation_sequence",
        ),
        Index("ix_chat_messages_conversation_id", "conversation_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    conversation_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("chat_conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    insufficient_context: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    conversation: Mapped[ChatConversation] = relationship(back_populates="messages")
    sources: Mapped[list[ChatMessageSource]] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        order_by="ChatMessageSource.source_id",
    )


class ChatRun(Base):
    __tablename__ = "chat_runs"
    __table_args__ = (
        CheckConstraint(
            "status in ('running', 'succeeded', 'failed')",
            name="ck_chat_runs_status",
        ),
        Index("ix_chat_runs_conversation_id", "conversation_id"),
        Index("ix_chat_runs_assistant_message_id", "assistant_message_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    conversation_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("chat_conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    assistant_message_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("chat_messages.id", ondelete="SET NULL"),
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="running")
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text)
    insufficient_context: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    conversation: Mapped[ChatConversation] = relationship(back_populates="runs")
    trace_events: Mapped[list[ChatTraceEvent]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="ChatTraceEvent.sequence",
    )
    sources: Mapped[list[ChatMessageSource]] = relationship(
        back_populates="run",
        order_by="ChatMessageSource.source_id",
    )


class ChatTraceEvent(Base):
    __tablename__ = "chat_trace_events"
    __table_args__ = (
        UniqueConstraint("run_id", "sequence", name="uq_chat_trace_events_sequence"),
        Index("ix_chat_trace_events_run_id", "run_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("chat_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(180), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    agent_name: Mapped[str | None] = mapped_column(String(120))
    tool_name: Mapped[str | None] = mapped_column(String(160))
    payload: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    run: Mapped[ChatRun] = relationship(back_populates="trace_events")


class ChatMessageSource(Base):
    __tablename__ = "chat_message_sources"
    __table_args__ = (
        UniqueConstraint(
            "message_id",
            "source_id",
            name="uq_chat_message_sources_message_source",
        ),
        Index("ix_chat_message_sources_message_id", "message_id"),
        Index("ix_chat_message_sources_run_id", "run_id"),
        Index("ix_chat_message_sources_document_id", "document_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    message_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("chat_messages.id", ondelete="CASCADE"),
        nullable=False,
    )
    run_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("chat_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_id: Mapped[str] = mapped_column(String(32), nullable=False)
    source: Mapped[str | None] = mapped_column(String(1024))
    content_hash: Mapped[str | None] = mapped_column(String(64))
    document_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("documents.id", ondelete="SET NULL"),
    )
    document_name: Mapped[str | None] = mapped_column(String(255))
    relative_raw_path: Mapped[str | None] = mapped_column(String(1024))
    qdrant_point_id: Mapped[str | None] = mapped_column(String(64))
    chunk_index: Mapped[int | None] = mapped_column(Integer)
    char_start: Mapped[int | None] = mapped_column(Integer)
    char_end: Mapped[int | None] = mapped_column(Integer)
    retrieval_query: Mapped[str | None] = mapped_column(Text)
    score: Mapped[float | None] = mapped_column(Float)
    excerpt: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    message: Mapped[ChatMessage] = relationship(back_populates="sources")
    run: Mapped[ChatRun] = relationship(back_populates="sources")
