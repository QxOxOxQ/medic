"""add chat conversations

Revision ID: 0004_chat_conversations
Revises: 0003_user_preferred_language
Create Date: 2026-06-16 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0004_chat_conversations"
down_revision: str | None = "0003_user_preferred_language"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "chat_conversations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_user_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=160), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["owner_user_id"], ["users.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_chat_conversations_owner_user_id",
        "chat_conversations",
        ["owner_user_id"],
    )
    op.create_index(
        "ix_chat_conversations_updated_at",
        "chat_conversations",
        ["updated_at"],
    )

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("insufficient_context", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "role in ('user', 'assistant')",
            name="ck_chat_messages_role",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["chat_conversations.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "conversation_id",
            "sequence",
            name="uq_chat_messages_conversation_sequence",
        ),
    )
    op.create_index(
        "ix_chat_messages_conversation_id",
        "chat_messages",
        ["conversation_id"],
    )

    op.create_table(
        "chat_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("assistant_message_id", sa.Uuid(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("insufficient_context", sa.Boolean(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status in ('running', 'succeeded', 'failed')",
            name="ck_chat_runs_status",
        ),
        sa.ForeignKeyConstraint(
            ["assistant_message_id"],
            ["chat_messages.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["chat_conversations.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_chat_runs_assistant_message_id",
        "chat_runs",
        ["assistant_message_id"],
    )
    op.create_index("ix_chat_runs_conversation_id", "chat_runs", ["conversation_id"])

    op.create_table(
        "chat_trace_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=180), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("agent_name", sa.String(length=120), nullable=True),
        sa.Column("tool_name", sa.String(length=160), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["chat_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "sequence", name="uq_chat_trace_events_sequence"),
    )
    op.create_index("ix_chat_trace_events_run_id", "chat_trace_events", ["run_id"])

    op.create_table(
        "chat_message_sources",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("message_id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("source_id", sa.String(length=32), nullable=False),
        sa.Column("source", sa.String(length=1024), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=True),
        sa.Column("document_id", sa.Uuid(), nullable=True),
        sa.Column("document_name", sa.String(length=255), nullable=True),
        sa.Column("relative_raw_path", sa.String(length=1024), nullable=True),
        sa.Column("qdrant_point_id", sa.String(length=64), nullable=True),
        sa.Column("chunk_index", sa.Integer(), nullable=True),
        sa.Column("char_start", sa.Integer(), nullable=True),
        sa.Column("char_end", sa.Integer(), nullable=True),
        sa.Column("retrieval_query", sa.Text(), nullable=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("excerpt", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["message_id"], ["chat_messages.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["chat_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "message_id",
            "source_id",
            name="uq_chat_message_sources_message_source",
        ),
    )
    op.create_index(
        "ix_chat_message_sources_document_id",
        "chat_message_sources",
        ["document_id"],
    )
    op.create_index(
        "ix_chat_message_sources_message_id",
        "chat_message_sources",
        ["message_id"],
    )
    op.create_index("ix_chat_message_sources_run_id", "chat_message_sources", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_chat_message_sources_run_id", table_name="chat_message_sources")
    op.drop_index(
        "ix_chat_message_sources_message_id",
        table_name="chat_message_sources",
    )
    op.drop_index(
        "ix_chat_message_sources_document_id",
        table_name="chat_message_sources",
    )
    op.drop_table("chat_message_sources")
    op.drop_index("ix_chat_trace_events_run_id", table_name="chat_trace_events")
    op.drop_table("chat_trace_events")
    op.drop_index("ix_chat_runs_conversation_id", table_name="chat_runs")
    op.drop_index("ix_chat_runs_assistant_message_id", table_name="chat_runs")
    op.drop_table("chat_runs")
    op.drop_index("ix_chat_messages_conversation_id", table_name="chat_messages")
    op.drop_table("chat_messages")
    op.drop_index(
        "ix_chat_conversations_updated_at",
        table_name="chat_conversations",
    )
    op.drop_index(
        "ix_chat_conversations_owner_user_id",
        table_name="chat_conversations",
    )
    op.drop_table("chat_conversations")
