"""add persistent pipeline runs

Revision ID: 0006_pipeline_runs
Revises: 0005_drop_preferred_language
Create Date: 2026-06-24 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0006_pipeline_runs"
down_revision: str | None = "0005_drop_preferred_language"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("chat_runs") as batch_op:
        batch_op.drop_constraint("ck_chat_runs_status", type_="check")
        batch_op.create_check_constraint(
            "ck_chat_runs_status",
            "status in ('running', 'succeeded', 'failed', 'interrupted')",
        )

    op.create_table(
        "pipeline_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_user_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status in ('queued', 'running', 'succeeded', 'failed', 'interrupted')",
            name="ck_pipeline_runs_status",
        ),
        sa.ForeignKeyConstraint(["owner_user_id"], ["users.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_pipeline_runs_owner_user_id",
        "pipeline_runs",
        ["owner_user_id"],
    )
    op.create_index("ix_pipeline_runs_status", "pipeline_runs", ["status"])
    op.create_index(
        "ix_pipeline_runs_created_at",
        "pipeline_runs",
        ["created_at"],
    )

    op.create_table(
        "pipeline_run_documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=True),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("document_name", sa.String(length=255), nullable=False),
        sa.Column("relative_raw_path", sa.String(length=1024), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("current_step", sa.String(length=32), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status in ('queued', 'running', 'succeeded', 'failed', 'skipped')",
            name="ck_pipeline_run_documents_status",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["pipeline_runs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "run_id",
            "position",
            name="uq_pipeline_run_documents_position",
        ),
    )
    op.create_index(
        "ix_pipeline_run_documents_run_id",
        "pipeline_run_documents",
        ["run_id"],
    )
    op.create_index(
        "ix_pipeline_run_documents_document_id",
        "pipeline_run_documents",
        ["document_id"],
    )

    op.create_table(
        "pipeline_run_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("step", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("counters", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["pipeline_runs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "run_id",
            "sequence",
            name="uq_pipeline_run_events_sequence",
        ),
    )
    op.create_index(
        "ix_pipeline_run_events_run_id",
        "pipeline_run_events",
        ["run_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_pipeline_run_events_run_id", table_name="pipeline_run_events")
    op.drop_table("pipeline_run_events")
    op.drop_index(
        "ix_pipeline_run_documents_document_id",
        table_name="pipeline_run_documents",
    )
    op.drop_index(
        "ix_pipeline_run_documents_run_id",
        table_name="pipeline_run_documents",
    )
    op.drop_table("pipeline_run_documents")
    op.drop_index("ix_pipeline_runs_created_at", table_name="pipeline_runs")
    op.drop_index("ix_pipeline_runs_status", table_name="pipeline_runs")
    op.drop_index("ix_pipeline_runs_owner_user_id", table_name="pipeline_runs")
    op.drop_table("pipeline_runs")

    with op.batch_alter_table("chat_runs") as batch_op:
        batch_op.drop_constraint("ck_chat_runs_status", type_="check")
        batch_op.create_check_constraint(
            "ck_chat_runs_status",
            "status in ('running', 'succeeded', 'failed')",
        )
