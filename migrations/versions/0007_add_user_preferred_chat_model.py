"""add user preferred chat model

Revision ID: 0007_user_preferred_chat_model
Revises: 0006_pipeline_runs
Create Date: 2026-06-27 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0007_user_preferred_chat_model"
down_revision: str | None = "0006_pipeline_runs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("preferred_chat_model", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("users", "preferred_chat_model")
