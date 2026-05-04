"""drop user preferred language

Revision ID: 0005_drop_preferred_language
Revises: 0004_chat_conversations
Create Date: 2026-06-17 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0005_drop_preferred_language"
down_revision: str | None = "0004_chat_conversations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("users") as batch_op:
        batch_op.drop_column("preferred_language")


def downgrade() -> None:
    with op.batch_alter_table("users") as batch_op:
        batch_op.add_column(
            sa.Column(
                "preferred_language",
                sa.String(length=2),
                nullable=False,
                server_default="en",
            )
        )
