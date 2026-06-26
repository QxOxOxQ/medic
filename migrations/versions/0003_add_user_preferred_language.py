"""add user preferred language

Revision ID: 0003_user_preferred_language
Revises: 0002_document_processing_error
Create Date: 2026-06-16 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0003_user_preferred_language"
down_revision: str | None = "0002_document_processing_error"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "preferred_language",
            sa.String(length=2),
            nullable=False,
            server_default="pl",
        ),
    )


def downgrade() -> None:
    op.drop_column("users", "preferred_language")
