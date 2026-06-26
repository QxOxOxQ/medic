from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

import rag.config as config_module


def alembic_config(database_url: str | None = None) -> Config:
    project_root = _migration_project_root()
    config_path = project_root / "alembic.ini"
    config = Config(str(config_path))
    config.set_main_option("script_location", str(project_root / "migrations"))
    config.set_main_option(
        "sqlalchemy.url",
        database_url or config_module.get_database_settings().database_url,
    )
    return config


def upgrade_database(database_url: str | None = None) -> None:
    command.upgrade(alembic_config(database_url), "head")


def migration_paths() -> tuple[Path, Path]:
    project_root = _migration_project_root()
    return project_root / "alembic.ini", project_root / "migrations"


def _migration_project_root() -> Path:
    configured_root = config_module.PROJECT_ROOT
    if (configured_root / "migrations").exists():
        return configured_root
    return Path(__file__).resolve().parents[2]
