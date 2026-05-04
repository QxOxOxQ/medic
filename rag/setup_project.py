from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from dotenv import dotenv_values

import rag.config as config
from rag.database.migrations import upgrade_database
from rag.database.repositories import UserRepository
from rag.database.session import SessionFactory, get_session_factory, session_scope


@dataclass(frozen=True)
class ProjectSetupSummary:
    env_file_created: bool
    env_file_exists: bool
    raw_documents_dir_created: bool
    parsed_markdown_dir_created: bool
    postgres_database_checked: bool
    qdrant_collection_checked: bool

    def as_report_line(self) -> str:
        env_status = "created" if self.env_file_created else "present"
        if not self.env_file_exists:
            env_status = "missing"

        return (
            f"env={env_status} "
            f"raw_dir={_created_or_present(self.raw_documents_dir_created)} "
            f"parsed_dir={_created_or_present(self.parsed_markdown_dir_created)} "
            f"postgres={'ready' if self.postgres_database_checked else 'skipped'} "
            f"qdrant={'ready' if self.qdrant_collection_checked else 'skipped'}"
        )


def setup_project(
    *,
    project_root: Path | None = None,
    document_settings: config.DocumentPreparationSettings | None = None,
    create_env_file: bool = True,
    setup_database: bool = True,
    setup_qdrant: bool = True,
    database_setup: Callable[[], None] | None = None,
    database_session_factory: SessionFactory | None = None,
    qdrant_setup: Callable[[], None] | None = None,
) -> ProjectSetupSummary:
    root = project_root or config.PROJECT_ROOT
    settings = document_settings or config.get_document_preparation_settings()

    env_file_created = False
    if create_env_file:
        env_file_created = _ensure_env_file(root)

    raw_documents_dir_created = _ensure_directory(settings.raw_documents_dir)
    parsed_markdown_dir_created = _ensure_directory(settings.parsed_markdown_dir)

    postgres_database_checked = False
    if setup_database:
        ensure_database = database_setup or upgrade_database
        ensure_database()
        session_factory = database_session_factory or get_session_factory()
        _seed_bootstrap_admin(root, session_factory)
        postgres_database_checked = True

    qdrant_collection_checked = False
    if setup_qdrant:
        ensure_collection = qdrant_setup or ensure_qdrant_collection
        ensure_collection()
        qdrant_collection_checked = True

    return ProjectSetupSummary(
        env_file_created=env_file_created,
        env_file_exists=(root / ".env").exists(),
        raw_documents_dir_created=raw_documents_dir_created,
        parsed_markdown_dir_created=parsed_markdown_dir_created,
        postgres_database_checked=postgres_database_checked,
        qdrant_collection_checked=qdrant_collection_checked,
    )


def ensure_qdrant_collection() -> None:
    from rag.qdrant import Qdrant

    Qdrant().setup_db()


def _ensure_env_file(project_root: Path) -> bool:
    env_path = project_root / ".env"
    if env_path.exists():
        return False

    env_example_path = project_root / ".env.example"
    if not env_example_path.exists():
        raise FileNotFoundError(f"Missing environment template: {env_example_path}")

    shutil.copyfile(env_example_path, env_path)
    return True


def _ensure_directory(directory_path: Path) -> bool:
    existed = directory_path.exists()
    directory_path.mkdir(parents=True, exist_ok=True)
    return not existed


def _seed_bootstrap_admin(
    project_root: Path,
    session_factory: SessionFactory,
) -> UUID:
    username, password = _bootstrap_admin_credentials(project_root)
    with session_scope(session_factory) as session:
        user = UserRepository(session).seed_admin(username=username, password=password)
        return user.id


def _bootstrap_admin_credentials(project_root: Path) -> tuple[str, str]:
    dotenv_settings = dotenv_values(project_root / ".env")

    def lookup(name: str) -> str | None:
        return os.getenv(name) or dotenv_settings.get(name)

    username = lookup("MEDIC_DASHBOARD_USERNAME")
    password = lookup("MEDIC_DASHBOARD_PASSWORD")
    missing = [
        name
        for name, value in (
            ("MEDIC_DASHBOARD_USERNAME", username),
            ("MEDIC_DASHBOARD_PASSWORD", password),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "Missing bootstrap admin settings: " + ", ".join(missing)
        )
    return username or "", password or ""


def _created_or_present(created: bool) -> str:
    return "created" if created else "present"
