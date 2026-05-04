from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import rag.config as config
from rag.database.repositories import DocumentRepository, UserRepository
from rag.database.session import SessionFactory, get_session_factory, session_scope


@dataclass(frozen=True)
class DemoSeedSummary:
    created: int
    existing: int
    owner_username: str

    def as_report_line(self) -> str:
        return (
            f"demo_documents_created={self.created} "
            f"demo_documents_existing={self.existing} "
            f"owner={self.owner_username}"
        )


def seed_demo_documents(
    *,
    project_root: Path | None = None,
    document_settings: config.DocumentPreparationSettings | None = None,
    database_session_factory: SessionFactory | None = None,
    demo_documents_dir: Path | None = None,
) -> DemoSeedSummary:
    root = project_root or config.PROJECT_ROOT
    settings = document_settings or config.get_document_preparation_settings()
    source_dir = demo_documents_dir or root / "demo_documents"
    source_paths = _demo_pdf_paths(source_dir)
    session_factory = database_session_factory or get_session_factory()

    settings.raw_documents_dir.mkdir(parents=True, exist_ok=True)
    with session_scope(session_factory) as session:
        admin = UserRepository(session).first_active_admin()
        if admin is None:
            raise ValueError("No active admin user found. Run setup first.")

        repository = DocumentRepository(session)
        created = 0
        existing = 0
        for source_path in source_paths:
            relative_raw_path = f"demo/{source_path.name}"
            target_path = settings.raw_documents_dir / relative_raw_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if not target_path.exists():
                shutil.copyfile(source_path, target_path)

            document = repository.get_by_relative_raw_path(relative_raw_path)
            if document is not None:
                existing += 1
                continue

            repository.create_uploaded_document(
                owner_user_id=admin.id,
                original_filename=source_path.name,
                relative_raw_path=relative_raw_path,
                byte_size=target_path.stat().st_size,
            )
            created += 1

        return DemoSeedSummary(
            created=created,
            existing=existing,
            owner_username=admin.username,
        )


def _demo_pdf_paths(source_dir: Path) -> tuple[Path, ...]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing demo documents directory: {source_dir}")

    paths = tuple(sorted(path for path in source_dir.glob("*.pdf") if path.is_file()))
    if not paths:
        raise FileNotFoundError(f"No demo PDF documents found in: {source_dir}")
    return paths
