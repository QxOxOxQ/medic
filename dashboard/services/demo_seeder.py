from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from dashboard.auth import AuthenticatedUser
from dashboard.services.document_storage import DocumentStorage
from rag.config import DocumentPreparationSettings, get_document_preparation_settings
from rag.database.repositories import DocumentRepository, UserRepository
from rag.full_process import FullProcess


DEMO_PDF_NAMES = (
    "synthetic_acl_rehab_demo.pdf",
    "synthetic_psoriasis_treatment_demo.pdf",
    "synthetic_glp1_remote_monitoring_demo.pdf",
)


class DemoSeedError(RuntimeError):
    """Raised when the synthetic demo corpus cannot be seeded."""


@dataclass(frozen=True)
class DemoSeedSummary:
    owner_username: str
    uploaded: tuple[str, ...]
    skipped_existing: tuple[str, ...]
    indexed_documents: int
    pipeline_failed: bool

    def as_report_line(self) -> str:
        return (
            f"owner={self.owner_username} "
            f"uploaded={len(self.uploaded)} "
            f"skipped_existing={len(self.skipped_existing)} "
            f"indexed_documents={self.indexed_documents} "
            f"pipeline={'failed' if self.pipeline_failed else 'ok'}"
        )


def seed_demo_documents(
    *,
    admin_username: str,
    database_session_factory: sessionmaker[Session],
    documents_dir: Path,
    settings: DocumentPreparationSettings | None = None,
    document_storage: DocumentStorage | None = None,
) -> DemoSeedSummary:
    """Upload and index the synthetic demo PDFs for the admin user.

    Idempotent by original filename: a demo PDF already owned by the admin is
    skipped, so re-running does not create duplicate document records.
    """
    resolved_settings = settings or get_document_preparation_settings()
    owner = _resolve_admin_owner(admin_username, database_session_factory)
    storage = document_storage or DocumentStorage(
        database_session_factory=database_session_factory,
    )

    existing_filenames = _existing_owned_filenames(owner.id, database_session_factory)
    uploaded: list[str] = []
    skipped: list[str] = []
    for pdf_name in DEMO_PDF_NAMES:
        if pdf_name in existing_filenames:
            skipped.append(pdf_name)
            continue
        _store_demo_pdf(
            pdf_name,
            documents_dir=documents_dir,
            owner=owner,
            storage=storage,
            settings=resolved_settings,
        )
        uploaded.append(pdf_name)

    pipeline_failed = False
    if uploaded:
        result = FullProcess(
            settings=resolved_settings,
            database_session_factory=database_session_factory,
        ).execute(owner_user_id=owner.id, print_summary=False)
        pipeline_failed = result.failed > 0

    return DemoSeedSummary(
        owner_username=owner.username,
        uploaded=tuple(uploaded),
        skipped_existing=tuple(skipped),
        indexed_documents=_indexed_document_count(owner.id, database_session_factory),
        pipeline_failed=pipeline_failed,
    )


def _store_demo_pdf(
    pdf_name: str,
    *,
    documents_dir: Path,
    owner: AuthenticatedUser,
    storage: DocumentStorage,
    settings: DocumentPreparationSettings,
) -> None:
    pdf_path = documents_dir / pdf_name
    if not pdf_path.is_file():
        raise DemoSeedError(f"Missing demo document: {pdf_path}")
    storage.save_uploaded_pdf(
        file_name=pdf_name,
        content=pdf_path.read_bytes(),
        owner=owner,
        settings=settings,
    )


def _resolve_admin_owner(
    admin_username: str,
    database_session_factory: sessionmaker[Session],
) -> AuthenticatedUser:
    with database_session_factory() as session:
        user = UserRepository(session).get_by_username(admin_username)
        if user is None:
            raise DemoSeedError(f"Admin user not found: {admin_username}")
        if not user.is_active:
            raise DemoSeedError(f"Admin user is not active: {admin_username}")
        return AuthenticatedUser(
            id=user.id,
            username=user.username,
            is_admin=user.is_admin,
        )


def _existing_owned_filenames(
    owner_user_id: UUID,
    database_session_factory: sessionmaker[Session],
) -> set[str]:
    with database_session_factory() as session:
        documents = DocumentRepository(session).list_for_owner(owner_user_id)
        return {document.original_filename for document in documents}


def _indexed_document_count(
    owner_user_id: UUID,
    database_session_factory: sessionmaker[Session],
) -> int:
    with database_session_factory() as session:
        documents = DocumentRepository(session).list_with_chunks_for_owner(
            owner_user_id
        )
        return sum(1 for document in documents if document.chunks)
