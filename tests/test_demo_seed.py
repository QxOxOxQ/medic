from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.orm import sessionmaker

from rag.config import DocumentPreparationSettings
from rag.database.migrations import upgrade_database
from rag.database.repositories import DocumentRepository, UserRepository
from rag.database.session import create_database_engine
from rag.demo_seed import seed_demo_documents


def test_seed_demo_documents_loads_synthetic_pdfs_for_admin(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    demo_dir = tmp_path / "demo_documents"
    failure_dir = demo_dir / "failure_cases"
    demo_dir.mkdir()
    failure_dir.mkdir()
    (demo_dir / "synthetic_first.pdf").write_bytes(b"%PDF-1.7 first")
    (demo_dir / "synthetic_second.pdf").write_bytes(b"%PDF-1.7 second")
    (failure_dir / "EXPECTED_PARSE_FAILURE_invalid_pdf.pdf").write_bytes(b"broken")
    factory = _database_session_factory(tmp_path)
    _seed_admin(factory)

    summary = seed_demo_documents(
        document_settings=settings,
        database_session_factory=factory,
        demo_documents_dir=demo_dir,
    )
    repeated = seed_demo_documents(
        document_settings=settings,
        database_session_factory=factory,
        demo_documents_dir=demo_dir,
    )

    assert summary.as_report_line() == (
        "demo_documents_created=2 demo_documents_existing=0 owner=admin"
    )
    assert repeated.as_report_line() == (
        "demo_documents_created=0 demo_documents_existing=2 owner=admin"
    )
    assert (settings.raw_documents_dir / "demo" / "synthetic_first.pdf").is_file()
    assert not (
        settings.raw_documents_dir
        / "demo"
        / "EXPECTED_PARSE_FAILURE_invalid_pdf.pdf"
    ).exists()
    with factory() as session:
        documents = DocumentRepository(session).list_for_owner(_admin_id(factory))
        assert [document.relative_raw_path for document in documents] == [
            "demo/synthetic_first.pdf",
            "demo/synthetic_second.pdf",
        ]


def test_seed_demo_documents_requires_admin_user(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    demo_dir = tmp_path / "demo_documents"
    demo_dir.mkdir()
    (demo_dir / "synthetic.pdf").write_bytes(b"%PDF-1.7")
    factory = _database_session_factory(tmp_path)

    with pytest.raises(ValueError, match="No active admin user"):
        seed_demo_documents(
            document_settings=settings,
            database_session_factory=factory,
            demo_documents_dir=demo_dir,
        )


def _settings(tmp_path: Path) -> DocumentPreparationSettings:
    return DocumentPreparationSettings(
        raw_documents_dir=tmp_path / "data" / "raw",
        parsed_markdown_dir=tmp_path / "data" / "parsed",
    )


def _database_session_factory(tmp_path: Path) -> sessionmaker:
    database_url = f"sqlite:///{tmp_path / 'demo-seed.db'}"
    upgrade_database(database_url)
    return sessionmaker(
        bind=create_database_engine(database_url),
        expire_on_commit=False,
        future=True,
    )


def _seed_admin(factory: sessionmaker) -> None:
    with factory() as session:
        UserRepository(session).seed_admin(username="admin", password="secret")
        session.commit()


def _admin_id(factory: sessionmaker):
    with factory() as session:
        admin = UserRepository(session).get_by_username("admin")
        assert admin is not None
        return admin.id
