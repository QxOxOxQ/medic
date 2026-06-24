from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy.orm import sessionmaker

from dashboard.auth import AuthenticatedUser
from dashboard.schemas import QdrantCleanupResult
from dashboard.services.document_catalog import DocumentCatalog
from dashboard.services.document_storage import DocumentOperationError, DocumentStorage
from rag.config import DocumentPreparationSettings
from rag.database.migrations import upgrade_database
from rag.database.repositories import DocumentRepository, UserRepository
from rag.database.session import create_database_engine
from rag.retrieval import search_results_from_response


def _settings(tmp_path: Path) -> DocumentPreparationSettings:
    return DocumentPreparationSettings(
        raw_documents_dir=tmp_path / "data" / "raw",
        parsed_markdown_dir=tmp_path / "data" / "parsed",
    )


def _write_file(path: Path, content: str | bytes = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def _database_session_factory(tmp_path: Path) -> sessionmaker:
    database_url = f"sqlite:///{tmp_path / 'dashboard-services.db'}"
    upgrade_database(database_url)
    engine = create_database_engine(database_url)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def _seed_user(factory: sessionmaker) -> AuthenticatedUser:
    with factory() as session:
        user = UserRepository(session).create_user(
            username="admin",
            password="secret",
            is_admin=True,
        )
        session.commit()
        return AuthenticatedUser(
            id=user.id,
            username=user.username,
            is_admin=user.is_admin,
        )


def _seed_document(
    factory: sessionmaker,
    owner: AuthenticatedUser,
    *,
    relative_raw_path: str = "report.pdf",
    original_filename: str = "report.pdf",
    parsed_markdown_path: str | None = "report.md",
    content_hash: str | None = "hash",
    byte_size: int | None = 9,
    processed_at: str | None = "2026-06-02T10:40:22Z",
    status: str = "prepared",
) -> None:
    with factory() as session:
        DocumentRepository(session).upsert_prepared_document(
            owner_user_id=owner.id,
            relative_raw_path=relative_raw_path,
            original_filename=original_filename,
            parsed_markdown_path=parsed_markdown_path,
            content_hash=content_hash,
            byte_size=byte_size,
            processed_at=_datetime_from_iso_z(processed_at),
            status=status,
        )
        session.commit()


def _datetime_from_iso_z(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class _UnavailableIndexReader:
    def indexed_content_hashes(
        self,
        content_hashes: set[str],
    ) -> tuple[set[str], str | None]:
        return set(), "qdrant unavailable"

    def status(self) -> dict:
        return {
            "available": False,
            "collection_name": None,
            "collection_exists": False,
            "points_count": None,
            "error": "qdrant unavailable",
        }


class _AvailableIndexReader:
    def __init__(self, indexed_hashes: set[str]) -> None:
        self._indexed_hashes = indexed_hashes

    def indexed_content_hashes(
        self,
        content_hashes: set[str],
    ) -> tuple[set[str], str | None]:
        return content_hashes & self._indexed_hashes, None

    def status(self) -> dict:
        return {
            "available": True,
            "collection_name": "documents",
            "collection_exists": True,
            "points_count": len(self._indexed_hashes),
            "error": None,
        }


class _CountingIndexReader(_AvailableIndexReader):
    def __init__(self, indexed_hashes: set[str]) -> None:
        super().__init__(indexed_hashes)
        self.calls = 0

    def indexed_content_hashes(
        self,
        content_hashes: set[str],
    ) -> tuple[set[str], str | None]:
        self.calls += 1
        return super().indexed_content_hashes(content_hashes)


class _IndexCleanup:
    def delete_content_hash(self, content_hash: str | None) -> QdrantCleanupResult:
        return QdrantCleanupResult(
            attempted=content_hash is not None,
            deleted=content_hash is not None,
        )


def test_document_catalog_reports_unverified_status_when_qdrant_is_unavailable(
    tmp_path,
) -> None:
    settings = _settings(tmp_path)
    factory = _database_session_factory(tmp_path)
    owner = _seed_user(factory)
    _write_file(settings.raw_documents_dir / "report.pdf", b"%PDF-1.4\n")
    _write_file(settings.parsed_markdown_dir / "report.md", "parsed")
    _seed_document(factory, owner)

    catalog = DocumentCatalog(
        index_reader=_UnavailableIndexReader(),
        database_session_factory=factory,
    )
    records, qdrant_error = catalog.list_records(settings, owner_user_id=owner.id)
    status = catalog.dashboard_status(settings, owner_user_id=owner.id)

    assert qdrant_error == "qdrant unavailable"
    assert records[0].status == "prepared_unverified"
    assert records[0].indexed is None
    assert records[0].display_name == "report.pdf"
    assert status.raw_pdf_count == 1
    assert status.parsed_markdown_count == 1
    assert status.document_count == 1
    assert status.qdrant["available"] is False


def test_document_catalog_marks_documents_indexed_by_content_hash(tmp_path) -> None:
    settings = _settings(tmp_path)
    factory = _database_session_factory(tmp_path)
    owner = _seed_user(factory)
    _write_file(settings.raw_documents_dir / "report.pdf", b"%PDF-1.4\n")
    _write_file(settings.parsed_markdown_dir / "report.md", "parsed")
    _seed_document(factory, owner)

    records, qdrant_error = DocumentCatalog(
        index_reader=_AvailableIndexReader({"hash"}),
        database_session_factory=factory,
    ).list_records(settings, owner_user_id=owner.id)

    assert qdrant_error is None
    assert records[0].status == "indexed"
    assert records[0].indexed is True


def test_document_detail_can_skip_qdrant_for_lazy_artifact_tabs(tmp_path) -> None:
    settings = _settings(tmp_path)
    factory = _database_session_factory(tmp_path)
    owner = _seed_user(factory)
    _write_file(settings.raw_documents_dir / "report.pdf", b"%PDF-1.4\n")
    _write_file(settings.parsed_markdown_dir / "report.md", "parsed")
    _seed_document(factory, owner, status="indexed")
    with factory() as session:
        document = DocumentRepository(session).get_by_relative_raw_path("report.pdf")
        assert document is not None
        document_id = document.id

    index_reader = _CountingIndexReader({"hash"})
    record = DocumentCatalog(
        index_reader=index_reader,
        database_session_factory=factory,
    ).record_by_id(
        settings,
        owner_user_id=owner.id,
        document_id=document_id,
        check_index=False,
    )

    assert record is not None
    assert record.status == "indexed"
    assert index_reader.calls == 0


def test_document_storage_upload_accepts_only_new_pdf_files(tmp_path) -> None:
    settings = _settings(tmp_path)
    storage = DocumentStorage(index_cleanup=_IndexCleanup())

    result = storage.save_uploaded_pdf(
        file_name="report.pdf",
        content=b"%PDF-1.4\n",
        settings=settings,
    )

    assert result == {"relative_raw_path": "report.pdf", "bytes": 9}
    assert (settings.raw_documents_dir / "report.pdf").read_bytes() == b"%PDF-1.4\n"
    with pytest.raises(DocumentOperationError, match="already exists"):
        storage.save_uploaded_pdf(
            file_name="report.pdf",
            content=b"%PDF-1.4\n",
            settings=settings,
        )
    with pytest.raises(DocumentOperationError, match="Only PDF"):
        storage.save_uploaded_pdf(
            file_name="notes.txt",
            content=b"plain",
            settings=settings,
        )


def test_document_storage_uploads_multiple_pdf_files(tmp_path) -> None:
    settings = _settings(tmp_path)
    storage = DocumentStorage(index_cleanup=_IndexCleanup())

    result = storage.save_uploaded_pdfs(
        [
            ("report.pdf", b"%PDF-1.4\nreport"),
            ("summary.pdf", b"%PDF-1.4\nsummary"),
        ],
        settings=settings,
    )

    assert result == [
        {"relative_raw_path": "report.pdf", "bytes": 15},
        {"relative_raw_path": "summary.pdf", "bytes": 16},
    ]
    assert (settings.raw_documents_dir / "report.pdf").read_bytes() == (
        b"%PDF-1.4\nreport"
    )
    assert (settings.raw_documents_dir / "summary.pdf").read_bytes() == (
        b"%PDF-1.4\nsummary"
    )


def test_document_storage_delete_blocks_traversal_and_removes_local_artifacts(
    tmp_path,
) -> None:
    settings = _settings(tmp_path)
    factory = _database_session_factory(tmp_path)
    owner = _seed_user(factory)
    _write_file(settings.raw_documents_dir / "report.pdf", b"%PDF-1.4\n")
    _write_file(settings.parsed_markdown_dir / "report.md", "parsed")
    _seed_document(factory, owner)
    storage = DocumentStorage(
        index_cleanup=_IndexCleanup(),
        database_session_factory=factory,
    )

    with pytest.raises(ValueError, match="PDF path inside the raw documents directory"):
        storage.delete_document("../report.pdf", owner=owner, settings=settings)
    result = storage.delete_document("report.pdf", owner=owner, settings=settings)

    assert result["deletion"]["raw_deleted"] is True
    assert result["deletion"]["parsed_deleted"] is True
    assert result["qdrant_cleanup"] == {
        "attempted": True,
        "deleted": True,
        "error": None,
    }
    assert not (settings.raw_documents_dir / "report.pdf").exists()
    assert not (settings.parsed_markdown_dir / "report.md").exists()
    with factory() as session:
        assert DocumentRepository(session).get_by_relative_raw_path("report.pdf") is None


def test_document_storage_delete_rejects_unsafe_database_markdown_path(
    tmp_path,
) -> None:
    settings = _settings(tmp_path)
    factory = _database_session_factory(tmp_path)
    owner = _seed_user(factory)
    outside_path = settings.parsed_markdown_dir.parent / "outside.md"
    _write_file(settings.raw_documents_dir / "report.pdf", b"%PDF-1.4\n")
    _write_file(outside_path, "outside")
    _seed_document(
        factory,
        owner,
        parsed_markdown_path="../outside.md",
    )
    storage = DocumentStorage(
        index_cleanup=_IndexCleanup(),
        database_session_factory=factory,
    )

    with pytest.raises(ValueError, match="markdown path"):
        storage.delete_document("report.pdf", owner=owner, settings=settings)

    assert (settings.raw_documents_dir / "report.pdf").exists()
    assert outside_path.read_text(encoding="utf-8") == "outside"
    with factory() as session:
        document = DocumentRepository(session).get_by_relative_raw_path("report.pdf")
        assert document is not None
        assert document.parsed_markdown_path == "../outside.md"


def test_search_results_are_filtered_to_current_user_documents(tmp_path) -> None:
    factory = _database_session_factory(tmp_path)
    owner = _seed_user(factory)
    _seed_document(
        factory,
        owner,
        parsed_markdown_path="nested/report.md",
        content_hash="allowed-hash",
    )
    response = SimpleNamespace(
        points=[
            _point("allowed-hash", "other.md", 0.9, "included by hash"),
            _point("stale-hash", "stale.md", 0.8, "excluded"),
            _point("another-hash", "report.md", 0.7, "included by file name"),
        ]
    )

    results = search_results_from_response(
        response,
        owner_user_id=owner.id,
        database_session_factory=factory,
        limit=10,
    )

    assert [result.content_hash for result in results] == [
        "allowed-hash",
        "another-hash",
    ]
    assert [result.document_name for result in results] == [
        "report.pdf",
        "report.pdf",
    ]
    assert [result.excerpt for result in results] == [
        "included by hash",
        "included by file name",
    ]


def test_search_results_are_empty_without_database_ownership(tmp_path) -> None:
    response = SimpleNamespace(
        points=[_point("stale-hash", "stale.md", 0.8, "stale content")]
    )

    results = search_results_from_response(response, limit=10)

    assert results == []


def _point(
    content_hash: str,
    source: str,
    score: float,
    content: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        score=score,
        payload={
            "content_hash": content_hash,
            "source": source,
            "content": content,
        },
    )
