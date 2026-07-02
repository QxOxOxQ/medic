import logging
from types import SimpleNamespace

import pymupdf
from sqlalchemy.orm import sessionmaker

import rag.config as settings_module
from rag.config import DocumentPreparationSettings
from rag.database.migrations import upgrade_database
from rag.database.repositories import DocumentRepository, UserRepository
from rag.database.session import create_database_engine
from rag.document_preparation import PreparationSummary, calculate_text_sha256
from rag.full_process import FullProcess


def test_full_process_execute_indexes_parsed_files_with_checksum_metadata(
    monkeypatch,
    tmp_path,
    capsys,
    caplog,
):
    caplog.set_level(logging.INFO, logger="rag.full_process")
    parsed_dir = tmp_path / "parsed"
    nested_dir = parsed_dir / "nested"
    parsed_dir.mkdir(parents=True)
    nested_dir.mkdir()
    (parsed_dir / "doc1.md").write_text("content 1", encoding="utf-8")
    (nested_dir / "doc2.md").write_text("content 2", encoding="utf-8")
    summary = PreparationSummary(scanned=2, skipped=2)
    indexed_calls = []

    monkeypatch.setattr(
        "rag.full_process.get_document_preparation_settings",
        lambda: SimpleNamespace(parsed_markdown_dir=parsed_dir),
    )

    def prepared_summary(**kwargs):
        assert kwargs["database_session_factory"] is None
        assert kwargs["owner_user_id"] is None
        return summary

    monkeypatch.setattr("rag.full_process.prepare_documents", prepared_summary)

    def record_index_text(*, text, source_metadata):
        indexed_calls.append({"text": text, "source_metadata": source_metadata})
        return 1

    monkeypatch.setattr(
        "rag.full_process.index_text",
        record_index_text,
    )

    result = FullProcess().execute()

    captured = capsys.readouterr()
    assert result == summary
    assert captured.out == (
        "scanned=2 prepared=0 reprepared=0 pruned=0 skipped=2 "
        "duplicates_removed=0 failed=0\n"
    )
    assert indexed_calls == [
        {
            "text": "content 1",
            "source_metadata": {
                "file_name": "doc1.md",
                "source": "doc1.md",
                "content_hash": calculate_text_sha256("content 1"),
            },
        },
        {
            "text": "content 2",
            "source_metadata": {
                "file_name": "doc2.md",
                "source": "nested/doc2.md",
                "content_hash": calculate_text_sha256("content 2"),
            },
        },
    ]
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "rag.full_process"
    ]
    assert messages == [
        "Starting ingestion",
        "Preparing documents",
        "Document preparation finished: scanned=2 prepared=0 reprepared=0 "
        "pruned=0 skipped=2 duplicates_removed=0 failed=0",
        f"Indexing parsed markdown files: directory={parsed_dir} files=2",
        "Indexing parsed markdown file 1/2: doc1.md",
        "Indexed parsed markdown file 1/2: doc1.md chunks=1",
        "Indexing parsed markdown file 2/2: nested/doc2.md",
        "Indexed parsed markdown file 2/2: nested/doc2.md chunks=1",
        "Finished ingestion: files=2 chunks=2",
    ]


def test_full_process_indexes_only_owner_documents_without_selection(
    monkeypatch,
    tmp_path,
) -> None:
    settings = DocumentPreparationSettings(
        raw_documents_dir=tmp_path / "data" / "raw",
        parsed_markdown_dir=tmp_path / "data" / "parsed",
    )
    (settings.parsed_markdown_dir / "owner.md").parent.mkdir(parents=True)
    (settings.parsed_markdown_dir / "owner.md").write_text(
        "owner content",
        encoding="utf-8",
    )
    (settings.parsed_markdown_dir / "other.md").write_text(
        "other content",
        encoding="utf-8",
    )
    database_url = f"sqlite:///{tmp_path / 'owner-indexing.db'}"
    upgrade_database(database_url)
    session_factory = sessionmaker(
        bind=create_database_engine(database_url),
        expire_on_commit=False,
        future=True,
    )
    with session_factory() as session:
        users = UserRepository(session)
        owner = users.create_user(username="owner", password="secret")
        other = users.create_user(username="other", password="secret")
        documents = DocumentRepository(session)
        documents.upsert_prepared_document(
            owner_user_id=owner.id,
            relative_raw_path="owner.pdf",
            original_filename="owner.pdf",
            parsed_markdown_path="owner.md",
            content_hash="owner-hash",
            byte_size=1,
            processed_at=None,
        )
        documents.upsert_prepared_document(
            owner_user_id=other.id,
            relative_raw_path="other.pdf",
            original_filename="other.pdf",
            parsed_markdown_path="other.md",
            content_hash="other-hash",
            byte_size=1,
            processed_at=None,
        )
        owner_id = owner.id
        session.commit()

    def prepared_summary(**kwargs):
        assert kwargs["owner_user_id"] == owner_id
        return PreparationSummary(scanned=1, skipped=1)

    monkeypatch.setattr("rag.full_process.prepare_documents", prepared_summary)
    indexed_sources = []

    def record_index_text(*, text, source_metadata):
        indexed_sources.append(source_metadata["source"])
        return 1

    FullProcess(
        settings=settings,
        database_session_factory=session_factory,
        indexer=record_index_text,
    ).execute(print_summary=False, owner_user_id=owner_id)

    assert indexed_sources == ["owner.md"]


def test_full_process_processes_uploaded_document_for_owner(
    monkeypatch,
    tmp_path,
) -> None:
    settings = DocumentPreparationSettings(
        raw_documents_dir=tmp_path / "data" / "raw",
        parsed_markdown_dir=tmp_path / "data" / "parsed",
    )
    _create_pdf(settings.raw_documents_dir / "report.pdf", "Manual raw document")
    database_url = f"sqlite:///{tmp_path / 'full-process.db'}"
    upgrade_database(database_url)
    engine = create_database_engine(database_url)
    monkeypatch.setenv(settings_module.SETTINGS["env"]["database_url"], database_url)

    with engine.connect():
        pass

    session_factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with session_factory() as session:
        admin = UserRepository(session).create_user(
            username="admin",
            password="secret",
            is_admin=True,
        )
        admin_id = admin.id
        DocumentRepository(session).create_uploaded_document(
            owner_user_id=admin_id,
            original_filename="report.pdf",
            relative_raw_path="report.pdf",
            byte_size=(settings.raw_documents_dir / "report.pdf").stat().st_size,
        )
        session.commit()

    indexed_sources = []

    def record_index_text(*, text, source_metadata):
        indexed_sources.append(source_metadata["source"])
        return 1

    result = FullProcess(
        settings=settings,
        database_session_factory=session_factory,
        indexer=record_index_text,
    ).execute(
        print_summary=False,
        owner_user_id=admin_id,
    )

    assert result.prepared == 1
    assert indexed_sources == ["report.md"]
    with session_factory() as session:
        document = DocumentRepository(session).get_by_relative_raw_path("report.pdf")
        assert document is not None
        assert document.owner_user_id == admin_id
        assert document.parsed_markdown_path == "report.md"
        assert document.status == "prepared"


def _create_pdf(pdf_path, text: str) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    document.save(pdf_path)
    document.close()
