from __future__ import annotations

import builtins
import logging
from types import SimpleNamespace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import pymupdf
import pytest
from sqlalchemy.orm import sessionmaker

import rag.document_preparation as document_preparation_module
from rag.config import DocumentPreparationSettings
from rag.database.migrations import upgrade_database
from rag.database.repositories import DocumentRepository, UserRepository
from rag.database.session import create_database_engine
from rag.document_preparation import (
    PreparationSummary,
    calculate_text_sha256,
    discover_raw_documents,
    prepare_documents,
)


def _create_pdf(pdf_path: Path, *lines: str) -> None:
    if pdf_path.exists():
        pdf_path.unlink()

    document = pymupdf.open()
    page = document.new_page()
    top = 72

    for index, line in enumerate(lines):
        page.insert_text((72, top + (index * 18)), line, fontsize=12)

    document.save(pdf_path)
    document.close()


def _settings_for(tmp_path: Path) -> DocumentPreparationSettings:
    return DocumentPreparationSettings(
        raw_documents_dir=tmp_path / "data" / "raw",
        parsed_markdown_dir=tmp_path / "data" / "parsed",
    )


def _database_session_factory(tmp_path: Path) -> sessionmaker:
    database_url = f"sqlite:///{tmp_path / 'documents.db'}"
    upgrade_database(database_url)
    return sessionmaker(
        bind=create_database_engine(database_url),
        expire_on_commit=False,
        future=True,
    )


def _seed_user(factory: sessionmaker) -> UUID:
    with factory() as session:
        user = UserRepository(session).create_user(username="admin", password="secret")
        session.commit()
        return user.id


def test_discover_raw_documents_returns_only_pdf_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    nested_dir = raw_dir / "nested"
    nested_dir.mkdir(parents=True)
    (raw_dir / "ignored.txt").write_text("ignore me", encoding="utf-8")
    (nested_dir / "second.md").write_text("ignore me", encoding="utf-8")
    (raw_dir / "first.pdf").write_bytes(b"%PDF-1.7")
    (nested_dir / "second.PDF").write_bytes(b"%PDF-1.7")

    documents = discover_raw_documents(raw_dir)

    assert documents == [
        raw_dir / "first.pdf",
        nested_dir / "second.PDF",
    ]


def test_parse_pdf_to_markdown_mutes_and_restores_mupdf_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_errors = pymupdf.TOOLS.mupdf_display_errors()
    original_warnings = pymupdf.TOOLS.mupdf_display_warnings()
    observed_states = []

    def fake_to_markdown(*args, **kwargs) -> str:
        observed_states.append(
            (
                pymupdf.TOOLS.mupdf_display_errors(),
                pymupdf.TOOLS.mupdf_display_warnings(),
                args,
                kwargs,
            )
        )
        return "Parsed markdown"

    monkeypatch.setattr(
        document_preparation_module.pymupdf4llm,
        "to_markdown",
        fake_to_markdown,
    )

    pymupdf.TOOLS.mupdf_display_errors(True)
    pymupdf.TOOLS.mupdf_display_warnings(True)
    try:
        markdown = document_preparation_module.parse_pdf_to_markdown(
            tmp_path / "report.pdf"
        )

        assert markdown == "Parsed markdown"
        assert observed_states == [
            (
                False,
                False,
                (str(tmp_path / "report.pdf"),),
                {
                    "use_ocr": True,
                    "force_ocr": False,
                    "show_progress": False,
                    "embed_images": False,
                    "write_images": False,
                },
            )
        ]
        assert pymupdf.TOOLS.mupdf_display_errors() is True
        assert pymupdf.TOOLS.mupdf_display_warnings() is True
    finally:
        pymupdf.TOOLS.mupdf_display_errors(original_errors)
        pymupdf.TOOLS.mupdf_display_warnings(original_warnings)


def test_parse_pdf_to_markdown_restores_mupdf_diagnostics_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_errors = pymupdf.TOOLS.mupdf_display_errors()
    original_warnings = pymupdf.TOOLS.mupdf_display_warnings()

    def failing_to_markdown(*args, **kwargs) -> str:
        raise RuntimeError("parser failed")

    monkeypatch.setattr(
        document_preparation_module.pymupdf4llm,
        "to_markdown",
        failing_to_markdown,
    )

    pymupdf.TOOLS.mupdf_display_errors(True)
    pymupdf.TOOLS.mupdf_display_warnings(True)
    try:
        with pytest.raises(RuntimeError, match="parser failed"):
            document_preparation_module.parse_pdf_to_markdown(tmp_path / "broken.pdf")

        assert pymupdf.TOOLS.mupdf_display_errors() is True
        assert pymupdf.TOOLS.mupdf_display_warnings() is True
    finally:
        pymupdf.TOOLS.mupdf_display_errors(original_errors)
        pymupdf.TOOLS.mupdf_display_warnings(original_warnings)


def test_parse_pdf_to_markdown_falls_back_when_tesseract_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []

    def fake_to_markdown(*args, **kwargs) -> str:
        calls.append(kwargs)
        if kwargs["use_ocr"]:
            raise RuntimeError("No tessdata specified and Tesseract is not installed")
        return "Parsed without OCR"

    monkeypatch.setattr(
        document_preparation_module.pymupdf4llm,
        "to_markdown",
        fake_to_markdown,
    )

    markdown = document_preparation_module.parse_pdf_to_markdown(tmp_path / "report.pdf")

    assert markdown == "Parsed without OCR"
    assert [call["use_ocr"] for call in calls] == [True, False]


def test_parse_pdf_to_markdown_accepts_ocr_with_more_recovered_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []
    noisy_words = " ".join("aa" for _ in range(25))
    noisy_markdown = f"{'�' * 120}\n{noisy_words}"
    ocr_markdown = "RecoveredClinicalText" * 6

    def fake_to_markdown(*args, **kwargs) -> str:
        calls.append(kwargs)
        if kwargs["force_ocr"]:
            return ocr_markdown
        return noisy_markdown

    monkeypatch.setattr(
        document_preparation_module.pymupdf4llm,
        "to_markdown",
        fake_to_markdown,
    )
    monkeypatch.setattr(document_preparation_module, "_has_ocr_support", lambda: True)

    markdown = document_preparation_module.parse_pdf_to_markdown(tmp_path / "report.pdf")

    assert markdown == ocr_markdown
    assert [call["force_ocr"] for call in calls] == [False, True]


def test_parse_pdf_to_markdown_uses_rapidocr_when_pymupdf_ocr_does_not_improve(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []
    noisy_markdown = f"{'�' * 160}\n|---|---|---|"
    recovered_markdown = "Badanie lekarskie wynik pacjent cholesterol calkowity 174 mg dl"

    def fake_to_markdown(*args, **kwargs) -> str:
        calls.append(kwargs)
        return noisy_markdown

    monkeypatch.setattr(
        document_preparation_module.pymupdf4llm,
        "to_markdown",
        fake_to_markdown,
    )
    monkeypatch.setattr(document_preparation_module, "_has_ocr_support", lambda: True)
    monkeypatch.setattr(
        document_preparation_module,
        "_to_markdown_with_rapidocr",
        lambda pdf_path: recovered_markdown,
    )

    markdown = document_preparation_module.parse_pdf_to_markdown(tmp_path / "report.pdf")

    assert markdown == recovered_markdown
    assert [call["force_ocr"] for call in calls] == [False, True]


def test_clean_extracted_markdown_removes_image_placeholders_and_garbage_lines() -> None:
    markdown = (
        "Useful clinical text\n"
        "**==> picture [140 x 21] intentionally omitted <==**\n"
        "**----- Start of picture text -----**<br>\n"
        "��������������������������������\n"
        "Text with one � recoverable character\n"
        "<br>**----- End of picture text -----**<br>\n"
    )

    cleaned = document_preparation_module._clean_extracted_markdown(markdown)

    assert cleaned == (
        "Useful clinical text\n"
        "\n"
        "Text with one  recoverable character"
    )


def test_low_quality_extraction_requires_ocr_when_noise_is_not_recoverable() -> None:
    raw_markdown = "��������������������������������\n|---|---|---|\n"

    with pytest.raises(ValueError, match="OCR is required"):
        document_preparation_module._validate_extraction_quality(
            pdf_path=Path("broken.pdf"),
            raw_markdown=raw_markdown,
            cleaned_markdown="|---|---|---|",
        )


def test_low_quality_extraction_accepts_noisy_text_when_enough_words_are_recovered() -> None:
    recovered_text = " ".join(f"word{index}" for index in range(30))

    document_preparation_module._validate_extraction_quality(
        pdf_path=Path("recoverable.pdf"),
        raw_markdown=f"{'�' * 100}\n{recovered_text}",
        cleaned_markdown=recovered_text,
    )


def test_has_ocr_support_returns_false_when_tessdata_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_find_spec = document_preparation_module.importlib.util.find_spec

    def find_spec_without_rapidocr(name: str):
        if name == "rapidocr_onnxruntime":
            return None
        return real_find_spec(name)

    def missing_tessdata() -> None:
        raise RuntimeError("No tessdata specified and Tesseract is not installed")

    monkeypatch.setattr(
        document_preparation_module.importlib.util,
        "find_spec",
        find_spec_without_rapidocr,
    )
    monkeypatch.setattr(document_preparation_module.pymupdf, "get_tessdata", missing_tessdata)

    assert document_preparation_module._has_ocr_support() is False


def test_has_ocr_support_uses_rapidocr_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def find_spec_with_rapidocr(name: str) -> object | None:
        if name == "rapidocr_onnxruntime":
            return object()
        return None

    def import_with_rapidocr(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "rapidocr_onnxruntime":
            return SimpleNamespace(RapidOCR=object)
        return real_import(name, globals, locals, fromlist, level)

    def unexpected_tessdata_lookup() -> None:
        raise AssertionError("RapidOCR should be enough to enable OCR support")

    monkeypatch.setattr(
        document_preparation_module.importlib.util,
        "find_spec",
        find_spec_with_rapidocr,
    )
    monkeypatch.setattr(builtins, "__import__", import_with_rapidocr)
    monkeypatch.setattr(
        document_preparation_module.pymupdf,
        "get_tessdata",
        unexpected_tessdata_lookup,
    )

    assert document_preparation_module._has_ocr_support() is True


def test_has_ocr_support_returns_false_when_rapidocr_native_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def find_spec_with_rapidocr(name: str) -> object | None:
        if name == "rapidocr_onnxruntime":
            return object()
        return None

    def import_with_broken_rapidocr(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "rapidocr_onnxruntime":
            raise ImportError("libxcb.so.1: cannot open shared object file")
        return real_import(name, globals, locals, fromlist, level)

    def missing_tessdata() -> None:
        raise RuntimeError("No tessdata specified and Tesseract is not installed")

    monkeypatch.setattr(
        document_preparation_module.importlib.util,
        "find_spec",
        find_spec_with_rapidocr,
    )
    monkeypatch.setattr(builtins, "__import__", import_with_broken_rapidocr)
    monkeypatch.setattr(document_preparation_module.pymupdf, "get_tessdata", missing_tessdata)

    assert document_preparation_module._has_ocr_support() is False


@pytest.fixture
def mock_settings(tmp_path):
    settings = _settings_for(tmp_path)
    with patch("rag.document_preparation.get_document_preparation_settings", return_value=settings):
        yield settings


def test_prepare_documents_converts_new_pdf_to_markdown(
    mock_settings,
    caplog,
) -> None:
    settings = mock_settings
    caplog.set_level(logging.INFO, logger="rag.document_preparation")
    source_pdf = settings.raw_documents_dir / "clinical" / "report.pdf"
    source_pdf.parent.mkdir(parents=True)
    _create_pdf(source_pdf, "Clinical Summary", "Patient is improving.")

    summary = prepare_documents()

    output_markdown = settings.parsed_markdown_dir / "clinical" / "report.md"

    assert summary == PreparationSummary(scanned=1, prepared=1)
    assert output_markdown.exists()
    assert output_markdown.read_text(encoding="utf-8").endswith("\n")
    assert "Clinical Summary" in output_markdown.read_text(encoding="utf-8")
    assert len(calculate_text_sha256(output_markdown.read_text(encoding="utf-8"))) == 64
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "rag.document_preparation"
    ]
    assert f"Discovering raw documents: directory={settings.raw_documents_dir}" in messages
    assert "Discovered raw documents: files=1" in messages
    assert "Preparing raw document 1/1: clinical/report.pdf" in messages
    assert "Prepared raw document: clinical/report.pdf parsed=clinical/report.md" in messages
    assert (
        "Finished document preparation: scanned=1 prepared=1 reprepared=0 pruned=0 skipped=0 duplicates_removed=0 failed=0"
        in messages
    )


def test_prepare_documents_skips_unchanged_pdf(mock_settings) -> None:
    settings = mock_settings
    source_pdf = settings.raw_documents_dir / "report.pdf"
    source_pdf.parent.mkdir(parents=True)
    _create_pdf(source_pdf, "Stable content")

    first_summary = prepare_documents()
    output_markdown = settings.parsed_markdown_dir / "report.md"
    markdown_mtime_before = output_markdown.stat().st_mtime_ns

    second_summary = prepare_documents()

    assert first_summary == PreparationSummary(scanned=1, prepared=1)
    assert second_summary == PreparationSummary(scanned=1, skipped=1)
    assert output_markdown.stat().st_mtime_ns == markdown_mtime_before


def test_prepare_documents_parses_unchanged_pdf_before_skipping(mock_settings) -> None:
    settings = mock_settings
    source_pdf = settings.raw_documents_dir / "report.pdf"
    source_pdf.parent.mkdir(parents=True)
    source_pdf.write_bytes(b"%PDF-1.7")
    calls = []

    def parser(pdf_path: Path) -> str:
        calls.append(pdf_path)
        return "Stable parsed content"

    first_summary = prepare_documents(parser=parser)
    second_summary = prepare_documents(parser=parser)
    output_markdown = settings.parsed_markdown_dir / "report.md"

    assert first_summary == PreparationSummary(scanned=1, prepared=1)
    assert second_summary == PreparationSummary(scanned=1, skipped=1)
    assert calls == [source_pdf, source_pdf]
    assert len(calculate_text_sha256(output_markdown.read_text(encoding="utf-8"))) == 64


def test_prepare_documents_reprocesses_changed_pdf(mock_settings) -> None:
    settings = mock_settings
    source_pdf = settings.raw_documents_dir / "report.pdf"
    source_pdf.parent.mkdir(parents=True)
    _create_pdf(source_pdf, "Version one")

    prepare_documents()
    first_hash = calculate_text_sha256(
        (settings.parsed_markdown_dir / "report.md").read_text(encoding="utf-8")
    )

    _create_pdf(source_pdf, "Version two")
    summary = prepare_documents()

    output_markdown = settings.parsed_markdown_dir / "report.md"
    second_hash = calculate_text_sha256(output_markdown.read_text(encoding="utf-8"))

    assert summary == PreparationSummary(scanned=1, reprepared=1)
    assert "Version two" in output_markdown.read_text(encoding="utf-8")
    assert first_hash != second_hash


def test_prepare_documents_continues_after_single_file_failure(mock_settings) -> None:
    settings = mock_settings
    valid_pdf = settings.raw_documents_dir / "valid.pdf"
    invalid_pdf = settings.raw_documents_dir / "broken.pdf"
    valid_pdf.parent.mkdir(parents=True)
    _create_pdf(valid_pdf, "Valid document")
    invalid_pdf.write_text("this is not a valid pdf", encoding="utf-8")

    summary = prepare_documents()

    assert summary.scanned == 2
    assert summary.prepared == 1
    assert summary.failed == 1
    assert summary.failed_paths == ["broken.pdf"]
    assert (settings.parsed_markdown_dir / "valid.md").exists()
    assert not (settings.parsed_markdown_dir / "broken.md").exists()


def test_prepare_documents_recovers_multiple_low_quality_pdfs_with_rapidocr(
    monkeypatch: pytest.MonkeyPatch,
    mock_settings,
) -> None:
    settings = mock_settings
    first_pdf = settings.raw_documents_dir / "synthetic_scan_one.pdf"
    second_pdf = settings.raw_documents_dir / "synthetic_scan_two.pdf"
    first_pdf.parent.mkdir(parents=True)
    first_pdf.write_bytes(b"%PDF-1.7")
    second_pdf.write_bytes(b"%PDF-1.7")
    noisy_markdown = f"{'�' * 160}\n|---|---|---|"

    def fake_to_markdown(*args, **kwargs) -> str:
        return noisy_markdown

    def fake_rapidocr_markdown(pdf_path: Path) -> str:
        return (
            f"Recovered clinical text for {pdf_path.name} "
            "badanie wynik pacjent cholesterol calkowity 174 mg dl"
        )

    monkeypatch.setattr(
        document_preparation_module.pymupdf4llm,
        "to_markdown",
        fake_to_markdown,
    )
    monkeypatch.setattr(document_preparation_module, "_has_ocr_support", lambda: True)
    monkeypatch.setattr(
        document_preparation_module,
        "_to_markdown_with_rapidocr",
        fake_rapidocr_markdown,
    )

    summary = prepare_documents()

    assert summary == PreparationSummary(scanned=2, prepared=2)
    assert (settings.parsed_markdown_dir / "synthetic_scan_one.md").exists()
    assert (settings.parsed_markdown_dir / "synthetic_scan_two.md").exists()


def test_prepare_documents_failed_reprocessing_removes_stale_markdown(
    mock_settings,
) -> None:
    settings = mock_settings
    source_pdf = settings.raw_documents_dir / "report.pdf"
    source_pdf.parent.mkdir(parents=True)
    _create_pdf(source_pdf, "Version one")

    first_summary = prepare_documents()
    output_markdown = settings.parsed_markdown_dir / "report.md"
    assert first_summary == PreparationSummary(scanned=1, prepared=1)
    assert output_markdown.exists()

    source_pdf.write_text("not a valid pdf anymore", encoding="utf-8")

    summary = prepare_documents()

    assert summary == PreparationSummary(scanned=1, failed=1, failed_paths=["report.pdf"])
    assert not output_markdown.exists()


def test_prepare_documents_persists_failed_document_error(tmp_path: Path) -> None:
    settings = _settings_for(tmp_path)
    broken_pdf = settings.raw_documents_dir / "broken.pdf"
    broken_pdf.parent.mkdir(parents=True)
    broken_pdf.write_bytes(b"%PDF-1.7\n")
    factory = _database_session_factory(tmp_path)
    user_id = _seed_user(factory)
    with factory() as session:
        DocumentRepository(session).create_uploaded_document(
            owner_user_id=user_id,
            original_filename="broken.pdf",
            relative_raw_path="broken.pdf",
            byte_size=broken_pdf.stat().st_size,
        )
        session.commit()

    def failing_parser(path: Path) -> str:
        raise RuntimeError(f"parser failed for {path.name}")

    summary = prepare_documents(
        parser=failing_parser,
        settings=settings,
        database_session_factory=factory,
        owner_user_id=user_id,
    )

    assert summary == PreparationSummary(
        scanned=1,
        failed=1,
        failed_paths=["broken.pdf"],
    )
    with factory() as session:
        document = DocumentRepository(session).get_by_relative_raw_path("broken.pdf")
        assert document is not None
        assert document.status == "failed"
        assert document.processing_error == "RuntimeError: parser failed for broken.pdf"
        assert document.parsed_markdown_path is None
        assert document.content_hash is None
        assert document.processed_at is not None


def test_expected_parse_failure_demo_pdf_is_marked_failed(tmp_path: Path) -> None:
    settings = _settings_for(tmp_path)
    demo_pdf = (
        Path(__file__).resolve().parents[1]
        / "demo_documents"
        / "failure_cases"
        / "EXPECTED_PARSE_FAILURE_invalid_pdf.pdf"
    )
    target_pdf = settings.raw_documents_dir / demo_pdf.name
    target_pdf.parent.mkdir(parents=True)
    target_pdf.write_bytes(demo_pdf.read_bytes())
    factory = _database_session_factory(tmp_path)
    user_id = _seed_user(factory)
    with factory() as session:
        DocumentRepository(session).create_uploaded_document(
            owner_user_id=user_id,
            original_filename=demo_pdf.name,
            relative_raw_path=demo_pdf.name,
            byte_size=target_pdf.stat().st_size,
        )
        session.commit()

    summary = prepare_documents(
        settings=settings,
        database_session_factory=factory,
        owner_user_id=user_id,
    )

    assert summary == PreparationSummary(
        scanned=1,
        failed=1,
        failed_paths=[demo_pdf.name],
    )
    with factory() as session:
        document = DocumentRepository(session).get_by_relative_raw_path(demo_pdf.name)
        assert document is not None
        assert document.status == "failed"
        assert document.processing_error
        assert document.parsed_markdown_path is None
        assert document.content_hash is None


def test_prepare_documents_prunes_deleted_raw_documents(
    tmp_path,
) -> None:
    settings = _settings_for(tmp_path)
    factory = _database_session_factory(tmp_path)
    user_id = _seed_user(factory)
    existing_pdf = settings.raw_documents_dir / "keep.pdf"
    deleted_pdf = settings.raw_documents_dir / "remove.pdf"
    existing_pdf.parent.mkdir(parents=True)
    _create_pdf(existing_pdf, "Keep me")
    _create_pdf(deleted_pdf, "Remove me")
    with factory() as session:
        repository = DocumentRepository(session)
        repository.create_uploaded_document(
            owner_user_id=user_id,
            original_filename="keep.pdf",
            relative_raw_path="keep.pdf",
            byte_size=existing_pdf.stat().st_size,
        )
        repository.create_uploaded_document(
            owner_user_id=user_id,
            original_filename="remove.pdf",
            relative_raw_path="remove.pdf",
            byte_size=deleted_pdf.stat().st_size,
        )
        session.commit()

    first_summary = prepare_documents(
        settings=settings,
        database_session_factory=factory,
        owner_user_id=user_id,
    )
    assert first_summary == PreparationSummary(scanned=2, prepared=2)

    deleted_pdf.unlink()

    summary = prepare_documents(
        settings=settings,
        database_session_factory=factory,
        owner_user_id=user_id,
    )

    assert summary == PreparationSummary(scanned=1, pruned=1, skipped=1)
    assert (settings.parsed_markdown_dir / "keep.md").exists()
    with factory() as session:
        repository = DocumentRepository(session)
        keep = repository.get_by_relative_raw_path("keep.pdf")
        remove = repository.get_by_relative_raw_path("remove.pdf")
        assert keep is not None
        assert keep.status == "prepared"
        assert remove is not None
        assert remove.status == "stale"


def test_prepare_documents_removes_duplicate_upload(tmp_path) -> None:
    settings = _settings_for(tmp_path)
    factory = _database_session_factory(tmp_path)
    user_id = _seed_user(factory)
    original_pdf = settings.raw_documents_dir / "first" / "report.pdf"
    duplicate_pdf = settings.raw_documents_dir / "second" / "report.pdf"
    original_pdf.parent.mkdir(parents=True)
    duplicate_pdf.parent.mkdir(parents=True)
    original_pdf.write_bytes(b"%PDF-1.7")
    duplicate_pdf.write_bytes(b"%PDF-1.7")
    with factory() as session:
        repository = DocumentRepository(session)
        repository.create_uploaded_document(
            owner_user_id=user_id,
            original_filename="report.pdf",
            relative_raw_path="first/report.pdf",
            byte_size=original_pdf.stat().st_size,
        )
        repository.create_uploaded_document(
            owner_user_id=user_id,
            original_filename="report-copy.pdf",
            relative_raw_path="second/report.pdf",
            byte_size=duplicate_pdf.stat().st_size,
        )
        session.commit()

    summary = prepare_documents(
        parser=lambda path: "Identical content",
        settings=settings,
        database_session_factory=factory,
        owner_user_id=user_id,
    )

    assert summary == PreparationSummary(scanned=2, prepared=1, duplicates_removed=1)
    assert original_pdf.exists()
    assert not duplicate_pdf.exists()
    assert not duplicate_pdf.parent.exists()
    with factory() as session:
        repository = DocumentRepository(session)
        assert repository.get_by_relative_raw_path("first/report.pdf") is not None
        assert repository.get_by_relative_raw_path("second/report.pdf") is None


def test_prepare_documents_keeps_reprocessed_document_with_matching_content(
    tmp_path,
) -> None:
    settings = _settings_for(tmp_path)
    factory = _database_session_factory(tmp_path)
    user_id = _seed_user(factory)
    first_pdf = settings.raw_documents_dir / "first" / "report.pdf"
    second_pdf = settings.raw_documents_dir / "second" / "other.pdf"
    first_pdf.parent.mkdir(parents=True)
    second_pdf.parent.mkdir(parents=True)
    first_pdf.write_bytes(b"%PDF-1.7")
    second_pdf.write_bytes(b"%PDF-1.7")
    with factory() as session:
        repository = DocumentRepository(session)
        repository.upsert_prepared_document(
            owner_user_id=user_id,
            relative_raw_path="first/report.pdf",
            original_filename="report.pdf",
            parsed_markdown_path=None,
            content_hash=calculate_text_sha256("Converged content\n"),
            byte_size=first_pdf.stat().st_size,
            processed_at=datetime.fromisoformat("2026-06-02T10:40:22+00:00"),
            status="prepared",
        )
        repository.upsert_prepared_document(
            owner_user_id=user_id,
            relative_raw_path="second/other.pdf",
            original_filename="other.pdf",
            parsed_markdown_path=None,
            content_hash="previous-different-hash",
            byte_size=second_pdf.stat().st_size,
            processed_at=datetime.fromisoformat("2026-06-02T10:40:22+00:00"),
            status="prepared",
        )
        session.commit()

    summary = prepare_documents(
        parser=lambda path: "Converged content",
        settings=settings,
        database_session_factory=factory,
        owner_user_id=user_id,
    )

    assert summary.duplicates_removed == 0
    assert first_pdf.exists()
    assert second_pdf.exists()
    with factory() as session:
        repository = DocumentRepository(session)
        assert repository.get_by_relative_raw_path("second/other.pdf") is not None


def test_prepare_documents_selected_paths_do_not_mark_unselected_documents_stale(
    tmp_path,
) -> None:
    settings = _settings_for(tmp_path)
    factory = _database_session_factory(tmp_path)
    user_id = _seed_user(factory)
    selected_pdf = settings.raw_documents_dir / "selected.pdf"
    stale_markdown = settings.parsed_markdown_dir / "stale.md"
    selected_pdf.parent.mkdir(parents=True)
    stale_markdown.parent.mkdir(parents=True)
    selected_pdf.write_bytes(b"%PDF-1.7")
    stale_markdown.write_text("stale parsed content", encoding="utf-8")
    with factory() as session:
        repository = DocumentRepository(session)
        repository.create_uploaded_document(
            owner_user_id=user_id,
            original_filename="selected.pdf",
            relative_raw_path="selected.pdf",
            byte_size=selected_pdf.stat().st_size,
        )
        repository.upsert_prepared_document(
            owner_user_id=user_id,
            relative_raw_path="stale.pdf",
            original_filename="stale.pdf",
            parsed_markdown_path="stale.md",
            content_hash="stale-hash",
            byte_size=None,
            processed_at=datetime.fromisoformat("2026-06-02T10:40:22+00:00"),
            status="prepared",
        )
        session.commit()

    summary = prepare_documents(
        parser=lambda path: f"Selected content from {path.name}",
        settings=settings,
        selected_raw_paths=["selected.pdf"],
        database_session_factory=factory,
        owner_user_id=user_id,
    )

    assert summary == PreparationSummary(scanned=1, prepared=1)
    assert stale_markdown.exists()
    with factory() as session:
        repository = DocumentRepository(session)
        selected = repository.get_by_relative_raw_path("selected.pdf")
        stale = repository.get_by_relative_raw_path("stale.pdf")
        assert selected is not None
        assert selected.status == "prepared"
        assert stale is not None
        assert stale.status == "prepared"
