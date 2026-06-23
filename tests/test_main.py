from pathlib import Path
import subprocess
import sys

import main as main_module
import pymupdf

import rag.config as settings_module
from rag.config import DocumentPreparationSettings
from rag.document_preparation import PreparationSummary


def _create_pdf(pdf_path: Path, text: str) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    document.save(pdf_path)
    document.close()


def test_main_without_arguments_prints_help() -> None:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "main.py")],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage: medic" in result.stdout


def test_main_setup_prepares_local_files_without_db(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".env.example").write_text("EXAMPLE=value\n", encoding="utf-8")

    exit_code = main_module.main(["setup", "--skip-db"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == (
        "env=created raw_dir=created parsed_dir=created "
        "postgres=skipped qdrant=skipped\n"
    )
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "EXAMPLE=value\n"
    assert (tmp_path / "data" / "raw").is_dir()
    assert (tmp_path / "data" / "parsed").is_dir()
    assert not (tmp_path / "data" / "ingestion_manifest.json").exists()


def test_main_prepare_returns_zero_and_prints_summary(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    settings = DocumentPreparationSettings(
        raw_documents_dir=tmp_path / "data" / "raw",
        parsed_markdown_dir=tmp_path / "data" / "parsed",
    )
    source_pdf = settings.raw_documents_dir / "report.pdf"
    source_pdf.parent.mkdir(parents=True)
    _create_pdf(source_pdf, "Prepared by CLI")
    monkeypatch.setattr(
        "rag.document_preparation.get_document_preparation_settings",
        lambda: settings,
    )

    class FailingFullProcess:
        def __init__(self) -> None:
            raise AssertionError("prepare should not run full ingestion")

    monkeypatch.setattr(main_module, "FullProcess", FailingFullProcess)

    exit_code = main_module.main(["prepare"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "scanned=1 prepared=1 reprepared=0 pruned=0 skipped=0 failed=0\n"

def test_main_ingest_runs_full_process(monkeypatch, capsys) -> None:
    def failing_prepare_documents() -> PreparationSummary:
        raise AssertionError("ingest should run FullProcess")

    class RecordingFullProcess:
        def execute(self) -> PreparationSummary:
            summary = PreparationSummary(scanned=1, prepared=1)
            print(summary.as_report_line())
            return summary

    monkeypatch.setattr(main_module, "FullProcess", RecordingFullProcess)
    monkeypatch.setattr(main_module, "prepare_documents", failing_prepare_documents)

    exit_code = main_module.main(["ingest"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "scanned=1 prepared=1 reprepared=0 pruned=0 skipped=0 failed=0\n"
