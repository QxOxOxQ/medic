from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_gitignore_ignores_generated_document_directories() -> None:
    ignored_paths = [
        "data/raw/example.pdf",
        "data/parsed/example.md",
        "tmp/pdfs/private-result.png",
    ]

    result = subprocess.run(
        ["git", "check-ignore", *ignored_paths],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == ignored_paths


def test_gitignore_ignores_legacy_ingestion_manifest() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "data/ingestion_manifest.json"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == ["data/ingestion_manifest.json"]
