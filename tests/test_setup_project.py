from __future__ import annotations

import stat
from pathlib import Path

from sqlalchemy.orm import sessionmaker

from rag.config import DocumentPreparationSettings
from rag.database.migrations import upgrade_database
from rag.database.session import create_database_engine
from rag.setup_project import setup_project


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _settings_for(tmp_path: Path) -> DocumentPreparationSettings:
    return DocumentPreparationSettings(
        raw_documents_dir=tmp_path / "data" / "raw",
        parsed_markdown_dir=tmp_path / "data" / "parsed",
    )


def test_setup_project_creates_runtime_files_and_checks_qdrant(tmp_path: Path) -> None:
    settings = _settings_for(tmp_path)
    (tmp_path / ".env.example").write_text(
        "MEDIC_DASHBOARD_USERNAME=admin\n"
        "MEDIC_DASHBOARD_PASSWORD=secret\n",
        encoding="utf-8",
    )
    qdrant_calls = []
    database_calls = []
    database_session_factory = _database_session_factory(tmp_path)

    summary = setup_project(
        project_root=tmp_path,
        document_settings=settings,
        database_setup=lambda: database_calls.append("setup"),
        database_session_factory=database_session_factory,
        qdrant_setup=lambda: qdrant_calls.append("setup"),
    )

    assert database_calls == ["setup"]
    assert qdrant_calls == ["setup"]
    assert "MEDIC_DASHBOARD_USERNAME=admin" in (tmp_path / ".env").read_text(
        encoding="utf-8"
    )
    assert settings.raw_documents_dir.is_dir()
    assert settings.parsed_markdown_dir.is_dir()
    assert summary.as_report_line() == (
        "env=created raw_dir=created parsed_dir=created "
        "postgres=ready qdrant=ready"
    )


def test_setup_project_preserves_existing_runtime_files_when_db_is_skipped(
    tmp_path: Path,
) -> None:
    settings = _settings_for(tmp_path)
    settings.raw_documents_dir.mkdir(parents=True)
    settings.parsed_markdown_dir.mkdir(parents=True)
    (tmp_path / ".env").write_text("EXISTING=value\n", encoding="utf-8")

    summary = setup_project(
        project_root=tmp_path,
        document_settings=settings,
        setup_database=False,
        setup_qdrant=False,
    )

    assert (tmp_path / ".env").read_text(encoding="utf-8") == "EXISTING=value\n"
    assert summary.as_report_line() == (
        "env=present raw_dir=present parsed_dir=present "
        "postgres=skipped qdrant=skipped"
    )


def test_setup_script_runs_dependency_sync_services_and_cli() -> None:
    script_path = PROJECT_ROOT / "scripts" / "setup.sh"
    script = script_path.read_text(encoding="utf-8")

    assert script_path.stat().st_mode & stat.S_IXUSR
    assert "uv sync" in script
    assert 'SERVICES+=("postgres")' in script
    assert 'SERVICES+=("qdrant")' not in script
    assert 'docker-compose up -d "${SERVICES[@]}"' in script
    assert 'docker compose up -d "${SERVICES[@]}"' in script
    assert "--no-services" in script
    assert "--skip-postgres" in script
    assert "--skip-qdrant" in script
    assert 'uv run python main.py setup "${SETUP_ARGS[@]}"' in script


def _database_session_factory(tmp_path: Path) -> sessionmaker:
    database_url = f"sqlite:///{tmp_path / 'setup.db'}"
    upgrade_database(database_url)
    engine = create_database_engine(database_url)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
