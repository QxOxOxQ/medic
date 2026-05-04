from pathlib import Path

from pytest import MonkeyPatch

from evaluation import config


def test_langfuse_settings_are_loaded_from_dotenv(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    (tmp_path / ".env").write_text(
        "LANGFUSE_PUBLIC_KEY=public-from-file\n"
        "LANGFUSE_SECRET_KEY=secret-from-file\n"
        "LANGFUSE_BASE_URL=https://example.langfuse.test\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)

    settings = config.get_evaluation_settings()

    assert settings.langfuse_public_key == "public-from-file"
    assert settings.langfuse_secret_key == "secret-from-file"
    assert settings.langfuse_base_url == "https://example.langfuse.test"


def test_process_environment_overrides_dotenv(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    (tmp_path / ".env").write_text(
        "LANGFUSE_PUBLIC_KEY=public-from-file\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public-from-process")

    settings = config.get_evaluation_settings()

    assert settings.langfuse_public_key == "public-from-process"
