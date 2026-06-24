import json
from pathlib import Path

import pytest

import rag.config as settings_module
from rag.config import (
    get_database_settings,
    get_chat_model_settings,
    get_document_preparation_settings,
    get_embedding_settings,
    get_fast_embedding_settings,
    get_openrouter_settings,
    get_qdrant_settings,
)


ENV_NAMES = settings_module.SETTINGS["env"]


def _dotenv(**values: str) -> str:
    return "".join(f"{ENV_NAMES[name]}={value}\n" for name, value in values.items())


def _clear_env(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    for name in names:
        monkeypatch.delenv(ENV_NAMES[name], raising=False)


def test_settings_source_is_plain_json() -> None:
    loaded_json = json.loads(settings_module.SETTINGS_PATH.read_text(encoding="utf-8"))

    assert settings_module.SETTINGS_PATH.name == "settings.json"
    assert loaded_json == settings_module.SETTINGS
    assert not settings_module.SETTINGS_PATH.with_suffix(".py").exists()


def test_get_qdrant_settings_reads_required_values_from_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
    _clear_env(
        monkeypatch,
        "qdrant_url",
        "qdrant_api_key",
        "qdrant_collection_name",
        "openrouter_api_key",
    )
    (tmp_path / ".env").write_text(
        _dotenv(
            qdrant_url="http://localhost:6333",
            qdrant_api_key="qdrant-key",
            openrouter_api_key="openrouter-key",
        )
    )

    settings = get_qdrant_settings()

    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_api_key == "qdrant-key"
    assert (
        settings.qdrant_collection_name
        == settings_module.SETTINGS["qdrant"]["collection_name"]
    )


def test_get_qdrant_settings_prefers_process_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv(ENV_NAMES["qdrant_url"], "https://qdrant.example:7443")
    monkeypatch.setenv(ENV_NAMES["qdrant_api_key"], "runtime-key")
    monkeypatch.setenv(ENV_NAMES["openrouter_api_key"], "runtime-openrouter-key")
    (tmp_path / ".env").write_text(
        _dotenv(
            qdrant_url="http://localhost:6333",
            qdrant_api_key="file-key",
            openrouter_api_key="file-openrouter-key",
        )
    )

    settings = get_qdrant_settings()

    assert settings.qdrant_url == "https://qdrant.example:7443"
    assert settings.qdrant_api_key == "runtime-key"


def test_get_qdrant_settings_allows_collection_name_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
    _clear_env(monkeypatch, "qdrant_url", "qdrant_api_key", "qdrant_collection_name")
    (tmp_path / ".env").write_text(
        _dotenv(
            qdrant_url="http://localhost:6333",
            qdrant_api_key="qdrant-key",
            qdrant_collection_name="demo_collection",
        )
    )

    settings = get_qdrant_settings()

    assert settings.qdrant_collection_name == "demo_collection"


def test_get_qdrant_settings_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
    _clear_env(monkeypatch, "qdrant_url", "qdrant_api_key", "qdrant_collection_name")
    (tmp_path / ".env").write_text(_dotenv(qdrant_url="https://qdrant.example"))

    with pytest.raises(ValueError, match=ENV_NAMES["qdrant_api_key"]):
        get_qdrant_settings()


def test_get_qdrant_settings_raises_for_missing_required_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
    _clear_env(monkeypatch, "qdrant_url", "qdrant_api_key", "openrouter_api_key")
    (tmp_path / ".env").write_text("")

    with pytest.raises(ValueError, match=ENV_NAMES["qdrant_url"]):
        get_qdrant_settings()


def test_get_qdrant_settings_does_not_reuse_values_from_previous_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    (first_root / ".env").write_text(
        _dotenv(
            qdrant_url="http://dev.local",
            qdrant_api_key="dev-key",
            openrouter_api_key="dev-openrouter-key",
        )
    )
    (second_root / ".env").write_text(
        _dotenv(
            qdrant_api_key="prod-key",
            openrouter_api_key="prod-openrouter-key",
        )
    )
    _clear_env(monkeypatch, "qdrant_url", "qdrant_api_key", "openrouter_api_key")

    monkeypatch.setattr(settings_module, "PROJECT_ROOT", first_root)
    assert get_qdrant_settings().qdrant_url == "http://dev.local"

    monkeypatch.setattr(settings_module, "PROJECT_ROOT", second_root)
    with pytest.raises(ValueError, match=ENV_NAMES["qdrant_url"]):
        get_qdrant_settings()


def test_get_qdrant_settings_reads_defaults_from_settings_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
    _clear_env(monkeypatch, "qdrant_url", "qdrant_api_key")
    (tmp_path / ".env").write_text(
        _dotenv(qdrant_url="http://localhost:6333", qdrant_api_key="qdrant-key")
    )

    settings = get_qdrant_settings()
    qdrant_config = settings_module.SETTINGS["qdrant"]

    assert settings.qdrant_collection_name == qdrant_config["collection_name"]
    assert settings.client_timeout_seconds == qdrant_config["client_timeout_seconds"]
    assert settings.dense_vector_name == qdrant_config["dense_vector"]["name"]
    assert settings.dense_vector_size == qdrant_config["dense_vector"]["size"]
    assert settings.sparse_vector_name == qdrant_config["sparse_vector"]["name"]
    assert settings.sparse_vector_model == qdrant_config["sparse_vector"]["model"]
    assert settings.sparse_vector_on_disk == qdrant_config["sparse_vector"]["on_disk"]
    assert settings.prefetch_limit == qdrant_config["prefetch_limit"]
    assert settings.quantization_encoding == qdrant_config["quantization"]["encoding"]
    assert settings.quantization_always_ram == qdrant_config["quantization"]["always_ram"]


def test_get_openrouter_settings_reads_required_values_from_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
    _clear_env(monkeypatch, "openrouter_api_key")
    (tmp_path / ".env").write_text(
        _dotenv(openrouter_api_key="file-openrouter-key")
    )

    settings = get_openrouter_settings()

    assert settings.api_key == "file-openrouter-key"
    assert settings.base_url == settings_module.SETTINGS["openrouter"]["base_url"]


def test_get_openrouter_settings_prefers_process_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv(ENV_NAMES["openrouter_api_key"], "runtime-openrouter-key")
    (tmp_path / ".env").write_text(
        _dotenv(openrouter_api_key="file-openrouter-key")
    )

    settings = get_openrouter_settings()

    assert settings.api_key == "runtime-openrouter-key"


def test_get_openrouter_settings_raises_for_missing_required_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
    _clear_env(monkeypatch, "openrouter_api_key")
    (tmp_path / ".env").write_text("")

    with pytest.raises(ValueError, match=ENV_NAMES["openrouter_api_key"]):
        get_openrouter_settings()


def test_get_chat_model_settings_reads_provider_agnostic_chat_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
    _clear_env(monkeypatch, "openrouter_api_key")
    (tmp_path / ".env").write_text(
        _dotenv(openrouter_api_key="file-openrouter-key")
    )

    settings = get_chat_model_settings()
    chat_config = settings_module.SETTINGS["chat"]

    assert settings.provider == chat_config["provider"]
    assert settings.model == chat_config["model"]
    assert settings.temperature == chat_config["temperature"]
    assert (
        settings.max_retrieval_queries
        == chat_config["max_retrieval_queries"]
    )
    assert settings.max_consultations == chat_config["max_consultations"]
    assert settings.max_review_rounds == chat_config["max_review_rounds"]
    assert settings.provider_options["api_key"] == "file-openrouter-key"
    assert (
        settings.provider_options["base_url"]
        == settings_module.SETTINGS["openrouter"]["base_url"]
    )


def test_get_database_settings_reads_required_value_from_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
    _clear_env(monkeypatch, "database_url")
    (tmp_path / ".env").write_text(_dotenv(database_url="postgresql://local/medic"))

    settings = get_database_settings()

    assert settings.database_url == "postgresql://local/medic"


def test_get_database_settings_prefers_process_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv(ENV_NAMES["database_url"], "postgresql://runtime/medic")
    (tmp_path / ".env").write_text(_dotenv(database_url="postgresql://file/medic"))

    settings = get_database_settings()

    assert settings.database_url == "postgresql://runtime/medic"


def test_get_database_settings_raises_for_missing_required_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
    _clear_env(monkeypatch, "database_url")
    (tmp_path / ".env").write_text("")

    with pytest.raises(ValueError, match=ENV_NAMES["database_url"]):
        get_database_settings()


def test_get_embedding_settings_uses_settings_hash() -> None:
    settings = get_embedding_settings()
    config = settings_module.SETTINGS["embedding"]

    assert settings.provider == config["provider"]
    assert settings.model == config["model"]


def test_get_document_preparation_settings_uses_project_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)

    settings = get_document_preparation_settings()

    assert settings.raw_documents_dir == tmp_path / "data" / "raw"
    assert settings.parsed_markdown_dir == tmp_path / "data" / "parsed"


def test_get_fast_embedding_settings_uses_settings_hash() -> None:
    settings = get_fast_embedding_settings()
    config = settings_module.SETTINGS["fast_embedding"]

    assert settings.default_model == config["default_model"]
    assert set(settings.models) == set(config["models"])
    assert settings.models[settings.default_model].model_name == (
        config["models"][settings.default_model]["model_name"]
    )


def test_python_modules_do_not_hardcode_environment_variable_names() -> None:
    ignored_parts = {".venv", "__pycache__"}
    offenders = []

    for path in settings_module.PROJECT_ROOT.rglob("*.py"):
        if ignored_parts.intersection(path.parts):
            continue

        text = path.read_text(encoding="utf-8")
        env_hits = sorted(env_name for env_name in ENV_NAMES.values() if env_name in text)
        if env_hits:
            offenders.append(
                f"{path.relative_to(settings_module.PROJECT_ROOT)}: {', '.join(env_hits)}"
            )

    assert offenders == []
