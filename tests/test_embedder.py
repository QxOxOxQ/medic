from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

import rag.embedding.embedder as embedder_module
import rag.config as settings_module
from rag.embedding.embedder import embed_texts
from rag.embedding.fast_embedding import FastEmbedding
from rag.config import (
    get_fast_embedding_settings,
    get_openrouter_settings,
)


ENV_NAMES = settings_module.SETTINGS["env"]


def test_embedder_uses_openrouter_client_from_clients_package() -> None:
    assert embedder_module.OpenRouterClient.__module__ == "clients.openrouter"


def test_embedder_calls_openrouter_client_with_selected_model() -> None:
    class FakeOpenRouterClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def embed_texts(
            self,
            *,
            model: str,
            texts: list[str],
        ) -> list[list[float]]:
            self.calls.append({"model": model, "texts": texts})
            return [[0.1, 0.2], [0.3, 0.4]]

    fake_client = FakeOpenRouterClient()
    embedder = embedder_module.Embedder(
        provider="openrouter",
        model="openai/text-embedding-3-small",
        openrouter_client=fake_client,
    )

    embeddings = embedder.embed_texts(["first chunk", "second chunk"])

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert fake_client.calls == [
        {
            "model": "openai/text-embedding-3-small",
            "texts": ["first chunk", "second chunk"],
        }
    ]


def test_embedder_explicit_provider_and_model_do_not_read_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOpenRouterClient:
        def embed_texts(
            self,
            *,
            model: str,
            texts: list[str],
        ) -> list[list[float]]:
            return [[float(len(model)), float(len(texts[0]))]]

    def _unexpected_settings_read():
        raise AssertionError("settings should not be read for explicit model choice")

    monkeypatch.setattr(
        embedder_module,
        "get_embedding_settings",
        _unexpected_settings_read,
    )

    embedder = embedder_module.Embedder(
        provider="openrouter",
        model="openai/text-embedding-3-small",
        openrouter_client=FakeOpenRouterClient(),
    )

    assert embedder.embed_texts(["abc"]) == [[29.0, 3.0]]


def test_embed_texts_uses_settings_selected_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOpenRouterClient:
        def embed_texts(
            self,
            *,
            model: str,
            texts: list[str],
        ) -> list[list[float]]:
            return [[float(len(model)), float(len(texts[0]))]]

    monkeypatch.setattr(
        embedder_module,
        "OpenRouterClient",
        FakeOpenRouterClient,
    )

    expected_model = settings_module.SETTINGS["embedding"]["model"]
    assert embed_texts(["abc"]) == [[float(len(expected_model)), 3.0]]


@pytest.mark.parametrize(
    ("texts", "message"),
    [
        ([], "texts must not be empty"),
        ([""], "texts must not contain empty values"),
        (["   "], "texts must not contain empty values"),
        (["valid", "\n\t"], "texts must not contain empty values"),
    ],
)
def test_embed_texts_rejects_empty_inputs(
    monkeypatch: pytest.MonkeyPatch,
    texts: list[str],
    message: str,
) -> None:
    def _unexpected_call() -> None:
        raise AssertionError("Embedder should not be created for invalid input")

    monkeypatch.setattr(embedder_module, "Embedder", _unexpected_call)

    with pytest.raises(ValueError, match=message):
        embed_texts(texts)


def test_embedder_uses_fast_embedding_dense_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTextEmbedding:
        created_with: list[str] = []

        def __init__(self, *, model_name: str) -> None:
            self.created_with.append(model_name)

        def embed(self, documents: list[str]):
            assert documents == ["clinical note"]
            return [[0.1, 0.2, 0.3]]

    monkeypatch.setitem(
        embedder_module.FAST_EMBEDDING_MODEL_CLASSES,
        embedder_module.TEXT_EMBEDDING_PROVIDER,
        FakeTextEmbedding,
    )

    embedder = embedder_module.Embedder(
        provider="fast_embedding",
        model="BAAI/bge-small-en-v1.5",
    )

    assert embedder.model_config.vector_size == 384
    assert embedder.model_config.is_multivector is False
    assert embedder.embed_texts(["clinical note"]) == [[0.1, 0.2, 0.3]]
    assert FakeTextEmbedding.created_with == ["BAAI/bge-small-en-v1.5"]


def test_embedder_uses_fast_embedding_colbert_multivector_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLateInteractionTextEmbedding:
        created_with: list[str] = []

        def __init__(self, *, model_name: str) -> None:
            self.created_with.append(model_name)

        def embed(self, documents: list[str]):
            assert documents == ["clinical note"]
            return [[[0.1, 0.2], [0.3, 0.4]]]

    monkeypatch.setitem(
        embedder_module.FAST_EMBEDDING_MODEL_CLASSES,
        embedder_module.LATE_INTERACTION_TEXT_EMBEDDING_PROVIDER,
        FakeLateInteractionTextEmbedding,
    )

    embedder = embedder_module.Embedder(
        provider="fast_embedding",
        model="colbert-ir/colbertv2.0",
    )

    assert embedder.model_config.vector_size == 128
    assert embedder.model_config.is_multivector is True
    assert embedder.embed_texts(["clinical note"]) == [[[0.1, 0.2], [0.3, 0.4]]]
    assert FakeLateInteractionTextEmbedding.created_with == [
        "colbert-ir/colbertv2.0"
    ]


def test_embedder_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        embedder_module.Embedder(provider="unknown", model="anything")


def test_embedder_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unknown embedding model"):
        embedder_module.Embedder(provider="openrouter", model="unknown/model")


def test_fast_embedding_resolves_models_from_settings() -> None:
    settings = get_fast_embedding_settings()

    model_class, model_name = FastEmbedding._resolve_model(settings.default_model)

    model_settings = settings.models[settings.default_model]
    assert model_name == model_settings.model_name
    assert model_class is FastEmbedding.MODEL_CLASSES[model_settings.provider]


def test_embed_texts_live_returns_float_embedding() -> None:
    live_test_env = ENV_NAMES["run_openrouter_live_tests"]
    if os.getenv(live_test_env) != "1":
        pytest.skip(f"set {live_test_env}=1 to run live OpenRouter tests")

    try:
        get_openrouter_settings()
    except ValueError:
        api_key_env = ENV_NAMES["openrouter_api_key"]
        pytest.skip(f"set {api_key_env} in env or .env to run live OpenRouter tests")

    embeddings = embed_texts(["live embedding smoke test"])

    assert len(embeddings) == 1
    assert embeddings[0]
    assert all(isinstance(value, float) for value in embeddings[0])


def test_main_prints_embedding_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    received_texts: list[str] = []

    def _fake_embed_texts(texts: list[str]) -> list[list[float]]:
        received_texts.extend(texts)
        return [[0.1, 0.2, 0.3]]

    monkeypatch.setattr(embedder_module, "embed_texts", _fake_embed_texts)

    embedder_module.main()

    captured = capsys.readouterr()
    assert received_texts == ["This is a sample text for a test embedding."]
    assert captured.out == "Generated 1 embedding(s); first embedding dimension: 3\n"


def test_embedder_imports_with_rag_dir_first_on_sys_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    rag_dir = project_root / "rag"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(rag_dir)!r}); "
                f"sys.path.insert(1, {str(project_root)!r}); "
                "import rag.embedder"
            ),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_embedder_executes_via_run_path_without_package_import_error() -> None:
    project_root = Path(__file__).resolve().parents[1]
    embedder_path = project_root / "rag" / "embedder.py"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import runpy; runpy.run_path({str(embedder_path)!r}, run_name='not_main')",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
