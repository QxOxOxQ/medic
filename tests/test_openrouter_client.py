from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import clients.openrouter as openrouter_module
import rag.config as settings_module
from clients.openrouter import OpenRouterClient, get_openrouter_client


ENV_NAMES = settings_module.SETTINGS["env"]


def test_get_openrouter_client_reads_api_key_from_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(openrouter_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv(ENV_NAMES["openrouter_api_key"], raising=False)
    (tmp_path / ".env").write_text(
        f"{ENV_NAMES['openrouter_api_key']}=file-openrouter-key\n"
    )

    client = get_openrouter_client()
    base_url = settings_module.SETTINGS["openrouter"]["base_url"]

    assert client.api_key == "file-openrouter-key"
    assert str(client.base_url) == f"{base_url}/"


def test_get_openrouter_client_prefers_process_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(openrouter_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv(ENV_NAMES["openrouter_api_key"], "runtime-openrouter-key")
    (tmp_path / ".env").write_text(
        f"{ENV_NAMES['openrouter_api_key']}=file-openrouter-key\n"
    )

    client = get_openrouter_client()
    base_url = settings_module.SETTINGS["openrouter"]["base_url"]

    assert client.api_key == "runtime-openrouter-key"
    assert str(client.base_url) == f"{base_url}/"


def test_openrouter_client_embeds_texts_through_openai_embeddings_api() -> None:
    class FakeEmbeddingsApi:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def create(
            self,
            *,
            model: str,
            input: list[str],
            encoding_format: str,
        ) -> SimpleNamespace:
            self.calls.append(
                {
                    "model": model,
                    "input": input,
                    "encoding_format": encoding_format,
                }
            )
            return SimpleNamespace(
                data=[
                    SimpleNamespace(embedding=[0.1, 0.2]),
                    SimpleNamespace(embedding=[0.3, 0.4]),
                ]
            )

    fake_embeddings_api = FakeEmbeddingsApi()
    fake_openai_client = SimpleNamespace(
        embeddings=fake_embeddings_api,
        api_key="test-key",
        base_url="https://openrouter.example/api/v1",
    )
    settings = SimpleNamespace(
        api_key="test-key",
        base_url="https://openrouter.example/api/v1",
    )
    client = OpenRouterClient(settings=settings, client=fake_openai_client)

    embeddings = client.embed_texts(
        model="openai/text-embedding-3-small",
        texts=["first", "second"],
    )

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert fake_embeddings_api.calls == [
        {
            "model": "openai/text-embedding-3-small",
            "input": ["first", "second"],
            "encoding_format": "float",
        }
    ]


def test_openrouter_client_chats_through_openai_chat_api() -> None:
    class FakeChatCompletionsApi:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def create(
            self,
            *,
            model: str,
            messages: list[dict[str, str]],
            temperature: float,
        ) -> SimpleNamespace:
            self.calls.append(
                {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }
            )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="agent answer"),
                    )
                ]
            )

    fake_chat_completions_api = FakeChatCompletionsApi()
    fake_openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=fake_chat_completions_api),
        api_key="test-key",
        base_url="https://openrouter.example/api/v1",
    )
    settings = SimpleNamespace(
        api_key="test-key",
        base_url="https://openrouter.example/api/v1",
    )
    client = OpenRouterClient(settings=settings, client=fake_openai_client)

    answer = client.chat(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.1,
    )

    assert answer == "agent answer"
    assert fake_chat_completions_api.calls == [
        {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.1,
        }
    ]
