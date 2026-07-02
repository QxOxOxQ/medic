from datetime import UTC, datetime
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.error import HTTPError

import pytest

import clients.openrouter as openrouter_module
import rag.config as settings_module
from clients.openrouter import OpenRouterApiError, OpenRouterClient, get_openrouter_client


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


def test_openrouter_client_parses_provider_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str | None]] = []
    payloads = {
        "/credits": (
            b'{"data":{"total_credits":100.5,"total_usage":25.75}}'
        ),
        "/key": (
            b'{"data":{"byok_usage":0,"byok_usage_daily":0,'
            b'"byok_usage_monthly":0,"byok_usage_weekly":0,'
            b'"creator_user_id":"user_1","expires_at":"2027-12-31T23:59:59Z",'
            b'"include_byok_in_limit":false,"is_free_tier":false,'
            b'"is_management_key":false,"is_provisioning_key":false,'
            b'"label":"sk-or-v1-test","limit":100,"limit_remaining":74.5,'
            b'"limit_reset":"monthly","rate_limit":{"interval":"1h",'
            b'"note":"deprecated","requests":1000},"usage":25.5,'
            b'"usage_daily":1.5,"usage_monthly":12.5,"usage_weekly":7.5}}'
        ),
        "/activity": (
            b'{"data":[{"byok_usage_inference":0.012,'
            b'"completion_tokens":125,"date":"2026-07-01",'
            b'"endpoint_id":"endpoint-1","model":"openai/gpt-4.1-mini",'
            b'"model_permaslug":"openai/gpt-4.1-mini",'
            b'"prompt_tokens":50,"provider_name":"OpenAI",'
            b'"reasoning_tokens":25,"requests":5,"usage":0.015}]}'
        ),
    }

    def fake_urlopen(request: Any, timeout: int) -> FakeResponse:
        del timeout
        path = str(request.full_url).removeprefix("https://openrouter.example")
        calls.append((path, request.headers.get("Authorization")))
        return FakeResponse(payloads[path])

    monkeypatch.setattr(openrouter_module, "urlopen", fake_urlopen)
    client = OpenRouterClient(
        settings=SimpleNamespace(
            api_key="runtime-key",
            management_api_key="management-key",
            base_url="https://openrouter.example",
        ),
        client=SimpleNamespace(api_key="runtime-key", base_url="https://openrouter.example"),
    )

    credits = client.get_credits()
    key = client.get_current_key()
    activity = client.get_activity()

    assert credits.total_credits == Decimal("100.5")
    assert credits.total_usage == Decimal("25.75")
    assert key.expires_at == datetime(2027, 12, 31, 23, 59, 59, tzinfo=UTC)
    assert key.limit_remaining == Decimal("74.5")
    assert activity[0].provider_name == "OpenAI"
    assert activity[0].reasoning_tokens == 25
    assert calls == [
        ("/credits", "Bearer management-key"),
        ("/key", "Bearer runtime-key"),
        ("/activity", "Bearer management-key"),
    ]


def test_openrouter_client_exposes_api_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: Any, timeout: int) -> FakeResponse:
        del timeout
        raise HTTPError(
            url=str(request.full_url),
            code=403,
            msg="Forbidden",
            hdrs={},
            fp=BytesIO(
                b'{"error":{"message":"Only management keys can perform this operation"}}'
            ),
        )

    monkeypatch.setattr(openrouter_module, "urlopen", fake_urlopen)
    client = OpenRouterClient(
        settings=SimpleNamespace(
            api_key="runtime-key",
            management_api_key=None,
            base_url="https://openrouter.example",
        ),
        client=SimpleNamespace(api_key="runtime-key", base_url="https://openrouter.example"),
    )

    with pytest.raises(OpenRouterApiError) as error:
        client.get_credits()

    assert error.value.status_code == 403
    assert error.value.message == "Only management keys can perform this operation"


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body
