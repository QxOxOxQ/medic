from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from dotenv import dotenv_values
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = PROJECT_ROOT / "rag" / "settings.json"


@dataclass(frozen=True)
class OpenRouterSettings:
    api_key: str
    base_url: str


class OpenRouterClient:
    def __init__(
        self,
        settings: OpenRouterSettings | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self.settings = settings or get_openrouter_settings()
        self._client = client or OpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
        )

    @property
    def api_key(self) -> Any:
        return self._client.api_key

    @property
    def base_url(self) -> Any:
        return self._client.base_url

    def embed_texts(self, *, model: str, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=model,
            input=texts,
            encoding_format="float",
        )
        return [item.embedding for item in response.data]

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
        temperature: float = 0.2,
    ) -> str:
        typed_messages = cast(
            list[ChatCompletionMessageParam],
            [dict(message) for message in messages],
        )
        response = self._client.chat.completions.create(
            model=model,
            messages=typed_messages,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        return content or ""


def get_openrouter_client() -> OpenRouterClient:
    return OpenRouterClient()


def get_openrouter_settings(
    *,
    settings: Mapping[str, Any] | None = None,
    project_root: Path | None = None,
) -> OpenRouterSettings:
    resolved_settings = settings or _load_settings()
    root = project_root or PROJECT_ROOT
    dotenv_settings = _load_environment(root)
    env_names = resolved_settings["env"]
    openrouter_config = resolved_settings["openrouter"]

    return OpenRouterSettings(
        api_key=_get_required_env(env_names["openrouter_api_key"], dotenv_settings),
        base_url=openrouter_config["base_url"],
    )


def _load_settings() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(SETTINGS_PATH.read_text(encoding="utf-8")),
    )


def _load_environment(project_root: Path) -> Mapping[str, str]:
    return {
        name: value
        for name, value in dotenv_values(project_root / ".env").items()
        if value is not None
    }


def _get_required_env(name: str, dotenv_settings: Mapping[str, str]) -> str:
    value = os.getenv(name) or dotenv_settings.get(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value
