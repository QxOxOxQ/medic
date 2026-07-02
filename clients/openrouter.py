from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import dotenv_values
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = PROJECT_ROOT / "rag" / "settings.json"


@dataclass(frozen=True)
class OpenRouterSettings:
    api_key: str
    base_url: str
    management_api_key: str | None = None


@dataclass(frozen=True)
class OpenRouterCreditsResponse:
    total_credits: Decimal
    total_usage: Decimal


@dataclass(frozen=True)
class OpenRouterKeyResponse:
    label: str
    usage: Decimal
    usage_daily: Decimal
    usage_weekly: Decimal
    usage_monthly: Decimal
    byok_usage: Decimal
    byok_usage_daily: Decimal
    byok_usage_weekly: Decimal
    byok_usage_monthly: Decimal
    include_byok_in_limit: bool
    is_free_tier: bool
    is_management_key: bool
    is_provisioning_key: bool
    limit: Decimal | None
    limit_remaining: Decimal | None
    limit_reset: str | None
    expires_at: datetime | None


@dataclass(frozen=True)
class OpenRouterActivityResponseItem:
    date: str
    model: str
    model_permaslug: str
    endpoint_id: str
    provider_name: str
    usage: Decimal
    byok_usage_inference: Decimal
    requests: int
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int


class OpenRouterApiError(RuntimeError):
    def __init__(self, *, status_code: int | None, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


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

    def get_credits(self) -> OpenRouterCreditsResponse:
        data = _required_mapping(self._get_json("/credits"))
        payload = _required_mapping(data["data"])
        return OpenRouterCreditsResponse(
            total_credits=_decimal(payload["total_credits"]),
            total_usage=_decimal(payload["total_usage"]),
        )

    def get_current_key(self) -> OpenRouterKeyResponse:
        data = _required_mapping(
            self._get_json("/key", use_management_key=False),
        )
        payload = _required_mapping(data["data"])
        return OpenRouterKeyResponse(
            label=str(payload["label"]),
            usage=_decimal(payload["usage"]),
            usage_daily=_decimal(payload["usage_daily"]),
            usage_weekly=_decimal(payload["usage_weekly"]),
            usage_monthly=_decimal(payload["usage_monthly"]),
            byok_usage=_decimal(payload["byok_usage"]),
            byok_usage_daily=_decimal(payload["byok_usage_daily"]),
            byok_usage_weekly=_decimal(payload["byok_usage_weekly"]),
            byok_usage_monthly=_decimal(payload["byok_usage_monthly"]),
            include_byok_in_limit=bool(payload["include_byok_in_limit"]),
            is_free_tier=bool(payload["is_free_tier"]),
            is_management_key=bool(payload["is_management_key"]),
            is_provisioning_key=bool(payload["is_provisioning_key"]),
            limit=_optional_decimal(payload.get("limit")),
            limit_remaining=_optional_decimal(payload.get("limit_remaining")),
            limit_reset=_optional_string(payload.get("limit_reset")),
            expires_at=_optional_datetime(payload.get("expires_at")),
        )

    def get_activity(self) -> tuple[OpenRouterActivityResponseItem, ...]:
        data = _required_mapping(self._get_json("/activity"))
        rows = data["data"]
        if not isinstance(rows, list):
            raise ValueError("OpenRouter activity response data must be a list")
        return tuple(_activity_item(row) for row in rows)

    def _get_json(
        self,
        path: str,
        *,
        use_management_key: bool = True,
    ) -> Mapping[str, Any]:
        request = Request(
            _endpoint_url(self.settings.base_url, path),
            headers={
                "Authorization": f"Bearer {self._api_key(use_management_key)}",
                "Accept": "application/json",
            },
            method="GET",
        )
        try:
            with urlopen(request, timeout=20) as response:
                return _decode_response(response.read())
        except HTTPError as error:
            raise _api_error_from_http_error(error) from error
        except (TimeoutError, URLError) as error:
            raise OpenRouterApiError(
                status_code=None,
                message="OpenRouter could not be reached.",
            ) from error

    def _api_key(self, use_management_key: bool) -> str:
        if use_management_key and self.settings.management_api_key:
            return self.settings.management_api_key
        return self.settings.api_key


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
        management_api_key=_get_optional_env(
            env_names.get("openrouter_management_api_key"),
            dotenv_settings,
        ),
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


def _get_optional_env(
    name: str | None,
    dotenv_settings: Mapping[str, str],
) -> str | None:
    if name is None:
        return None
    value = os.getenv(name) or dotenv_settings.get(name)
    return value or None


def _endpoint_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _decode_response(body: bytes) -> Mapping[str, Any]:
    decoded = json.loads(body.decode("utf-8"))
    return _required_mapping(decoded)


def _required_mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("OpenRouter response payload must be an object")
    return cast(Mapping[str, Any], value)


def _decimal(value: object) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"Invalid decimal value from OpenRouter: {value}") from error


def _optional_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    return _decimal(value)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _activity_item(value: object) -> OpenRouterActivityResponseItem:
    payload = _required_mapping(value)
    return OpenRouterActivityResponseItem(
        date=str(payload["date"]),
        model=str(payload["model"]),
        model_permaslug=str(payload["model_permaslug"]),
        endpoint_id=str(payload["endpoint_id"]),
        provider_name=str(payload["provider_name"]),
        usage=_decimal(payload["usage"]),
        byok_usage_inference=_decimal(payload["byok_usage_inference"]),
        requests=int(payload["requests"]),
        prompt_tokens=int(payload["prompt_tokens"]),
        completion_tokens=int(payload["completion_tokens"]),
        reasoning_tokens=int(payload["reasoning_tokens"]),
    )


def _api_error_from_http_error(error: HTTPError) -> OpenRouterApiError:
    body = error.read()
    message = _error_message(body) or f"OpenRouter returned HTTP {error.code}."
    return OpenRouterApiError(status_code=error.code, message=message)


def _error_message(body: bytes) -> str | None:
    try:
        payload = _decode_response(body)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    raw_error = payload.get("error")
    if not isinstance(raw_error, Mapping):
        return None
    message = raw_error.get("message")
    return str(message) if message else None
