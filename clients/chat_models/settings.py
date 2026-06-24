from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = PROJECT_ROOT / "rag" / "settings.json"


@dataclass(frozen=True)
class ChatModelSettings:
    provider: str
    model: str
    temperature: float
    max_retrieval_queries: int
    max_consultations: int
    max_review_rounds: int
    provider_options: Mapping[str, Any]


def get_chat_model_settings(
    *,
    settings: Mapping[str, Any] | None = None,
    project_root: Path | None = None,
) -> ChatModelSettings:
    resolved_settings = settings or _load_settings()
    root = project_root or PROJECT_ROOT
    dotenv_settings = _load_environment(root)
    env_names = resolved_settings["env"]
    chat_config = resolved_settings["chat"]
    provider = chat_config["provider"]

    return ChatModelSettings(
        provider=provider,
        model=chat_config["model"],
        temperature=float(chat_config.get("temperature", 0.2)),
        max_retrieval_queries=int(chat_config.get("max_retrieval_queries", 6)),
        max_consultations=int(chat_config.get("max_consultations", 4)),
        max_review_rounds=int(chat_config.get("max_review_rounds", 3)),
        provider_options=_provider_options(
            provider=provider,
            settings=resolved_settings,
            env_names=env_names,
            dotenv_settings=dotenv_settings,
        ),
    )


def _provider_options(
    *,
    provider: str,
    settings: Mapping[str, Any],
    env_names: Mapping[str, str],
    dotenv_settings: Mapping[str, str],
) -> Mapping[str, Any]:
    if provider == "openrouter":
        return {
            "api_key": _get_required_env(
                env_names["openrouter_api_key"],
                dotenv_settings,
            ),
            "base_url": settings["openrouter"]["base_url"],
        }
    return {}


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
