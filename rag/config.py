from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from dotenv import dotenv_values

from clients.chat_models import ChatModelSettings
from clients.chat_models import get_chat_model_settings as _get_chat_model_settings
from clients.openrouter import OpenRouterSettings
from clients.openrouter import get_openrouter_settings as _get_openrouter_settings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = Path(__file__).with_name("settings.json")


def load_settings() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(SETTINGS_PATH.read_text(encoding="utf-8")),
    )


SETTINGS = load_settings()


@dataclass(frozen=True)
class QdrantSettings:
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection_name: str
    client_timeout_seconds: int
    dense_vector_name: str
    dense_vector_size: int
    sparse_vector_name: str
    sparse_vector_model: str
    sparse_vector_on_disk: bool
    prefetch_limit: int
    quantization_encoding: str
    quantization_always_ram: bool


@dataclass(frozen=True)
class DatabaseSettings:
    database_url: str


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str
    model: str


@dataclass(frozen=True)
class DocumentPreparationSettings:
    raw_documents_dir: Path
    parsed_markdown_dir: Path


@dataclass(frozen=True)
class FastEmbeddingModelSettings:
    provider: str
    model_name: str


@dataclass(frozen=True)
class FastEmbeddingSettings:
    default_model: str
    models: Mapping[str, FastEmbeddingModelSettings]


def _load_environment() -> Mapping[str, str]:
    return {
        name: value
        for name, value in dotenv_values(PROJECT_ROOT / ".env").items()
        if value is not None
    }


def _get_required_env(name: str, dotenv_settings: Mapping[str, str]) -> str:
    value = os.getenv(name) or dotenv_settings.get(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def get_qdrant_settings() -> QdrantSettings:
    dotenv_settings = _load_environment()
    qdrant_config = SETTINGS["qdrant"]
    env_names = SETTINGS["env"]

    return QdrantSettings(
        qdrant_url=_get_required_env(env_names["qdrant_url"], dotenv_settings),
        qdrant_api_key=_get_required_env(
            env_names["qdrant_api_key"],
            dotenv_settings,
        ),
        qdrant_collection_name=(
            os.getenv(env_names["qdrant_collection_name"])
            or dotenv_settings.get(env_names["qdrant_collection_name"])
            or qdrant_config["collection_name"]
        ),
        client_timeout_seconds=qdrant_config["client_timeout_seconds"],
        dense_vector_name=qdrant_config["dense_vector"]["name"],
        dense_vector_size=qdrant_config["dense_vector"]["size"],
        sparse_vector_name=qdrant_config["sparse_vector"]["name"],
        sparse_vector_model=qdrant_config["sparse_vector"]["model"],
        sparse_vector_on_disk=qdrant_config["sparse_vector"]["on_disk"],
        prefetch_limit=qdrant_config["prefetch_limit"],
        quantization_encoding=qdrant_config["quantization"]["encoding"],
        quantization_always_ram=qdrant_config["quantization"]["always_ram"],
    )


def get_openrouter_settings() -> OpenRouterSettings:
    return _get_openrouter_settings(settings=SETTINGS, project_root=PROJECT_ROOT)


def get_chat_model_settings() -> ChatModelSettings:
    return _get_chat_model_settings(settings=SETTINGS, project_root=PROJECT_ROOT)


def get_database_settings() -> DatabaseSettings:
    dotenv_settings = _load_environment()
    env_names = SETTINGS["env"]
    return DatabaseSettings(
        database_url=_get_required_env(env_names["database_url"], dotenv_settings),
    )


def get_embedding_settings() -> EmbeddingSettings:
    embedding_config = SETTINGS["embedding"]
    return EmbeddingSettings(
        provider=embedding_config["provider"],
        model=embedding_config["model"],
    )


def get_document_preparation_settings() -> DocumentPreparationSettings:
    document_config = SETTINGS["document_preparation"]
    data_dir = PROJECT_ROOT / document_config["data_dir"]
    return DocumentPreparationSettings(
        raw_documents_dir=data_dir / document_config["raw_documents_dir"],
        parsed_markdown_dir=data_dir / document_config["parsed_markdown_dir"],
    )


def get_fast_embedding_settings() -> FastEmbeddingSettings:
    fast_embedding_config = SETTINGS["fast_embedding"]
    return FastEmbeddingSettings(
        default_model=fast_embedding_config["default_model"],
        models={
            model: FastEmbeddingModelSettings(**config)
            for model, config in fast_embedding_config["models"].items()
        },
    )
