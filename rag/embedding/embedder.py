from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastembed import LateInteractionTextEmbedding, TextEmbedding

from clients.openrouter import OpenRouterClient
from rag.config import get_embedding_settings


EmbeddingProvider: TypeAlias = Literal["openrouter", "fast_embedding"]
EmbeddingKind: TypeAlias = Literal["dense", "multivector"]
EmbeddingVector: TypeAlias = list[float] | list[list[float]]
FastEmbeddingProvider: TypeAlias = Literal[
    "text_embedding",
    "late_interaction_text_embedding",
]
FastEmbeddingModelClass: TypeAlias = (
    type[TextEmbedding] | type[LateInteractionTextEmbedding]
)

OPENROUTER_PROVIDER: EmbeddingProvider = "openrouter"
FAST_EMBEDDING_PROVIDER: EmbeddingProvider = "fast_embedding"
TEXT_EMBEDDING_PROVIDER: FastEmbeddingProvider = "text_embedding"
LATE_INTERACTION_TEXT_EMBEDDING_PROVIDER: FastEmbeddingProvider = (
    "late_interaction_text_embedding"
)
DENSE_EMBEDDING: EmbeddingKind = "dense"
MULTIVECTOR_EMBEDDING: EmbeddingKind = "multivector"

OPENAI_TEXT_EMBEDDING_3_SMALL = "openai/text-embedding-3-small"
BGE_SMALL_EN_V15 = "BAAI/bge-small-en-v1.5"
COLBERT_V2 = "colbert-ir/colbertv2.0"


@dataclass(frozen=True)
class EmbeddingModelConfig:
    provider: EmbeddingProvider
    model: str
    vector_size: int
    kind: EmbeddingKind
    fast_embedding_provider: FastEmbeddingProvider | None = None

    @property
    def is_multivector(self) -> bool:
        return self.kind == MULTIVECTOR_EMBEDDING


class _EmbeddingBackend(Protocol):
    def embed_texts(self, texts: list[str]) -> list[EmbeddingVector]:
        ...


EMBEDDING_MODELS: dict[str, Mapping[str, EmbeddingModelConfig]] = {
    OPENROUTER_PROVIDER: {
        OPENAI_TEXT_EMBEDDING_3_SMALL: EmbeddingModelConfig(
            provider=OPENROUTER_PROVIDER,
            model=OPENAI_TEXT_EMBEDDING_3_SMALL,
            vector_size=1536,
            kind=DENSE_EMBEDDING,
        )
    },
    FAST_EMBEDDING_PROVIDER: {
        BGE_SMALL_EN_V15: EmbeddingModelConfig(
            provider=FAST_EMBEDDING_PROVIDER,
            model=BGE_SMALL_EN_V15,
            vector_size=384,
            kind=DENSE_EMBEDDING,
            fast_embedding_provider=TEXT_EMBEDDING_PROVIDER,
        ),
        COLBERT_V2: EmbeddingModelConfig(
            provider=FAST_EMBEDDING_PROVIDER,
            model=COLBERT_V2,
            vector_size=128,
            kind=MULTIVECTOR_EMBEDDING,
            fast_embedding_provider=LATE_INTERACTION_TEXT_EMBEDDING_PROVIDER,
        ),
    },
}

FAST_EMBEDDING_MODEL_CLASSES: dict[str, FastEmbeddingModelClass] = {
    TEXT_EMBEDDING_PROVIDER: TextEmbedding,
    LATE_INTERACTION_TEXT_EMBEDDING_PROVIDER: LateInteractionTextEmbedding,
}


class Embedder:
    """Embeds text with the selected provider/model pair."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        openrouter_client: OpenRouterClient | None = None,
    ) -> None:
        provider, model = _resolve_requested_model(provider, model)
        self.model_config = resolve_embedding_model(provider, model)
        self.provider = self.model_config.provider
        self.model = self.model_config.model
        self._backend = _create_backend(self.model_config, openrouter_client)

    def embed_text(self, text: str) -> EmbeddingVector:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[EmbeddingVector]:
        _validate_texts(texts)
        return self._backend.embed_texts(texts)


class _OpenRouterEmbeddingBackend:
    def __init__(
        self,
        model_config: EmbeddingModelConfig,
        client: OpenRouterClient | None,
    ) -> None:
        self._model = model_config.model
        self._client = client or OpenRouterClient()

    def embed_texts(self, texts: list[str]) -> list[EmbeddingVector]:
        embeddings: list[EmbeddingVector] = []
        embeddings.extend(self._client.embed_texts(model=self._model, texts=texts))
        return embeddings


class _FastEmbeddingBackend:
    def __init__(self, model_config: EmbeddingModelConfig) -> None:
        self._model_config = model_config
        self._model = self._create_model(model_config)

    def embed_texts(self, texts: list[str]) -> list[EmbeddingVector]:
        return [
            _normalize_embedding(raw_embedding, self._model_config)
            for raw_embedding in self._model.embed(texts)
        ]

    @staticmethod
    def _create_model(model_config: EmbeddingModelConfig) -> Any:
        if model_config.fast_embedding_provider is None:
            raise ValueError(
                f"FastEmbedding provider missing for model: {model_config.model}"
            )

        model_class = FAST_EMBEDDING_MODEL_CLASSES[model_config.fast_embedding_provider]
        return model_class(model_name=model_config.model)


def resolve_embedding_model(provider: str, model: str) -> EmbeddingModelConfig:
    try:
        provider_models = EMBEDDING_MODELS[provider]
    except KeyError as error:
        available_providers = ", ".join(sorted(EMBEDDING_MODELS))
        raise ValueError(
            f"Unknown embedding provider: {provider}. "
            f"Available providers: {available_providers}"
        ) from error

    try:
        return provider_models[model]
    except KeyError as error:
        available_models = ", ".join(sorted(provider_models))
        raise ValueError(
            f"Unknown embedding model for provider {provider}: {model}. "
            f"Available models: {available_models}"
        ) from error


def get_selected_embedding_model() -> EmbeddingModelConfig:
    settings = get_embedding_settings()
    return resolve_embedding_model(settings.provider, settings.model)


def embed_texts(texts: list[str]) -> list[EmbeddingVector]:
    _validate_texts(texts)
    return Embedder().embed_texts(texts)


def _resolve_requested_model(
    provider: str | None,
    model: str | None,
) -> tuple[str, str]:
    if provider is not None and model is not None:
        return provider, model

    settings = get_embedding_settings()
    return provider or settings.provider, model or settings.model


def _create_backend(
    model_config: EmbeddingModelConfig,
    openrouter_client: OpenRouterClient | None,
) -> _EmbeddingBackend:
    if model_config.provider == OPENROUTER_PROVIDER:
        return _OpenRouterEmbeddingBackend(model_config, openrouter_client)

    if model_config.provider == FAST_EMBEDDING_PROVIDER:
        return _FastEmbeddingBackend(model_config)

    raise ValueError(f"Unknown embedding provider: {model_config.provider}")


def _validate_texts(texts: list[str]) -> None:
    if not texts:
        raise ValueError("texts must not be empty")

    if any(not text.strip() for text in texts):
        raise ValueError("texts must not contain empty values")


def _normalize_embedding(
    raw_embedding: Any,
    model_config: EmbeddingModelConfig,
) -> EmbeddingVector:
    if model_config.is_multivector:
        return [_normalize_vector(vector) for vector in _as_list(raw_embedding)]

    return _normalize_vector(raw_embedding)


def _normalize_vector(raw_vector: Any) -> list[float]:
    return list(_as_list(raw_vector))


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def main() -> None:
    sample_texts = ["To jest przykładowy tekst do testowego embeddingu."]
    embeddings = embed_texts(sample_texts)
    print(
        f"Generated {len(embeddings)} embedding(s); "
        f"first embedding dimension: {len(embeddings[0])}"
    )


if __name__ == "__main__":
    main()
