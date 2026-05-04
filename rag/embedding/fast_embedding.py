from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, ClassVar, TypeAlias

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastembed import LateInteractionTextEmbedding, TextEmbedding
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from rag.config import get_fast_embedding_settings


EmbeddingModel: TypeAlias = TextEmbedding | LateInteractionTextEmbedding
EmbeddingModelClass: TypeAlias = type[TextEmbedding] | type[LateInteractionTextEmbedding]
EmbeddingVector: TypeAlias = NDArray[np.float32]


class FastEmbedding(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    model: str = Field(
        default_factory=lambda: get_fast_embedding_settings().default_model
    )

    MODEL_CLASSES: ClassVar[dict[str, EmbeddingModelClass]] = {
        "text_embedding": TextEmbedding,
        "late_interaction_text_embedding": LateInteractionTextEmbedding,
    }

    _embedding_model: EmbeddingModel = PrivateAttr()

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        model_class, model_name = self._resolve_model(self.model)
        self._embedding_model = model_class(model_name=model_name)

    @classmethod
    def _resolve_model(cls, model: str) -> tuple[EmbeddingModelClass, str]:
        settings = get_fast_embedding_settings()
        try:
            model_settings = settings.models[model]
        except KeyError as error:
            available_models = ", ".join(sorted(settings.models))
            raise ValueError(
                f"Unknown embedding model: {model}. "
                f"Available models: {available_models}"
            ) from error

        try:
            model_class = cls.MODEL_CLASSES[model_settings.provider]
        except KeyError as error:
            raise ValueError(
                f"Unknown embedding provider: {model_settings.provider}"
            ) from error

        return model_class, model_settings.model_name

    def embed_text(self, document: str) -> EmbeddingVector:
        return self.embed_documents([document])[0]

    def embed_documents(self, documents: list[str]) -> list[EmbeddingVector]:
        if not documents:
            raise ValueError("documents cannot be empty")

        return [
            np.asarray(embedding, dtype=np.float32)
            for embedding in self._embedding_model.embed(documents)
        ]


if __name__ == "__main__":
    text = "fastembed is supported by and maintained by Qdrant."

    embedder = FastEmbedding(model="colbert")
    embedding = embedder.embed_text(text)

    print(type(embedding))
    print(embedding.shape)
    print(embedding)
