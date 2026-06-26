from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

from langchain_core.language_models.chat_models import BaseChatModel

from clients.chat_models.openrouter import create_openrouter_chat_model
from clients.chat_models.settings import ChatModelSettings, get_chat_model_settings


class ChatModelConfigurationError(ValueError):
    """Raised when chat model configuration cannot build a tool-capable model."""


ProviderAdapter = Callable[[ChatModelSettings], BaseChatModel]


class ChatModelFactory:
    def __init__(
        self,
        registry: Mapping[str, ProviderAdapter] | None = None,
    ) -> None:
        self._registry = dict(registry or {"openrouter": create_openrouter_chat_model})

    def create(
        self,
        settings: ChatModelSettings | None = None,
        *,
        model: str | None = None,
    ) -> BaseChatModel:
        resolved_settings = settings or get_chat_model_settings()
        if model is not None and model != resolved_settings.model:
            resolved_settings = replace(resolved_settings, model=model)
        try:
            adapter = self._registry[resolved_settings.provider]
        except KeyError as error:
            raise ChatModelConfigurationError(
                f"Unknown chat provider: {resolved_settings.provider}"
            ) from error

        chat_model = adapter(resolved_settings)
        if not callable(getattr(chat_model, "bind_tools", None)):
            raise ChatModelConfigurationError(
                f"Chat provider '{resolved_settings.provider}' does not support tool calling"
            )
        return chat_model
