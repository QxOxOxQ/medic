from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openrouter import ChatOpenRouter
from pydantic import SecretStr

from clients.chat_models.settings import ChatModelSettings


def create_openrouter_chat_model(settings: ChatModelSettings) -> BaseChatModel:
    return ChatOpenRouter(
        api_key=SecretStr(str(settings.provider_options["api_key"])),
        base_url=str(settings.provider_options["base_url"]),
        model=settings.model,
        temperature=settings.temperature,
    )
