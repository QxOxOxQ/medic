from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel

from clients.chat_models import (
    ChatModelConfigurationError,
    ChatModelFactory,
    ChatModelSettings,
)


def _settings(provider: str = "openrouter") -> ChatModelSettings:
    return ChatModelSettings(
        provider=provider,
        model="openai/gpt-4o-mini",
        temperature=0.2,
        max_tool_iterations=3,
        max_review_rounds=1,
        provider_options={
            "api_key": "test-openrouter-key",
            "base_url": "https://openrouter.example/api/v1",
        },
    )


def test_chat_model_factory_creates_openrouter_model_from_configuration() -> None:
    model = ChatModelFactory().create(_settings())

    assert isinstance(model, BaseChatModel)
    assert model.model_name == "openai/gpt-4o-mini"
    assert model.temperature == 0.2
    assert callable(model.bind_tools)


def test_chat_model_factory_fails_fast_for_unknown_provider() -> None:
    with pytest.raises(ChatModelConfigurationError, match="Unknown chat provider"):
        ChatModelFactory().create(_settings(provider="unknown"))


def test_agents_do_not_import_provider_specific_chat_classes() -> None:
    agents_dir = Path(__file__).resolve().parents[1] / "agents"
    source = "\n".join(path.read_text(encoding="utf-8") for path in agents_dir.glob("*.py"))

    assert "ChatOpenRouter" not in source
    assert "OpenRouterClient" not in source
