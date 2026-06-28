from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel

from clients.chat_models import (
    DEFAULT_CHAT_MODEL_KEY,
    SELECTABLE_CHAT_MODELS,
    ChatModelConfigurationError,
    ChatModelFactory,
    ChatModelSettings,
    is_valid_chat_model_key,
    resolve_chat_model,
)


def _settings(provider: str = "openrouter") -> ChatModelSettings:
    return ChatModelSettings(
        provider=provider,
        model="openai/gpt-4o-mini",
        temperature=0.2,
        max_retrieval_queries=6,
        max_consultations=4,
        max_review_rounds=3,
        provider_options={
            "api_key": "test-openrouter-key",
            "base_url": "https://openrouter.example/api/v1",
        },
        agent_models={"professor": "openai/gpt-4o-mini"},
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


def test_selectable_chat_models_expose_all_choices() -> None:
    keys = {model.key for model in SELECTABLE_CHAT_MODELS}

    assert keys == {"deepseek", "deepseek-v4", "openai", "gemini", "claude-opus"}
    assert DEFAULT_CHAT_MODEL_KEY == "openai"
    assert resolve_chat_model("deepseek").model_id == "deepseek/deepseek-chat-v3.1"
    assert resolve_chat_model("deepseek-v4").model_id == "deepseek/deepseek-v4"
    assert resolve_chat_model("gemini").model_id == "google/gemini-3.5-flash"
    assert resolve_chat_model("claude-opus").model_id == "anthropic/claude-opus-4-8"


def test_resolve_chat_model_falls_back_to_default_for_unknown_keys() -> None:
    assert resolve_chat_model(None).key == DEFAULT_CHAT_MODEL_KEY
    assert resolve_chat_model("nope").key == DEFAULT_CHAT_MODEL_KEY
    assert is_valid_chat_model_key("deepseek") is True
    assert is_valid_chat_model_key("nope") is False


def test_build_chat_models_applies_override_to_every_agent() -> None:
    from backend.factory import _build_chat_models

    default_model, overrides = _build_chat_models(
        _settings(),
        override_model_id="deepseek/deepseek-chat-v3.1",
    )

    assert default_model.model_name == "deepseek/deepseek-chat-v3.1"
    assert overrides["professor"].label == "deepseek/deepseek-chat-v3.1"
    assert overrides["professor"].model.model_name == "deepseek/deepseek-chat-v3.1"


def test_build_chat_models_without_override_uses_settings() -> None:
    from backend.factory import _build_chat_models

    default_model, overrides = _build_chat_models(_settings())

    assert default_model.model_name == "openai/gpt-4o-mini"
    assert overrides["professor"].label == "openai/gpt-4o-mini"


def test_agents_do_not_import_provider_specific_chat_classes() -> None:
    agents_dir = Path(__file__).resolve().parents[1] / "agents"
    source = "\n".join(path.read_text(encoding="utf-8") for path in agents_dir.glob("*.py"))

    assert "ChatOpenRouter" not in source
    assert "OpenRouterClient" not in source
