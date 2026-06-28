from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SelectableChatModel:
    """A chat model a user may pick, mapped to its OpenRouter model id."""

    key: str
    label: str
    model_id: str


DEEPSEEK = SelectableChatModel(
    key="deepseek",
    label="DeepSeek V3.1",
    model_id="deepseek/deepseek-chat-v3.1",
)
DEEPSEEK_V4 = SelectableChatModel(
    key="deepseek-v4",
    label="DeepSeek V4",
    model_id="deepseek/deepseek-v4",
)
OPENAI = SelectableChatModel(
    key="openai",
    label="OpenAI GPT-4.1 mini",
    model_id="openai/gpt-4.1-mini",
)
GEMINI = SelectableChatModel(
    key="gemini",
    label="Gemini 3.5 Flash",
    model_id="google/gemini-3.5-flash",
)
CLAUDE_OPUS = SelectableChatModel(
    key="claude-opus",
    label="Claude Opus 4.8",
    model_id="anthropic/claude-opus-4-8",
)

SELECTABLE_CHAT_MODELS: tuple[SelectableChatModel, ...] = (DEEPSEEK, DEEPSEEK_V4, OPENAI, GEMINI, CLAUDE_OPUS)
DEFAULT_CHAT_MODEL_KEY = OPENAI.key

_MODELS_BY_KEY = {model.key: model for model in SELECTABLE_CHAT_MODELS}


def is_valid_chat_model_key(key: str) -> bool:
    return key in _MODELS_BY_KEY


def resolve_chat_model(key: str | None) -> SelectableChatModel:
    """Return the selected model, falling back to the default for unknown keys."""
    if key is None:
        return _MODELS_BY_KEY[DEFAULT_CHAT_MODEL_KEY]
    return _MODELS_BY_KEY.get(key, _MODELS_BY_KEY[DEFAULT_CHAT_MODEL_KEY])
