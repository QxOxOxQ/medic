from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel


@dataclass(frozen=True)
class RoutedModel:
    """A chat model paired with the human-readable id used to display it."""

    model: BaseChatModel
    label: str
