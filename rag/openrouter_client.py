from __future__ import annotations

from clients.openrouter import OpenRouterClient
from rag.config import get_openrouter_settings


def get_openrouter_client() -> OpenRouterClient:
    return OpenRouterClient(settings=get_openrouter_settings())


__all__ = ["OpenRouterClient", "get_openrouter_client"]
