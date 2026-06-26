from __future__ import annotations

from typing import Protocol

from langchain_core.tools import BaseTool


class AgentTool(Protocol):
    name: str
    description: str

    def to_langchain_tool(self) -> BaseTool:
        ...
