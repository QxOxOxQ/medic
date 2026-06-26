from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Protocol

from langchain_core.runnables import RunnableConfig

from agents.models import AgentAnswer, AgentRequest


class AgentObservability(Protocol):
    def trace(self, request: AgentRequest) -> AbstractContextManager[None]: ...

    def model_config(self, *, agent_name: str, phase: str) -> RunnableConfig | None: ...

    def tool_config(
        self, *, agent_name: str, tool_name: str
    ) -> RunnableConfig | None: ...

    def complete(self, answer: AgentAnswer) -> None: ...

    def close(self) -> None: ...


class NullAgentObservability:
    def trace(self, request: AgentRequest) -> AbstractContextManager[None]:
        del request
        return nullcontext()

    def model_config(self, *, agent_name: str, phase: str) -> RunnableConfig | None:
        del agent_name, phase
        return None

    def tool_config(self, *, agent_name: str, tool_name: str) -> RunnableConfig | None:
        del agent_name, tool_name
        return None

    def complete(self, answer: AgentAnswer) -> None:
        del answer

    def close(self) -> None:
        return None
