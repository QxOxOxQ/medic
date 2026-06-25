from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage

from agents.model_gateway import AgentModelGateway
from agents.models import AgentExecutionError
from agents.observability import NullAgentObservability
from agents.trace import AgentTraceRecorder


class _FlakyChatModel:
    """Chat model stub that fails a fixed number of times before succeeding."""

    def __init__(self, *, failures: int, content: str = "final answer") -> None:
        self._remaining_failures = failures
        self._content = content
        self.calls = 0

    def invoke(self, messages: Any, config: Any = None) -> AIMessage:
        del messages, config
        self.calls += 1
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise RuntimeError("Provider returned error")
        return AIMessage(content=self._content)


def _gateway(chat_model: Any, *, max_attempts: int) -> AgentModelGateway:
    return AgentModelGateway(
        chat_model=chat_model,
        observability=NullAgentObservability(),
        trace_recorder=AgentTraceRecorder(),
        max_attempts=max_attempts,
        retry_backoff_seconds=0.0,
    )


def _call_text(gateway: AgentModelGateway) -> str:
    return gateway.text(
        system_prompt="system",
        user_prompt="user",
        agent_name="professor",
        phase="synthesis",
    )


def test_text_retries_transient_errors_then_succeeds() -> None:
    chat_model = _FlakyChatModel(failures=2)
    gateway = _gateway(chat_model, max_attempts=3)

    answer = _call_text(gateway)

    assert answer == "final answer"
    assert chat_model.calls == 3


def test_text_gives_up_after_max_attempts() -> None:
    chat_model = _FlakyChatModel(failures=5)
    gateway = _gateway(chat_model, max_attempts=3)

    with pytest.raises(AgentExecutionError):
        _call_text(gateway)

    assert chat_model.calls == 3


def test_text_does_not_retry_when_max_attempts_is_one() -> None:
    chat_model = _FlakyChatModel(failures=5)
    gateway = _gateway(chat_model, max_attempts=1)

    with pytest.raises(AgentExecutionError):
        _call_text(gateway)

    assert chat_model.calls == 1


def test_text_succeeds_on_first_attempt_without_retry() -> None:
    chat_model = _FlakyChatModel(failures=0)
    gateway = _gateway(chat_model, max_attempts=3)

    answer = _call_text(gateway)

    assert answer == "final answer"
    assert chat_model.calls == 1


def test_retry_records_each_failed_attempt_in_trace() -> None:
    chat_model = _FlakyChatModel(failures=2)
    trace_recorder = AgentTraceRecorder()
    gateway = AgentModelGateway(
        chat_model=chat_model,
        observability=NullAgentObservability(),
        trace_recorder=trace_recorder,
        max_attempts=3,
        retry_backoff_seconds=0.0,
    )

    _call_text(gateway)

    retry_events = [
        event for event in trace_recorder.events() if event.status == "retrying"
    ]
    assert len(retry_events) == 2
