from __future__ import annotations

import logging
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from pytest import LogCaptureFixture

from agents.model_gateway import AgentModelGateway
from agents.model_router import RoutedModel
from agents.models import AgentExecutionError
from agents.observability import NullAgentObservability
from agents.trace import AgentTraceRecorder


class _LabeledChatModel:
    """Chat model stub that records how many times it was invoked."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls = 0

    def invoke(self, messages: Any, config: Any = None) -> AIMessage:
        del messages, config
        self.calls += 1
        return AIMessage(content=self._content)


class _FlakyChatModel:
    """Chat model stub that fails a fixed number of times before succeeding."""

    def __init__(
        self,
        *,
        failures: int,
        content: str = "final answer",
        error_message: str = "Provider returned error",
    ) -> None:
        self._remaining_failures = failures
        self._content = content
        self._error_message = error_message
        self.calls = 0

    def invoke(self, messages: Any, config: Any = None) -> AIMessage:
        del messages, config
        self.calls += 1
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise RuntimeError(self._error_message)
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


def test_text_failure_sanitizes_public_error_and_logs_detail(
    caplog: LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR, logger="medic.agents.model_gateway")
    raw_error = "Provider returned API key sk-test-123 and path /tmp/secret"
    chat_model = _FlakyChatModel(failures=1, error_message=raw_error)
    trace_recorder = AgentTraceRecorder()
    gateway = AgentModelGateway(
        chat_model=chat_model,  # type: ignore[arg-type]
        observability=NullAgentObservability(),
        trace_recorder=trace_recorder,
        max_attempts=1,
        retry_backoff_seconds=0.0,
        default_label="model-a",
    )

    with pytest.raises(AgentExecutionError) as caught:
        _call_text(gateway)

    public_error = str(caught.value)
    assert public_error == (
        "professor model call during synthesis via model-a failed. "
        "This is usually temporary — please try again in a moment."
    )
    assert raw_error not in public_error
    failed_event = next(
        event for event in trace_recorder.events() if event.status == "failed"
    )
    assert failed_event.payload["error"] == public_error
    assert raw_error not in str(failed_event.payload["error"])
    assert raw_error in caplog.text


def test_text_succeeds_on_first_attempt_without_retry() -> None:
    chat_model = _FlakyChatModel(failures=0)
    gateway = _gateway(chat_model, max_attempts=3)

    answer = _call_text(gateway)

    assert answer == "final answer"
    assert chat_model.calls == 1


def test_text_routes_to_per_agent_model_and_records_model_label() -> None:
    default_model = _LabeledChatModel("default-model answer")
    specialist_model = _LabeledChatModel("specialist answer")
    trace_recorder = AgentTraceRecorder()
    gateway = AgentModelGateway(
        chat_model=default_model,  # type: ignore[arg-type]
        observability=NullAgentObservability(),
        trace_recorder=trace_recorder,
        model_overrides={
            "orthopedist": RoutedModel(
                model=specialist_model,  # type: ignore[arg-type]
                label="model-b",
            ),
        },
        default_label="model-a",
    )

    professor_answer = gateway.text(
        system_prompt="s",
        user_prompt="u",
        agent_name="professor",
        phase="synthesis",
    )
    specialist_answer = gateway.text(
        system_prompt="s",
        user_prompt="u",
        agent_name="orthopedist",
        phase="consultation",
    )

    assert professor_answer == "default-model answer"
    assert specialist_answer == "specialist answer"
    assert default_model.calls == 1
    assert specialist_model.calls == 1
    labelled = {
        (event.agent_name, event.payload.get("model"))
        for event in trace_recorder.events()
        if event.event_type == "model_call" and event.status == "succeeded"
    }
    assert ("professor", "model-a") in labelled
    assert ("orthopedist", "model-b") in labelled


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
    for event in retry_events:
        assert event.payload["error"] == (
            "A temporary issue reaching the model provider interrupted this step; "
            "retrying automatically."
        )
        assert "Provider returned error" not in str(event.payload["error"])
