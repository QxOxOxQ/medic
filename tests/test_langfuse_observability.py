from __future__ import annotations

from contextlib import nullcontext
from uuid import uuid4

import pytest
from langchain_core.callbacks import BaseCallbackHandler

from agents.models import AgentAnswer, AgentRequest, ChatHistoryMessage
from agents.observability import NullAgentObservability
from backend.factory import _build_agent_graph
from clients.chat_models.settings import ChatModelSettings
from observability.config import (
    ObservabilityConfigurationError,
    load_langfuse_tracing_settings,
)
from observability.langfuse import (
    LangfuseAgentObservability,
    _redact_trace_data,
    build_agent_observability,
)


class FakeSpan:
    def __init__(self) -> None:
        self.updates: list[dict[str, object]] = []

    def update(
        self,
        *,
        output: object | None = None,
        metadata: object | None = None,
    ) -> object:
        self.updates.append({"output": output, "metadata": metadata})
        return self


class FakeClient:
    def __init__(self) -> None:
        self.span = FakeSpan()
        self.starts: list[dict[str, object]] = []
        self.flush_calls = 0
        self.shutdown_calls = 0

    def start_as_current_observation(self, **kwargs: object):
        self.starts.append(dict(kwargs))
        return nullcontext(self.span)

    def flush(self) -> None:
        self.flush_calls += 1

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class FakeAttributePropagator:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object):
        self.calls.append(dict(kwargs))
        return nullcontext()


class FakeCallbackHandler(BaseCallbackHandler):
    pass


class FakeChatModel:
    def bind_tools(self, tools: list[object], **kwargs: object) -> FakeChatModel:
        del tools, kwargs
        return self


class FakeChatModelFactory:
    def create(
        self,
        settings: ChatModelSettings | None = None,
        *,
        model: str | None = None,
    ) -> FakeChatModel:
        del settings, model
        return FakeChatModel()


def test_tracing_settings_are_disabled_and_private_by_default() -> None:
    settings = load_langfuse_tracing_settings(environment={})

    assert settings.enabled is False
    assert settings.capture_content is False
    assert settings.environment == "development"
    assert settings.sample_rate == 1.0
    assert isinstance(
        build_agent_observability(settings=settings), NullAgentObservability
    )


def test_enabled_tracing_requires_credentials_and_valid_sampling() -> None:
    with pytest.raises(ObservabilityConfigurationError, match="are required"):
        load_langfuse_tracing_settings(
            environment={"MEDIC_LANGFUSE_TRACING_ENABLED": "true"}
        )

    with pytest.raises(ObservabilityConfigurationError, match="between 0 and 1"):
        load_langfuse_tracing_settings(
            environment={"MEDIC_LANGFUSE_SAMPLE_RATE": "1.1"}
        )


def test_zero_sample_rate_disables_observability_without_building_client() -> None:
    settings = load_langfuse_tracing_settings(
        environment={
            "MEDIC_LANGFUSE_TRACING_ENABLED": "true",
            "MEDIC_LANGFUSE_SAMPLE_RATE": "0",
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_SECRET_KEY": "sk-test",
        }
    )

    assert isinstance(
        build_agent_observability(settings=settings),
        NullAgentObservability,
    )


def test_langfuse_observability_correlates_and_redacts_agent_run() -> None:
    client = FakeClient()
    propagator = FakeAttributePropagator()
    observability = LangfuseAgentObservability(
        client=client,
        callback_factory=FakeCallbackHandler,
        attribute_propagator=propagator,
        trace_id_factory=lambda seed: f"trace-{seed}",
        capture_content=False,
        prompt_version="agents-v1",
    )
    user_id = uuid4()
    session_id = uuid4()
    execution_id = uuid4()
    request = AgentRequest(
        question="Sensitive medical question",
        conversation_messages=(ChatHistoryMessage(role="user", content="History"),),
        user_id=user_id,
        session_id=session_id,
        execution_id=execution_id,
    )
    answer = AgentAnswer(
        answer="Sensitive medical answer",
        agents=("cardiometabolic_internist",),
        sources=(),
        insufficient_context=False,
    )

    with observability.trace(request):
        model_config = observability.model_config(
            agent_name="cardiometabolic_internist",
            phase="specialist",
        )
        tool_config = observability.tool_config(
            agent_name="cardiometabolic_internist",
            tool_name="search_user_medical_documents",
        )
        observability.complete(answer)

    assert client.starts[0]["name"] == "medical-agent-response"
    assert client.starts[0]["as_type"] == "agent"
    assert client.starts[0]["input"] == "[REDACTED]"
    assert client.starts[0]["trace_context"] == {"trace_id": f"trace-{execution_id}"}
    assert propagator.calls[0]["session_id"] == str(session_id)
    assert propagator.calls[0]["user_id"] != str(user_id)
    assert model_config is not None
    assert model_config["run_name"] == "cardiometabolic_internist-specialist"
    assert tool_config is not None
    assert tool_config["run_name"].endswith("search_user_medical_documents")
    assert client.span.updates[0]["output"] == "[REDACTED]"
    assert observability.model_config(agent_name="agent", phase="phase") is None

    observability.close()
    assert client.flush_calls == 1
    assert client.shutdown_calls == 1


def test_nested_observability_uses_current_trace_and_does_not_own_client() -> None:
    client = FakeClient()
    observability = LangfuseAgentObservability(
        client=client,
        callback_factory=FakeCallbackHandler,
        attribute_propagator=FakeAttributePropagator(),
        capture_content=True,
        prompt_version="agents-v1",
        close_client=False,
    )

    with observability.trace(AgentRequest(question="Synthetic question")):
        pass
    observability.close()

    assert client.starts[0]["trace_context"] is None
    assert client.flush_calls == 0
    assert client.shutdown_calls == 0


def test_content_capture_is_explicit_and_masker_redacts_all_payloads() -> None:
    client = FakeClient()
    observability = LangfuseAgentObservability(
        client=client,
        callback_factory=FakeCallbackHandler,
        attribute_propagator=FakeAttributePropagator(),
        trace_id_factory=lambda seed: seed,
        capture_content=True,
        prompt_version="agents-v1",
    )
    request = AgentRequest(question="Question")

    with observability.trace(request):
        observability.complete(
            AgentAnswer(
                answer="Answer",
                agents=(),
                sources=(),
                insufficient_context=False,
            )
        )

    assert client.starts[0]["input"] == {"question": "Question", "conversation": []}
    assert client.span.updates[0]["output"] == {"answer": "Answer"}
    assert _redact_trace_data(data={"secret": "value"}) == "[REDACTED]"


def test_backend_factory_injects_observability_into_agent_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = ChatModelSettings(
        provider="test",
        model="test-model",
        temperature=0.0,
        max_retrieval_queries=2,
        max_consultations=2,
        max_review_rounds=1,
        provider_options={},
    )
    monkeypatch.setattr("backend.factory.get_chat_model_settings", lambda: settings)
    monkeypatch.setattr("backend.factory.ChatModelFactory", FakeChatModelFactory)
    observability = NullAgentObservability()

    graph = _build_agent_graph(
        retriever=object(),
        owner_user_id=uuid4(),
        retrieval_limit=3,
        observability=observability,
    )

    assert graph._observability is observability
