from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Protocol, cast

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig

from agents.models import AgentAnswer, AgentRequest
from agents.observability import AgentObservability, NullAgentObservability
from observability.config import (
    LangfuseTracingSettings,
    load_langfuse_tracing_settings,
)

if TYPE_CHECKING:
    from langfuse import Langfuse


TRACE_NAME = "medical-agent-response"
TRACE_TAGS = ["medical-agent", "rag"]


class _Observation(Protocol):
    def update(
        self,
        *,
        output: object | None = None,
        metadata: object | None = None,
    ) -> object: ...


class _LangfuseClient(Protocol):
    def start_as_current_observation(
        self,
        *,
        name: str,
        as_type: str,
        trace_context: dict[str, str] | None,
        input: object,
        metadata: object,
        version: str,
    ) -> AbstractContextManager[object]: ...

    def flush(self) -> None: ...

    def shutdown(self) -> None: ...


class _AttributePropagator(Protocol):
    def __call__(
        self,
        *,
        user_id: str | None,
        session_id: str | None,
        tags: list[str],
        trace_name: str,
    ) -> AbstractContextManager[Any]: ...


class LangfuseAgentObservability:
    def __init__(
        self,
        *,
        client: _LangfuseClient,
        callback_factory: Callable[[], BaseCallbackHandler],
        attribute_propagator: _AttributePropagator,
        capture_content: bool,
        prompt_version: str,
        close_client: bool = True,
        trace_id_factory: Callable[[str], str] | None = None,
    ) -> None:
        self._client = client
        self._callback_factory = callback_factory
        self._attribute_propagator = attribute_propagator
        self._trace_id_factory = trace_id_factory
        self._capture_content = capture_content
        self._prompt_version = prompt_version
        self._close_client = close_client
        self._handler: ContextVar[BaseCallbackHandler | None] = ContextVar(
            "langfuse_agent_handler",
            default=None,
        )
        self._span: ContextVar[_Observation | None] = ContextVar(
            "langfuse_agent_span",
            default=None,
        )

    @contextmanager
    def trace(self, request: AgentRequest) -> Iterator[None]:
        trace_context = self._trace_context(request)
        with self._client.start_as_current_observation(
            name=TRACE_NAME,
            as_type="agent",
            trace_context=trace_context,
            input=self._trace_input(request),
            metadata={"execution_id": str(request.execution_id)},
            version=self._prompt_version,
        ) as observation:
            span_token = self._span.set(cast(_Observation, observation))
            handler_token = self._handler.set(self._callback_factory())
            with self._attribute_propagator(
                user_id=_pseudonymous_user_id(request),
                session_id=_session_id(request),
                tags=TRACE_TAGS,
                trace_name=TRACE_NAME,
            ):
                try:
                    yield
                finally:
                    self._handler.reset(handler_token)
                    self._span.reset(span_token)

    def model_config(self, *, agent_name: str, phase: str) -> RunnableConfig | None:
        handler = self._handler.get()
        if handler is None:
            return None
        return {
            "callbacks": [handler],
            "run_name": f"{agent_name}-{phase}",
            "tags": ["model", f"agent:{agent_name}", f"phase:{phase}"],
            "metadata": {
                "agent_name": agent_name,
                "phase": phase,
                "prompt_version": self._prompt_version,
            },
        }

    def tool_config(self, *, agent_name: str, tool_name: str) -> RunnableConfig | None:
        handler = self._handler.get()
        if handler is None:
            return None
        return {
            "callbacks": [handler],
            "run_name": f"{agent_name}-{tool_name}",
            "tags": ["tool", f"agent:{agent_name}", f"tool:{tool_name}"],
            "metadata": {
                "agent_name": agent_name,
                "tool_name": tool_name,
                "prompt_version": self._prompt_version,
            },
        }

    def complete(self, answer: AgentAnswer) -> None:
        span = self._span.get()
        if span is None:
            return
        span.update(
            output=self._trace_output(answer),
            metadata={
                "agents": list(answer.agents),
                "source_count": len(answer.sources),
                "insufficient_context": answer.insufficient_context,
            },
        )

    def close(self) -> None:
        if not self._close_client:
            return
        self._client.flush()
        self._client.shutdown()

    def _trace_context(self, request: AgentRequest) -> dict[str, str] | None:
        if self._trace_id_factory is None:
            return None
        return {"trace_id": self._trace_id_factory(str(request.execution_id))}

    def _trace_input(self, request: AgentRequest) -> object:
        if self._capture_content:
            return {
                "question": request.question,
                "conversation": [
                    {"role": message.role, "content": message.content}
                    for message in request.conversation_messages
                ],
            }
        return "[REDACTED]"

    def _trace_output(self, answer: AgentAnswer) -> object:
        if self._capture_content:
            return {"answer": answer.answer}
        return "[REDACTED]"


def build_agent_observability(
    *,
    settings: LangfuseTracingSettings | None = None,
    prompt_version: str = "agents-v2",
) -> AgentObservability:
    resolved = settings or load_langfuse_tracing_settings()
    if not resolved.enabled or resolved.sample_rate == 0.0:
        return NullAgentObservability()

    # Import after configuration is loaded so the singleton is initialized explicitly.
    from langfuse import Langfuse, propagate_attributes
    from langfuse.langchain import CallbackHandler

    client = Langfuse(
        public_key=resolved.public_key,
        secret_key=resolved.secret_key,
        base_url=resolved.base_url,
        environment=resolved.environment,
        sample_rate=resolved.sample_rate,
        mask=None if resolved.capture_content else _redact_trace_data,
    )
    return LangfuseAgentObservability(
        client=cast(_LangfuseClient, client),
        callback_factory=lambda: CallbackHandler(public_key=resolved.public_key),
        attribute_propagator=cast(_AttributePropagator, propagate_attributes),
        trace_id_factory=lambda seed: Langfuse.create_trace_id(seed=seed),
        capture_content=resolved.capture_content,
        prompt_version=prompt_version,
    )


def build_nested_agent_observability(
    *,
    client: Langfuse,
    public_key: str,
    prompt_version: str,
    capture_content: bool,
) -> AgentObservability:
    from langfuse import propagate_attributes
    from langfuse.langchain import CallbackHandler

    return LangfuseAgentObservability(
        client=cast(_LangfuseClient, client),
        callback_factory=lambda: CallbackHandler(public_key=public_key),
        attribute_propagator=cast(_AttributePropagator, propagate_attributes),
        capture_content=capture_content,
        prompt_version=prompt_version,
        close_client=False,
    )


def _redact_trace_data(*, data: object, **kwargs: object) -> object:
    del kwargs
    if data is None:
        return None
    return "[REDACTED]"


def _pseudonymous_user_id(request: AgentRequest) -> str | None:
    if request.user_id is None:
        return None
    digest = sha256(str(request.user_id).encode("ascii")).hexdigest()
    return f"medic-{digest}"


def _session_id(request: AgentRequest) -> str | None:
    if request.session_id is None:
        return None
    return str(request.session_id)
