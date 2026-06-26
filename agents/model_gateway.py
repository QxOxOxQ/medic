from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import TypeVar, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from agents.contracts import (
    ConsultationReport,
    ResearchPlan,
    ReviewDecision,
    SpecialistTask,
)
from agents.model_router import RoutedModel
from agents.models import AgentExecutionError
from agents.observability import AgentObservability
from agents.structured_output import (
    ConsultationReportPayload,
    DocumentExpansionPayload,
    ResearchPlanPayload,
    ReviewDecisionPayload,
    TaskPlanPayload,
)
from agents.trace import AgentTraceRecorder


PayloadT = TypeVar("PayloadT", bound=BaseModel)
_ResultT = TypeVar("_ResultT")


class AgentModelGateway:
    def __init__(
        self,
        *,
        chat_model: BaseChatModel,
        observability: AgentObservability,
        trace_recorder: AgentTraceRecorder,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 0.5,
        model_overrides: Mapping[str, RoutedModel] | None = None,
        default_label: str | None = None,
    ) -> None:
        self._chat_model = chat_model
        self._observability = observability
        self._trace_recorder = trace_recorder
        self._max_attempts = max(1, max_attempts)
        self._retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self._model_overrides = dict(model_overrides or {})
        self._default_label = default_label

    def _model_for(self, agent_name: str) -> BaseChatModel:
        routed = self._model_overrides.get(agent_name)
        return routed.model if routed is not None else self._chat_model

    def _label_for(self, agent_name: str) -> str | None:
        routed = self._model_overrides.get(agent_name)
        return routed.label if routed is not None else self._default_label

    def research_plan(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_queries: int,
        agent_name: str,
        phase: str,
    ) -> ResearchPlan:
        payload = self._structured(
            ResearchPlanPayload,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            agent_name=agent_name,
            phase=phase,
        )
        return payload.to_domain(max_queries=max_queries)

    def task_plan(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_language: str,
        agent_name: str,
        phase: str,
    ) -> tuple[SpecialistTask, ...]:
        payload = self._structured(
            TaskPlanPayload,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            agent_name=agent_name,
            phase=phase,
        )
        return payload.to_domain(response_language=response_language)

    def consultation_report(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        agent_name: str,
        phase: str,
    ) -> ConsultationReport:
        payload = self._structured(
            ConsultationReportPayload,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            agent_name=agent_name,
            phase=phase,
        )
        return payload.to_domain()

    def review_decision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_language: str,
        agent_name: str,
        phase: str,
    ) -> ReviewDecision:
        payload = self._structured(
            ReviewDecisionPayload,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            agent_name=agent_name,
            phase=phase,
        )
        return payload.to_domain(response_language=response_language)

    def select_full_documents(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        valid_source_ids: set[str],
        max_documents: int,
        agent_name: str,
        phase: str,
    ) -> tuple[str, ...]:
        payload = self._structured(
            DocumentExpansionPayload,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            agent_name=agent_name,
            phase=phase,
        )
        return payload.to_domain(
            valid_source_ids=valid_source_ids,
            max_documents=max_documents,
        )

    def _structured(
        self,
        schema: type[PayloadT],
        *,
        system_prompt: str,
        user_prompt: str,
        agent_name: str,
        phase: str,
    ) -> PayloadT:
        messages = _messages(system_prompt, user_prompt)
        config = self._observability.model_config(
            agent_name=agent_name,
            phase=phase,
        )
        try:
            runnable = self._model_for(agent_name).with_structured_output(
                schema,
                method="function_calling",
            )
            response = self._invoke_with_retry(
                lambda: runnable.invoke(list(messages), config=config),
                agent_name=agent_name,
                phase=phase,
            )
        except Exception as error:
            self._record_failure(agent_name=agent_name, phase=phase, error=error)
            raise AgentExecutionError("Agent structured model call failed") from error

        if not isinstance(response, schema):
            invalid_error = TypeError(
                f"Structured response must be {schema.__name__}, "
                f"got {type(response).__name__}"
            )
            self._record_failure(
                agent_name=agent_name,
                phase=phase,
                error=invalid_error,
            )
            raise AgentExecutionError("Agent returned invalid structured output")

        self._record_success(
            agent_name=agent_name,
            phase=phase,
            message_count=len(messages),
            structured_schema=schema.__name__,
        )
        return response

    def text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        agent_name: str,
        phase: str,
    ) -> str:
        messages = _messages(system_prompt, user_prompt)
        config = self._observability.model_config(
            agent_name=agent_name,
            phase=phase,
        )
        try:
            response = self._invoke_with_retry(
                lambda: self._model_for(agent_name).invoke(list(messages), config=config),
                agent_name=agent_name,
                phase=phase,
            )
        except Exception as error:
            self._record_failure(agent_name=agent_name, phase=phase, error=error)
            raise AgentExecutionError("Agent model call failed") from error

        if not isinstance(response, AIMessage):
            error = TypeError("Agent model returned an unsupported message")
            self._record_failure(agent_name=agent_name, phase=phase, error=error)
            raise AgentExecutionError(str(error))

        self._record_success(
            agent_name=agent_name,
            phase=phase,
            message_count=len(messages),
        )
        return _message_content(response).strip()

    def _record_success(
        self,
        *,
        agent_name: str,
        phase: str,
        message_count: int,
        structured_schema: str | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "phase": phase,
            "message_count": message_count,
        }
        if structured_schema is not None:
            payload["structured_schema"] = structured_schema
        self._add_model_label(payload, agent_name)
        self._trace_recorder.record(
            event_type="model_call",
            title="Model call completed",
            status="succeeded",
            agent_name=agent_name,
            payload=payload,
        )

    def _record_failure(
        self,
        *,
        agent_name: str,
        phase: str,
        error: Exception,
    ) -> None:
        payload: dict[str, object] = {"phase": phase, "error": str(error)}
        self._add_model_label(payload, agent_name)
        self._trace_recorder.record(
            event_type="model_call",
            title="Model call failed",
            status="failed",
            agent_name=agent_name,
            payload=payload,
        )

    def _add_model_label(self, payload: dict[str, object], agent_name: str) -> None:
        label = self._label_for(agent_name)
        if label is not None:
            payload["model"] = label

    def _invoke_with_retry(
        self,
        operation: Callable[[], _ResultT],
        *,
        agent_name: str,
        phase: str,
    ) -> _ResultT:
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return operation()
            except Exception as error:
                last_error = error
                if attempt >= self._max_attempts:
                    break
                self._record_retry(
                    agent_name=agent_name,
                    phase=phase,
                    attempt=attempt,
                    error=error,
                )
                self._sleep_before_retry(attempt)
        if last_error is None:
            raise AgentExecutionError("Model call produced no result")
        raise last_error

    def _sleep_before_retry(self, attempt: int) -> None:
        if self._retry_backoff_seconds <= 0:
            return
        time.sleep(self._retry_backoff_seconds * (2 ** (attempt - 1)))

    def _record_retry(
        self,
        *,
        agent_name: str,
        phase: str,
        attempt: int,
        error: Exception,
    ) -> None:
        payload: dict[str, object] = {
            "phase": phase,
            "attempt": attempt,
            "error": str(error),
        }
        self._add_model_label(payload, agent_name)
        self._trace_recorder.record(
            event_type="model_call",
            title="Model call retrying after transient error",
            status="retrying",
            agent_name=agent_name,
            payload=payload,
        )


def _message_content(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(_content_part_to_text(part) for part in content)


def _content_part_to_text(part: str | dict[str, object]) -> str:
    if isinstance(part, str):
        return part
    text = part.get("text")
    if isinstance(text, str):
        return text
    return str(cast(object, part))


def _messages(system_prompt: str, user_prompt: str) -> list[BaseMessage]:
    from langchain_core.messages import HumanMessage, SystemMessage

    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
