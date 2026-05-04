from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.tool import ToolCall, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from agents.models import (
    AgentAnswer,
    AgentExecutionError,
    AgentRequest,
    UnknownAgentError,
)
from agents.observability import AgentObservability, NullAgentObservability
from agents.profiles import AgentProfile, AgentRegistry
from agents.trace import AgentTraceRecorder
from tools.source_ledger import SourceLedger


AGENT_PROMPT_VERSION = "agents-v1"


INSUFFICIENT_CONTEXT_ANSWER = (
    "I could not find enough context in the documentation to prepare a "
    "source-grounded answer."
)


@dataclass(frozen=True)
class SpecialistAnswer:
    agent_name: str
    answer: str
    insufficient_context: bool


class AgentGraph:
    def __init__(
        self,
        *,
        chat_model: BaseChatModel,
        tools: list[BaseTool],
        source_ledger: SourceLedger,
        max_tool_iterations: int,
        max_review_rounds: int = 0,
        registry: AgentRegistry | None = None,
        trace_recorder: AgentTraceRecorder | None = None,
        observability: AgentObservability | None = None,
    ) -> None:
        self._registry = registry or AgentRegistry()
        self._tools = tools
        self._tools_by_name = {tool.name: tool for tool in tools}
        self._source_ledger = source_ledger
        self._max_tool_iterations = max_tool_iterations
        self._max_review_rounds = max_review_rounds
        self._plain_model = chat_model
        self._tool_model = chat_model.bind_tools(self._tools)
        self._required_tool_model = chat_model.bind_tools(
            self._tools,
            tool_choice=self._required_tool_choice(),
        )
        self._trace_recorder = trace_recorder or AgentTraceRecorder()
        self._observability = observability or NullAgentObservability()

    def answer(self, request: AgentRequest) -> AgentAnswer:
        with self._observability.trace(request):
            return self._answer(request)

    def _answer(self, request: AgentRequest) -> AgentAnswer:
        try:
            profiles = self._select_profiles(request)
            specialist_answers = [
                self._run_specialist(profile, request) for profile in profiles
            ]
            specialist_answers = self._review_specialists(
                profiles,
                request,
                specialist_answers,
            )
            answer, insufficient_context = self._synthesize_answer(
                request,
                specialist_answers=specialist_answers,
            )
        except AgentExecutionError:
            raise
        except UnknownAgentError:
            raise
        except Exception as error:
            self._trace_recorder.record(
                event_type="error",
                title="Agent execution failed",
                status="failed",
                payload={"error": str(error)},
            )
            raise AgentExecutionError("Agent execution failed") from error

        result = AgentAnswer(
            answer=answer,
            agents=tuple(profile.name for profile in profiles),
            sources=self._source_ledger.sources(),
            insufficient_context=insufficient_context,
            trace_events=self._trace_recorder.events(),
        )
        self._observability.complete(result)
        return result

    def _select_profiles(self, request: AgentRequest) -> tuple[AgentProfile, ...]:
        profiles = self._registry.select_many(
            question=request.question,
            requested_agent=request.requested_agent,
        )
        self._trace_recorder.record(
            event_type="coordinator",
            title="Coordinator selected specialists",
            status="succeeded",
            agent_name="coordinator",
            payload={
                "requested_agent": request.requested_agent,
                "selected_agents": [profile.name for profile in profiles],
            },
        )
        return profiles

    def _run_specialist(
        self,
        profile: AgentProfile,
        request: AgentRequest,
        *,
        feedback: str | None = None,
    ) -> SpecialistAnswer:
        self._trace_recorder.record(
            event_type="agent",
            title=f"{profile.display_name} started",
            status="running",
            agent_name=profile.name,
        )
        messages = _to_langchain_messages(
            profile.build_messages(
                question=request.question,
                conversation_context=_conversation_context(request),
            )
        )
        if feedback:
            messages.append(HumanMessage(content=_revision_instruction(feedback)))
        tool_iterations = 0
        last_ai_message: AIMessage | None = None

        while True:
            model = self._model_for_iteration(tool_iterations)
            last_ai_message = self._invoke_model(
                model,
                messages,
                agent_name=profile.name,
                phase="specialist",
            )
            messages.append(last_ai_message)

            if not last_ai_message.tool_calls:
                break
            if tool_iterations >= self._max_tool_iterations:
                break

            messages.extend(
                self._tool_messages(
                    last_ai_message.tool_calls,
                    agent_name=profile.name,
                )
            )
            tool_iterations += 1

        answer = _answer_from_message(last_ai_message)
        insufficient_context = not answer
        if insufficient_context:
            answer = _insufficient_context_answer()

        self._trace_recorder.record(
            event_type="agent",
            title=f"{profile.display_name} finished",
            status="succeeded" if not insufficient_context else "insufficient_context",
            agent_name=profile.name,
            payload={
                "tool_iterations": tool_iterations,
                "answer_preview": answer[:240],
            },
        )
        return SpecialistAnswer(
            agent_name=profile.name,
            answer=answer,
            insufficient_context=insufficient_context,
        )

    def _review_specialists(
        self,
        profiles: tuple[AgentProfile, ...],
        request: AgentRequest,
        specialist_answers: list[SpecialistAnswer],
    ) -> list[SpecialistAnswer]:
        if self._max_review_rounds <= 0:
            return specialist_answers

        answers = list(specialist_answers)
        profile_by_name = {profile.name: profile for profile in profiles}
        for _ in range(self._max_review_rounds):
            all_approved = True
            for index, current in enumerate(answers):
                if current.insufficient_context:
                    continue
                feedback = self._review_specialist_answer(
                    profile_by_name[current.agent_name],
                    request,
                    current,
                )
                if feedback is None:
                    continue
                all_approved = False
                answers[index] = self._run_specialist(
                    profile_by_name[current.agent_name],
                    request,
                    feedback=feedback,
                )
            if all_approved:
                break
        return answers

    def _review_specialist_answer(
        self,
        profile: AgentProfile,
        request: AgentRequest,
        specialist_answer: SpecialistAnswer,
    ) -> str | None:
        messages = [
            SystemMessage(content=_review_system_prompt()),
            HumanMessage(
                content=_review_user_prompt(
                    request,
                    profile=profile,
                    specialist_answer=specialist_answer,
                    source_blocks=[
                        source.prompt_block()
                        for source in self._source_ledger.sources()
                    ],
                )
            ),
        ]
        response = self._invoke_model(
            self._plain_model,
            messages,
            agent_name="coordinator",
            phase="review",
        )
        verdict = _message_content(response).strip()
        requests_more_work = _review_requests_more_work(verdict)
        self._trace_recorder.record(
            event_type="review",
            title=f"Coordinator reviewed {profile.display_name}",
            status="needs_revision" if requests_more_work else "approved",
            agent_name="coordinator",
            payload={
                "reviewed_agent": profile.name,
                "verdict_preview": verdict[:240],
            },
        )
        if not requests_more_work:
            return None
        return verdict

    def _model_for_iteration(self, tool_iterations: int) -> Runnable[Any, AIMessage]:
        if tool_iterations == 0:
            return self._required_tool_model
        return self._tool_model

    def _invoke_model(
        self,
        model: Runnable[Any, AIMessage] | BaseChatModel,
        messages: list[BaseMessage],
        *,
        agent_name: str,
        phase: str,
    ) -> AIMessage:
        config = self._observability.model_config(
            agent_name=agent_name,
            phase=phase,
        )
        try:
            if config is None:
                response = model.invoke(messages)
            else:
                response = model.invoke(messages, config=config)
        except Exception as error:
            self._trace_recorder.record(
                event_type="model_call",
                title="Model call failed",
                status="failed",
                agent_name=agent_name,
                payload={"phase": phase, "error": str(error)},
            )
            raise AgentExecutionError("Agent model call failed") from error

        self._trace_recorder.record(
            event_type="model_call",
            title="Model call completed",
            status="succeeded",
            agent_name=agent_name,
            payload={
                "phase": phase,
                "message_count": len(messages),
                "requested_tool_calls": len(getattr(response, "tool_calls", ()) or ()),
            },
        )
        if not isinstance(response, AIMessage):
            raise AgentExecutionError("Agent model returned an unsupported message")
        return response

    def _tool_messages(
        self,
        tool_calls: list[ToolCall],
        *,
        agent_name: str,
    ) -> list[ToolMessage]:
        messages: list[ToolMessage] = []
        for tool_call in tool_calls:
            name = str(tool_call.get("name", ""))
            tool = self._tools_by_name.get(name)
            if tool is None:
                raise AgentExecutionError(f"Unknown tool requested: {name}")

            args = tool_call.get("args", {})
            self._trace_recorder.record(
                event_type="tool_call",
                title="Tool requested",
                status="running",
                agent_name=agent_name,
                tool_name=name,
                payload={"args": args if isinstance(args, dict) else {}},
            )
            try:
                config = self._observability.tool_config(
                    agent_name=agent_name,
                    tool_name=name,
                )
                if config is None:
                    content = tool.invoke(args)
                else:
                    content = tool.invoke(args, config=config)
            except Exception as error:
                self._trace_recorder.record(
                    event_type="tool_call",
                    title="Tool failed",
                    status="failed",
                    agent_name=agent_name,
                    tool_name=name,
                    payload={"error": str(error)},
                )
                raise AgentExecutionError("Agent tool call failed") from error
            messages.append(
                ToolMessage(
                    content=str(content),
                    tool_call_id=str(tool_call.get("id", "")),
                    name=name,
                )
            )
        return messages

    def _synthesize_answer(
        self,
        request: AgentRequest,
        *,
        specialist_answers: list[SpecialistAnswer],
    ) -> tuple[str, bool]:
        sources = self._source_ledger.sources()
        if not sources:
            return _insufficient_context_answer(), True
        if all(answer.insufficient_context for answer in specialist_answers):
            return _insufficient_context_answer(), True

        messages = [
            SystemMessage(content=_synthesis_system_prompt()),
            HumanMessage(
                content=_synthesis_user_prompt(
                    request,
                    specialist_answers=specialist_answers,
                    source_blocks=[source.prompt_block() for source in sources],
                )
            ),
        ]
        response = self._invoke_model(
            self._plain_model,
            messages,
            agent_name="coordinator",
            phase="synthesis",
        )
        answer = _message_content(response).strip()
        insufficient_context = not answer
        if insufficient_context:
            answer = _insufficient_context_answer()

        self._trace_recorder.record(
            event_type="synthesis",
            title="Coordinator synthesized final answer",
            status="succeeded" if not insufficient_context else "insufficient_context",
            agent_name="coordinator",
            payload={
                "specialist_count": len(specialist_answers),
                "source_count": len(sources),
            },
        )
        return answer, insufficient_context

    def _required_tool_choice(self) -> str | None:
        if not self._tools:
            return None
        return self._tools[0].name


def _to_langchain_messages(messages: list[dict[str, str]]) -> list[BaseMessage]:
    converted: list[BaseMessage] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "user":
            converted.append(HumanMessage(content=content))
        else:
            raise ValueError(f"Unsupported message role: {role}")
    return converted


def _conversation_context(request: AgentRequest) -> str:
    lines: list[str] = []
    for message in request.conversation_messages:
        label = "User" if message.role == "user" else "Assistant"
        lines.append(f"{label}: {message.content}")
    return "\n".join(lines)


def _answer_from_message(message: AIMessage | None) -> str:
    if message is None or message.tool_calls:
        return ""
    return _message_content(message).strip()


def _message_content(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(_content_part_to_text(part) for part in content)


def _content_part_to_text(part: str | dict[str, Any]) -> str:
    if isinstance(part, str):
        return part
    text = part.get("text")
    if isinstance(text, str):
        return text
    return str(part)


def _insufficient_context_answer() -> str:
    return INSUFFICIENT_CONTEXT_ANSWER


def _synthesis_system_prompt() -> str:
    return (
        "You are the coordinator for a medical-documentation RAG assistant. "
        "Synthesize specialist notes into one concise answer. Use only the "
        "provided sources, cite claims inline as [S1], [S2], and state clearly "
        "when the sources are insufficient."
    )


def _synthesis_user_prompt(
    request: AgentRequest,
    *,
    specialist_answers: list[SpecialistAnswer],
    source_blocks: list[str],
) -> str:
    specialist_block = "\n\n".join(
        f"{answer.agent_name}: {answer.answer}" for answer in specialist_answers
    )
    return (
        "Response language: English\n\n"
        f"Question:\n{request.question}\n\n"
        f"Recent conversation:\n{_conversation_context(request) or '-'}\n\n"
        f"Specialist notes:\n{specialist_block or '-'}\n\n"
        f"Sources:\n{chr(10).join(source_blocks)}"
    )


REVIEW_APPROVED_TOKEN = "APPROVED"


def _review_system_prompt() -> str:
    return (
        "You are the coordinator for a medical-documentation RAG assistant and "
        "the final reviewer of a specialist's draft answer. Review the draft "
        "strictly. Check that every clinical claim is grounded in the provided "
        "sources and cited as [S1], [S2], that the assessment is realistic and "
        "complete, that doubts or conflicting evidence are resolved, and that "
        "no red flag is missed. "
        f"If the draft is complete and trustworthy, reply with exactly "
        f"{REVIEW_APPROVED_TOKEN} and nothing else. Otherwise reply with "
        "concise, specific instructions telling the specialist what to fix, "
        "which claims need grounding, and which additional document searches "
        "or analysis to perform."
    )


def _review_user_prompt(
    request: AgentRequest,
    *,
    profile: AgentProfile,
    specialist_answer: SpecialistAnswer,
    source_blocks: list[str],
) -> str:
    return (
        f"Specialist under review: {profile.display_name}\n\n"
        f"User question:\n{request.question}\n\n"
        f"Specialist draft answer:\n{specialist_answer.answer}\n\n"
        f"Available sources:\n{chr(10).join(source_blocks) or '-'}"
    )


def _review_requests_more_work(verdict: str) -> bool:
    normalized = verdict.strip()
    if not normalized:
        return False
    return normalized.upper() != REVIEW_APPROVED_TOKEN


def _revision_instruction(feedback: str) -> str:
    return (
        "A coordinator reviewed your previous draft and requires revisions "
        "before it can be accepted. Address every point below. Call "
        "search_user_medical_documents again with focused queries when more "
        "evidence is needed, keep claims grounded in the cited sources, and "
        "produce an improved final answer.\n\n"
        f"Coordinator feedback:\n{feedback}"
    )
