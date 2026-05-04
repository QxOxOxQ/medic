from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from agents.graph import AgentGraph, INSUFFICIENT_CONTEXT_ANSWER
from agents.models import AgentAnswer, AgentRequest, UnknownAgentError
from agents.observability import AgentObservability
from agents.profiles import AgentRegistry, load_profiles
from agents.trace import AgentTraceRecorder
from rag.retrieval import SearchResult
from tools import RagSearchTool, SourceLedger


class ScriptedChatModel:
    def __init__(self, responses: Sequence[AIMessage]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []
        self.binds: list[dict[str, object]] = []

    def bind_tools(
        self,
        tools: list[object],
        *,
        tool_choice: str | None = None,
    ) -> BoundScriptedChatModel:
        self.binds.append({"tools": tools, "tool_choice": tool_choice})
        return BoundScriptedChatModel(self, tool_choice=tool_choice)

    def next_response(self) -> AIMessage:
        return self._responses.pop(0)

    def invoke(
        self,
        messages: list[object],
        config: RunnableConfig | None = None,
    ) -> AIMessage:
        self.calls.append(
            {
                "messages": list(messages),
                "tool_choice": None,
                "config": config,
            }
        )
        return self.next_response()


class BoundScriptedChatModel:
    def __init__(
        self,
        parent: ScriptedChatModel,
        *,
        tool_choice: str | None,
    ) -> None:
        self._parent = parent
        self._tool_choice = tool_choice

    def invoke(
        self,
        messages: list[object],
        config: RunnableConfig | None = None,
    ) -> AIMessage:
        self._parent.calls.append(
            {
                "messages": list(messages),
                "tool_choice": self._tool_choice,
                "config": config,
            }
        )
        return self._parent.next_response()


class RecordingRetriever:
    def __init__(self, results: Sequence[SearchResult]) -> None:
        self._results = tuple(results)
        self.calls: list[dict[str, object]] = []

    def search(
        self,
        *,
        query: str,
        limit: int,
        owner_user_id: UUID | None = None,
    ) -> Sequence[SearchResult]:
        self.calls.append(
            {
                "query": query,
                "limit": limit,
                "owner_user_id": owner_user_id,
            }
        )
        return self._results


class RecordingObservability:
    def __init__(self) -> None:
        self.traced_requests: list[AgentRequest] = []
        self.model_calls: list[tuple[str, str]] = []
        self.tool_calls: list[tuple[str, str]] = []
        self.completed_answers: list[AgentAnswer] = []

    def trace(self, request: AgentRequest) -> AbstractContextManager[None]:
        self.traced_requests.append(request)
        return nullcontext()

    def model_config(self, *, agent_name: str, phase: str) -> RunnableConfig | None:
        self.model_calls.append((agent_name, phase))
        return {"metadata": {"phase": phase}}

    def tool_config(self, *, agent_name: str, tool_name: str) -> RunnableConfig | None:
        self.tool_calls.append((agent_name, tool_name))
        return {"metadata": {"tool_name": tool_name}}

    def complete(self, answer: AgentAnswer) -> None:
        self.completed_answers.append(answer)

    def close(self) -> None:
        return None


def _source_result() -> SearchResult:
    return SearchResult(
        score=0.91,
        source="report.md",
        document_name="Clinical Report",
        content_hash="hash",
        excerpt="LDL cholesterol is elevated in the lipid panel.",
    )


def _tool_call(query: str = "LDL cholesterol trend", limit: int = 2) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_user_medical_documents",
                "args": {"query": query, "limit": limit},
                "id": "call-1",
            }
        ],
    )


def _graph(
    *,
    chat_model: ScriptedChatModel,
    retriever: RecordingRetriever,
    owner_user_id: UUID | None = None,
    max_tool_iterations: int = 3,
    max_review_rounds: int = 0,
    observability: AgentObservability | None = None,
) -> AgentGraph:
    ledger = SourceLedger()
    trace_recorder = AgentTraceRecorder()
    rag_tool = RagSearchTool(
        retriever=retriever,
        owner_user_id=owner_user_id or uuid4(),
        source_ledger=ledger,
        default_limit=5,
        trace_recorder=trace_recorder,
    )
    return AgentGraph(
        chat_model=chat_model,
        tools=[rag_tool.to_langchain_tool()],
        source_ledger=ledger,
        max_tool_iterations=max_tool_iterations,
        max_review_rounds=max_review_rounds,
        trace_recorder=trace_recorder,
        observability=observability,
    )


def test_agent_graph_routes_question_and_runs_rag_tool_loop() -> None:
    retriever = RecordingRetriever([_source_result()])
    chat_model = ScriptedChatModel(
        [
            _tool_call(),
            AIMessage(content="The lipid panel includes elevated LDL [S1]."),
            AIMessage(content="Final: the lipid panel includes elevated LDL [S1]."),
        ]
    )
    graph = _graph(chat_model=chat_model, retriever=retriever)

    answer = graph.answer(
        AgentRequest(
            question="Is the lipid panel concerning?",
            requested_agent="internist",
        )
    )

    assert answer.answer == "Final: the lipid panel includes elevated LDL [S1]."
    assert answer.agents == ("cardiometabolic_internist",)
    assert answer.sources[0].id == "S1"
    assert answer.sources[0].source == "report.md"
    assert answer.sources[0].document_name == "Clinical Report"
    assert answer.sources[0].content_hash == "hash"
    assert answer.sources[0].score == 0.91
    assert (
        answer.sources[0].excerpt == "LDL cholesterol is elevated in the lipid panel."
    )
    assert answer.insufficient_context is False
    assert retriever.calls[0]["query"] == "LDL cholesterol trend"
    assert retriever.calls[0]["limit"] == 2
    assert chat_model.calls[0]["tool_choice"] == "search_user_medical_documents"
    assert any(
        isinstance(message, ToolMessage) for message in chat_model.calls[1]["messages"]
    )
    event_types = {event.event_type for event in answer.trace_events}
    assert {"coordinator", "agent", "model_call", "tool_call", "tool"}.issubset(
        event_types
    )


def test_agent_graph_passes_model_and_tool_runs_to_observability() -> None:
    retriever = RecordingRetriever([_source_result()])
    chat_model = ScriptedChatModel(
        [
            _tool_call(),
            AIMessage(content="Specialist answer [S1]."),
            AIMessage(content="Final answer [S1]."),
        ]
    )
    observability = RecordingObservability()
    graph = _graph(
        chat_model=chat_model,
        retriever=retriever,
        observability=observability,
    )
    request = AgentRequest(question="Question", requested_agent="internist")

    answer = graph.answer(request)

    assert observability.traced_requests == [request]
    assert observability.model_calls == [
        ("cardiometabolic_internist", "specialist"),
        ("cardiometabolic_internist", "specialist"),
        ("coordinator", "synthesis"),
    ]
    assert observability.tool_calls == [
        ("cardiometabolic_internist", "search_user_medical_documents")
    ]
    assert observability.completed_answers == [answer]
    assert chat_model.calls[0]["config"] == {"metadata": {"phase": "specialist"}}


def test_agent_graph_coordinator_reviews_and_requests_specialist_revision() -> None:
    retriever = RecordingRetriever([_source_result()])
    chat_model = ScriptedChatModel(
        [
            _tool_call(),
            AIMessage(content="Draft answer [S1]."),
            AIMessage(content="Add the documented LDL value and cite it."),
            _tool_call(query="LDL value"),
            AIMessage(content="Revised answer with LDL value [S1]."),
            AIMessage(content="Final: revised answer with LDL value [S1]."),
        ]
    )
    graph = _graph(
        chat_model=chat_model,
        retriever=retriever,
        max_review_rounds=1,
    )

    answer = graph.answer(
        AgentRequest(
            question="Is the lipid panel concerning?",
            requested_agent="internist",
        )
    )

    assert answer.answer == "Final: revised answer with LDL value [S1]."
    assert answer.insufficient_context is False
    assert len(retriever.calls) == 2
    review_events = [
        event for event in answer.trace_events if event.event_type == "review"
    ]
    assert review_events
    assert review_events[0].status == "needs_revision"
    revision_messages = [
        message
        for call in chat_model.calls
        for message in call["messages"]
        if "Coordinator feedback" in getattr(message, "content", "")
    ]
    assert revision_messages


def test_agent_graph_coordinator_approves_specialist_without_revision() -> None:
    retriever = RecordingRetriever([_source_result()])
    chat_model = ScriptedChatModel(
        [
            _tool_call(),
            AIMessage(content="Draft answer [S1]."),
            AIMessage(content="APPROVED"),
            AIMessage(content="Final: draft answer [S1]."),
        ]
    )
    graph = _graph(
        chat_model=chat_model,
        retriever=retriever,
        max_review_rounds=1,
    )

    answer = graph.answer(
        AgentRequest(
            question="Is the lipid panel concerning?",
            requested_agent="internist",
        )
    )

    assert answer.answer == "Final: draft answer [S1]."
    assert len(retriever.calls) == 1
    review_events = [
        event for event in answer.trace_events if event.event_type == "review"
    ]
    assert review_events
    assert review_events[0].status == "approved"


def test_agent_graph_coordinator_requires_exact_approved_verdict() -> None:
    retriever = RecordingRetriever([_source_result()])
    chat_model = ScriptedChatModel(
        [
            _tool_call(),
            AIMessage(content="Draft answer [S1]."),
            AIMessage(content="APPROVED but add the LDL value."),
            _tool_call(query="LDL value"),
            AIMessage(content="Revised answer with LDL value [S1]."),
            AIMessage(content="Final: revised answer with LDL value [S1]."),
        ]
    )
    graph = _graph(
        chat_model=chat_model,
        retriever=retriever,
        max_review_rounds=1,
    )

    answer = graph.answer(
        AgentRequest(
            question="Is the lipid panel concerning?",
            requested_agent="internist",
        )
    )

    assert answer.answer == "Final: revised answer with LDL value [S1]."
    assert len(retriever.calls) == 2
    review_events = [
        event for event in answer.trace_events if event.event_type == "review"
    ]
    assert review_events
    assert review_events[0].status == "needs_revision"


def test_agent_graph_rejects_unknown_requested_profile() -> None:
    retriever = RecordingRetriever([_source_result()])
    chat_model = ScriptedChatModel([AIMessage(content="unused")])
    graph = _graph(chat_model=chat_model, retriever=retriever)

    with pytest.raises(UnknownAgentError, match="Unknown agent"):
        graph.answer(
            AgentRequest(
                question="Question",
                requested_agent="unknown-specialist",
            )
        )


def test_agent_graph_requires_rag_tool_before_final_answer() -> None:
    retriever = RecordingRetriever([_source_result()])
    chat_model = ScriptedChatModel([AIMessage(content="Answer without sources.")])
    graph = _graph(chat_model=chat_model, retriever=retriever)

    answer = graph.answer(AgentRequest(question="What does the result mean?"))

    assert answer.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert answer.insufficient_context is True
    assert answer.sources == ()
    assert retriever.calls == []


def test_agent_graph_returns_insufficient_context_when_rag_finds_no_sources() -> None:
    retriever = RecordingRetriever([])
    chat_model = ScriptedChatModel(
        [
            _tool_call(query="missing document"),
            AIMessage(content="There is no source-grounded answer."),
        ]
    )
    graph = _graph(chat_model=chat_model, retriever=retriever)

    answer = graph.answer(AgentRequest(question="Question"))

    assert answer.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert answer.insufficient_context is True
    assert answer.sources == ()
    assert retriever.calls[0]["query"] == "missing document"


def test_agent_graph_stops_tool_loop_at_configured_limit() -> None:
    retriever = RecordingRetriever([_source_result()])
    chat_model = ScriptedChatModel([_tool_call(), _tool_call(query="second search")])
    graph = _graph(
        chat_model=chat_model,
        retriever=retriever,
        max_tool_iterations=1,
    )

    answer = graph.answer(AgentRequest(question="Question"))

    assert answer.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert answer.insufficient_context is True
    assert len(retriever.calls) == 1
    assert answer.sources[0].id == "S1"


def test_agent_profiles_are_loaded_from_config_and_markdown_prompts(
    tmp_path: Path,
) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "base.md").write_text("Base prompt from markdown.", encoding="utf-8")
    (prompts_dir / "custom.md").write_text(
        "Custom agent prompt from markdown.",
        encoding="utf-8",
    )
    profiles_path = tmp_path / "profiles.json"
    profiles_path.write_text(
        json.dumps(
            {
                "system_prompt": "base.md",
                "default_agent": "custom_agent",
                "profiles": [
                    {
                        "name": "custom_agent",
                        "display_name": "Custom Agent",
                        "aliases": ["custom"],
                        "keywords": ["custom-keyword"],
                        "prompt": "custom.md",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    profiles = load_profiles(
        profiles_path=profiles_path,
        prompts_dir=prompts_dir,
    )
    registry = AgentRegistry(profiles=profiles)
    profile = registry.select(question="custom-keyword", requested_agent=None)
    messages = profile.build_messages(question="Question")

    assert profile.name == "custom_agent"
    assert "Base prompt from markdown." in messages[0]["content"]
    assert "Custom agent prompt from markdown." in messages[0]["content"]
    assert messages[1]["content"] == "Response language: English\n\nQuestion:\nQuestion"


def test_agent_registry_routes_english_keywords() -> None:
    registry = AgentRegistry()

    assert (
        registry.select(
            question="The patient has psoriasis and a rash.",
            requested_agent=None,
        ).name
        == "dermatologist"
    )
    assert (
        registry.select(
            question="Knee effusion continues after ACL rehabilitation.",
            requested_agent=None,
        ).name
        == "orthopedist"
    )


def test_agent_registry_selects_all_profiles_for_broad_question() -> None:
    registry = AgentRegistry()

    profiles = registry.select_many(
        question="Summarize all medical problems in the records.",
        requested_agent=None,
    )

    assert [profile.name for profile in profiles] == [
        "orthopedist",
        "neurologist",
        "dermatologist",
        "cardiometabolic_internist",
    ]
