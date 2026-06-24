from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from agents.graph import AgentGraph
from agents.observability import AgentObservability
from agents.trace import AgentTraceRecorder
from agents.trace import AgentTraceSink
from backend.chat_use_cases import ChatConversationUseCase
from backend.use_cases import AnswerQuestionUseCase
from clients.chat_models import ChatModelFactory, get_chat_model_settings
from rag.retrieval import RetrievalService
from observability import build_agent_observability
from rag.database.chat_store import SqlAlchemyChatConversationStore
from tools import ObservedRagSearchPort, RagSearchTool, SourceLedger


def build_answer_question_use_case(
    *,
    database_session_factory: sessionmaker[Session],
    retriever: RetrievalService | None = None,
    observability: AgentObservability | None = None,
) -> AnswerQuestionUseCase:
    return AnswerQuestionUseCase(
        agent_runner_factory=build_agent_runner_factory(
            database_session_factory=database_session_factory,
            retriever=retriever,
            observability=observability,
        ),
    )


def build_chat_conversation_use_case(
    *,
    database_session_factory: sessionmaker[Session],
    observability: AgentObservability | None = None,
) -> ChatConversationUseCase:
    return ChatConversationUseCase(
        agent_runner_factory=build_agent_runner_factory(
            database_session_factory=database_session_factory,
            observability=observability,
        ),
        conversation_store=SqlAlchemyChatConversationStore(database_session_factory),
    )


def build_agent_runner_factory(
    *,
    database_session_factory: sessionmaker[Session],
    retriever: RetrievalService | None = None,
    observability: AgentObservability | None = None,
) -> DefaultAgentRunnerFactory:
    resolved_retriever = retriever or RetrievalService(
        database_session_factory=database_session_factory
    )
    resolved_observability = observability or build_agent_observability()
    return DefaultAgentRunnerFactory(
        retriever=resolved_retriever,
        observability=resolved_observability,
    )


class DefaultAgentRunnerFactory:
    def __init__(
        self,
        *,
        retriever: RetrievalService,
        observability: AgentObservability,
    ) -> None:
        self._retriever = retriever
        self._observability = observability

    def __call__(
        self,
        *,
        owner_user_id: UUID,
        retrieval_limit: int,
        trace_sink: AgentTraceSink | None = None,
    ) -> AgentGraph:
        return _build_agent_graph(
            retriever=self._retriever,
            owner_user_id=owner_user_id,
            retrieval_limit=retrieval_limit,
            observability=self._observability,
            trace_sink=trace_sink,
        )


def _build_agent_graph(
    *,
    retriever: RetrievalService,
    owner_user_id: UUID,
    retrieval_limit: int,
    observability: AgentObservability,
    trace_sink: AgentTraceSink | None = None,
) -> AgentGraph:
    chat_settings = get_chat_model_settings()
    source_ledger = SourceLedger()
    trace_recorder = AgentTraceRecorder(trace_sink)
    rag_tool = RagSearchTool(
        retriever=retriever,
        owner_user_id=owner_user_id,
        source_ledger=source_ledger,
        default_limit=retrieval_limit,
        trace_recorder=trace_recorder,
    )
    return AgentGraph(
        chat_model=ChatModelFactory().create(chat_settings),
        search_port=ObservedRagSearchPort(
            tool=rag_tool,
            observability=observability,
            agent_name="professor",
        ),
        max_retrieval_queries=chat_settings.max_retrieval_queries,
        max_consultations=chat_settings.max_consultations,
        max_review_rounds=chat_settings.max_review_rounds,
        trace_recorder=trace_recorder,
        observability=observability,
    )
