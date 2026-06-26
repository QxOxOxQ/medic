from __future__ import annotations

from uuid import UUID

from langchain_core.language_models.chat_models import BaseChatModel
from sqlalchemy.orm import Session, sessionmaker

from agents.graph import AgentGraph
from agents.model_router import RoutedModel
from agents.observability import AgentObservability
from agents.trace import AgentTraceRecorder
from agents.trace import AgentTraceSink
from backend.chat_use_cases import ChatConversationUseCase
from backend.full_document_reader import ParsedMarkdownDocumentReader
from backend.use_cases import AnswerQuestionUseCase
from clients.chat_models import ChatModelFactory, get_chat_model_settings
from clients.chat_models.settings import ChatModelSettings
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
        database_session_factory=database_session_factory,
    )


class DefaultAgentRunnerFactory:
    def __init__(
        self,
        *,
        retriever: RetrievalService,
        observability: AgentObservability,
        database_session_factory: sessionmaker[Session] | None = None,
    ) -> None:
        self._retriever = retriever
        self._observability = observability
        self._database_session_factory = database_session_factory

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
            database_session_factory=self._database_session_factory,
            trace_sink=trace_sink,
        )


def _build_agent_graph(
    *,
    retriever: RetrievalService,
    owner_user_id: UUID,
    retrieval_limit: int,
    observability: AgentObservability,
    database_session_factory: sessionmaker[Session] | None = None,
    trace_sink: AgentTraceSink | None = None,
) -> AgentGraph:
    chat_settings = get_chat_model_settings()
    default_model, model_overrides = _build_chat_models(chat_settings)
    source_ledger = SourceLedger()
    trace_recorder = AgentTraceRecorder(trace_sink)
    rag_tool = RagSearchTool(
        retriever=retriever,
        owner_user_id=owner_user_id,
        source_ledger=source_ledger,
        default_limit=retrieval_limit,
        trace_recorder=trace_recorder,
    )
    full_document_reader = (
        ParsedMarkdownDocumentReader(
            database_session_factory=database_session_factory,
            owner_user_id=owner_user_id,
        )
        if database_session_factory is not None
        else None
    )
    return AgentGraph(
        chat_model=default_model,
        search_port=ObservedRagSearchPort(
            tool=rag_tool,
            observability=observability,
            agent_name="professor",
        ),
        max_retrieval_queries=chat_settings.max_retrieval_queries,
        max_consultations=chat_settings.max_consultations,
        max_review_rounds=chat_settings.max_review_rounds,
        max_full_documents=chat_settings.max_full_documents,
        trace_recorder=trace_recorder,
        observability=observability,
        full_document_reader=full_document_reader,
        model_overrides=model_overrides,
        default_label=chat_settings.model,
    )


def _build_chat_models(
    chat_settings: ChatModelSettings,
) -> tuple[BaseChatModel, dict[str, RoutedModel]]:
    factory = ChatModelFactory()
    default_model = factory.create(chat_settings)
    models_by_id: dict[str, BaseChatModel] = {chat_settings.model: default_model}
    overrides: dict[str, RoutedModel] = {}
    for role, model_id in chat_settings.agent_models.items():
        if model_id not in models_by_id:
            models_by_id[model_id] = factory.create(chat_settings, model=model_id)
        overrides[role] = RoutedModel(model=models_by_id[model_id], label=model_id)
    return default_model, overrides
