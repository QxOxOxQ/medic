from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from agents.models import AgentAnswer, AgentRequest, AgentSource
from backend.use_cases import AnswerQuestionUseCase
from dashboard.app import create_app
from dashboard.auth import AuthSettings
from rag.database.migrations import upgrade_database
from rag.database.repositories import UserRepository
from rag.database.session import create_database_engine


class RecordingAgentRunner:
    def __init__(self, answer: AgentAnswer | None = None) -> None:
        self._answer = answer
        self.requests: list[AgentRequest] = []

    def answer(self, request: AgentRequest) -> AgentAnswer:
        self.requests.append(request)
        return self._answer or AgentAnswer(
            answer="Agent answer [S1].",
            agents=("cardiometabolic_internist",),
            sources=(
                AgentSource(
                    id="S1",
                    source="report.md",
                    document_name="Clinical Report",
                    content_hash="hash",
                    score=0.82,
                    excerpt="LDL cholesterol is elevated.",
                ),
            ),
            insufficient_context=False,
        )


class RecordingAgentRunnerFactory:
    def __init__(self, runner: RecordingAgentRunner) -> None:
        self._runner = runner
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        *,
        owner_user_id: UUID,
        retrieval_limit: int,
    ) -> RecordingAgentRunner:
        self.calls.append(
            {
                "owner_user_id": owner_user_id,
                "retrieval_limit": retrieval_limit,
            }
        )
        return self._runner


def _database_session_factory(tmp_path: Path) -> sessionmaker:
    database_url = f"sqlite:///{tmp_path / 'ask.db'}"
    upgrade_database(database_url)
    engine = create_database_engine(database_url)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with factory() as session:
        UserRepository(session).seed_admin(username="admin", password="secret")
        session.commit()
    return factory


def _client(
    tmp_path: Path,
    use_case: AnswerQuestionUseCase,
) -> TestClient:
    app = create_app(
        auth_settings=AuthSettings(
            username="admin",
            password="secret",
            session_secret="test-session-secret",
            cookie_secure=False,
        ),
        database_session_factory=_database_session_factory(tmp_path),
        answer_question_use_case=use_case,
    )
    client = TestClient(app)
    response = client.post(
        "/login",
        data={"username": "admin", "password": "secret"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    return client


def test_ask_endpoint_returns_agent_answer_with_sources(tmp_path: Path) -> None:
    runner = RecordingAgentRunner()
    runner_factory = RecordingAgentRunnerFactory(runner)
    use_case = AnswerQuestionUseCase(agent_runner_factory=runner_factory)
    client = _client(tmp_path, use_case)

    response = client.post(
        "/api/ask",
        json={
            "question": "Is the lipid panel concerning?",
            "limit": 3,
            "specialist": "internist",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["answer"] == "Agent answer [S1]."
    assert payload["agents"] == ["cardiometabolic_internist"]
    assert payload["insufficient_context"] is False
    assert {
        "id": "S1",
        "source": "report.md",
        "document_name": "Clinical Report",
        "content_hash": "hash",
        "score": 0.82,
        "excerpt": "LDL cholesterol is elevated.",
    }.items() <= payload["sources"][0].items()
    assert payload["sources"][0]["qdrant_point_id"] is None
    assert payload["sources"][0]["retrieval_query"] is None
    assert runner.requests[0].question == "Is the lipid panel concerning?"
    assert runner.requests[0].requested_agent == "internist"
    assert runner_factory.calls[0]["retrieval_limit"] == 3
    assert isinstance(runner_factory.calls[0]["owner_user_id"], UUID)


def test_ask_endpoint_returns_agent_insufficient_context(
    tmp_path: Path,
) -> None:
    runner = RecordingAgentRunner(
        AgentAnswer(
            answer="I could not find enough context in the documentation to prepare a source-grounded answer.",
            agents=("cardiometabolic_internist",),
            sources=(),
            insufficient_context=True,
        )
    )
    use_case = AnswerQuestionUseCase(
        agent_runner_factory=RecordingAgentRunnerFactory(runner),
    )
    client = _client(tmp_path, use_case)

    response = client.post("/api/ask", json={"question": "What next?", "limit": 3})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["insufficient_context"] is True
    assert payload["sources"] == []
    assert runner.requests[0].question == "What next?"


def test_ask_endpoint_payload_contains_chat_ui_rendering_contract(
    tmp_path: Path,
) -> None:
    runner = RecordingAgentRunner()
    use_case = AnswerQuestionUseCase(
        agent_runner_factory=RecordingAgentRunnerFactory(runner),
    )
    client = _client(tmp_path, use_case)

    response = client.post("/api/ask", json={"question": "Summarize my result."})

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) >= {
        "ok",
        "answer",
        "agents",
        "sources",
        "insufficient_context",
        "trace_events",
    }
    assert isinstance(payload["answer"], str)
    assert isinstance(payload["agents"], list)
    assert isinstance(payload["sources"], list)
    assert isinstance(payload["insufficient_context"], bool)
    assert set(payload["sources"][0]) >= {
        "id",
        "source",
        "document_name",
        "content_hash",
        "score",
        "excerpt",
        "qdrant_point_id",
        "document_id",
        "relative_raw_path",
        "chunk_index",
        "char_start",
        "char_end",
        "retrieval_query",
    }
    assert isinstance(payload["trace_events"], list)


def test_ask_endpoint_rejects_empty_question(tmp_path: Path) -> None:
    use_case = AnswerQuestionUseCase(
        agent_runner_factory=RecordingAgentRunnerFactory(RecordingAgentRunner()),
    )
    client = _client(tmp_path, use_case)

    response = client.post("/api/ask", json={"question": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "Question is required"


def test_ask_endpoint_returns_unavailable_when_agent_initialization_fails(
    tmp_path: Path,
) -> None:
    def _failing_agent_runner(
        *,
        owner_user_id: UUID,
        retrieval_limit: int,
    ) -> RecordingAgentRunner:
        raise RuntimeError("missing client configuration")

    use_case = AnswerQuestionUseCase(agent_runner_factory=_failing_agent_runner)
    client = _client(tmp_path, use_case)

    response = client.post("/api/ask", json={"question": "What does the result mean?"})

    assert response.status_code == 503
    assert response.json() == {"ok": False, "error": "Agent execution failed"}
