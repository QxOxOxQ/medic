from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from agents.models import AgentAnswer, AgentRequest, AgentSource, AgentTraceEvent
from backend.chat_use_cases import ChatConversationUseCase
from dashboard.app import create_app
from dashboard.auth import AuthSettings
from rag.database.migrations import upgrade_database
from rag.database.repositories import UserRepository
from rag.database.session import create_database_engine


class RecordingAgentRunner:
    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []

    def answer(self, request: AgentRequest) -> AgentAnswer:
        self.requests.append(request)
        return AgentAnswer(
            answer="Source-grounded answer [S1].",
            agents=("cardiometabolic_internist",),
            sources=(
                AgentSource(
                    id="S1",
                    source="parsed/report.md",
                    content_hash="hash",
                    document_name="report.pdf",
                    score=0.91,
                    excerpt="LDL cholesterol is elevated.",
                    qdrant_point_id="point-1",
                    relative_raw_path="raw/report.pdf",
                    chunk_index=1,
                    char_start=0,
                    char_end=32,
                    retrieval_query="LDL trend",
                ),
            ),
            insufficient_context=False,
            trace_events=(
                AgentTraceEvent(
                    sequence=1,
                    event_type="coordinator",
                    title="Coordinator selected specialists",
                    status="succeeded",
                    agent_name="coordinator",
                    payload={"selected_agents": ["cardiometabolic_internist"]},
                ),
            ),
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
    database_url = f"sqlite:///{tmp_path / 'chat.db'}"
    upgrade_database(database_url)
    engine = create_database_engine(database_url)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with factory() as session:
        users = UserRepository(session)
        users.seed_admin(username="admin", password="secret")
        users.create_user(username="other", password="secret")
        session.commit()
    return factory


def _client(
    tmp_path: Path,
    *,
    runner: RecordingAgentRunner,
) -> TestClient:
    session_factory = _database_session_factory(tmp_path)
    use_case = ChatConversationUseCase(
        agent_runner_factory=RecordingAgentRunnerFactory(runner),
        database_session_factory=session_factory,
    )
    app = create_app(
        auth_settings=AuthSettings(
            username="admin",
            password="secret",
            session_secret="test-session-secret",
            cookie_secure=False,
        ),
        database_session_factory=session_factory,
        chat_conversation_use_case=use_case,
    )
    return TestClient(app)


def _login(client: TestClient, username: str = "admin") -> None:
    response = client.post(
        "/login",
        data={"username": username, "password": "secret"},
        follow_redirects=False,
    )
    assert response.status_code == 303


def test_chat_conversation_can_be_created_loaded_and_continued(tmp_path: Path) -> None:
    runner = RecordingAgentRunner()
    client = _client(tmp_path, runner=runner)
    _login(client)

    created = client.post(
        "/api/chat/conversations",
        json={"question": "Is LDL high?", "limit": 3},
    )

    assert created.status_code == 200
    conversation = created.json()["conversation"]
    conversation_id = conversation["id"]
    assert conversation["title"] == "Is LDL high?"
    assert [message["role"] for message in conversation["messages"]] == [
        "user",
        "assistant",
    ]
    assistant = conversation["messages"][1]
    assert assistant["sources"][0]["source_id"] == "S1"
    assert assistant["sources"][0]["qdrant_point_id"] == "point-1"
    assert assistant["trace_events"][0]["event_type"] == "coordinator"

    listed = client.get("/api/chat/conversations")
    assert listed.status_code == 200
    assert listed.json()["conversations"][0]["id"] == conversation_id

    loaded = client.get(f"/api/chat/conversations/{conversation_id}")
    assert loaded.status_code == 200
    assert len(loaded.json()["conversation"]["messages"]) == 2

    continued = client.post(
        f"/api/chat/conversations/{conversation_id}/messages",
        json={"question": "What about glucose?", "limit": 4},
    )

    assert continued.status_code == 200
    assert len(continued.json()["conversation"]["messages"]) == 4
    assert runner.requests[0].user_id is not None
    assert runner.requests[0].session_id == UUID(conversation_id)
    assert runner.requests[0].execution_id != runner.requests[1].execution_id
    assert runner.requests[1].conversation_messages[0].role == "user"
    assert runner.requests[1].conversation_messages[1].role == "assistant"


def test_chat_conversation_is_scoped_to_logged_in_user(tmp_path: Path) -> None:
    runner = RecordingAgentRunner()
    client = _client(tmp_path, runner=runner)
    _login(client, "admin")
    created = client.post(
        "/api/chat/conversations",
        json={"question": "Is LDL high?"},
    )
    conversation_id = created.json()["conversation"]["id"]

    other_client = TestClient(client.app)
    _login(other_client, "other")

    response = other_client.get(f"/api/chat/conversations/{conversation_id}")

    assert response.status_code == 404
