from __future__ import annotations

from pathlib import Path
from uuid import UUID
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from agents.models import AgentAnswer, AgentRequest, AgentSource, AgentTraceEvent
from backend.chat_use_cases import ChatConversationUseCase
from dashboard.app import create_app
from dashboard.auth import AuthSettings
from rag.database.chat_store import SqlAlchemyChatConversationStore
from rag.database.migrations import upgrade_database
from rag.database.repositories import UserRepository
from rag.database.session import create_database_engine


class UnusedAgentRunner:
    def answer(self, request: AgentRequest) -> AgentAnswer:
        del request
        raise AssertionError("Read-only conversation tests must not run the agent")


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


def _client(tmp_path: Path) -> TestClient:
    session_factory = _database_session_factory(tmp_path)
    store = SqlAlchemyChatConversationStore(session_factory)
    use_case = ChatConversationUseCase(
        agent_runner_factory=lambda **_: UnusedAgentRunner(),
        conversation_store=store,
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


def _owner_id(client: TestClient, username: str = "admin") -> UUID:
    with client.app.state.database_session_factory() as session:
        user = UserRepository(session).get_by_username(username)
        assert user is not None
        return user.id


def _seed_completed_conversation(
    client: TestClient,
    *,
    username: str = "admin",
    question: str = "Is LDL high?",
    answer: AgentAnswer | None = None,
) -> str:
    owner_user_id = _owner_id(client, username=username)
    store = SqlAlchemyChatConversationStore(client.app.state.database_session_factory)
    started = store.start_conversation(owner_user_id=owner_user_id, question=question)
    detail = store.complete_run(
        owner_user_id=owner_user_id,
        conversation_id=started.conversation_id,
        run_id=started.run_id,
        answer=answer or _single_source_answer(),
    )
    assert detail is not None
    return str(detail.id)


def test_chat_conversation_read_api_lists_and_loads_conversation(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    _login(client)
    conversation_id = _seed_completed_conversation(client)

    listed = client.get("/api/chat/conversations")
    loaded = client.get(f"/api/chat/conversations/{conversation_id}")

    assert listed.status_code == 200
    assert listed.json()["conversations"][0]["id"] == conversation_id
    assert loaded.status_code == 200
    conversation = loaded.json()["conversation"]
    assert conversation["title"] == "Is LDL high?"
    assert [message["role"] for message in conversation["messages"]] == [
        "user",
        "assistant",
    ]
    assistant = conversation["messages"][1]
    assert assistant["sources"][0]["source_id"] == "S1"
    assert assistant["sources"][0]["qdrant_point_id"] == "point-1"
    assert assistant["trace_events"][0]["event_type"] == "coordinator"


def test_chat_conversation_read_api_marks_only_cited_sources_as_used(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    _login(client)
    conversation_id = _seed_completed_conversation(
        client,
        question="Compare A and B",
        answer=_two_source_answer(),
    )

    loaded = client.get(f"/api/chat/conversations/{conversation_id}")

    assert loaded.status_code == 200
    assistant = loaded.json()["conversation"]["messages"][1]
    by_id = {source["source_id"]: source for source in assistant["sources"]}
    assert by_id["S1"]["used"] is True
    assert by_id["S2"]["used"] is False


def test_chat_conversation_read_api_is_scoped_to_logged_in_user(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    _login(client, "admin")
    conversation_id = _seed_completed_conversation(client, username="admin")

    other_client = TestClient(client.app)
    _login(other_client, "other")

    response = other_client.get(f"/api/chat/conversations/{conversation_id}")

    assert response.status_code == 404


def test_legacy_chat_execution_endpoints_are_removed(tmp_path: Path) -> None:
    client = _client(tmp_path)
    _login(client)
    conversation_id = uuid4()

    ask = client.post("/api/ask", json={"question": "Is LDL high?"})
    create = client.post(
        "/api/chat/conversations",
        json={"question": "Is LDL high?"},
    )
    append = client.post(
        f"/api/chat/conversations/{conversation_id}/messages",
        json={"question": "What about glucose?"},
    )

    assert ask.status_code == 404
    assert create.status_code == 405
    assert append.status_code == 404
    paths = client.app.openapi()["paths"]
    assert "/api/ask" not in paths
    assert "post" not in paths["/api/chat/conversations"]
    assert "/api/chat/conversations/{conversation_id}/messages" not in paths


def _single_source_answer() -> AgentAnswer:
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


def _two_source_answer() -> AgentAnswer:
    return AgentAnswer(
        answer="Grounded on [S1] only.",
        agents=(),
        sources=(
            AgentSource(
                id="S1",
                source="a.md",
                content_hash="hash-a",
                document_name="A.pdf",
                score=0.92,
                excerpt="Cited record.",
            ),
            AgentSource(
                id="S2",
                source="b.md",
                content_hash="hash-b",
                document_name="B.pdf",
                score=0.40,
                excerpt="Checked but unused record.",
            ),
        ),
        insufficient_context=False,
        trace_events=(),
    )
