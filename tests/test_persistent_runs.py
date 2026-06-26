from __future__ import annotations

from pathlib import Path

from agents.models import AgentTraceEvent
from rag.database.chat_store import SqlAlchemyChatConversationStore
from rag.database.migrations import upgrade_database
from rag.database.pipeline_store import SqlAlchemyPipelineRunRepository
from rag.database.repositories import DocumentRepository, UserRepository
from rag.database.session import create_database_engine
from sqlalchemy.orm import sessionmaker


def _session_factory(tmp_path: Path) -> sessionmaker:
    database_url = f"sqlite:///{tmp_path / 'runs.db'}"
    upgrade_database(database_url)
    engine = create_database_engine(database_url)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def test_pipeline_history_and_events_are_persistent_and_owner_scoped(
    tmp_path: Path,
) -> None:
    factory = _session_factory(tmp_path)
    with factory() as session:
        users = UserRepository(session)
        owner = users.create_user(username="owner", password="secret")
        other = users.create_user(username="other", password="secret")
        document = DocumentRepository(session).create_uploaded_document(
            owner_user_id=owner.id,
            original_filename="report.pdf",
            relative_raw_path="owner/report.pdf",
            byte_size=100,
        )
        session.commit()
        owner_id = owner.id
        other_id = other.id
        document_id = document.id

    repository = SqlAlchemyPipelineRunRepository(factory)
    created = repository.create(
        owner_user_id=owner_id,
        document_ids=(document_id,),
    )
    repository.start(run_id=created.run.id)
    repository.append_event(
        run_id=created.run.id,
        payload={
            "step": "prepare",
            "status": "running",
            "message": "Preparing report",
            "result": {"path": "owner/report.pdf"},
        },
    )
    repository.append_event(
        run_id=created.run.id,
        payload={
            "step": "index",
            "status": "succeeded",
            "message": "Indexed report",
            "result": {"path": "owner/report.pdf"},
        },
    )
    repository.finish(
        run_id=created.run.id,
        status="succeeded",
        summary="1 document indexed",
        error=None,
    )

    persisted = repository.get_for_owner(
        owner_user_id=owner_id,
        run_id=created.run.id,
    )
    hidden = repository.get_for_owner(
        owner_user_id=other_id,
        run_id=created.run.id,
    )
    replay = repository.events_after(
        owner_user_id=owner_id,
        run_id=created.run.id,
        sequence=1,
    )

    assert persisted is not None
    assert persisted.status == "succeeded"
    assert persisted.documents[0].document_name == "report.pdf"
    assert persisted.documents[0].status == "succeeded"
    assert [event.sequence for event in persisted.events] == [1, 2]
    assert replay is not None
    assert [event.sequence for event in replay] == [2]
    assert hidden is None


def test_active_pipeline_and_chat_runs_are_interrupted_after_restart(
    tmp_path: Path,
) -> None:
    factory = _session_factory(tmp_path)
    with factory() as session:
        user = UserRepository(session).create_user(
            username="owner",
            password="secret",
        )
        session.commit()
        owner_id = user.id

    pipeline_repository = SqlAlchemyPipelineRunRepository(factory)
    pipeline = pipeline_repository.create(
        owner_user_id=owner_id,
        document_ids=(),
    )
    pipeline_repository.start(run_id=pipeline.run.id)

    chat_store = SqlAlchemyChatConversationStore(factory)
    chat = chat_store.start_conversation(
        owner_user_id=owner_id,
        question="What is in the document?",
    )
    chat_store.trace_sink(run_id=chat.run_id).record(
        AgentTraceEvent(
            sequence=1,
            event_type="coordinator",
            title="Coordinator selected specialists",
            status="succeeded",
        )
    )

    assert pipeline_repository.interrupt_active_runs() == 1
    assert chat_store.interrupt_active_runs() == 1

    interrupted_pipeline = pipeline_repository.get_for_owner(
        owner_user_id=owner_id,
        run_id=pipeline.run.id,
    )
    interrupted_chat = chat_store.run_view(
        owner_user_id=owner_id,
        run_id=chat.run_id,
    )
    trace = chat_store.trace_events_after(
        owner_user_id=owner_id,
        run_id=chat.run_id,
        sequence=0,
    )

    assert interrupted_pipeline is not None
    assert interrupted_pipeline.status == "interrupted"
    assert interrupted_chat is not None
    assert interrupted_chat.status == "interrupted"
    assert trace is not None
    assert trace[0].as_dict()["phase"] == "coordinator"
