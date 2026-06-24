from __future__ import annotations

from types import SimpleNamespace

from sqlalchemy import inspect
from sqlalchemy.orm import sessionmaker

from rag.database.migrations import upgrade_database
from rag.database.repositories import ChunkInput, DocumentRepository, UserRepository
from rag.database.session import create_database_engine
from rag.retrieval import search_results_from_response


def _database_session_factory(database_url: str) -> sessionmaker:
    upgrade_database(database_url)
    engine = create_database_engine(database_url)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def test_alembic_upgrade_creates_postgres_app_tables(tmp_path) -> None:
    database_url = f"sqlite:///{tmp_path / 'migrated.db'}"
    upgrade_database(database_url)
    engine = create_database_engine(database_url)

    inspector = inspect(engine)

    assert {
        "alembic_version",
        "users",
        "documents",
        "document_chunks",
        "chat_conversations",
        "chat_messages",
        "chat_runs",
        "chat_trace_events",
        "chat_message_sources",
        "pipeline_runs",
        "pipeline_run_documents",
        "pipeline_run_events",
    }.issubset(set(inspector.get_table_names()))
    user_columns = {column["name"] for column in inspector.get_columns("users")}
    assert user_columns >= {
        "id",
        "username",
        "password_hash",
        "is_active",
        "is_admin",
    }
    removed_language_column = "preferred" + "_language"
    assert removed_language_column not in user_columns
    assert {column["name"] for column in inspector.get_columns("document_chunks")} >= {
        "document_id",
        "chunk_index",
        "content",
        "qdrant_point_id",
    }
    assert {column["name"] for column in inspector.get_columns("documents")} >= {
        "processing_error",
    }
    assert {
        "owner_user_id",
        "title",
    }.issubset({column["name"] for column in inspector.get_columns("chat_conversations")})
    assert {
        "conversation_id",
        "role",
        "content",
        "sequence",
    }.issubset({column["name"] for column in inspector.get_columns("chat_messages")})
    assert {
        "run_id",
        "event_type",
        "payload",
    }.issubset({column["name"] for column in inspector.get_columns("chat_trace_events")})


def test_repositories_authenticate_and_store_document_chunks(tmp_path) -> None:
    factory = _database_session_factory(f"sqlite:///{tmp_path / 'repository.db'}")

    with factory() as session:
        users = UserRepository(session)
        admin = users.seed_admin(username="Admin", password="secret")
        documents = DocumentRepository(session)
        documents.mark_processing_failed(
            owner_user_id=admin.id,
            relative_raw_path="raw/report.pdf",
            original_filename="report.pdf",
            byte_size=123,
            processed_at=None,
            processing_error="RuntimeError: parser failed",
        )
        document = documents.upsert_prepared_document(
            owner_user_id=admin.id,
            relative_raw_path="raw/report.pdf",
            original_filename="report.pdf",
            parsed_markdown_path="raw/report.md",
            content_hash="hash",
            byte_size=123,
            processed_at=None,
        )
        assert document.owner_user_id == admin.id
        assert document.processing_error is None
        documents.upsert_chunks_for_parsed_path(
            parsed_markdown_path="raw/report.md",
            chunks=[
                ChunkInput(
                    chunk_index=1,
                    char_start=0,
                    char_end=11,
                    content="hello world",
                    qdrant_point_id="point-1",
                )
            ],
        )
        session.commit()

    with factory() as session:
        users = UserRepository(session)
        assert users.authenticate(username="admin", password="secret") is not None
        assert users.authenticate(username="admin", password="wrong") is None
        document = DocumentRepository(session).get_by_parsed_markdown_path(
            "raw/report.md"
        )
        assert document is not None
        assert document.status == "indexed"
        assert document.chunks[0].qdrant_point_id == "point-1"


def test_search_results_are_filtered_to_document_owner(tmp_path) -> None:
    factory = _database_session_factory(f"sqlite:///{tmp_path / 'search.db'}")
    with factory() as session:
        users = UserRepository(session)
        owner = users.create_user(username="owner", password="secret")
        other = users.create_user(username="other", password="secret")
        documents = DocumentRepository(session)
        documents.upsert_prepared_document(
            owner_user_id=owner.id,
            relative_raw_path="owner/report.pdf",
            original_filename="report.pdf",
            parsed_markdown_path="owner/report.md",
            content_hash="owner-hash",
            byte_size=1,
            processed_at=None,
        )
        documents.upsert_prepared_document(
            owner_user_id=other.id,
            relative_raw_path="other/report.pdf",
            original_filename="report.pdf",
            parsed_markdown_path="other/report.md",
            content_hash="other-hash",
            byte_size=1,
            processed_at=None,
        )
        documents.upsert_chunks_for_parsed_path(
            parsed_markdown_path="owner/report.md",
            chunks=[
                ChunkInput(1, 0, 5, "owner chunk", "owner-point"),
            ],
        )
        documents.upsert_chunks_for_parsed_path(
            parsed_markdown_path="other/report.md",
            chunks=[
                ChunkInput(1, 0, 5, "other chunk", "other-point"),
            ],
        )
        owner_id = owner.id
        session.commit()

    response = SimpleNamespace(
        points=[
            _point("owner-point", "owner-hash", "owner/report.md", "owner chunk"),
            _point("other-point", "other-hash", "other/report.md", "other chunk"),
        ]
    )

    results = search_results_from_response(
        response,
        owner_user_id=owner_id,
        database_session_factory=factory,
    )

    assert [result.excerpt for result in results] == ["owner chunk"]


def test_search_results_match_owned_document_by_source_basename(tmp_path) -> None:
    factory = _database_session_factory(f"sqlite:///{tmp_path / 'source.db'}")
    with factory() as session:
        user = UserRepository(session).create_user(username="owner", password="secret")
        DocumentRepository(session).upsert_prepared_document(
            owner_user_id=user.id,
            relative_raw_path="nested/report.pdf",
            original_filename="report.pdf",
            parsed_markdown_path="nested/report.md",
            content_hash="owner-hash",
            byte_size=1,
            processed_at=None,
        )
        owner_id = user.id
        session.commit()

    response = SimpleNamespace(
        points=[
            _point(None, "unknown-hash", "report.md", "source-only owner chunk"),
        ]
    )

    results = search_results_from_response(
        response,
        owner_user_id=owner_id,
        database_session_factory=factory,
    )

    assert [result.excerpt for result in results] == ["source-only owner chunk"]


def test_search_results_fall_back_to_owned_hash_when_point_id_is_not_recorded(
    tmp_path,
) -> None:
    factory = _database_session_factory(f"sqlite:///{tmp_path / 'hash-fallback.db'}")
    with factory() as session:
        user = UserRepository(session).create_user(username="owner", password="secret")
        DocumentRepository(session).upsert_prepared_document(
            owner_user_id=user.id,
            relative_raw_path="nested/report.pdf",
            original_filename="report.pdf",
            parsed_markdown_path="nested/report.md",
            content_hash="owner-hash",
            byte_size=1,
            processed_at=None,
        )
        owner_id = user.id
        session.commit()

    response = SimpleNamespace(
        points=[
            _point("qdrant-only-point", "owner-hash", "nested/report.md", "owner chunk"),
        ]
    )

    results = search_results_from_response(
        response,
        owner_user_id=owner_id,
        database_session_factory=factory,
    )

    assert [result.excerpt for result in results] == ["owner chunk"]
    assert results[0].qdrant_point_id == "qdrant-only-point"
    assert results[0].document_name == "report.pdf"


def _point(
    point_id: str | None,
    content_hash: str,
    source: str,
    content: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=point_id,
        score=0.9,
        payload={
            "content_hash": content_hash,
            "source": source,
            "content": content,
        },
    )
