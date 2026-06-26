from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from rag.config import get_database_settings


SessionFactory = sessionmaker[Session]


def get_session_factory(database_url: str | None = None) -> SessionFactory:
    url = database_url or get_database_settings().database_url
    return _session_factory_for_url(url)


@lru_cache(maxsize=8)
def _session_factory_for_url(database_url: str) -> SessionFactory:
    engine = create_database_engine(database_url)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def create_database_engine(database_url: str) -> Engine:
    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(
        database_url,
        future=True,
        pool_pre_ping=True,
        connect_args=connect_args,
    )


@contextmanager
def session_scope(session_factory: SessionFactory | None = None) -> Iterator[Session]:
    factory = session_factory or get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
