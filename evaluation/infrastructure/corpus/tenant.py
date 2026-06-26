from __future__ import annotations

from uuid import UUID, uuid4

from sqlalchemy.orm import Session, sessionmaker

from rag.database.repositories import UserRepository


EVALUATION_USERNAME = "__evaluation__"


class EvaluationTenantProvisioner:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def ensure_tenant(self) -> UUID:
        with self._session_factory() as session:
            repository = UserRepository(session)
            user = repository.get_by_username(EVALUATION_USERNAME)
            if user is None:
                user = repository.create_user(
                    username=EVALUATION_USERNAME,
                    password=uuid4().hex,
                    is_active=False,
                )
            session.commit()
            return UUID(str(user.id))
