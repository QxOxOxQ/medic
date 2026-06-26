from __future__ import annotations

import shutil
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from evaluation.application.errors import CorpusProvisioningError
from evaluation.application.models import SeededEvaluationCorpus
from evaluation.domain.suite import EvaluationProfile
from rag.database.repositories import DocumentRepository


class EvaluationDocumentSeeder:
    def __init__(
        self,
        *,
        project_root: Path,
        raw_documents_dir: Path,
        session_factory: sessionmaker[Session],
    ) -> None:
        self._project_root = project_root
        self._raw_documents_dir = raw_documents_dir
        self._session_factory = session_factory

    def seed(
        self,
        profile: EvaluationProfile,
        *,
        corpus_fingerprint: str,
        owner_user_id: UUID,
    ) -> SeededEvaluationCorpus:
        document_ids: set[UUID] = set()
        relative_raw_paths: set[str] = set()
        source_keys: set[str] = set()
        with self._session_factory() as session:
            repository = DocumentRepository(session)
            for relative_source in profile.document_paths:
                source = self._project_root / relative_source
                if not source.is_file():
                    raise CorpusProvisioningError(
                        f"Missing evaluation document: {relative_source}"
                    )
                relative_target = Path(profile.id) / corpus_fingerprint / source.name
                target = self._raw_documents_dir / relative_target
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                document_id = self._ensure_document(
                    repository,
                    owner_user_id=owner_user_id,
                    source=source,
                    relative_target=relative_target.as_posix(),
                )
                document_ids.add(document_id)
                relative_raw_paths.add(relative_target.as_posix())
                source_keys.add(source.name)
            session.commit()
        return SeededEvaluationCorpus(
            document_ids=frozenset(document_ids),
            relative_raw_paths=frozenset(relative_raw_paths),
            source_keys=frozenset(source_keys),
        )

    @staticmethod
    def _ensure_document(
        repository: DocumentRepository,
        *,
        owner_user_id: UUID,
        source: Path,
        relative_target: str,
    ) -> UUID:
        existing = repository.get_by_relative_raw_path(relative_target)
        if existing is not None:
            if existing.owner_user_id != owner_user_id:
                raise CorpusProvisioningError(
                    f"Evaluation path belongs to another user: {relative_target}"
                )
            return UUID(str(existing.id))
        created = repository.create_uploaded_document(
            owner_user_id=owner_user_id,
            original_filename=source.name,
            relative_raw_path=relative_target,
            byte_size=source.stat().st_size,
        )
        return UUID(str(created.id))
