from __future__ import annotations

from functools import partial
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from evaluation.application.errors import CorpusProvisioningError
from evaluation.domain.suite import EvaluationProfile
from rag.config import DocumentPreparationSettings
from rag.full_process import FullProcess
from rag.indexer import index_text
from rag.qdrant import Qdrant


class EvaluationIndexRebuilder:
    def __init__(
        self,
        *,
        raw_documents_dir: Path,
        parsed_markdown_dir: Path,
        session_factory: sessionmaker[Session],
    ) -> None:
        self._settings = DocumentPreparationSettings(
            raw_documents_dir=raw_documents_dir,
            parsed_markdown_dir=parsed_markdown_dir,
        )
        self._session_factory = session_factory

    def rebuild(
        self,
        profile: EvaluationProfile,
        *,
        corpus_fingerprint: str,
        owner_user_id: UUID,
        collection_name: str,
    ) -> None:
        qdrant = Qdrant(collection_name=collection_name)
        if qdrant.collection_exists(collection_name):
            qdrant.delete_collection(collection_name)
        bound_indexer = partial(
            index_text,
            database_session_factory=self._session_factory,
            qdrant=qdrant,
        )
        summary = FullProcess(
            settings=self._settings,
            database_session_factory=self._session_factory,
            indexer=bound_indexer,
        ).execute(
            print_summary=False,
            selected_raw_paths=[
                f"{profile.id}/{corpus_fingerprint}/{relative_path.rsplit('/', 1)[-1]}"
                for relative_path in profile.document_paths
            ],
            owner_user_id=owner_user_id,
        )
        if summary.failed:
            raise CorpusProvisioningError(
                f"Evaluation corpus preparation failed for {summary.failed} document(s)"
            )
