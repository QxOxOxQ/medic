from __future__ import annotations

import re
from uuid import UUID

from evaluation.application.errors import CorpusProvisioningError
from evaluation.application.models import ReadyEvaluationCorpus
from evaluation.application.ports import (
    CollectionGuard,
    CollectionInspector,
    CorpusFingerprintProvider,
    DocumentSeeder,
    IndexRebuilder,
    TenantProvisioner,
)
from evaluation.domain.suite import EvaluationProfile


class EnsureEvaluationCorpus:
    def __init__(
        self,
        *,
        collection_prefix: str,
        fingerprint_provider: CorpusFingerprintProvider,
        collection_guard: CollectionGuard,
        tenant_provisioner: TenantProvisioner,
        document_seeder: DocumentSeeder,
        index_rebuilder: IndexRebuilder,
        collection_inspector: CollectionInspector,
    ) -> None:
        self._collection_prefix = collection_prefix
        self._fingerprint_provider = fingerprint_provider
        self._collection_guard = collection_guard
        self._tenant_provisioner = tenant_provisioner
        self._document_seeder = document_seeder
        self._index_rebuilder = index_rebuilder
        self._collection_inspector = collection_inspector

    def execute(self, profile: EvaluationProfile) -> ReadyEvaluationCorpus:
        fingerprint = self._fingerprint_provider.calculate(profile)
        collection_name = _collection_name(
            self._collection_prefix,
            profile.id,
            fingerprint,
        )
        self._collection_guard.validate(collection_name)
        owner_user_id = self._tenant_provisioner.ensure_tenant()
        seeded = self._document_seeder.seed(
            profile,
            corpus_fingerprint=fingerprint,
            owner_user_id=owner_user_id,
        )
        if not self._is_ready(collection_name, seeded.document_ids):
            self._index_rebuilder.rebuild(
                profile,
                corpus_fingerprint=fingerprint,
                owner_user_id=owner_user_id,
                collection_name=collection_name,
            )
        if not self._is_ready(collection_name, seeded.document_ids):
            raise CorpusProvisioningError("Evaluation collection is incomplete")
        return ReadyEvaluationCorpus(
            owner_user_id=owner_user_id,
            collection_name=collection_name,
            fingerprint=fingerprint,
            seeded=seeded,
        )

    def _is_ready(self, collection_name: str, document_ids: frozenset[UUID]) -> bool:
        return self._collection_inspector.is_ready(
            collection_name=collection_name,
            document_ids=document_ids,
        )


def _collection_name(prefix: str, profile_id: str, fingerprint: str) -> str:
    normalized_prefix = re.sub(r"[^a-zA-Z0-9_-]", "_", prefix).strip("_")
    normalized_profile = re.sub(r"[^a-zA-Z0-9_-]", "_", profile_id).strip("_")
    if not normalized_prefix or not normalized_profile:
        raise CorpusProvisioningError("Evaluation collection components cannot be empty")
    return f"{normalized_prefix}_{normalized_profile}_{fingerprint[:12]}"
