from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evaluation.application.errors import CorpusProvisioningError
from evaluation.domain.suite import EvaluationProfile
from rag.chunking.process_text import MARKDOWN_CHUNK_OVERLAP, MARKDOWN_CHUNK_SIZE
from rag.config import SETTINGS


class CorpusFingerprintCalculator:
    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root

    def calculate(self, profile: EvaluationProfile) -> str:
        digest = hashlib.sha256()
        digest.update(profile.id.encode("utf-8"))
        digest.update(profile.version.encode("utf-8"))
        digest.update(self._pipeline_configuration())
        for relative_path in profile.document_paths:
            path = self._project_root / relative_path
            if not path.is_file():
                raise CorpusProvisioningError(
                    f"Missing evaluation document: {relative_path}"
                )
            digest.update(relative_path.encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    @staticmethod
    def _pipeline_configuration() -> bytes:
        payload = {
            "chunk_size": MARKDOWN_CHUNK_SIZE,
            "chunk_overlap": MARKDOWN_CHUNK_OVERLAP,
            "embedding": SETTINGS["embedding"],
            "fast_embedding": SETTINGS["fast_embedding"],
            "qdrant": SETTINGS["qdrant"],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
