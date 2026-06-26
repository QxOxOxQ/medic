from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

from rag.config import PROJECT_ROOT


@dataclass(frozen=True)
class EvaluationSettings:
    collection_prefix: str
    profile_directory: Path
    bootstrap_directory: Path
    calibration_path: Path
    raw_documents_dir: Path
    parsed_markdown_dir: Path
    judge_model: str
    judge_provider: str
    judge_prompt_version: str
    embedding_model: str
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_base_url: str
    langfuse_environment: str
    confirmation_timeout_seconds: int

    def bootstrap_path(self, profile_id: str) -> Path:
        return self.bootstrap_directory / f"{profile_id.replace('-', '_')}.json"


def get_evaluation_settings() -> EvaluationSettings:
    data_root = PROJECT_ROOT / "data" / "evaluation"
    dotenv_settings = dotenv_values(PROJECT_ROOT / ".env")
    return EvaluationSettings(
        collection_prefix=os.getenv("MEDIC_EVAL_QDRANT_PREFIX", "medic_eval"),
        profile_directory=PROJECT_ROOT / "evaluation" / "profiles",
        bootstrap_directory=PROJECT_ROOT / "evaluation" / "suites",
        calibration_path=(
            PROJECT_ROOT / "evaluation" / "suites" / "judge_calibration_v1.json"
        ),
        raw_documents_dir=data_root / "raw",
        parsed_markdown_dir=data_root / "parsed",
        judge_model=os.getenv("MEDIC_EVAL_JUDGE_MODEL", "openai/gpt-4.1-mini"),
        judge_provider="OpenAI",
        judge_prompt_version="ragas-0.4.3-default-v1",
        embedding_model=os.getenv(
            "MEDIC_EVAL_EMBEDDING_MODEL", "openai/text-embedding-3-small"
        ),
        langfuse_public_key=_environment_value(
            "LANGFUSE_PUBLIC_KEY", dotenv_settings, ""
        ),
        langfuse_secret_key=_environment_value(
            "LANGFUSE_SECRET_KEY", dotenv_settings, ""
        ),
        langfuse_base_url=_environment_value(
            "LANGFUSE_BASE_URL", dotenv_settings, "https://cloud.langfuse.com"
        ),
        langfuse_environment=_environment_value(
            "LANGFUSE_TRACING_ENVIRONMENT", dotenv_settings, "evaluation"
        ),
        confirmation_timeout_seconds=int(
            os.getenv("MEDIC_EVAL_CONFIRMATION_TIMEOUT_SECONDS", "30")
        ),
    )


def _environment_value(
    name: str,
    dotenv_settings: Mapping[str, str | None],
    default: str,
) -> str:
    value = os.getenv(name) or dotenv_settings.get(name)
    return value or default
