from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ENVIRONMENT_PATTERN = re.compile(r"^(?!langfuse)[a-z0-9-_]{1,40}$")
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


class ObservabilityConfigurationError(ValueError):
    """Raised when runtime tracing configuration is invalid."""


@dataclass(frozen=True)
class LangfuseTracingSettings:
    enabled: bool
    public_key: str
    secret_key: str
    base_url: str
    environment: str
    capture_content: bool
    sample_rate: float


def load_langfuse_tracing_settings(
    *,
    environment: Mapping[str, str] | None = None,
) -> LangfuseTracingSettings:
    values = _runtime_environment() if environment is None else environment
    settings = LangfuseTracingSettings(
        enabled=_boolean(values, "MEDIC_LANGFUSE_TRACING_ENABLED", default=False),
        public_key=values.get("LANGFUSE_PUBLIC_KEY", "").strip(),
        secret_key=values.get("LANGFUSE_SECRET_KEY", "").strip(),
        base_url=values.get(
            "LANGFUSE_BASE_URL",
            "https://cloud.langfuse.com",
        ).strip(),
        environment=values.get(
            "MEDIC_LANGFUSE_ENVIRONMENT",
            "development",
        ).strip(),
        capture_content=_boolean(
            values,
            "MEDIC_LANGFUSE_CAPTURE_CONTENT",
            default=False,
        ),
        sample_rate=_sample_rate(values),
    )
    _validate(settings)
    return settings


def _runtime_environment() -> dict[str, str]:
    dotenv_environment = {
        name: value
        for name, value in dotenv_values(PROJECT_ROOT / ".env").items()
        if value is not None
    }
    return {**dotenv_environment, **os.environ}


def _boolean(values: Mapping[str, str], name: str, *, default: bool) -> bool:
    raw_value = values.get(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ObservabilityConfigurationError(f"{name} must be a boolean value")


def _sample_rate(values: Mapping[str, str]) -> float:
    raw_value = values.get("MEDIC_LANGFUSE_SAMPLE_RATE", "1.0")
    try:
        return float(raw_value)
    except ValueError as error:
        raise ObservabilityConfigurationError(
            "MEDIC_LANGFUSE_SAMPLE_RATE must be a number"
        ) from error


def _validate(settings: LangfuseTracingSettings) -> None:
    if not 0.0 <= settings.sample_rate <= 1.0:
        raise ObservabilityConfigurationError(
            "MEDIC_LANGFUSE_SAMPLE_RATE must be between 0 and 1"
        )
    if not _ENVIRONMENT_PATTERN.fullmatch(settings.environment):
        raise ObservabilityConfigurationError(
            "MEDIC_LANGFUSE_ENVIRONMENT must be a valid Langfuse environment"
        )
    if not settings.enabled:
        return
    if not settings.public_key or not settings.secret_key:
        raise ObservabilityConfigurationError(
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required when tracing is enabled"
        )
    if not settings.base_url:
        raise ObservabilityConfigurationError(
            "LANGFUSE_BASE_URL is required when tracing is enabled"
        )
