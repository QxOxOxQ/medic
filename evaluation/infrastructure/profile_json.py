from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.domain.errors import InvalidSuiteError
from evaluation.domain.suite import EvaluationProfile
from evaluation.domain.values import MetricName, Score, Threshold, ThresholdScope


class JsonProfileRepository:
    def __init__(self, profile_directory: Path) -> None:
        self._profile_directory = profile_directory

    def get(self, profile_id: str) -> EvaluationProfile:
        path = self._profile_directory / f"{profile_id.replace('-', '_')}.json"
        if not path.is_file() or path.parent != self._profile_directory:
            raise InvalidSuiteError(f"Unknown evaluation profile: {profile_id}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("Profile root must be an object")
            return _profile(payload)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise InvalidSuiteError(f"Invalid evaluation profile: {path.name}") from error


def profile_payload(profile: EvaluationProfile) -> dict[str, object]:
    return {
        "id": profile.id,
        "version": profile.version,
        "dataset_name": profile.dataset_name,
        "document_paths": list(profile.document_paths),
        "retrieval_limit": profile.retrieval_limit,
        "gate_version": profile.gate_version,
        "agent_prompt_version": profile.agent_prompt_version,
        "thresholds": [
            {
                "metric": threshold.metric.value,
                "minimum": threshold.minimum.value,
                "scope": threshold.scope.value,
            }
            for threshold in profile.thresholds
        ],
    }


def _profile(payload: dict[str, Any]) -> EvaluationProfile:
    document_paths = payload["document_paths"]
    thresholds = payload["thresholds"]
    if not isinstance(document_paths, list) or not isinstance(thresholds, list):
        raise TypeError("Profile collections must be arrays")
    return EvaluationProfile(
        id=_required_string(payload, "id"),
        version=_required_string(payload, "version"),
        dataset_name=_required_string(payload, "dataset_name"),
        document_paths=tuple(_string(value) for value in document_paths),
        thresholds=tuple(_threshold(value) for value in thresholds),
        gate_version=_required_string(payload, "gate_version"),
        agent_prompt_version=_required_string(payload, "agent_prompt_version"),
        retrieval_limit=int(payload.get("retrieval_limit", 5)),
    )


def _threshold(value: object) -> Threshold:
    if not isinstance(value, dict):
        raise TypeError("Threshold must be an object")
    return Threshold(
        metric=MetricName(_required_string(value, "metric")),
        minimum=Score(float(value["minimum"])),
        scope=ThresholdScope(str(value.get("scope", "aggregate"))),
    )


def _required_string(payload: dict[str, Any], key: str) -> str:
    return _string(payload[key])


def _string(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("Expected a non-empty string")
    return value
