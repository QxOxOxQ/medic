from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from langfuse import Langfuse
from langfuse.api import DatasetItem
from langfuse.api.core import ApiError

from evaluation.application.errors import (
    EvaluationConfigurationError,
    EvaluationDatasetError,
)
from evaluation.application.models import DatasetBootstrapResult
from evaluation.domain.suite import EvaluationProfile
from evaluation.infrastructure.langfuse_codec import (
    non_empty_string,
    object_value,
    string_list,
)


class LangfuseDatasetBootstrapper:
    def __init__(self, client: Langfuse) -> None:
        self._client = client

    def authenticate(self) -> None:
        try:
            authenticated = self._client.auth_check()
        except Exception as error:
            raise EvaluationConfigurationError("Langfuse authentication failed") from error
        if not authenticated:
            raise EvaluationConfigurationError("Langfuse credentials are invalid")

    def bootstrap(
        self,
        *,
        profile: EvaluationProfile,
        manifest_path: str,
    ) -> DatasetBootstrapResult:
        manifest = _read_manifest(Path(manifest_path), profile=profile)
        self._ensure_dataset(profile)
        dataset = self._load_dataset(profile.dataset_name)
        existing = {item.id: item for item in dataset.items}
        created = 0
        verified = 0
        for payload in manifest:
            item_id = _dataset_item_id(profile.dataset_name, payload["id"])
            expected = _bootstrap_payload(payload)
            current = existing.get(item_id)
            if current is not None:
                _verify_existing_item(current, expected)
                verified += 1
                continue
            self._client.create_dataset_item(
                id=item_id,
                dataset_name=profile.dataset_name,
                input=expected["input"],
                expected_output=expected["expected_output"],
                metadata=expected["metadata"],
            )
            created += 1
        self._client.flush()
        return DatasetBootstrapResult(
            dataset_name=profile.dataset_name,
            created_items=created,
            verified_items=verified,
        )

    def _ensure_dataset(self, profile: EvaluationProfile) -> None:
        try:
            self._client.api.datasets.get(dataset_name=profile.dataset_name)
            return
        except ApiError as error:
            if error.status_code != 404:
                raise EvaluationDatasetError("Cannot inspect Langfuse dataset") from error
        self._client.create_dataset(
            name=profile.dataset_name,
            description=f"Medic synthetic evaluation dataset for {profile.id}",
            metadata={"profile_id": profile.id, "synthetic_only": True},
            input_schema=_input_schema(),
            expected_output_schema=_expected_output_schema(),
        )

    def _load_dataset(self, name: str) -> Any:
        try:
            return self._client.get_dataset(name)
        except Exception as error:
            raise EvaluationDatasetError(f"Cannot load Langfuse dataset: {name}") from error


def _read_manifest(
    path: Path,
    *,
    profile: EvaluationProfile,
) -> tuple[dict[str, Any], ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        root = object_value(payload, "bootstrap manifest")
        if root.get("id") != profile.id or root.get("version") != profile.version:
            raise ValueError("Manifest profile identity does not match")
        cases = root["cases"]
        if not isinstance(cases, list) or not cases:
            raise TypeError("Manifest cases must be a non-empty array")
        resolved = tuple(object_value(case, "bootstrap case") for case in cases)
        case_ids = [non_empty_string(case.get("id"), "case id") for case in resolved]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Bootstrap case ids must be unique")
        return resolved
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise EvaluationDatasetError(f"Invalid bootstrap manifest: {path}") from error


def _bootstrap_payload(payload: dict[str, Any]) -> dict[str, dict[str, object]]:
    case_id = non_empty_string(payload.get("id"), "case id")
    requested_agent = payload.get("requested_agent")
    if requested_agent is not None and not isinstance(requested_agent, str):
        raise EvaluationDatasetError("requested_agent must be a string")
    input_value: dict[str, object] = {
        "id": case_id,
        "question": non_empty_string(payload.get("question"), "question"),
        "expected_source_keys": string_list(payload.get("expected_source_keys", [])),
        "answerable": _boolean(payload.get("answerable"), "answerable"),
        "requested_agent": requested_agent,
        "tags": string_list(payload.get("tags", [])),
    }
    expected_output: dict[str, object] = {
        "reference_answer": non_empty_string(
            payload.get("reference_answer"),
            "reference answer",
        )
    }
    return {
        "input": input_value,
        "expected_output": expected_output,
        "metadata": {
            "case_id": case_id,
            "manifest_item_hash": _canonical_hash(
                {"input": input_value, "expected_output": expected_output}
            ),
        },
    }


def _verify_existing_item(
    item: DatasetItem,
    expected: dict[str, dict[str, object]],
) -> None:
    actual = {
        "input": item.input,
        "expected_output": item.expected_output,
        "metadata": item.metadata,
    }
    if _canonical_json(actual) != _canonical_json(expected):
        raise EvaluationDatasetError(f"Bootstrap item drift detected: {item.id}")


def _dataset_item_id(dataset_name: str, case_id: object) -> str:
    value = non_empty_string(case_id, "case id")
    return str(uuid5(NAMESPACE_URL, f"https://medic.local/{dataset_name}/{value}"))


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise EvaluationDatasetError(f"{label} must be boolean")
    return value


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _input_schema() -> dict[str, object]:
    return {
        "type": "object",
        "required": ["id", "question", "expected_source_keys", "answerable"],
        "properties": {
            "id": {"type": "string"},
            "question": {"type": "string"},
            "expected_source_keys": {"type": "array", "items": {"type": "string"}},
            "answerable": {"type": "boolean"},
            "requested_agent": {"type": ["string", "null"]},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
    }


def _expected_output_schema() -> dict[str, object]:
    return {
        "type": "object",
        "required": ["reference_answer"],
        "properties": {"reference_answer": {"type": "string"}},
    }
