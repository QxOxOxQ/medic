from __future__ import annotations

from typing import Any, cast

from langfuse import Evaluation
from langfuse.api import DatasetItem

from evaluation.application.errors import (
    EvaluationDatasetError,
    EvaluationExecutionError,
)
from evaluation.application.models import EvaluationRunSummary
from evaluation.domain.report import CaseResult
from evaluation.domain.suite import EvaluationCase
from evaluation.domain.values import SourceKey


def case_from_item(item: DatasetItem) -> EvaluationCase:
    input_value = object_value(item.input, "dataset input")
    expected_output = object_value(item.expected_output, "expected output")
    expected_sources = string_list(input_value.get("expected_source_keys", []))
    tags = string_list(input_value.get("tags", []))
    answerable = input_value.get("answerable")
    if not isinstance(answerable, bool):
        raise EvaluationDatasetError("Dataset answerable must be boolean")
    requested_agent = input_value.get("requested_agent")
    if requested_agent is not None and not isinstance(requested_agent, str):
        raise EvaluationDatasetError("Dataset requested_agent must be a string")
    return EvaluationCase(
        id=non_empty_string(input_value.get("id"), "case id"),
        question=non_empty_string(input_value.get("question"), "question"),
        reference_answer=non_empty_string(
            expected_output.get("reference_answer"),
            "reference answer",
        ),
        expected_source_keys=tuple(SourceKey(value) for value in expected_sources),
        answerable=answerable,
        requested_agent=requested_agent,
        tags=tuple(tags),
    )


def case_result_payload(result: CaseResult) -> dict[str, object]:
    return {
        "case_id": result.case_id,
        "answer": result.answer.answer,
        "insufficient_context": result.answer.insufficient_context,
        "latency_ms": result.answer.latency_ms,
        "metrics": [
            {
                "name": metric.metric.value,
                "score": metric.score.value,
                "raw_result_json": metric.raw_result_json,
            }
            for metric in result.metrics
        ],
        "retrieval_sources": [
            {
                "rank": item.rank,
                "source_key": item.source_key.value,
                "score": item.score,
                "excerpt": item.excerpt,
                "document_id": str(item.document_id),
                "relative_raw_path": item.relative_raw_path,
            }
            for item in result.retrieval.items
        ],
        "answer_sources": [
            {
                "id": context.id,
                "source_key": context.source_key.value,
                "score": context.score,
                "excerpt": context.excerpt,
                "retrieval_query": context.retrieval_query,
                "document_id": str(context.document_id),
                "relative_raw_path": context.relative_raw_path,
            }
            for context in result.answer.contexts
        ],
    }


def metric_evaluations(output: object) -> list[Evaluation]:
    payload = object_value(output, "task output")
    raw_metrics = payload.get("metrics")
    if not isinstance(raw_metrics, list) or not raw_metrics:
        raise EvaluationExecutionError("Task output has no metrics")
    evaluations: list[Evaluation] = []
    for raw_metric in raw_metrics:
        metric = object_value(raw_metric, "metric")
        evaluations.append(
            Evaluation(
                name=non_empty_string(metric.get("name"), "metric name"),
                value=float(metric["score"]),
                metadata={"raw_result_json": metric.get("raw_result_json")},
            )
        )
    return evaluations


def summary_evaluations(summary: EvaluationRunSummary) -> list[Evaluation]:
    violations = [
        {
            "metric": violation.metric.value,
            "actual": violation.actual.value if violation.actual else None,
            "required": violation.required.value,
            "case_id": violation.case_id,
        }
        for violation in summary.decision.violations
    ]
    aggregate = [
        Evaluation(name=metric.metric.value, value=metric.score.value)
        for metric in summary.aggregate_metrics
    ]
    return [
        *aggregate,
        Evaluation(
            name="gate_pass",
            value=summary.decision.passed,
            data_type="BOOLEAN",
            metadata={"violations": violations},
        ),
        Evaluation(name="gate_violation_count", value=len(violations)),
    ]


def case_id_from_output(output: object) -> str:
    return non_empty_string(object_value(output, "task output").get("case_id"), "case id")


def object_value(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvaluationDatasetError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def non_empty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationDatasetError(f"{label} must be a non-empty string")
    return value


def string_list(value: object) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise EvaluationDatasetError("Expected an array of strings")
    return cast(list[str], value)
