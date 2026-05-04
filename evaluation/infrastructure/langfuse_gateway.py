from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from langfuse import Evaluation, Langfuse
from langfuse.api import DatasetItem
from langfuse.experiment import ExperimentItemResult

from evaluation.application.errors import (
    EvaluationConfigurationError,
    EvaluationDatasetError,
    EvaluationExecutionError,
    EvaluationPublishingError,
)
from evaluation.application.models import (
    ExperimentExecution,
    EvaluationRunSummary,
    PublishedScore,
    ScorePublication,
)
from evaluation.application.ports import CaseTask, RunSummarizer
from evaluation.domain.report import CaseResult
from evaluation.domain.suite import EvaluationProfile
from evaluation.infrastructure.langfuse_codec import (
    case_from_item,
    case_id_from_output,
    case_result_payload,
    metric_evaluations,
    summary_evaluations,
)


class LangfuseExperimentGateway:
    def __init__(
        self,
        client: Langfuse,
        *,
        confirmation_timeout_seconds: int,
    ) -> None:
        self._client = client
        self._confirmation_timeout_seconds = confirmation_timeout_seconds

    def authenticate(self) -> None:
        try:
            authenticated = self._client.auth_check()
        except Exception as error:
            raise EvaluationConfigurationError("Langfuse authentication failed") from error
        if not authenticated:
            raise EvaluationConfigurationError("Langfuse credentials are invalid")

    def execute(
        self,
        *,
        profile: EvaluationProfile,
        dataset_version: datetime,
        run_name: str,
        metadata: dict[str, str],
        task: CaseTask,
        summarize: RunSummarizer,
    ) -> ExperimentExecution:
        dataset = self._load_dataset(profile.dataset_name, dataset_version)
        if not dataset.items:
            raise EvaluationDatasetError("Evaluation dataset is empty")
        case_results: dict[str, CaseResult] = {}
        summaries: list[EvaluationRunSummary] = []
        expected_count = len(dataset.items)
        task_function = self._task(task, case_results)
        run_evaluator = self._run_evaluator(
            expected_count=expected_count,
            case_results=case_results,
            summarize=summarize,
            summaries=summaries,
        )
        try:
            result = self._client.run_experiment(
                name=profile.id,
                run_name=run_name,
                description=f"Medic synthetic evaluation {profile.version}",
                data=dataset.items,
                task=task_function,
                evaluators=[self._evaluate_item],
                run_evaluators=[run_evaluator],
                max_concurrency=1,
                metadata=metadata,
                _dataset_version=dataset_version,
            )
        except EvaluationDatasetError:
            raise
        except Exception as error:
            raise EvaluationExecutionError("Langfuse experiment failed") from error
        self._validate_execution(result.item_results, expected_count, summaries)
        self._validate_published_metrics(result.item_results, case_results)
        if result.dataset_run_id is None:
            raise EvaluationPublishingError("Langfuse did not create a dataset run")
        return ExperimentExecution(
            run_id=result.experiment_id,
            run_name=result.run_name,
            dataset_version=dataset_version,
            dataset_name=profile.dataset_name,
            dataset_run_id=result.dataset_run_id,
            dataset_run_url=result.dataset_run_url,
            expected_item_count=expected_count,
            cases=tuple(
                case_results[case_id_from_output(item_result.output)]
                for item_result in result.item_results
            ),
            summary=summaries[0],
            score_publications=_score_publications(
                result.item_results,
                result.run_evaluations,
                dataset_run_id=result.dataset_run_id,
            ),
        )

    def flush_and_verify(self, execution: ExperimentExecution) -> None:
        try:
            self._client.flush()
        except Exception as error:
            raise EvaluationPublishingError("Failed to flush Langfuse events") from error
        deadline = time.monotonic() + self._confirmation_timeout_seconds
        while time.monotonic() < deadline:
            if self._run_is_complete(execution):
                return
            time.sleep(0.5)
        raise EvaluationPublishingError("Langfuse publication confirmation timed out")

    def _load_dataset(self, name: str, version: datetime) -> Any:
        try:
            return self._client.get_dataset(name, version=version)
        except Exception as error:
            raise EvaluationDatasetError(f"Cannot load Langfuse dataset: {name}") from error

    @staticmethod
    def _task(
        task: CaseTask,
        case_results: dict[str, CaseResult],
    ) -> Callable[..., Awaitable[dict[str, object]]]:
        async def execute_item(
            *,
            item: DatasetItem,
            **_: object,
        ) -> dict[str, object]:
            case = case_from_item(item)
            result = await asyncio.to_thread(task, case)
            if case.id in case_results:
                raise EvaluationDatasetError(f"Duplicate evaluation case: {case.id}")
            case_results[case.id] = result
            return case_result_payload(result)

        return execute_item

    @staticmethod
    def _evaluate_item(*, output: object, **_: object) -> list[Evaluation]:
        return metric_evaluations(output)

    @staticmethod
    def _run_evaluator(
        *,
        expected_count: int,
        case_results: dict[str, CaseResult],
        summarize: RunSummarizer,
        summaries: list[EvaluationRunSummary],
    ) -> Callable[..., list[Evaluation]]:
        def evaluate_run(
            *,
            item_results: list[ExperimentItemResult],
            **_: object,
        ) -> list[Evaluation]:
            if len(item_results) != expected_count:
                return [
                    Evaluation(
                        name="execution_complete",
                        value=False,
                        data_type="BOOLEAN",
                        comment="One or more evaluation items failed",
                    )
                ]
            ordered = tuple(
                case_results[case_id_from_output(item_result.output)]
                for item_result in item_results
            )
            summary = summarize(ordered)
            summaries.append(summary)
            return summary_evaluations(summary)

        return evaluate_run

    @staticmethod
    def _validate_execution(
        item_results: list[ExperimentItemResult],
        expected_count: int,
        summaries: list[EvaluationRunSummary],
    ) -> None:
        if len(item_results) != expected_count or not summaries:
            raise EvaluationExecutionError("Langfuse experiment did not complete all items")

    @staticmethod
    def _validate_published_metrics(
        item_results: list[ExperimentItemResult],
        case_results: dict[str, CaseResult],
    ) -> None:
        for item_result in item_results:
            case_id = case_id_from_output(item_result.output)
            expected = {metric.metric.value for metric in case_results[case_id].metrics}
            published = {evaluation.name for evaluation in item_result.evaluations}
            if published != expected:
                raise EvaluationExecutionError(
                    f"Langfuse scores are incomplete for case {case_id}"
                )

    def _run_is_complete(self, execution: ExperimentExecution) -> bool:
        try:
            run = self._client.api.datasets.get_run(
                dataset_name=execution.dataset_name,
                run_name=execution.run_name,
            )
            scores_complete = all(
                self._publication_is_complete(publication)
                for publication in execution.score_publications
            )
        except Exception:
            return False
        return (
            run.id == execution.dataset_run_id
            and len(run.dataset_run_items) == execution.expected_item_count
            and scores_complete
        )

    def _publication_is_complete(self, publication: ScorePublication) -> bool:
        actual_scores = self._scores(publication)
        return all(
            any(_score_matches(expected, actual) for actual in actual_scores)
            for expected in publication.scores
        )

    def _scores(self, publication: ScorePublication) -> list[Any]:
        page = 1
        scores: list[Any] = []
        while True:
            response = self._client.api.scores.get_many(
                page=page,
                limit=100,
                trace_id=publication.trace_id,
                dataset_run_id=publication.dataset_run_id,
            )
            scores.extend(response.data)
            if page >= response.meta.total_pages:
                return scores
            page += 1


def _score_publications(
    item_results: list[ExperimentItemResult],
    run_evaluations: list[Evaluation],
    *,
    dataset_run_id: str,
) -> tuple[ScorePublication, ...]:
    item_publications = tuple(
        ScorePublication(
            trace_id=item_result.trace_id,
            dataset_run_id=None,
            scores=_published_scores(item_result.evaluations),
        )
        for item_result in item_results
    )
    return (
        *item_publications,
        ScorePublication(
            trace_id=None,
            dataset_run_id=dataset_run_id,
            scores=_published_scores(run_evaluations),
        ),
    )


def _published_scores(evaluations: list[Evaluation]) -> tuple[PublishedScore, ...]:
    return tuple(
        PublishedScore(
            name=evaluation.name,
            value=float(evaluation.value),
            data_type=evaluation.data_type or "NUMERIC",
        )
        for evaluation in evaluations
    )


def _score_matches(expected: PublishedScore, actual: Any) -> bool:
    return (
        actual.name == expected.name
        and actual.data_type == expected.data_type
        and math.isclose(float(actual.value), expected.value, abs_tol=1e-9)
    )
