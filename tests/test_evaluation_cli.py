from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast

import pytest

from evaluation.application.errors import EvaluationConfigurationError
from evaluation.application.models import (
    EvaluationFingerprints,
    EvaluationOutcome,
    EvaluationRunSummary,
    ExperimentExecution,
)
from evaluation.domain.quality import GateDecision, GateViolation
from evaluation.domain.report import EvaluationReport
from evaluation.domain.values import MetricName, Score
from evaluation.factory import EvaluationServices
from evaluation.presentation.cli import run_evaluation_cli


@pytest.mark.parametrize(("passed", "expected_code"), [(True, 0), (False, 1)])
def test_evaluation_cli_returns_quality_status_codes(
    passed: bool,
    expected_code: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    services = cast(
        EvaluationServices,
        SimpleNamespace(run=FakeRun(outcome=_outcome(passed=passed))),
    )

    code = run_evaluation_cli(
        services,
        profile_id="profile",
        dataset_version="2026-06-18T00:00:00Z",
    )

    assert code == expected_code
    expected_status = "passed" if passed else "failed_quality"
    assert f'"status": "{expected_status}"' in capsys.readouterr().out


def test_evaluation_cli_returns_two_for_execution_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    services = cast(
        EvaluationServices,
        SimpleNamespace(run=FakeRun(error=EvaluationConfigurationError("missing"))),
    )

    code = run_evaluation_cli(
        services,
        profile_id="profile",
        dataset_version=None,
    )

    assert code == 2
    assert '"status": "failed_error"' in capsys.readouterr().out


class FakeRun:
    def __init__(
        self,
        *,
        outcome: EvaluationOutcome | None = None,
        error: Exception | None = None,
    ) -> None:
        self._outcome = outcome
        self._error = error

    def execute(self, **_: object) -> EvaluationOutcome:
        if self._error is not None:
            raise self._error
        assert self._outcome is not None
        return self._outcome


def _outcome(*, passed: bool) -> EvaluationOutcome:
    now = datetime.now(UTC)
    violations = ()
    if not passed:
        violations = (
            GateViolation(
                metric=MetricName.FAITHFULNESS,
                actual=Score(0.5),
                required=Score(0.9),
            ),
        )
    decision = GateDecision(passed=passed, violations=violations)
    summary = EvaluationRunSummary(aggregate_metrics=(), decision=decision)
    execution = ExperimentExecution(
        run_id="run",
        run_name="run-name",
        dataset_version=now,
        dataset_name="medic/test",
        dataset_run_id="dataset-run",
        dataset_run_url=None,
        expected_item_count=0,
        cases=(),
        summary=summary,
        score_publications=(),
    )
    return EvaluationOutcome(
        execution=execution,
        report=EvaluationReport(
            run_id="run",
            profile_id="profile",
            profile_version="1",
            started_at=now,
            finished_at=now,
            cases=(),
            aggregate_metrics=(),
        ),
        fingerprints=EvaluationFingerprints("p", "c", "s", "j"),
    )
