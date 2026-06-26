from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from evaluation.application.errors import EvaluationApplicationError
from evaluation.application.bootstrap import BootstrapEvaluationDataset
from evaluation.application.models import EvaluationOutcome
from evaluation.domain.errors import EvaluationDomainError
from evaluation.factory import EvaluationServices


logger = logging.getLogger(__name__)


def run_evaluation_cli(
    services: EvaluationServices,
    *,
    profile_id: str,
    dataset_version: str | None,
) -> int:
    try:
        outcome = services.run.execute(
            profile_id=profile_id,
            dataset_version=_dataset_version(dataset_version),
        )
    except (EvaluationApplicationError, EvaluationDomainError, ValueError) as error:
        _print_error(error)
        return 2
    except Exception:
        logger.exception("Unexpected evaluation failure")
        _print_error(RuntimeError("Unexpected evaluation failure"))
        return 2
    print(json.dumps(_outcome_payload(outcome), indent=2, ensure_ascii=False))
    return 0 if outcome.execution.summary.decision.passed else 1


def run_calibration_cli(services: EvaluationServices) -> int:
    try:
        result = services.calibration.execute()
    except (EvaluationApplicationError, EvaluationDomainError, ValueError) as error:
        _print_error(error)
        return 2
    except Exception:
        logger.exception("Unexpected evaluation calibration failure")
        _print_error(RuntimeError("Unexpected evaluation calibration failure"))
        return 2
    print(
        json.dumps(
            {
                "passed": result.passed,
                "good_score": result.good_score,
                "bad_score": result.bad_score,
            },
            indent=2,
        )
    )
    return 0 if result.passed else 1


def run_bootstrap_cli(
    service: BootstrapEvaluationDataset,
    *,
    profile_id: str,
) -> int:
    try:
        result = service.execute(profile_id)
    except (EvaluationApplicationError, EvaluationDomainError, ValueError) as error:
        _print_error(error)
        return 2
    except Exception:
        logger.exception("Unexpected evaluation bootstrap failure")
        _print_error(RuntimeError("Unexpected evaluation bootstrap failure"))
        return 2
    print(
        json.dumps(
            {
                "dataset_name": result.dataset_name,
                "created_items": result.created_items,
                "verified_items": result.verified_items,
            },
            indent=2,
        )
    )
    return 0


def _dataset_version(value: str | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Dataset version must include a timezone")
    return parsed.astimezone(UTC)


def _outcome_payload(outcome: EvaluationOutcome) -> dict[str, object]:
    decision = outcome.execution.summary.decision
    return {
        "status": "passed" if decision.passed else "failed_quality",
        "run_id": outcome.execution.run_id,
        "run_name": outcome.execution.run_name,
        "dataset_name": outcome.execution.dataset_name,
        "dataset_version": outcome.execution.dataset_version.isoformat(),
        "dataset_run_id": outcome.execution.dataset_run_id,
        "dataset_run_url": outcome.execution.dataset_run_url,
        "fingerprints": {
            "profile": outcome.fingerprints.profile,
            "corpus": outcome.fingerprints.corpus,
            "system": outcome.fingerprints.system,
            "judge": outcome.fingerprints.judge,
        },
        "aggregate_metrics": {
            metric.metric.value: metric.score.value
            for metric in outcome.report.aggregate_metrics
        },
        "violations": [
            {
                "metric": violation.metric.value,
                "actual": violation.actual.value if violation.actual else None,
                "required": violation.required.value,
                "case_id": violation.case_id,
            }
            for violation in decision.violations
        ],
    }


def _print_error(error: Exception) -> None:
    print(
        json.dumps(
            {"status": "failed_error", "error": str(error)},
            ensure_ascii=False,
        )
    )
