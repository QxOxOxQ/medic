class EvaluationApplicationError(RuntimeError):
    """Base error for evaluation use cases."""


class CorpusIsolationError(EvaluationApplicationError):
    """Raised when evaluation could affect a production collection."""


class CorpusProvisioningError(EvaluationApplicationError):
    """Raised when the evaluation corpus cannot be prepared."""


class MetricEvaluationError(EvaluationApplicationError):
    """Raised when a metric cannot produce a valid score."""


class EvaluationExecutionError(EvaluationApplicationError):
    """Raised when the system under test cannot execute a case."""


class JudgeNotCalibratedError(EvaluationApplicationError):
    """Raised when quality gates are requested for an uncalibrated judge."""


class EvaluationConfigurationError(EvaluationApplicationError):
    """Raised when evaluation dependencies are not configured."""


class EvaluationDatasetError(EvaluationApplicationError):
    """Raised when a hosted evaluation dataset is invalid or unavailable."""


class EvaluationPublishingError(EvaluationApplicationError):
    """Raised when an experiment cannot be persisted or verified."""
