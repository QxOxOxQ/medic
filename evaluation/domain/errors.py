class EvaluationDomainError(ValueError):
    """Base error for invalid evaluation domain state."""


class InvalidSuiteError(EvaluationDomainError):
    """Raised when an evaluation suite violates its invariants."""


class InvalidScoreError(EvaluationDomainError):
    """Raised when a metric score is not finite or outside [0, 1]."""
