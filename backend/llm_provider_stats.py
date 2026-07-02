from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Protocol


LOGGER = logging.getLogger(__name__)
USD = "USD"
OPENROUTER_PROVIDER_KEY = "openrouter"


@dataclass(frozen=True)
class Money:
    amount: Decimal
    currency: str = USD

    @classmethod
    def zero(cls) -> Money:
        return cls(amount=Decimal("0"))

    def __add__(self, other: Money) -> Money:
        if self.currency != other.currency:
            raise ValueError("Cannot add money with different currencies")
        return Money(amount=self.amount + other.amount, currency=self.currency)

    def __sub__(self, other: Money) -> Money:
        if self.currency != other.currency:
            raise ValueError("Cannot subtract money with different currencies")
        return Money(amount=self.amount - other.amount, currency=self.currency)


@dataclass(frozen=True)
class ProviderIssue:
    section: str
    message: str


@dataclass(frozen=True)
class OpenRouterCredits:
    total_credits: Money
    total_usage: Money

    @property
    def remaining_credits(self) -> Money:
        return self.total_credits - self.total_usage


@dataclass(frozen=True)
class OpenRouterKeyStats:
    label: str
    usage: Money
    usage_daily: Money
    usage_weekly: Money
    usage_monthly: Money
    byok_usage: Money
    byok_usage_daily: Money
    byok_usage_weekly: Money
    byok_usage_monthly: Money
    include_byok_in_limit: bool
    is_free_tier: bool
    is_management_key: bool
    is_provisioning_key: bool
    limit: Money | None
    limit_remaining: Money | None
    limit_reset: str | None
    expires_at: datetime | None


@dataclass(frozen=True)
class OpenRouterActivityItem:
    date: str
    model: str
    model_permaslug: str
    endpoint_id: str
    provider_name: str
    usage: Money
    byok_usage: Money
    requests: int
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int


@dataclass(frozen=True)
class ProviderActivityTotals:
    usage: Money
    byok_usage: Money
    requests: int
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int

    @classmethod
    def empty(cls) -> ProviderActivityTotals:
        return cls(
            usage=Money.zero(),
            byok_usage=Money.zero(),
            requests=0,
            prompt_tokens=0,
            completion_tokens=0,
            reasoning_tokens=0,
        )

    def add(self, item: OpenRouterActivityItem) -> ProviderActivityTotals:
        return ProviderActivityTotals(
            usage=self.usage + item.usage,
            byok_usage=self.byok_usage + item.byok_usage,
            requests=self.requests + item.requests,
            prompt_tokens=self.prompt_tokens + item.prompt_tokens,
            completion_tokens=self.completion_tokens + item.completion_tokens,
            reasoning_tokens=self.reasoning_tokens + item.reasoning_tokens,
        )


@dataclass(frozen=True)
class ModelActivitySummary:
    model: str
    provider_name: str
    totals: ProviderActivityTotals
    last_activity_date: str | None


@dataclass(frozen=True)
class ProviderActivitySummary:
    provider_name: str
    totals: ProviderActivityTotals
    last_activity_date: str | None


@dataclass(frozen=True)
class OpenRouterActivitySummary:
    window_label: str
    completed_utc_days: int
    totals: ProviderActivityTotals
    top_models: tuple[ModelActivitySummary, ...]
    top_providers: tuple[ProviderActivitySummary, ...]


@dataclass(frozen=True)
class ConfiguredModel:
    key: str
    label: str
    model_id: str


@dataclass(frozen=True)
class LLMProviderConfiguration:
    chat_provider: str
    chat_model: str
    embedding_provider: str
    embedding_model: str
    agent_models: Mapping[str, str]
    selectable_models: tuple[ConfiguredModel, ...]


@dataclass(frozen=True)
class OpenRouterProviderStats:
    provider_key: str
    provider_name: str
    status: str
    message: str | None
    issues: tuple[ProviderIssue, ...]
    credits: OpenRouterCredits | None
    api_key: OpenRouterKeyStats | None
    activity: OpenRouterActivitySummary | None


@dataclass(frozen=True)
class LLMProviderStats:
    generated_at: datetime
    configuration: LLMProviderConfiguration
    providers: tuple[OpenRouterProviderStats, ...]


class ProviderStatsGatewayError(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.public_message = message


class OpenRouterProviderStatsPort(Protocol):
    def credits(self) -> OpenRouterCredits: ...

    def current_key(self) -> OpenRouterKeyStats: ...

    def activity(self) -> tuple[OpenRouterActivityItem, ...]: ...


class Clock(Protocol):
    def __call__(self) -> datetime: ...


class GetLLMProviderStatsUseCase:
    def __init__(
        self,
        *,
        openrouter: OpenRouterProviderStatsPort,
        configuration: LLMProviderConfiguration,
        clock: Clock | None = None,
    ) -> None:
        self._openrouter = openrouter
        self._configuration = configuration
        self._clock = clock or utc_now

    def execute(self) -> LLMProviderStats:
        credits, credits_issue = _capture("balance", self._openrouter.credits)
        key_stats, key_issue = _capture("api_key", self._openrouter.current_key)
        activity_items, activity_issue = _capture("activity", self._openrouter.activity)
        issues = tuple(
            issue
            for issue in (credits_issue, key_issue, activity_issue)
            if issue is not None
        )
        activity = (
            _summarize_activity(activity_items)
            if activity_items is not None
            else None
        )

        return LLMProviderStats(
            generated_at=self._clock(),
            configuration=self._configuration,
            providers=(
                OpenRouterProviderStats(
                    provider_key=OPENROUTER_PROVIDER_KEY,
                    provider_name="OpenRouter",
                    status=_provider_status(
                        successful_sections=(credits, key_stats, activity),
                        issues=issues,
                    ),
                    message=_provider_message(issues),
                    issues=issues,
                    credits=credits,
                    api_key=key_stats,
                    activity=activity,
                ),
            ),
        )


def utc_now() -> datetime:
    return datetime.now(UTC)


def _capture[T](
    section: str,
    action: CallableWithoutArgs[T],
) -> tuple[T | None, ProviderIssue | None]:
    try:
        return action(), None
    except ProviderStatsGatewayError as error:
        return None, ProviderIssue(section=section, message=error.public_message)
    except Exception:
        LOGGER.exception("Unexpected provider statistics failure for %s", section)
        return None, ProviderIssue(
            section=section,
            message="Provider statistics are temporarily unavailable.",
        )


class CallableWithoutArgs[T](Protocol):
    def __call__(self) -> T: ...


def _provider_status(
    *,
    successful_sections: Iterable[object | None],
    issues: tuple[ProviderIssue, ...],
) -> str:
    successes = sum(section is not None for section in successful_sections)
    if successes == 0:
        return "unavailable"
    if issues:
        return "degraded"
    return "available"


def _provider_message(issues: tuple[ProviderIssue, ...]) -> str | None:
    if not issues:
        return None
    if len(issues) == 1:
        return issues[0].message
    return "Some provider statistics could not be refreshed."


def _summarize_activity(
    items: tuple[OpenRouterActivityItem, ...],
) -> OpenRouterActivitySummary:
    return OpenRouterActivitySummary(
        window_label="Last 30 completed UTC days",
        completed_utc_days=30,
        totals=_activity_totals(items),
        top_models=_top_models(items),
        top_providers=_top_providers(items),
    )


def _activity_totals(items: Iterable[OpenRouterActivityItem]) -> ProviderActivityTotals:
    totals = ProviderActivityTotals.empty()
    for item in items:
        totals = totals.add(item)
    return totals


def _top_models(
    items: tuple[OpenRouterActivityItem, ...],
) -> tuple[ModelActivitySummary, ...]:
    grouped: dict[str, tuple[str, ProviderActivityTotals, str | None]] = {}
    for item in items:
        provider_name, totals, last_activity_date = grouped.get(
            item.model,
            (item.provider_name, ProviderActivityTotals.empty(), None),
        )
        grouped[item.model] = (
            provider_name,
            totals.add(item),
            _latest_date(last_activity_date, item.date),
        )

    summaries = (
        ModelActivitySummary(
            model=model,
            provider_name=provider_name,
            totals=totals,
            last_activity_date=last_activity_date,
        )
        for model, (provider_name, totals, last_activity_date) in grouped.items()
    )
    return tuple(sorted(summaries, key=_activity_sort_key)[:10])


def _top_providers(
    items: tuple[OpenRouterActivityItem, ...],
) -> tuple[ProviderActivitySummary, ...]:
    grouped: dict[str, tuple[ProviderActivityTotals, str | None]] = {}
    for item in items:
        totals, last_activity_date = grouped.get(
            item.provider_name,
            (ProviderActivityTotals.empty(), None),
        )
        grouped[item.provider_name] = (
            totals.add(item),
            _latest_date(last_activity_date, item.date),
        )

    summaries = (
        ProviderActivitySummary(
            provider_name=provider_name,
            totals=totals,
            last_activity_date=last_activity_date,
        )
        for provider_name, (totals, last_activity_date) in grouped.items()
    )
    return tuple(sorted(summaries, key=_activity_sort_key)[:10])


def _activity_sort_key(
    summary: ModelActivitySummary | ProviderActivitySummary,
) -> tuple[Decimal, int]:
    return (-summary.totals.usage.amount, -summary.totals.requests)


def _latest_date(current: str | None, candidate: str) -> str:
    if current is None:
        return candidate
    return max(current, candidate)
