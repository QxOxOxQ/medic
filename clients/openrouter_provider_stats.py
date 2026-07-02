from __future__ import annotations

from collections.abc import Callable

from backend.llm_provider_stats import (
    Money,
    OpenRouterActivityItem,
    OpenRouterCredits,
    OpenRouterKeyStats,
    OpenRouterProviderStatsPort,
    ProviderStatsGatewayError,
)
from clients.openrouter import (
    OpenRouterActivityResponseItem,
    OpenRouterApiError,
    OpenRouterClient,
    OpenRouterCreditsResponse,
    OpenRouterKeyResponse,
)


class OpenRouterProviderStatsGateway(OpenRouterProviderStatsPort):
    def __init__(
        self,
        *,
        client_factory: Callable[[], OpenRouterClient] | None = None,
    ) -> None:
        self._client_factory = client_factory or OpenRouterClient

    def credits(self) -> OpenRouterCredits:
        response = self._call(lambda client: client.get_credits())
        return _credits(response)

    def current_key(self) -> OpenRouterKeyStats:
        response = self._call(lambda client: client.get_current_key())
        return _key_stats(response)

    def activity(self) -> tuple[OpenRouterActivityItem, ...]:
        response = self._call(lambda client: client.get_activity())
        return tuple(_activity_item(item) for item in response)

    def _call[T](self, action: Callable[[OpenRouterClient], T]) -> T:
        try:
            return action(self._client_factory())
        except ValueError as error:
            raise ProviderStatsGatewayError(str(error)) from error
        except OpenRouterApiError as error:
            raise ProviderStatsGatewayError(_public_error(error)) from error


def _credits(response: OpenRouterCreditsResponse) -> OpenRouterCredits:
    return OpenRouterCredits(
        total_credits=Money(response.total_credits),
        total_usage=Money(response.total_usage),
    )


def _key_stats(response: OpenRouterKeyResponse) -> OpenRouterKeyStats:
    return OpenRouterKeyStats(
        label=response.label,
        usage=Money(response.usage),
        usage_daily=Money(response.usage_daily),
        usage_weekly=Money(response.usage_weekly),
        usage_monthly=Money(response.usage_monthly),
        byok_usage=Money(response.byok_usage),
        byok_usage_daily=Money(response.byok_usage_daily),
        byok_usage_weekly=Money(response.byok_usage_weekly),
        byok_usage_monthly=Money(response.byok_usage_monthly),
        include_byok_in_limit=response.include_byok_in_limit,
        is_free_tier=response.is_free_tier,
        is_management_key=response.is_management_key,
        is_provisioning_key=response.is_provisioning_key,
        limit=Money(response.limit) if response.limit is not None else None,
        limit_remaining=(
            Money(response.limit_remaining)
            if response.limit_remaining is not None
            else None
        ),
        limit_reset=response.limit_reset,
        expires_at=response.expires_at,
    )


def _activity_item(response: OpenRouterActivityResponseItem) -> OpenRouterActivityItem:
    return OpenRouterActivityItem(
        date=response.date,
        model=response.model,
        model_permaslug=response.model_permaslug,
        endpoint_id=response.endpoint_id,
        provider_name=response.provider_name,
        usage=Money(response.usage),
        byok_usage=Money(response.byok_usage_inference),
        requests=response.requests,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        reasoning_tokens=response.reasoning_tokens,
    )


def _public_error(error: OpenRouterApiError) -> str:
    if error.status_code == 401:
        return "OpenRouter authentication failed."
    if error.status_code == 403:
        return "OpenRouter management-key permissions are required for this statistic."
    if error.status_code is None:
        return error.message
    return f"OpenRouter returned HTTP {error.status_code}."
