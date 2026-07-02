from __future__ import annotations

from typing import cast

from fastapi import APIRouter, HTTPException, Request, status

from backend.llm_provider_stats import (
    GetLLMProviderStatsUseCase,
    LLMProviderStats,
    Money,
    OpenRouterProviderStats,
    ProviderActivityTotals,
)
from dashboard.api_models import LLMProviderStatsResponse
from dashboard.dependencies import current_user


router = APIRouter(prefix="/api/admin")


@router.get("/llm-providers", response_model=LLMProviderStatsResponse)
def llm_provider_stats(request: Request) -> dict[str, object]:
    user = current_user(request)
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    use_case = cast(
        GetLLMProviderStatsUseCase,
        request.app.state.llm_provider_stats_use_case,
    )
    return _stats(use_case.execute())


def _stats(stats: LLMProviderStats) -> dict[str, object]:
    return {
        "ok": True,
        "generated_at": stats.generated_at,
        "configuration": {
            "chat_provider": stats.configuration.chat_provider,
            "chat_model": stats.configuration.chat_model,
            "embedding_provider": stats.configuration.embedding_provider,
            "embedding_model": stats.configuration.embedding_model,
            "agent_models": [
                {"agent_name": agent_name, "model_id": model_id}
                for agent_name, model_id in sorted(
                    stats.configuration.agent_models.items()
                )
            ],
            "selectable_models": [
                {
                    "key": model.key,
                    "label": model.label,
                    "model_id": model.model_id,
                }
                for model in stats.configuration.selectable_models
            ],
        },
        "providers": [_provider(provider) for provider in stats.providers],
    }


def _provider(provider: OpenRouterProviderStats) -> dict[str, object]:
    return {
        "provider_key": provider.provider_key,
        "provider_name": provider.provider_name,
        "status": provider.status,
        "message": provider.message,
        "issues": [
            {"section": issue.section, "message": issue.message}
            for issue in provider.issues
        ],
        "credits": (
            {
                "total_credits": _money(provider.credits.total_credits),
                "total_usage": _money(provider.credits.total_usage),
                "remaining_credits": _money(provider.credits.remaining_credits),
            }
            if provider.credits is not None
            else None
        ),
        "api_key": (
            {
                "label": provider.api_key.label,
                "usage": _money(provider.api_key.usage),
                "usage_daily": _money(provider.api_key.usage_daily),
                "usage_weekly": _money(provider.api_key.usage_weekly),
                "usage_monthly": _money(provider.api_key.usage_monthly),
                "byok_usage": _money(provider.api_key.byok_usage),
                "byok_usage_daily": _money(provider.api_key.byok_usage_daily),
                "byok_usage_weekly": _money(provider.api_key.byok_usage_weekly),
                "byok_usage_monthly": _money(provider.api_key.byok_usage_monthly),
                "include_byok_in_limit": provider.api_key.include_byok_in_limit,
                "is_free_tier": provider.api_key.is_free_tier,
                "is_management_key": provider.api_key.is_management_key,
                "is_provisioning_key": provider.api_key.is_provisioning_key,
                "limit": _optional_money(provider.api_key.limit),
                "limit_remaining": _optional_money(
                    provider.api_key.limit_remaining,
                ),
                "limit_reset": provider.api_key.limit_reset,
                "expires_at": provider.api_key.expires_at,
            }
            if provider.api_key is not None
            else None
        ),
        "activity": (
            {
                "window_label": provider.activity.window_label,
                "completed_utc_days": provider.activity.completed_utc_days,
                "totals": _totals(provider.activity.totals),
                "top_models": [
                    {
                        "model": model.model,
                        "provider_name": model.provider_name,
                        "totals": _totals(model.totals),
                        "last_activity_date": model.last_activity_date,
                    }
                    for model in provider.activity.top_models
                ],
                "top_providers": [
                    {
                        "provider_name": item.provider_name,
                        "totals": _totals(item.totals),
                        "last_activity_date": item.last_activity_date,
                    }
                    for item in provider.activity.top_providers
                ],
            }
            if provider.activity is not None
            else None
        ),
    }


def _totals(totals: ProviderActivityTotals) -> dict[str, object]:
    return {
        "usage": _money(totals.usage),
        "byok_usage": _money(totals.byok_usage),
        "requests": totals.requests,
        "prompt_tokens": totals.prompt_tokens,
        "completion_tokens": totals.completion_tokens,
        "reasoning_tokens": totals.reasoning_tokens,
    }


def _optional_money(money: Money | None) -> dict[str, str] | None:
    if money is None:
        return None
    return _money(money)


def _money(money: Money) -> dict[str, str]:
    return {"amount": format(money.amount, "f"), "currency": money.currency}
