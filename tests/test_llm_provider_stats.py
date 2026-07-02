from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from backend.llm_provider_stats import (
    GetLLMProviderStatsUseCase,
    LLMProviderConfiguration,
    Money,
    OpenRouterActivityItem,
    OpenRouterCredits,
    OpenRouterKeyStats,
    ProviderStatsGatewayError,
)
from dashboard.app import create_app
from dashboard.auth import AuthSettings
from rag.database.migrations import upgrade_database
from rag.database.repositories import UserRepository
from rag.database.session import create_database_engine


class FakeOpenRouterStats:
    def __init__(self, *, fail_credits: bool = False) -> None:
        self._fail_credits = fail_credits

    def credits(self) -> OpenRouterCredits:
        if self._fail_credits:
            raise ProviderStatsGatewayError("OpenRouter credits unavailable.")
        return OpenRouterCredits(
            total_credits=Money(Decimal("100.50")),
            total_usage=Money(Decimal("25.75")),
        )

    def current_key(self) -> OpenRouterKeyStats:
        return OpenRouterKeyStats(
            label="sk-or-v1-test",
            usage=Money(Decimal("25.50")),
            usage_daily=Money(Decimal("1.50")),
            usage_weekly=Money(Decimal("7.50")),
            usage_monthly=Money(Decimal("12.50")),
            byok_usage=Money(Decimal("0")),
            byok_usage_daily=Money(Decimal("0")),
            byok_usage_weekly=Money(Decimal("0")),
            byok_usage_monthly=Money(Decimal("0")),
            include_byok_in_limit=False,
            is_free_tier=False,
            is_management_key=False,
            is_provisioning_key=False,
            limit=Money(Decimal("100")),
            limit_remaining=Money(Decimal("74.50")),
            limit_reset="monthly",
            expires_at=datetime(2027, 12, 31, 23, 59, tzinfo=UTC),
        )

    def activity(self) -> tuple[OpenRouterActivityItem, ...]:
        return (
            OpenRouterActivityItem(
                date="2026-07-01",
                model="openai/gpt-4.1-mini",
                model_permaslug="openai/gpt-4.1-mini",
                endpoint_id="endpoint-1",
                provider_name="OpenAI",
                usage=Money(Decimal("0.015")),
                byok_usage=Money(Decimal("0")),
                requests=5,
                prompt_tokens=50,
                completion_tokens=125,
                reasoning_tokens=0,
            ),
        )


class UnexpectedCreditsFailureStats(FakeOpenRouterStats):
    def credits(self) -> OpenRouterCredits:
        raise RuntimeError("Credits parser regression")


def test_llm_provider_stats_use_case_degrades_partial_provider_failures() -> None:
    use_case = GetLLMProviderStatsUseCase(
        openrouter=FakeOpenRouterStats(fail_credits=True),
        configuration=_configuration(),
        clock=lambda: datetime(2026, 7, 2, tzinfo=UTC),
    )

    stats = use_case.execute()
    provider = stats.providers[0]

    assert provider.status == "degraded"
    assert provider.credits is None
    assert provider.api_key is not None
    assert provider.activity is not None
    assert provider.issues[0].section == "balance"


def test_llm_provider_stats_use_case_logs_unexpected_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR, logger="backend.llm_provider_stats")
    use_case = GetLLMProviderStatsUseCase(
        openrouter=UnexpectedCreditsFailureStats(),
        configuration=_configuration(),
        clock=lambda: datetime(2026, 7, 2, tzinfo=UTC),
    )

    stats = use_case.execute()
    provider = stats.providers[0]

    assert provider.status == "degraded"
    assert provider.credits is None
    assert provider.issues[0].message == (
        "Provider statistics are temporarily unavailable."
    )
    assert any(
        record.exc_info
        and record.message == "Unexpected provider statistics failure for balance"
        for record in caplog.records
    )


def test_llm_provider_stats_route_requires_admin(tmp_path: Path) -> None:
    client = _client(tmp_path, username="operator", is_admin=False)
    _login(client, username="operator")

    response = client.get("/api/admin/llm-providers")

    assert response.status_code == 403


def test_llm_provider_stats_route_returns_provider_snapshot(tmp_path: Path) -> None:
    client = _client(tmp_path, username="admin", is_admin=True)
    _login(client, username="admin")

    response = client.get("/api/admin/llm-providers")

    assert response.status_code == 200
    body = response.json()
    provider = body["providers"][0]
    assert provider["status"] == "available"
    assert provider["credits"]["remaining_credits"] == {
        "amount": "74.75",
        "currency": "USD",
    }
    assert provider["activity"]["totals"]["requests"] == 5
    assert body["configuration"]["chat_model"] == "openai/gpt-4.1-mini"


def _client(tmp_path: Path, *, username: str, is_admin: bool) -> TestClient:
    session_factory = _session_factory(tmp_path, username=username, is_admin=is_admin)
    app = create_app(
        auth_settings=AuthSettings(
            username="admin",
            password="secret",
            session_secret="test-session-secret",
            cookie_secure=False,
        ),
        database_session_factory=session_factory,
        llm_provider_stats_use_case=GetLLMProviderStatsUseCase(
            openrouter=FakeOpenRouterStats(),
            configuration=_configuration(),
            clock=lambda: datetime(2026, 7, 2, tzinfo=UTC),
        ),
    )
    return TestClient(app)


def _session_factory(
    tmp_path: Path,
    *,
    username: str,
    is_admin: bool,
) -> sessionmaker:
    database_url = f"sqlite:///{tmp_path / 'llm-provider-stats.db'}"
    upgrade_database(database_url)
    engine = create_database_engine(database_url)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with factory() as session:
        UserRepository(session).create_user(
            username=username,
            password="secret",
            is_admin=is_admin,
        )
        session.commit()
    return factory


def _login(client: TestClient, *, username: str) -> None:
    response = client.post(
        "/login",
        data={"username": username, "password": "secret"},
        follow_redirects=False,
    )
    assert response.status_code == 303


def _configuration() -> LLMProviderConfiguration:
    return LLMProviderConfiguration(
        chat_provider="openrouter",
        chat_model="openai/gpt-4.1-mini",
        embedding_provider="openrouter",
        embedding_model="openai/text-embedding-3-small",
        agent_models={"professor": "openai/gpt-4.1-mini"},
        selectable_models=(),
    )
