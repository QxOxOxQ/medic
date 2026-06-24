import os
import shutil
import subprocess
from pathlib import Path

import pytest

from rag.config import SETTINGS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCKER_COMPOSE = shutil.which("docker-compose")
ENV_NAMES = SETTINGS["env"]


pytestmark = pytest.mark.skipif(
    DOCKER_COMPOSE is None,
    reason="docker-compose is not installed",
)


def _copy_compose_file(tmp_path: Path) -> Path:
    compose_path = tmp_path / "docker-compose.yml"
    compose_path.write_text((PROJECT_ROOT / "docker-compose.yml").read_text())
    return compose_path


def test_dockerfile_installs_rapidocr_native_runtime_libraries() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    for package in (
        "libglib2.0-0",
        "libgl1",
        "libsm6",
        "libxext6",
        "libxrender1",
        "libxcb1",
    ):
        assert package in dockerfile


def test_dockerfile_builds_a_minimal_non_root_runtime_image() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert " AS python-builder" in dockerfile
    assert "FROM node:24-bookworm-slim AS frontend-builder" in dockerfile
    assert "npm run build" in dockerfile
    assert "/app/dashboard/static/dist" in dockerfile
    assert "FROM python:3.12-slim-bookworm AS runtime" in dockerfile
    assert "COPY . ." not in dockerfile
    assert "USER medic:medic" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "/healthz" in dockerfile
    assert "COPY --chown=medic:medic evaluation" not in dockerfile
    assert "COPY --chown=medic:medic demo_documents" not in dockerfile


def test_development_compose_uses_dedicated_uv_image_and_frontend_watcher() -> None:
    override = (PROJECT_ROOT / "docker-compose.dev.yml").read_text(
        encoding="utf-8"
    )
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "image: medic-development:local" in override
    assert "target: development" in override
    assert "uv sync --locked --all-extras --no-dev" in override
    assert "uv run python -m uvicorn" in override
    assert "image: node:24-bookworm-slim" in override
    assert "npm ci && npm run dev" in override
    assert "$(COMPOSE_DEV) up --build" in makefile
    assert "$(COMPOSE_DEV) build --no-cache app" in makefile


def test_dockerignore_excludes_local_and_secret_material() -> None:
    ignored = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    for pattern in (".aider*", ".env*", ".idea", "data", "sk-*", "tests"):
        assert pattern in ignored


def test_docker_compose_configures_app_with_local_postgres_and_remote_qdrant(
    tmp_path: Path,
) -> None:
    assert DOCKER_COMPOSE is not None
    _copy_compose_file(tmp_path)

    result = subprocess.run(
        [DOCKER_COMPOSE, "config"],
        cwd=tmp_path,
        env={
            **os.environ,
            ENV_NAMES["openrouter_api_key"]: "openrouter-key",
            ENV_NAMES["qdrant_url"]: "https://qdrant.example",
            ENV_NAMES["qdrant_api_key"]: "qdrant-key",
            ENV_NAMES["qdrant_collection_name"]: "demo-collection",
            "MEDIC_DASHBOARD_USERNAME": "admin",
            "MEDIC_DASHBOARD_PASSWORD": "secret",
            "MEDIC_SESSION_SECRET": "local-test-secret",
            "MEDIC_DASHBOARD_COOKIE_SECURE": "false",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "\n  app:" in result.stdout
    assert "\n  postgres:" in result.stdout
    assert "\n  qdrant:" not in result.stdout
    assert "context: " in result.stdout
    assert (
        f"{ENV_NAMES['database_url']}: "
        "postgresql+psycopg://medic:medic@postgres:5432/medic"
    ) in result.stdout
    assert f"{ENV_NAMES['openrouter_api_key']}: openrouter-key" in result.stdout
    assert f"{ENV_NAMES['qdrant_url']}: https://qdrant.example" in result.stdout
    assert f"{ENV_NAMES['qdrant_api_key']}: qdrant-key" in result.stdout
    assert f"{ENV_NAMES['qdrant_collection_name']}: demo-collection" in result.stdout
    assert "MEDIC_DASHBOARD_USERNAME: admin" in result.stdout
    assert "MEDIC_DASHBOARD_PASSWORD: secret" in result.stdout
    assert "MEDIC_SESSION_SECRET: local-test-secret" in result.stdout
    assert "MEDIC_DASHBOARD_COOKIE_SECURE: \"false\"" in result.stdout
    assert "condition: service_healthy" in result.stdout
    assert "published: \"8000\"" in result.stdout
    assert "source: demo_data" in result.stdout
    assert "target: /app/data" in result.stdout
    assert "python main.py setup --no-create-env" in result.stdout
    assert "python main.py seed-demo" not in result.stdout
    assert "python main.py dashboard --host 0.0.0.0 --port 8000" in result.stdout
    assert "http://127.0.0.1:8000/healthz" in result.stdout
    assert "restart: unless-stopped" in result.stdout
    assert "image: postgres:18-alpine" in result.stdout
    assert "POSTGRES_DB: medic" in result.stdout
    assert "POSTGRES_USER: medic" in result.stdout
    assert "POSTGRES_PASSWORD: medic" in result.stdout
    assert "published: \"5432\"" in result.stdout
    assert "target: /var/lib/postgresql" in result.stdout
    assert "\n  demo_data:" in result.stdout


def test_docker_compose_requires_remote_qdrant_url(tmp_path: Path) -> None:
    assert DOCKER_COMPOSE is not None
    _copy_compose_file(tmp_path)

    env = os.environ.copy()
    env[ENV_NAMES["openrouter_api_key"]] = "openrouter-key"
    env[ENV_NAMES["qdrant_api_key"]] = "qdrant-key"
    env["MEDIC_DASHBOARD_USERNAME"] = "admin"
    env["MEDIC_DASHBOARD_PASSWORD"] = "secret"
    env["MEDIC_SESSION_SECRET"] = "local-test-secret"
    env["MEDIC_DASHBOARD_COOKIE_SECURE"] = "false"
    env.pop(ENV_NAMES["qdrant_url"], None)
    result = subprocess.run(
        [DOCKER_COMPOSE, "config"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert f"{ENV_NAMES['qdrant_url']} is required" in (
        result.stderr or result.stdout
    )


def test_docker_compose_requires_remote_qdrant_api_key(tmp_path: Path) -> None:
    assert DOCKER_COMPOSE is not None
    _copy_compose_file(tmp_path)

    env = os.environ.copy()
    env[ENV_NAMES["openrouter_api_key"]] = "openrouter-key"
    env[ENV_NAMES["qdrant_url"]] = "https://qdrant.example"
    env["MEDIC_DASHBOARD_USERNAME"] = "admin"
    env["MEDIC_DASHBOARD_PASSWORD"] = "secret"
    env["MEDIC_SESSION_SECRET"] = "local-test-secret"
    env["MEDIC_DASHBOARD_COOKIE_SECURE"] = "false"
    env.pop(ENV_NAMES["qdrant_api_key"], None)
    result = subprocess.run(
        [DOCKER_COMPOSE, "config"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert f"{ENV_NAMES['qdrant_api_key']} is required" in (
        result.stderr or result.stdout
    )


def test_docker_compose_requires_dashboard_cookie_secure(tmp_path: Path) -> None:
    assert DOCKER_COMPOSE is not None
    _copy_compose_file(tmp_path)

    env = os.environ.copy()
    env[ENV_NAMES["openrouter_api_key"]] = "openrouter-key"
    env[ENV_NAMES["qdrant_url"]] = "https://qdrant.example"
    env[ENV_NAMES["qdrant_api_key"]] = "qdrant-key"
    env["MEDIC_DASHBOARD_USERNAME"] = "admin"
    env["MEDIC_DASHBOARD_PASSWORD"] = "secret"
    env["MEDIC_SESSION_SECRET"] = "local-test-secret"
    env.pop("MEDIC_DASHBOARD_COOKIE_SECURE", None)
    result = subprocess.run(
        [DOCKER_COMPOSE, "config"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "MEDIC_DASHBOARD_COOKIE_SECURE is required" in (
        result.stderr or result.stdout
    )
