import json
from pathlib import Path

from rag.config import SETTINGS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_api_schema_generation_uses_environment_uv_cache() -> None:
    package = json.loads((PROJECT_ROOT / "package.json").read_text(encoding="utf-8"))

    assert (
        package["scripts"]["api:schema"]
        == "uv run python -m scripts.export_openapi"
    )


def test_runtime_image_smoke_test_initializes_database_before_dashboard() -> None:
    script = (
        PROJECT_ROOT / "scripts" / "smoke_runtime_image.sh"
    ).read_text(encoding="utf-8")
    setup_command = "python main.py setup --skip-qdrant --no-create-env"

    assert setup_command in script
    assert "sqlite:////app/data/medic-smoke.db" in script
    assert script.index(setup_command) < script.index("docker run --detach")


def test_production_compose_publishes_app_over_http() -> None:
    compose = (PROJECT_ROOT / "docker-compose.prod.yml").read_text(encoding="utf-8")

    assert 'image: "${MEDIC_IMAGE:?MEDIC_IMAGE is required}"' in compose
    assert "build:" not in compose
    assert "POSTGRES_PASSWORD is required" in compose
    assert "postgresql+psycopg://" in compose
    assert "python main.py setup --no-create-env" in compose
    assert '"8000:8000"' in compose
    assert "caddy" not in compose
    assert "duckdns" not in compose
    assert "MEDIC_DOMAIN" not in compose
    assert 'MEDIC_DASHBOARD_COOKIE_SECURE: "false"' in compose
    assert 'FORWARDED_ALLOW_IPS: "*"' in compose
    assert "postgres_data:/var/lib/postgresql" in compose
    assert "demo_data:/app/data" in compose


def test_production_compose_requires_database_secrets() -> None:
    compose = (PROJECT_ROOT / "docker-compose.prod.yml").read_text(encoding="utf-8")
    database_url_env = SETTINGS["env"]["database_url"]

    assert (
        database_url_env
        + ': "postgresql+psycopg://${POSTGRES_USER:-medic}:'
        "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD is required}"
        '@postgres:5432/${POSTGRES_DB:-medic}"'
    ) in compose
    assert 'POSTGRES_PASSWORD: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD is required}"' in compose
    assert "postgresql+psycopg://medic:medic@postgres:5432/medic" not in compose
    assert "POSTGRES_PASSWORD: medic" not in compose


def test_ci_workflow_configures_qdrant_for_quality_tests() -> None:
    workflow = (
        PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")
    qdrant_url_env = SETTINGS["env"]["qdrant_url"]
    qdrant_api_key_env = SETTINGS["env"]["qdrant_api_key"]

    assert f"{qdrant_url_env}: http://127.0.0.1:6333" in workflow
    assert f"{qdrant_api_key_env}: test-qdrant-key" in workflow
    assert 'node-version: "24"' in workflow
    assert "make verify" in workflow
    assert 'bash scripts/smoke_runtime_image.sh "${IMAGE_SHA}"' in workflow


def test_ci_workflow_verifies_and_publishes_image() -> None:
    workflow = (
        PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")

    assert "pull_request:" in workflow
    assert "push:" in workflow
    assert "pull_request_target" not in workflow
    # Documentation-only pushes skip the build/deploy pipeline.
    assert "paths-ignore:" in workflow
    assert '- "**.md"' in workflow
    assert '- "docs/**"' in workflow
    assert "ghcr.io" in workflow
    assert "IMAGE_NAME: qxoxoxq/medic" in workflow
    assert 'node-version: "24"' in workflow
    assert "make verify" in workflow
    assert 'bash scripts/smoke_runtime_image.sh "${IMAGE_SHA}"' in workflow
    assert (
        "if: (github.event_name == 'push' || github.event_name == 'workflow_dispatch') "
        "&& github.ref == 'refs/heads/main'"
    ) in workflow
    assert 'docker push "${IMAGE_SHA}"' in workflow
    assert 'docker push "${IMAGE_LATEST}"' in workflow
    assert "packages: write" in workflow
    assert 'docker login "${REGISTRY}"' in workflow
    assert "MEDIC_RUN_LIVE_EVALUATION" not in workflow
    assert "evaluation-bootstrap-dataset" not in workflow
    assert "evaluation-calibrate" not in workflow
    assert "main.py evaluate" not in workflow


def test_ci_workflow_deploys_to_production_only_from_main_on_self_hosted() -> None:
    workflow = (
        PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")

    # The deploy job is the only self-hosted job; verify and image stay on
    # disposable GitHub-hosted runners so pull-request code never reaches the
    # production host.
    assert "runs-on: [self-hosted, Linux, X64]" in workflow
    assert workflow.count("runs-on: ubuntu-latest") == 2
    assert "environment: production" in workflow

    # Deploy is gated to pushes to main only — no pull request can deploy.
    assert (
        "if: github.event_name == 'push' && github.ref == 'refs/heads/main'"
    ) in workflow

    # Same-repo deploy uses the built-in token, never a cross-repo PAT.
    assert "DEPLOY_DISPATCH_TOKEN" not in workflow
    assert "secrets.GITHUB_TOKEN" in workflow

    # Deploy drives the production compose stack from the repository root.
    assert (
        "install -m 0644 docker-compose.prod.yml /opt/medic/docker-compose.prod.yml"
        in workflow
    )
    assert "docker compose -f docker-compose.prod.yml" in workflow

    # The insecure trigger that would expose secrets to forks is never used.
    assert "pull_request_target" not in workflow


def test_evaluation_workflow_runs_independently_on_demand() -> None:
    workflow = (
        PROJECT_ROOT / ".github" / "workflows" / "evaluation.yml"
    ).read_text(encoding="utf-8")

    assert "name: Live RAG Evaluation" in workflow
    assert "workflow_dispatch:" in workflow
    assert "suite:" in workflow
    assert "default: medical-demo-v1" in workflow
    assert "group: live-rag-evaluation" in workflow
    assert "live-rag-evaluation-${{ github.ref }}" not in workflow
    assert "if: github.ref == 'refs/heads/main'" in workflow
    assert "MEDIC_RUN_LIVE_EVALUATION" not in workflow
    assert "evaluation-bootstrap-dataset --suite \"${EVALUATION_SUITE}\"" in workflow
    assert "evaluation-calibrate" in workflow
    assert "evaluate --suite \"${EVALUATION_SUITE}\"" in workflow
