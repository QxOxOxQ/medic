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


def test_production_compose_uses_published_image_without_demo_seed() -> None:
    compose = (PROJECT_ROOT / "docker-compose.prod.yml").read_text(encoding="utf-8")

    assert 'image: "${MEDIC_IMAGE:?MEDIC_IMAGE is required}"' in compose
    assert "build:" not in compose
    assert "POSTGRES_PASSWORD is required" in compose
    assert "postgresql+psycopg://" in compose
    assert "python main.py setup --no-create-env" in compose
    assert "seed-demo" not in compose
    assert '"0.0.0.0:8000:8000"' in compose
    assert '"127.0.0.1:8000:8000"' not in compose
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
    assert "bash scripts/smoke_runtime_image.sh medic:test" in workflow


def test_deploy_workflow_verifies_before_push_and_oci_deploy() -> None:
    workflow = (
        PROJECT_ROOT / ".github" / "workflows" / "deploy.yml"
    ).read_text(encoding="utf-8")
    qdrant_url_env = SETTINGS["env"]["qdrant_url"]

    assert "pull_request:" in workflow
    assert "push:" in workflow
    assert "pull_request_target" not in workflow
    assert "ghcr.io" in workflow
    assert "IMAGE_NAME: qxoxoxq/medic" in workflow
    assert 'node-version: "24"' in workflow
    assert "make verify" in workflow
    assert 'bash scripts/smoke_runtime_image.sh "${IMAGE_SHA}"' in workflow
    assert (
        "if: (github.event_name == 'push' || github.event_name == 'workflow_dispatch') "
        "&& github.ref == 'refs/heads/main'"
    ) in workflow
    assert "always() && needs.image.result == 'success'" in workflow
    assert "MEDIC_RUN_LIVE_EVALUATION" not in workflow
    assert "evaluation-bootstrap-dataset" not in workflow
    assert "evaluation-calibrate" not in workflow
    assert "main.py evaluate" not in workflow
    assert "docker push \"${IMAGE_SHA}\"" in workflow
    assert "docker push \"${IMAGE_LATEST}\"" in workflow
    assert "runs-on: [self-hosted, Linux, X64]" in workflow
    assert "install -m 0644 docker-compose.prod.yml /opt/medic/docker-compose.prod.yml" in workflow
    assert "cat > /opt/medic/.env <<EOF" in workflow
    assert "POSTGRES_PASSWORD=${POSTGRES_PASSWORD}" in workflow
    assert f"{qdrant_url_env}=${{QDRANT_URL}}" in workflow
    assert "chmod 600 /opt/medic/.env" in workflow
    assert "test -f .env" in workflow
    assert "ssh " not in workflow
    assert "scp " not in workflow
    assert "docker login \"${REGISTRY}\"" in workflow
    assert "docker compose -f docker-compose.prod.yml pull" in workflow
    assert "docker compose -f docker-compose.prod.yml up -d" in workflow
    assert "for attempt in $(seq 1 30); do" in workflow
    assert "curl -fsS http://127.0.0.1:8000/healthz" in workflow
    assert 'if test "${attempt}" -eq 30; then' in workflow
    assert "docker compose -f docker-compose.prod.yml logs app" in workflow
    assert "sleep 2" in workflow


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
