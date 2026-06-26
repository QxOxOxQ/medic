#!/usr/bin/env bash

set -euo pipefail

readonly image="${1:?Usage: smoke_runtime_image.sh IMAGE}"
readonly run_suffix="${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}-${GITHUB_JOB:-image}-$$"
readonly container_name="medic-smoke-${run_suffix}"
readonly volume_name="medic-smoke-data-${run_suffix}"
readonly database_url="sqlite:////app/data/medic-smoke.db"

cleanup() {
  docker rm --force "${container_name}" >/dev/null 2>&1 || true
  docker volume rm --force "${volume_name}" >/dev/null 2>&1 || true
}

runtime_environment=(
  --env OPENROUTER_API_KEY=test-openrouter-key
  --env QdrantURL=http://127.0.0.1:6333
  --env QdrantApiKey=test-qdrant-key
  --env MEDIC_QDRANT_COLLECTION=test-collection
  --env "MEDIC_DATABASE_URL=${database_url}"
  --env MEDIC_DASHBOARD_USERNAME=admin
  --env MEDIC_DASHBOARD_PASSWORD=test-password
  --env MEDIC_SESSION_SECRET=test-session-secret
  --env MEDIC_DASHBOARD_COOKIE_SECURE=false
)

trap cleanup EXIT
docker volume create "${volume_name}" >/dev/null

docker run --rm \
  --volume "${volume_name}:/app/data" \
  "${runtime_environment[@]}" \
  "${image}" \
  python main.py setup --skip-qdrant --no-create-env

docker run --detach \
  --name "${container_name}" \
  --publish 127.0.0.1:18000:8000 \
  --volume "${volume_name}:/app/data" \
  "${runtime_environment[@]}" \
  "${image}" >/dev/null

for attempt in $(seq 1 30); do
  if curl --fail --silent http://127.0.0.1:18000/healthz >/dev/null; then
    break
  fi
  if test "${attempt}" -eq 30; then
    docker logs "${container_name}"
    exit 1
  fi
  sleep 1
done

test "$(curl --silent --output /dev/null --write-out '%{http_code}' http://127.0.0.1:18000/login)" = "200"
