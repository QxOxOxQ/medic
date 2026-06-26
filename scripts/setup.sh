#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

START_SERVICES=1
START_POSTGRES=1
SETUP_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --no-services)
      START_SERVICES=0
      ;;
    --skip-db)
      START_SERVICES=0
      SETUP_ARGS+=("$arg")
      ;;
    --skip-postgres)
      START_POSTGRES=0
      SETUP_ARGS+=("$arg")
      ;;
    --skip-qdrant)
      SETUP_ARGS+=("$arg")
      ;;
    *)
      SETUP_ARGS+=("$arg")
      ;;
  esac
done

uv sync --all-extras

if [[ ! -f .env && -f .env.example ]]; then
  cp .env.example .env
fi

if [[ "$START_SERVICES" == "1" ]]; then
  SERVICES=()
  if [[ "$START_POSTGRES" == "1" ]]; then
    SERVICES+=("postgres")
  fi

  if [[ "${#SERVICES[@]}" == "0" ]]; then
    echo "No local services selected for startup."
  elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose up -d "${SERVICES[@]}"
  elif command -v docker >/dev/null 2>&1; then
    docker compose up -d "${SERVICES[@]}"
  else
    echo "Docker Compose is not available; skipping local service startup." >&2
  fi
fi

uv run python main.py setup "${SETUP_ARGS[@]}"
