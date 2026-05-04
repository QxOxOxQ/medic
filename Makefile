COMPOSE ?= docker-compose
COMPOSE_DEV = $(COMPOSE) -f docker-compose.yml -f docker-compose.dev.yml

.PHONY: dev dev-build verify

dev:
	$(COMPOSE_DEV) up

dev-build:
	$(COMPOSE_DEV) up --build

verify:
	uv run --extra evaluation ruff check .
	uv run --extra evaluation mypy
	uv run --extra evaluation pytest -q
