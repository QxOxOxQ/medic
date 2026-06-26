COMPOSE ?= docker-compose
COMPOSE_DEV = $(COMPOSE) -f docker-compose.yml -f docker-compose.dev.yml

.PHONY: dev dev-build frontend-build frontend-test verify

dev:
	$(COMPOSE_DEV) up --build

dev-build:
	$(COMPOSE_DEV) build --no-cache app
	$(COMPOSE_DEV) up --force-recreate

frontend-build:
	npm ci
	npm run build

frontend-test:
	npm run api:types
	npm run typecheck
	npm test

verify:
	npm ci
	npm run api:types
	npm run typecheck
	npm test
	npm run build
	uv run --extra evaluation ruff check .
	uv run --extra evaluation mypy
	uv run --extra evaluation pytest -q
