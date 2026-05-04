FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

FROM python:3.12-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libxcb1 \
    && groupadd --gid 10001 medic \
    && useradd --create-home --uid 10001 --gid medic medic \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder --chown=medic:medic /app/.venv /app/.venv
COPY --chown=medic:medic main.py alembic.ini ./
COPY --chown=medic:medic agents ./agents
COPY --chown=medic:medic backend ./backend
COPY --chown=medic:medic clients ./clients
COPY --chown=medic:medic dashboard ./dashboard
COPY --chown=medic:medic demo_documents ./demo_documents
COPY --chown=medic:medic migrations ./migrations
COPY --chown=medic:medic observability ./observability
COPY --chown=medic:medic rag ./rag
COPY --chown=medic:medic tools ./tools

RUN mkdir -p /app/data \
    && chown medic:medic /app/data

USER medic:medic

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/healthz', timeout=3).close()"]

CMD ["python", "main.py", "dashboard", "--host", "0.0.0.0", "--port", "8000"]
