# Deployment

This document contains information on advanced configuration, production deployment, and infrastructure setup.

## Portable Docker Image

Build the standalone runtime image without development and evaluation tooling:

```bash
docker build -t medic:local .
```

The image runs the dashboard on `0.0.0.0:8000` as the unprivileged user
`medic` (UID/GID `10001`). It expects PostgreSQL, Qdrant, OpenRouter, and
authentication configuration through environment variables. Run setup before
the dashboard when using the image outside Compose:

```bash
docker run --rm --env-file runtime.env \
  --volume medic_data:/app/data \
  medic:local python main.py setup --no-create-env

docker run --rm --env-file runtime.env \
  --publish 0.0.0.0:8000:8000 \
  --volume medic_data:/app/data \
  medic:local
```

`MEDIC_DATABASE_URL` and remote service URLs in `runtime.env` must be reachable
from inside the container. A bind-mounted `/app/data` directory must be writable
by UID `10001`. The liveness endpoint is `GET /healthz` and returns HTTP `204`.

The runtime image intentionally excludes RAGAS and the `evaluation/` package.
Install the evaluation extra for local evaluation commands:

```bash
uv sync --extra evaluation
```

Never reuse credentials that appeared in a local assistant history or an older
image. Revoke them at the provider and supply newly generated values through the
runtime environment.

## Production Compose

`docker-compose.prod.yml` documents the production stack. The dashboard is
published directly on port `8000` and served over plain HTTP at
`http://<public-ip>:8000/`. Because traffic is unencrypted,
`MEDIC_DASHBOARD_COOKIE_SECURE` is `false` (Secure cookies require HTTPS); put a
reverse proxy with a real domain in front to re-enable TLS. Pushing to `main`
runs the `deploy` job in `.github/workflows/ci.yml`, which builds and publishes
the runtime image and then deploys it on the self-hosted runner using the
built-in `GITHUB_TOKEN`. Keeping deployment in this public repository is safe
because the `deploy` job runs only on pushes to `main` and the self-hosted
runner is used exclusively by that job, so pull-request code only ever runs on
disposable GitHub-hosted runners.

Keep `/opt/medic/.env` on the OCI host and set at least:

```env
MEDIC_IMAGE=ghcr.io/qxoxoxq/medic:sha-...
POSTGRES_PASSWORD=replace-with-a-long-random-password
MEDIC_DATABASE_URL=postgresql+psycopg://medic:replace-with-a-long-random-password@postgres:5432/medic
OPENROUTER_API_KEY=...
OPENROUTER_MANAGEMENT_API_KEY=...
QdrantURL=https://your-qdrant-cluster-url
QdrantApiKey=...
MEDIC_DASHBOARD_USERNAME=admin
MEDIC_DASHBOARD_PASSWORD=replace-with-a-long-random-password
MEDIC_SESSION_SECRET=replace-with-a-long-random-secret
```

If the PostgreSQL password contains URL-reserved characters, percent-encode it
inside `MEDIC_DATABASE_URL`.

`OPENROUTER_MANAGEMENT_API_KEY` can be managed as a GitHub Actions environment
secret named `OPENROUTER_MANAGEMENT_API_KEY` on the `production` environment.
The deploy job syncs that secret into `/opt/medic/.env` on the self-hosted
runner before restarting the Compose stack.

The OCI Network Security Group attached to the instance must allow ingress from
`0.0.0.0/0` to TCP port `8000`. The `deploy` job opens the matching host
`firewalld` rule automatically.

### OCI GitHub Actions runner service

Register the runner against this `QxOxOxQ/medic` repository (Settings → Actions →
Runners → New self-hosted runner, Linux x64) with the default `self-hosted`,
`Linux`, `X64` labels. The `deploy` job targets those labels and deploys the
image built in the same run (`ghcr.io/qxoxoxq/medic:sha-<40-character-commit>`)
automatically on every push to `main`.

Install the self-hosted runner as a `systemd` service so it starts after VM
reboots and remains available after the SSH session closes:

```bash
cd /opt/github-actions-runner
sudo ./svc.sh install opc
sudo ./svc.sh start
sudo ./svc.sh status
```

Confirm that the service is enabled and active:

```bash
sudo systemctl is-enabled "$(cat .service)"
sudo systemctl is-active "$(cat .service)"
```

Use the generated service name to inspect recent logs:

```bash
sudo journalctl -u "$(cat .service)" --no-pager -n 100
```

`./run.sh` starts the runner interactively and stops when its terminal or SSH
session ends. Use it only for temporary diagnostics, not for production
deployments.

### Langfuse application tracing

Set valid `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and
`LANGFUSE_BASE_URL` values, then enable dashboard agent tracing:

```env
MEDIC_LANGFUSE_TRACING_ENABLED=true
```

Each agent response becomes a `medical-agent-response` trace. LangChain model
generations and RAG tool calls are nested beneath it, so Langfuse captures model
names, token usage, latency, errors, and the agent/tool hierarchy. Conversation
IDs group multi-turn requests in the Langfuse Sessions view, while user IDs are
SHA-256 pseudonyms rather than database identifiers. `MEDIC_LANGFUSE_ENVIRONMENT`
keeps application traces separate from the evaluation environment, and
`MEDIC_LANGFUSE_SAMPLE_RATE` controls ingestion volume.

Medical prompt, conversation, tool, and response content is masked before
transmission by default. Set `MEDIC_LANGFUSE_CAPTURE_CONTENT=true` only after the
target Langfuse deployment and data-handling policy are approved for that data.
See the Langfuse documentation for [LangChain tracing](https://langfuse.com/docs/integrations/langchain/python),
[sessions](https://langfuse.com/docs/tracing-features/sessions), and
[client-side masking](https://langfuse.com/docs/observability/features/masking).
