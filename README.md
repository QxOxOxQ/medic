# Medic RAG

A RAG pipeline for preparing PDF documents, indexing them in Qdrant, and asking a source-grounded medical-documentation assistant through a web dashboard.

## Requirements

- Docker and Docker Compose for the packaged application workflow.
- `uv` 0.11.24 and Python 3.12+ for local CLI development outside Docker.
- Node.js 24 LTS for frontend development outside Docker.
- A reachable PostgreSQL database, provided by Docker Compose for local runs.
- A remote or hosted Qdrant deployment.
- An OpenRouter API key for model calls.

## Configuration

The Compose workflow in this repository is for local development and portfolio
demo use. It is not a production hosting configuration.

The fastest application start after configuring real environment values:

```bash
cp .env.example .env
docker-compose up --build
```

Before starting, set your own `OPENROUTER_API_KEY`, remote `QdrantURL`, `QdrantApiKey`, and dashboard credentials in `.env`. Docker Compose builds the application image, starts local PostgreSQL, runs Alembic migrations, seeds the admin user, and checks or creates the configured collection in remote Qdrant. Compose does not start local Qdrant and does not mount the local `data/` directory in demo mode.

The dashboard is available at:

```text
http://127.0.0.1:8000/
```

Helper setup for local CLI usage outside Docker:

```bash
./scripts/setup.sh
```

The script syncs dependencies with `uv`, creates `.env` from `.env.example` when missing, starts local PostgreSQL with `docker-compose`, runs Alembic migrations, seeds the admin user, and checks or creates the configured Qdrant collection.

Application setup command:

```bash
uv run python main.py setup
```

Important variables:

```env
OPENROUTER_API_KEY=...
QdrantURL=https://your-qdrant-cluster-url
QdrantApiKey=...
MEDIC_QDRANT_COLLECTION=hybrid_medic_demo_documents
MEDIC_EVAL_QDRANT_PREFIX=medic_eval
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
MEDIC_LANGFUSE_TRACING_ENABLED=false
MEDIC_LANGFUSE_ENVIRONMENT=development
MEDIC_LANGFUSE_SAMPLE_RATE=1.0
MEDIC_LANGFUSE_CAPTURE_CONTENT=false
MEDIC_DATABASE_URL=postgresql+psycopg://medic:medic@127.0.0.1:5432/medic
MEDIC_DASHBOARD_USERNAME=admin
MEDIC_DASHBOARD_PASSWORD=change-me
MEDIC_SESSION_SECRET=replace-with-a-long-random-secret
MEDIC_DASHBOARD_COOKIE_SECURE=false
```

`MEDIC_DATABASE_URL` is used by local CLI commands. Inside the application container, Docker Compose overrides it to `postgresql+psycopg://medic:medic@postgres:5432/medic`.

`QdrantURL` must point to a remote or hosted Qdrant deployment. `QdrantApiKey` is required so the demo fails fast when configuration is incomplete. The project intentionally does not start local Qdrant.

`MEDIC_QDRANT_COLLECTION` defaults to the demo collection `hybrid_medic_demo_documents`. Use a separate collection for demos so synthetic documents are not mixed with private or development indexes. Compose uses fixed ports `5432` and `8000`.

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

`docker-compose.prod.yml` documents the production stack. Caddy is the only
public entry point on ports `80` and `443`; the dashboard remains internal on
port `8000`. Caddy obtains and renews the public TLS certificate automatically.
Point `MEDIC_DOMAIN`'s DNS `A` record at the OCI VM's public IPv4 address. The
public repository only verifies and publishes the runtime image. Deployment runs
manually from
the private `QxOxOxQ/medic-deploy` repository so public pull requests cannot
reach the self-hosted runner.

Keep `/opt/medic/.env` on the OCI host and set at least:

```env
MEDIC_IMAGE=ghcr.io/qxoxoxq/medic:sha-...
MEDIC_DOMAIN=medic.example.com
POSTGRES_PASSWORD=replace-with-a-long-random-password
MEDIC_DATABASE_URL=postgresql+psycopg://medic:replace-with-a-long-random-password@postgres:5432/medic
OPENROUTER_API_KEY=...
QdrantURL=https://your-qdrant-cluster-url
QdrantApiKey=...
MEDIC_DASHBOARD_USERNAME=admin
MEDIC_DASHBOARD_PASSWORD=replace-with-a-long-random-password
MEDIC_SESSION_SECRET=replace-with-a-long-random-secret
```

If the PostgreSQL password contains URL-reserved characters, percent-encode it
inside `MEDIC_DATABASE_URL`.

Before the first deployment, open public TCP ports `80` and `443` in both
Oracle Linux `firewalld` and the OCI Network Security Group. Follow
[`docs/oci-public-https.md`](docs/oci-public-https.md). Remove any public
port-`8000` rule after HTTPS verification succeeds.

### OCI GitHub Actions runner service

Register the runner against the private `QxOxOxQ/medic-deploy` repository, not
this public repository. Trigger its **Deploy OCI** workflow manually and provide
the immutable image tag produced here, for example `sha-<40-character-commit>`.

Install the already configured self-hosted runner as a `systemd` service so it
starts after VM reboots and remains available after the SSH session closes:

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

## Dashboard

Portfolio/demo mode builds an image from the current code and runs the application as a Docker user would see it:

```bash
docker-compose up --build --force-recreate
```

In this mode, application data is stored in the Docker volume `demo_data`, not in the local `data/` directory. This avoids accidentally showing private documents used during development.

Open:

```text
http://127.0.0.1:8000/
```

Log in with:

```env
MEDIC_DASHBOARD_USERNAME
MEDIC_DASHBOARD_PASSWORD
```

Login is verified in PostgreSQL. `main.py setup` creates the admin user from those variables when it does not already exist. Compose runs setup automatically before starting the dashboard, but it does not load synthetic demo documents.

The dashboard is English-only. There is no language selector and agent answers are requested in English.

The primary dashboard is a Preact/TypeScript application with these workspaces:

- `Overview` — health, workflow, latest pipeline run and latest conversation.
- `Documents` — upload, server-side pagination, filters, bulk actions and
  markdown/chunk/index inspectors.
- `Pipeline` — persistent run history and live SSE progress.
- `Assistant` — asynchronous agent runs, live trace and source verification.
- `Retrieval` — direct inspection of ranked Qdrant results.
- `Admin` — SQLAdmin for authorized administrators.

The previous server-rendered dashboard remains temporarily available at
`http://127.0.0.1:8000/legacy` as a migration fallback.

API contracts are defined with Pydantic and exported through OpenAPI. Regenerate
the checked-in TypeScript schema after changing a contract:

```bash
npm run api:types
```

The detailed architecture and rollout criteria are documented in
[`docs/ui-ux-implementation-plan.md`](docs/ui-ux-implementation-plan.md).

### Admin SQL dashboard

Admin users can open the SQLAdmin panel at:

```text
http://127.0.0.1:8000/admin
```

Access requires an active PostgreSQL user with `is_admin=true`. The panel exposes
CRUD for the application SQL tables, including users. User passwords entered
through the admin form are stored as Argon2 hashes. Treat this panel as a
technical administration tool: direct edits to document, chunk, and chat records
do not automatically update local PDF files, markdown files, or Qdrant points.

## Docker Development Mode

For day-to-day work, use the development override. Python runs with Uvicorn
reload and the Node.js 24 service rebuilds Vite assets in watch mode:

```bash
make dev-build
```

For later starts after ordinary code changes:

```bash
make dev
```

`make dev` performs a cached build so it cannot accidentally reuse the packaged
runtime image. After changes to `pyproject.toml`, `uv.lock`, `Dockerfile`, or
image configuration, force a clean development image rebuild:

```bash
make dev-build
```

Do not run `docker-compose down -v` unless you intentionally want to delete PostgreSQL volume data.

## How It Works

The dashboard shows directory and index state:

- `data/raw` - source PDFs.
- `data/parsed` - markdown generated from PDFs.
- PostgreSQL - users, document ownership, document metadata, pipeline statuses, processing errors, and chunks. This is the document manifest.
- Qdrant - vector index used by RAG search.

Main actions:

- `Add PDF` stores a file under `data/raw/<document_id>` and creates a document record for the logged-in user.
- `Delete` removes the PDF, matching markdown, document record, and attempts Qdrant cleanup by `content_hash`.
- `Run pipeline` prepares and indexes the selected documents for the logged-in user.
- `Pipeline` shows live status for each processing step.
- `Medical agent` lets the user ask questions against the current index and shows readable source names.

If Qdrant is unavailable, the dashboard still shows local documents and clearly reports the index problem.

## Demo Documents

`demo_documents/` contains three synthetic PDFs that can be uploaded manually:

- `synthetic_acl_rehab_demo.pdf`
- `synthetic_psoriasis_treatment_demo.pdf`
- `synthetic_glp1_remote_monitoring_demo.pdf`

After the first Compose start, log in and add your own PDFs. To rehearse the synthetic demo path, upload the three PDFs below, click `Run pipeline`, then ask a question in the `Medical agent` panel.

Suggested 5-minute demo path:

1. Log in with the account from `.env`.
2. Upload the three synthetic documents and show them in the `Documents` table.
3. Run `Run pipeline` and wait for the final status.
4. Ask these questions:
   - `What are the progression criteria after ACL reconstruction?`
   - `In the fictional comparison, was phototherapy or biologic treatment better?`
   - `What changed with remote monitoring in the GLP-1 study?`
5. Show the agent answer, selected specialist, `[S1]` citations, source list, and source excerpt preview.

`demo_documents/failure_cases/EXPECTED_PARSE_FAILURE_invalid_pdf.pdf` is intentionally broken and only supports testing the `failed` status with a readable `processing_error`. It is not part of the main demo path.

## CLI Commands

Run the CLI outside the container after:

```bash
./scripts/setup.sh
```

Prepare PDFs to markdown:

```bash
uv run python main.py prepare
```

Full preparation and indexing:

```bash
uv run python main.py ingest
```

Run the full evaluation suite synchronously:

```bash
uv run python main.py evaluation-bootstrap-dataset --suite medical-demo-v1
uv run python main.py evaluation-calibrate
uv run python main.py evaluate --suite medical-demo-v1
```

The bootstrap command creates or verifies the versioned synthetic dataset in
Langfuse Cloud. Evaluation runs initialize their own tracing configuration;
dashboard conversations are exported only when `MEDIC_LANGFUSE_TRACING_ENABLED`
is true. The calibration command verifies
that the configured RAGAS judge separates a known supported answer from a known
incorrect answer. `evaluate` exits with `0` for PASS, `1` for a quality-gate
failure, and `2` for an execution or configuration failure.

Each corpus fingerprint gets immutable raw/parsed paths and a Qdrant collection
named from `MEDIC_EVAL_QDRANT_PREFIX`. Old evaluation collections are retained
for reproducibility. Delete obsolete `medic_eval_*` collections manually through
your Qdrant administration tooling after confirming that no historical run needs
them. Full runs use live Langfuse, OpenRouter, and Qdrant services and incur model
and embedding costs.

To run evaluation independently in GitHub Actions, open **Actions**, select
**Live RAG Evaluation**, choose **Run workflow**, select the `main` branch and
suite, then start the workflow. Runs selected from other branches are skipped,
and evaluation runs are serialized to protect shared external resources. The
repository must define these Actions secrets:
`OPENROUTER_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`,
`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL`.
Evaluation is independent from the **Deploy OCI** workflow and does not run
automatically during deployment.

Dashboard:

```bash
uv run python main.py dashboard --host 127.0.0.1 --port 8000
```

Setup without external stores:

```bash
uv run python main.py setup --skip-db
```

You can skip them separately:

```bash
uv run python main.py setup --skip-postgres
uv run python main.py setup --skip-qdrant
```

## Tests

```bash
make verify
```

This runs Ruff, project-wide strict mypy, and pytest. Default tests do not
require live OpenRouter, live Qdrant, or live PostgreSQL. It also runs TypeScript
type checking, Vitest, and the production Vite build. Run integration tests
against external services only after setting the required environment variables.

Browser accessibility and responsive checks require a running dashboard:

```bash
npm run test:e2e
```
