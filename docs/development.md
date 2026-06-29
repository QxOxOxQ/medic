# Development & operations

Local development, the Docker workflows, the CLI, and the demo walkthrough.
For the short overview see the [README](../README.md); for production see
[deployment.md](deployment.md); for the quality gates see
[evaluation.md](evaluation.md).

## Requirements

- Docker and Docker Compose for the packaged application workflow.
- `uv` 0.11.24 and Python 3.12+ for local CLI development outside Docker.
- Node.js 24 LTS for frontend development outside Docker.
- A reachable PostgreSQL database, provided by Docker Compose for local runs.
- A remote or hosted Qdrant deployment.
- An OpenRouter API key for model calls.

The Compose workflow in this repository is for local development and portfolio
demo use. It is not a production hosting configuration.

## Configuration

Copy the template and fill in real values before the first start:

```bash
cp .env.example .env
```

Set at least your own `OPENROUTER_API_KEY`, remote `QdrantURL`, `QdrantApiKey`,
and dashboard credentials. Important variables:

```env
OPENROUTER_API_KEY=...
QdrantURL=https://your-qdrant-cluster-url
QdrantApiKey=...
MEDIC_QDRANT_COLLECTION=hybrid_medic_documents
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

- `MEDIC_DATABASE_URL` is used by local CLI commands. Inside the application
  container, Docker Compose overrides it to
  `postgresql+psycopg://medic:medic@postgres:5432/medic`.
- `QdrantURL` must point to a remote or hosted Qdrant deployment. `QdrantApiKey`
  is required so startup fails fast when configuration is incomplete. The
  project intentionally does not start local Qdrant.
- `MEDIC_QDRANT_COLLECTION` selects the Qdrant collection. Use a dedicated
  collection per environment so documents are not mixed between separate
  indexes. Compose uses fixed ports `5432` and `8000`.

## Packaged / demo mode

```bash
docker-compose up --build
```

Docker Compose builds the application image, starts local PostgreSQL, runs
Alembic migrations, seeds the admin user, and checks or creates the configured
collection in remote Qdrant. It does not start local Qdrant and does not mount
the local `data/` directory in demo mode. The dashboard is then available at
<http://127.0.0.1:8000/>.

Portfolio/demo mode builds an image from the current code and runs the
application as a Docker user would see it:

```bash
docker-compose up --build --force-recreate
```

In this mode application data is stored in the Docker volume `demo_data`, not in
the local `data/` directory. This avoids accidentally showing private documents
used during development. Log in with `MEDIC_DASHBOARD_USERNAME` /
`MEDIC_DASHBOARD_PASSWORD`. Login is verified in PostgreSQL; `main.py setup`
creates the admin user from those variables when it does not already exist.
Compose runs setup automatically before starting the dashboard, but it does not
load synthetic demo documents.

## Docker development mode

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
image configuration, force a clean development image rebuild with `make
dev-build`.

> Do not run `docker-compose down -v` unless you intentionally want to delete
> PostgreSQL volume data.

## Local CLI (outside Docker)

Helper setup for local CLI usage:

```bash
./scripts/setup.sh
```

The script syncs dependencies with `uv`, creates `.env` from `.env.example` when
missing, starts local PostgreSQL with `docker-compose`, runs Alembic migrations,
seeds the admin user, and checks or creates the configured Qdrant collection.

Equivalent application setup command:

```bash
uv run python main.py setup
```

Prepare PDFs to markdown:

```bash
uv run python main.py prepare
```

Full preparation and indexing:

```bash
uv run python main.py ingest
```

Run the dashboard directly:

```bash
uv run python main.py dashboard --host 127.0.0.1 --port 8000
```

Setup without external stores (skip both, or each independently):

```bash
uv run python main.py setup --skip-db
uv run python main.py setup --skip-postgres
uv run python main.py setup --skip-qdrant
```

## Data layout & dashboard actions

The dashboard shows directory and index state:

- `data/raw` — source PDFs.
- `data/parsed` — markdown generated from PDFs.
- PostgreSQL — users, document ownership, document metadata, pipeline statuses,
  processing errors, and chunks. This is the document manifest.
- Qdrant — vector index used by RAG search.

Main actions:

- `Add PDF` stores a file under `data/raw/<document_id>` and creates a document
  record for the logged-in user.
- `Delete` removes the PDF, matching markdown, document record, and attempts
  Qdrant cleanup by `content_hash`.
- `Run pipeline` prepares and indexes the selected documents for the logged-in
  user.
- `Pipeline` shows live status for each processing step.
- `Medical agent` lets the user ask questions against the current index and shows
  readable source names.

If Qdrant is unavailable, the dashboard still shows local documents and clearly
reports the index problem.

## Dashboard

The dashboard is English-only. There is no language selector and agent answers
are requested in English. The primary dashboard is a Preact/TypeScript
application with these workspaces:

- `Overview` — health, workflow, latest pipeline run and latest conversation.
- `Documents` — upload, server-side pagination, filters, bulk actions and
  markdown/chunk/index inspectors.
- `Pipeline` — persistent run history and live SSE progress.
- `Assistant` — asynchronous agent runs, live trace and source verification.
- `Retrieval` — direct inspection of ranked Qdrant results.
- `Admin` — SQLAdmin for authorized administrators.

API contracts are defined with Pydantic and exported through OpenAPI. Regenerate
the checked-in TypeScript schema after changing a contract:

```bash
npm run api:types
```

### Admin SQL dashboard

Admin users can open the SQLAdmin panel at <http://127.0.0.1:8000/admin>. Access
requires an active PostgreSQL user with `is_admin=true`. The panel exposes CRUD
for the application SQL tables, including users. User passwords entered through
the admin form are stored as Argon2 hashes. Treat this panel as a technical
administration tool: direct edits to document, chunk, and chat records do not
automatically update local PDF files, markdown files, or Qdrant points.

## Working with documents

Log in with the account from `.env`, then in the `Documents` workspace:

1. Upload one or more PDFs (`Add PDF`); they appear in the `Documents` table.
2. Run `Run pipeline` and wait for the final status.
3. In `Assistant`, ask a question grounded in those documents. Inspect the agent
   answer, the selected specialist, the `[S1]` citations, the source list, and
   the source excerpt preview.

Each user only ever retrieves their own documents — retrieval and full-document
reads are filtered by `owner_user_id` at every layer.

### Test fixtures

`demo_documents/` holds synthetic PDFs used only as fixtures for the RAG quality
evaluation (`evaluation/`, suite `medical-demo-v1`); they are never seeded into
the application. `demo_documents/failure_cases/EXPECTED_PARSE_FAILURE_invalid_pdf.pdf`
is intentionally broken and exercises the `failed` status with a readable
`processing_error`.

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
