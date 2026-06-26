# Evaluation & quality gates

Retrieval and answer quality are measured with [RAGAS](https://docs.ragas.io)
metrics and traced through [Langfuse](https://langfuse.com). Evaluation is a
gate, not a demo: `evaluate` exits non-zero when a suite drops below threshold,
so it can fail a CI pipeline.

## Running a suite

```bash
uv run python main.py evaluation-bootstrap-dataset --suite medical-demo-v1
uv run python main.py evaluation-calibrate
uv run python main.py evaluate --suite medical-demo-v1
```

- `evaluation-bootstrap-dataset` creates or verifies the versioned synthetic
  dataset in Langfuse Cloud.
- `evaluation-calibrate` verifies that the configured RAGAS judge separates a
  known supported answer from a known incorrect answer.
- `evaluate` runs the suite and exits with `0` for PASS, `1` for a quality-gate
  failure, and `2` for an execution or configuration failure.

Evaluation runs initialize their own tracing configuration; dashboard
conversations are exported only when `MEDIC_LANGFUSE_TRACING_ENABLED` is true.
Full runs use live Langfuse, OpenRouter, and Qdrant services and incur model and
embedding costs.

## Evaluation collections

Each corpus fingerprint gets immutable raw/parsed paths and a Qdrant collection
named from `MEDIC_EVAL_QDRANT_PREFIX`. Old evaluation collections are retained
for reproducibility. Delete obsolete `medic_eval_*` collections manually through
your Qdrant administration tooling after confirming that no historical run needs
them.

## GitHub Actions

To run evaluation independently in GitHub Actions, open **Actions**, select
**Live RAG Evaluation**, choose **Run workflow**, select the `main` branch and
suite, then start the workflow. Runs selected from other branches are skipped,
and evaluation runs are serialized to protect shared external resources.

The repository must define these Actions secrets:
`OPENROUTER_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL`.

Evaluation is independent from the **Deploy OCI** workflow and does not run
automatically during deployment.
