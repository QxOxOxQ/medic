---
name: business-docs
description: Use this skill when the user asks to create, update, or audit business-facing documentation for the application, including what it does, how the RAG pipeline and dashboard work, user workflows, capabilities, data flow, external systems, or operational behavior.
---

# Business Docs

## Required Context

Before writing documentation, inspect the current project sources:

- `README.md` and `pyproject.toml`
- Dashboard routes, templates, static files, and services
- Backend routes, dependencies, factories, and use cases
- RAG preparation, chunking, indexing, retrieval, search, manifest, and configuration modules
- Database models, repositories, sessions, migrations, and document synchronization
- Agent profiles, prompts, and tests that describe expected behavior

Use targeted searches and file reads. Do not rely on memory when the repository can answer the question.

## Documentation Defaults

- Write in Polish by default.
- Write for business and product readers first, while staying technically accurate.
- Prefer clear process descriptions over code-level detail.
- Do not invent product behavior, compliance claims, medical claims, or operational guarantees that are not supported by the repository.
- Preserve useful existing documentation when updating files.

## Output Files

Create or update these Markdown files unless the user asks for a different target:

- `docs/business/overview.md`
- `docs/business/how-it-works.md`

## Content Requirements

`docs/business/overview.md` should cover:

- Product purpose
- Intended users and main use cases
- Core capabilities
- Main business concepts and data objects
- External systems and dependencies
- Current limitations and explicit non-goals visible from the codebase

`docs/business/how-it-works.md` should cover:

- Setup and login flow
- PDF upload and ownership flow
- Document preparation flow
- Ingestion and indexing flow
- Search or chat flow
- Document deletion flow
- Data storage locations and external system responsibilities
- Failure states such as unavailable Qdrant, database errors, invalid input, or missing configuration

## Workflow

1. Gather source facts from the repository.
2. Map the user-visible workflows end to end.
3. Draft or update the two docs with business-readable language.
4. Cross-check every workflow against the source files before finalizing.
5. Report which files changed and what source areas informed the documentation.
