# RAG Implementation Weaknesses

## Purpose

This document tracks known weaknesses in Medic RAG's retrieval implementation. Each entry is a self-contained work item: problem, effect, code locations, suggested fix direction, and completion criteria.

Scope covers the pipeline end to end: PDF preparation, chunking, embedding, Qdrant indexing, retrieval, per-user filtering, and the agent's cited answer.

## Status At A Glance (2026-07-01)

| # | Topic | Status | Priority |
| - | --- | --- | --- |
| 1 | Hybrid RRF in the production Searcher | Resolved | - |
| 2 | Per-user filtering in Qdrant (prefilter) | Resolved | - |
| 3 | Oversampling before filtering | Resolved (moot after #2) | - |
| 4 | No reranking or quality threshold on results | Open | High |
| 5 | No retrieval evaluation set | Open | High |
| 6 | Chunking too simple for medical documents | Open | Medium |
| 7 | `rag/indexer.py` has too many responsibilities | Open | Medium |
| 8 | Architectural layers are mixed | Open | Medium |
| 9 | No context-free retrieval mode | Open | Low |
| 10 | Infrastructure error handling too generic | Open | Medium |
| 11 | No embedding-model migration process | Open | Low |
| 12 | Missing answer-quality observability | Open (partial signal at the synthesis layer) | Medium |
| 13 | Index deletion relies on `content_hash` alone | Open | High |
| 14 | Typing too loose at the retrieval/indexer boundaries | Open | Low |
| 15 | General medical questions limited by forced source requirement | Partially resolved | - |

Priority reflects risk to clinical answer quality and data isolation, not backlog order — see "Proposed Work Order" at the end of this document for the suggested sequence.

## 1. The Production Searcher Doesn't Use Hybrid RRF

**Status (2026-06-28): RESOLVED.** `Searcher.search()` now delegates to `Qdrant.hybrid_search_with_rrf()`, which runs a dense prefetch + sparse (BM25) prefetch + `Fusion.RRF`. Confirmed by the test `test_qdrant_hybrid_search_uses_dense_sparse_prefetch_and_rrf`. The description below is kept as historical context.

Problem: the application creates Qdrant collections with both dense and sparse vectors, and indexing writes both, but the main search path used by the backend only queried the dense vector. The hybrid RRF method existed but wasn't wired into `Searcher`.

Effect: the advertised hybrid retrieval was incomplete in practice. Queries containing rare medical terms, test names, codes, lab values, or exact document wording could get worse recall than the dense+sparse index should provide.

Code locations:

- `rag/searcher.py` - `Searcher.search()` built a single dense query.
- `rag/qdrant.py` - `hybrid_search_with_rrf()` implements dense+sparse prefetch and RRF, but wasn't the main path.
- `rag/indexer.py` - `_index_chunks()` writes the sparse vector via `models.Document` when `sparse_vector_name` is set.

Fix direction:

- Change `Searcher.search()` to use the hybrid dense+sparse RRF search.
- Make sure the result keeps the same contract `RetrievalService` expects.
- Keep the Searcher testable by injecting a client or provider.

Completion criteria:

- A unit test confirms the production `Searcher` sends Qdrant a dense prefetch, a sparse prefetch, and `Fusion.RRF`.
- The API/agent test still passes with no change to the sources contract.
- Documentation no longer needs to treat hybrid RAG as an unwired feature.

## 2. Per-User Filtering Happens Only After The Qdrant Search

**Status (2026-06-28): RESOLVED.** Indexing now writes `owner_user_id` into the point payload (with a KEYWORD index), and `hybrid_search_with_rrf()` applies a `Filter` on `owner_user_id` to every `Prefetch` — the filter is applied before top-k, not after. PostgreSQL-side filtering is kept as a second line of defense. Isolation was confirmed: one owner's query never returns another owner's points, even when they're the best semantic match. Existing points without the field are skipped (fail-closed) and need reindexing.

Problem: Qdrant returned a global top-k across the whole collection, and only afterward did `RetrievalService` filter results through PostgreSQL to check whether points belonged to the current user.

Effect: with multiple users, other users' documents could occupy the top-k slots. After filtering, the backend could return few or no results even though the current user's own documents contained the answer. This was a retrieval-quality problem, not a direct data leak, since results were filtered before being shown to the agent.

Code locations:

- `rag/retrieval.py` - `RetrievalService.search()` fetched results from the provider with no owner filter.
- `rag/retrieval.py` - `search_results_from_response()` filtered results on the application side.
- `rag/database/repositories.py` - `ownership_for_search()` checks in PostgreSQL which points are allowed for the user.

Fix direction:

- Add `owner_user_id` or a stable `document_id` to Qdrant point payloads.
- Pass the Qdrant filter at search time, before top-k is computed.
- Keep the existing DB-side filtering as a second line of defense.

Completion criteria:

- New Qdrant points carry a field that allows filtering by owner.
- The Searcher accepts a user context or a separate query object carrying the filter.
- A test shows that with other users' highly-scored points present, the user still gets their own results.

## 3. No Oversampling Before Filtering Results

**Status (2026-06-28): RESOLVED by the target fix.** Prefiltering by `owner_user_id` in Qdrant was introduced (see #2), so prefetch candidates are already scoped to the owner and the DB post-filter effectively cuts nothing. Oversampling is no longer needed for the multi-tenant isolation problem. The description below is kept as historical context.

Problem: `RetrievalService` asked the provider for exactly `limit` results, then removed results that didn't belong to the user. There was no candidate buffer before filtering.

Effect: even while DB-side filtering remained temporarily necessary, a small limit could produce empty answers. With a limit of 5 in a global collection, it was enough for the first 5 points to belong to other documents.

Code locations:

- `rag/retrieval.py` - `response = self._search_provider.search(query, k=limit)`.
- `rag/retrieval.py` - `_database_search_results()` only truncated after collecting allowed results.

Fix direction:

- Until the Qdrant prefilter existed, fetch more candidates than the final limit, e.g. `limit * 5` with an upper bound.
- Apply the final limit only after ownership filtering.
- Eventually replace this with `owner_user_id` prefiltering in Qdrant.

Completion criteria:

- A test covers the case where the first Qdrant results belong to another user but later points belong to the user.
- The API ultimately returns at most the requested limit.
- Oversampling has a safety cap so it doesn't cause expensive queries.

## 4. No Reranking Or Quality Threshold On Results

Problem: the system hands the agent results with no additional reranking and no minimum relevance threshold. The agent gets Qdrant's top-k by score, with no separate check on whether a chunk actually answers the question.

Effect: answers can be based on chunks that are topically similar but don't answer the specific question. In the medical domain this raises the risk of imprecise interpretation.

Code locations:

- `rag/searcher.py` - no reranking after the Qdrant query.
- `rag/retrieval.py` - `SearchResult` carries a score but applies no thresholds.
- `tools/rag_search.py` - the agent tool records every result the retriever returns.

Fix direction:

- Add a reranking stage for candidates, or at least score thresholds for results.
- Consider a separate `RetrievalRanker` as an application port, so reranking isn't mixed into the Qdrant adapter.
- For low confidence, return the agent fewer sources or an explicit insufficient-retrieval state.

Completion criteria:

- A test exists for a below-threshold result that doesn't reach the agent's sources.
- A test exists for multiple candidates confirming stable ordering after reranking.
- The agent trace shows the candidate count before and after reranking.

## 5. No Retrieval Evaluation Set

Problem: tests check technical contracts but don't measure search quality. There's no set of questions with expected documents, chunks, or a minimum recall@k.

Effect: changes to chunking, embeddings, Qdrant, filters, or prompts can degrade retrieval with no visible test failure. The system can keep passing tests while answering worse.

Code locations:

- `tests/test_searcher.py` - checks query construction, not result quality.
- `tests/test_indexer.py` - checks indexing and payload, not search effectiveness.
- `rag/measurement/performance.py` - measures performance and overlap, but isn't a production regression set for the medical demo.

Fix direction:

- Add a small evaluation set based on the synthetic demo documents.
- For each question, record the expected document, and ideally the expected excerpt or `content_hash`.
- Run the evaluation locally without live OpenRouter where possible, via a deterministic test embedding or a local fast embedding.

Completion criteria:

- An evaluation file exists with expected sources for each question.
- A test or command reports recall@k for key demo scenarios.
- The regression set covers ACL, psoriasis/phototherapy, and GLP-1 remote monitoring questions.

## 6. Chunking Is Too Simple For Medical Documents

Problem: chunking uses a fixed markdown chunk size and overlap. There's no semantic model of sections, dates, lab tables, headers, units, or the relationship between a result and its reference range.

Effect: important information can be split apart or mixed together. For lab results, the risk is losing context: the test name, result, unit, range, and date should land in the same chunk.

Code locations:

- `rag/chunking/process_text.py` - `MARKDOWN_CHUNK_SIZE = 800`, `MARKDOWN_CHUNK_OVERLAP = 120`.
- `rag/indexer.py` - `chunks_from_text()` relies on `ProcessText.markdown_chunking()`.
- `tests/test_process_text.py` - only tests simple table and long-text behavior.

Fix direction:

- Add a chunker that keeps whole tables and clinical sections together when they're under the size limit.
- Carry section headers into chunk metadata.
- Add dedicated tests for long tables, results with units, dates, and reference ranges.

Completion criteria:

- A test shows a lab record doesn't lose its unit, range, or date.
- Chunk metadata includes at least source, character range, and optional section headers.
- The dashboard still shows correct chunk previews.

## 7. `rag/indexer.py` Has Too Many Responsibilities

Problem: one module is responsible for chunking, embedding, embedding validation, Qdrant collection creation, point construction, payload preview, `content_hash` idempotency, syncing chunks to PostgreSQL, and progress events.

Effect: every change to indexing carries high cognitive cost and regression risk. It's harder to test application logic, the Qdrant adapter, and DB sync in isolation.

Code locations:

- `rag/indexer.py` - the central indexing file.
- `rag/full_process.py` - calls `index_text()` as the main indexer.
- `rag/markdown_indexing.py` - wraps markdown indexing, but the logic itself still lives in `rag/indexer.py`.

Fix direction:

- Split the module into smaller roles:
  - `ChunkDocumentUseCase` or `MarkdownChunker`.
  - `EmbeddingService`, or a port for embeddings.
  - `VectorIndexWriter` as the Qdrant adapter.
  - `IndexedChunkSynchronizer` for PostgreSQL.
  - `IndexDocumentUseCase` as a thin orchestrator.

Completion criteria:

- Each new class has a single responsibility and a simple contract.
- Tests exist at the chunker, Qdrant writer, and use-case levels.
- `FullProcess`'s public contract stays compatible with the dashboard.

## 8. Architectural Layers Are Mixed

Problem: RAG code mixes application logic with infrastructure. Modules under `rag` import Qdrant, SQLAlchemy, repositories, LangChain splitters, and Pydantic directly. This doesn't meet the standard in `AGENTS.md`, which requires separating domain, application, infrastructure, and presentation.

Effect: it's harder to swap out Qdrant, the embedding provider, the database, or the chunking approach without touching process logic. It also makes testability and strict typing harder to maintain.

Code locations:

- `rag/indexer.py` - imports Qdrant, the SQLAlchemy sessionmaker, repositories, Qdrant models, and chunking.
- `rag/chunking/process_text.py` - the chunking model inherits from Pydantic's `BaseModel`.
- `rag/retrieval.py` - the retrieval use case knows about SQLAlchemy's `sessionmaker` and the concrete `DocumentRepository`.

Fix direction:

- Define application ports for search, index writes, and the document repository.
- Move Qdrant/SQLAlchemy dependencies into infrastructure adapters.
- Keep framework-independent types and protocols in the application layer.

Completion criteria:

- Core use cases don't import Qdrant or SQLAlchemy.
- The chunker doesn't require Pydantic.
- Use-case tests use fake ports instead of monkeypatching global classes.

## 9. No Sane Context-Free Retrieval Mode

Problem: `search_results_from_response()` returns an empty list if there's no `owner_user_id` and `database_session_factory`. This is safe for the dashboard, but it makes the retriever hard to use outside the logged-in flow.

Effect: the CLI, diagnostic tools, or future RAG tests can't easily reuse the same retriever without a full user and database context. This creates pressure to bypass `RetrievalService` and use Qdrant directly.

Code locations:

- `rag/retrieval.py` - `search_results_from_response()` ends with `return []` when there's no ownership context.
- `dashboard/services/search_service.py` - composes `RetrievalService` (not inheritance) and assumes the dashboard's context.

Fix direction:

- Name the current use case explicitly as `UserScopedRetrievalService`.
- For diagnostics, add a separate, explicit admin or dev-only mode that doesn't pretend to be the production path.
- Don't relax the dashboard's safeguards.

Completion criteria:

- Class and test names clearly distinguish user-scoped retrieval from diagnostic retrieval.
- Missing user context produces an explicit error on the production path, not a silent empty result.
- Any diagnostic mode is kept separate from user-facing endpoints.

## 10. Infrastructure Error Handling Is Too Generic

Problem: many places catch broad `Exception` and return `str(error)` to the calling layer. That's fine for a demo, but it gives no stable classification of operational errors.

Effect: the UI and backend have limited ability to distinguish between missing configuration, a Qdrant timeout, an embedding-provider error, a PDF parsing error, or a database error. It's harder to build retries, alerts, and safe user-facing messages.

Code locations:

- `dashboard/services/qdrant_index.py` - status, delete, and preview all catch broad errors.
- `dashboard/routes/search.py` - the `/api/search` endpoint catches `Exception`.
- `backend/use_cases.py` and `backend/chat_use_cases.py` - wrap many errors as `AgentExecutionError`.

Fix direction:

- Introduce explicit application error types: `VectorStoreUnavailable`, `EmbeddingProviderUnavailable`, `DocumentParsingFailed`, `RetrievalFailed`.
- Log the technical detail server-side, but return a controlled message to the UI.
- Keep diagnostic detail in the trace or logs, not in ad hoc JSON responses.

Completion criteria:

- Endpoints return stable error codes and categories.
- Tests cover a Qdrant timeout, missing configuration, and an embedding error.
- The user sees an understandable message, and the log carries the technical detail.

## 11. Embedding Model And Collection Configuration Can Drift Easily

Problem: the Qdrant collection is validated for vector dimension and type, but changing the embedding model still requires a deliberate migration or a new collection. There's no clear reindexing process documented after a model change.

Effect: after a model change, you can get a configuration error or an ambiguous state where old points and new expectations don't match. That's acceptable for a demo, but further development needs an operational process for it.

Code locations:

- `rag/qdrant.py` - `_validate_collection_vectors()` checks an existing collection's configuration.
- `rag/indexer.py` - `_validate_collection_vectors()` repeats similar validation.
- `rag/settings.json` - holds the selected embedding model and vector names.

Fix direction:

- Add a documented process: changing the model means a new collection, or a migration plus a full reindex.
- Consider storing `embedding_model` and `embedding_provider` in the payload or document metadata.
- Consolidate collection validation into a single adapter.

Completion criteria:

- The README or docs describe the embedding-model-change procedure.
- A test confirms a readable error on an incompatible collection.
- Collection-configuration validation isn't duplicated across multiple places.

## 12. Missing Answer Quality Observability

**Status (2026-07-01): PARTIAL PROGRESS.** `agents/professor.py` now has `insufficient_reason()` (`not_record_grounded` / `no_sources` / `review_incomplete` / `evidence_insufficient` / `sufficient`), recorded in the synthesis trace. That's a useful but shallow signal at the synthesis/review layer — the granular retrieval-funnel metrics described below (candidate count from Qdrant, count rejected by the owner filter, search mode) are still missing. The description below remains accurate for the retrieval layer.

Problem: the system records agent trace events and sources, but doesn't record retrieval-quality metrics, candidate counts before filtering, rejected-result counts, which search mode was used, or insufficient-context reasons.

Effect: when the agent answers poorly or reports insufficient context, it's hard to quickly tell whether parsing, chunking, the index, the owner filter, the Searcher, the prompt, or the model is at fault.

Code locations:

- `tools/rag_search.py` - the trace records query, limit, and source count.
- `agents/graph.py` - the trace records model calls, tool calls, and synthesis.
- `rag/retrieval.py` - no explicit metrics for candidates before and after filtering.

Fix direction:

- Add to the retrieval trace: candidate count from Qdrant, count after the owner filter, search mode, final limit.
- Record the insufficient-context reason, e.g. no candidates, candidates filtered out, low score, provider error.
- Surface the basic information in the dashboard's trace panel.

Completion criteria:

- The trace for a RAG query shows the full retrieval funnel.
- An agent test checks that the RAG event includes candidate count and source count.
- Debugging an empty answer doesn't require manually querying Qdrant.

## 13. Index Deletion Relies Mainly On `content_hash`

Problem: when deleting a document, the system removes Qdrant points by `content_hash`. That works for the current model, but `content_hash` isn't an owner or document identifier.

Effect: if two users have identical document content, deleting by `content_hash` alone can remove points that are logically shared by more than one document record. The current `content_hash`-based deduplication can also get in the way of precise document management.

Code locations:

- `dashboard/services/document_storage.py` - delete triggers cleanup by `content_hash`.
- `dashboard/services/qdrant_index.py` - `delete_content_hash()` removes points with a `content_hash` filter.
- `rag/indexer.py` - `_content_hash_exists()` skips embedding if the content hash already exists.

Fix direction:

- Add `document_id` and `owner_user_id` to the Qdrant point payload.
- Consider a point per document+chunk even for identical content, or an explicit reference table for deduplication.
- Delete by `document_id` on the user's document path.

Completion criteria:

- A test shows two documents with the same `content_hash` belonging to different users.
- Deleting one user's document does not remove another user's index entries.
- Qdrant cleanup filters by `document_id`, or by the `owner_user_id` + `document_id` pair.

## 14. Typing And Contracts Are Too Loose In Places

Problem: many boundaries use `Any`, dictionaries, and raw Qdrant objects instead of dedicated application types. This makes strict typing harder to maintain and increases regression risk when the Qdrant client or response models change.

Effect: contract errors can only surface at runtime. It's also harder to understand which fields are required in a point payload, a search result, or a trace event.

Code locations:

- `rag/retrieval.py` - the response and point are typed `Any`.
- `rag/indexer.py` - many vector helpers operate on `Any`.
- `dashboard/services/qdrant_preview.py` and `dashboard/services/qdrant_index.py` - Qdrant adapters return dictionaries.

Fix direction:

- Introduce dataclasses for `VectorSearchCandidate`, `IndexedChunkPayload`, `RetrievalTrace`.
- Confine `Any` to infrastructure adapters.
- Add conversions from Qdrant types to application types at the adapter boundary.

Completion criteria:

- The retrieval use case doesn't operate directly on Qdrant objects.
- The point payload has an explicit type and a serialization test.
- `mypy --strict` has fewer exceptions, or doesn't require loose types in the application layer.

## 15. General Medical Questions Are Limited By The Forced Source Requirement

**Status (2026-07-01): PARTIALLY RESOLVED.** `insufficient_reason()` in `agents/professor.py` returns `not_record_grounded` for `general_information` mode, so `_insufficient_context` is `False` — the runtime no longer blocks general medical questions for lack of sources. `agents/prompts/base.md` instructs the specialist not to imply that a general answer came from the user's documents. Open part: the UI (`AssistantView.tsx`) only shows an `insufficient context` badge — there's no separate, explicit indicator for "answer grounded in your documents" vs. "general information," so one of the completion criteria below still isn't met.

Problem (historical context): the prompt allowed general medical information, but the runtime ultimately returned insufficient context whenever no sources were recorded. That was fine for questions about the user's documents but limited general questions.

Effect: a user could ask for a general medical explanation, and the system would treat the lack of document sources as an inability to answer. This may have been an intentional product decision, but it should be named explicitly.

Code locations:

- `agents/prompts/base.md` - distinguishes document questions from general questions.
- `agents/graph.py` - `_synthesize_answer()` required sources to be present.
- `docs/agent-prompt-authoring-brief.md` - notes that prompt-only changes may not be enough to support general questions.

Fix direction:

- Make a product decision: document-grounded only, or also a general medical assistant.
- If document-grounded only, tighten the prompt and UI to say so.
- If also general-purpose, add a separate answer path with no document citations, clearly labeled as general information.

Completion criteria:

- A test distinguishes a document question from a general question.
- The UI shows whether an answer is grounded in documents or is general information.
- Document questions still require sources and citations.

## Proposed Work Order

1. ~~Fix the search path: wire up hybrid RRF and add tests.~~ DONE (#1).
2. ~~Add per-user filtering in Qdrant~~ DONE — prefiltering by `owner_user_id` in Qdrant (#2, #3).
3. Build a retrieval regression set for the demo documents (#5).
4. Split `rag/indexer.py` into smaller components (#7).
5. Clean up the ports-and-adapters architecture (#8).
6. Add retrieval observability and error categories (#10, #12).
7. Improve medical chunking and chunk metadata (#6).
8. ~~Tighten the general-medical-question policy~~ PARTIALLY DONE — the runtime no longer blocks it; the UI indicator is still missing (#15).
