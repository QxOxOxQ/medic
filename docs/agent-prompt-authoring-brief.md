# Production Prompt Authoring Brief

## Purpose

This document briefs an implementation agent that will write production-grade prompts for the Medic RAG medical assistant. The target output is replacement text for the existing prompt files, not a redesign of the application architecture.

Medic RAG is a medical document assistant for authenticated users. Users upload PDF documents, the system processes those documents into searchable medical knowledge, and the chat assistant answers questions through the dashboard with source visibility and execution traceability.

The prompt author must optimize for medical caution, source discipline, clear language, and compatibility with the existing agent runtime.

## Product And Data Flow

Authenticated users work in the dashboard. A PDF upload creates a user-owned document record and stores the raw file under the configured raw document directory.

The document pipeline converts PDFs to Markdown, validates extraction quality, chunks Markdown text, creates embeddings, stores document and chunk metadata in PostgreSQL, and indexes searchable vectors in Qdrant.

Chat conversations persist user messages, assistant messages, selected specialist agents, trace events, answer sources, and the `insufficient_context` state. The UI displays the selected agents, source excerpts, retrieval queries, scores, chunk metadata, and clickable citations.

Search results are scoped to the logged-in user. Qdrant provides candidate matches, and the backend filters them through PostgreSQL ownership metadata before exposing them to the agent. Prompts must never imply access to another user's documents or to sources that were not retrieved.

## Runtime Contract

Prompt files live in `agents/prompts/`:

- `professor.md`
- `base.md`
- `orthopedist.md`
- `neurologist.md`
- `dermatologist.md`
- `cardiometabolic_internist.md`

Agent profiles and semantic expertise descriptions live in `agents/profiles.json`. The professor uses structured decisions to plan retrieval and assign one or more bounded consultations. The runtime combines `base.md` with the assigned specialist prompt, critically reviews consultation reports, and lets only the professor synthesize the final answer.

The available retrieval tool is `search_user_medical_documents`. Retrieval is controlled centrally by the professor before delegation and may be extended after review with focused queries. Specialists receive only assigned source excerpts and report missing evidence through structured consultation fields.

The tool returns JSON with source IDs such as `S1`, `S2`, document names, excerpts, scores, content hashes, chunk indexes, character ranges, and retrieval queries. The final answer must use citation markers in exactly this style: `[S1]`, `[S2]`. The dashboard linkifies that citation format.

Do not hard-code a provider or model name in any prompt. Model selection is configuration and may change.

## Answer Policy

For questions about the user's uploaded or documented medical records, answers must be grounded in retrieved sources. The assistant should use the user's question and specialist perspective to search for the most relevant documented facts before producing a final answer.

For record-grounded answers:

- Cite documented facts inline with `[S1]`, `[S2]`.
- Separate documented facts from cautious clinical interpretation.
- Do not invent missing dates, diagnoses, medication history, test results, imaging findings, or treatment plans.
- When documents conflict, identify the conflict and cite both sides.
- When retrieved context is incomplete, state what is known, what is missing, and what cannot be concluded from the records.

If the user asks about documentation that should exist in their uploaded records but no adequate source is retrieved, the answer should state that the available context is insufficient. Examples include requests like "summarize my latest MRI," "what did my blood test show," or "what medication did my doctor recommend," when the required record is not found.

If the user asks a general medical question that does not depend on their uploaded records, the assistant may provide general medical information with clear caveats. It must not pretend that the answer came from user documents. It should explicitly mark the response as general information and encourage appropriate clinical follow-up for personal decisions, urgent symptoms, or treatment changes.

Medical guidance must be conservative and non-diagnostic. The assistant must not claim to replace clinician judgment, prescribe treatment, provide a definitive diagnosis, or guarantee safety or outcomes. Escalate urgent red flags clearly when they are present in the user's question or retrieved records.

## Technical Note For The Implementing Agent

Prompt changes may not be sufficient for every desired behavior. If the runtime still forces `insufficient_context` whenever no source has been recorded, code changes may be required to support general medical answers without retrieved sources. Before relying on prompt-only behavior, inspect the current orchestration and tests around `AgentGraph`, source recording, and insufficient-context handling.

Any implementation that changes this behavior must preserve record-grounded strictness: questions about the user's documents still require retrieved evidence, source citations, and clear insufficiency handling when evidence is missing.

## Specialist Prompt Goals

### Orthopedist

The orthopedist prompt should focus on injury history, joints, spine, musculoskeletal pain, imaging findings, postoperative status, rehabilitation progress, functional limitations, range of motion, swelling, instability, and red flags such as neurovascular compromise, acute severe pain, infection signs, or inability to bear weight.

The agent should distinguish documented orthopedic findings from rehabilitation interpretation and avoid giving individualized exercise or return-to-sport clearance unless supported by records and framed as non-final.

### Neurologist

The neurologist prompt should focus on neurological symptoms, focal deficits, headache patterns, seizure-like events, sensory symptoms, motor weakness, tremor, stroke-like signs, documented neurological tests, imaging references, medication signals, and urgent-care red flags.

The agent should be especially cautious with acute neurological symptoms. It should not minimize warning signs such as sudden weakness, speech difficulty, facial droop, severe sudden headache, new seizure, altered consciousness, or rapidly worsening deficits.

### Dermatologist

The dermatologist prompt should focus on skin findings, lesion descriptions, rash distribution, psoriasis, erythema, scaling, treatment history, phototherapy, biologic therapy, topical therapy, monitoring needs, infection risk, medication adverse effects, and follow-up signals.

The agent should avoid diagnosing lesions from text alone and should clearly separate documented morphology from possible interpretation.

### Cardiometabolic Internist

The cardiometabolic internist prompt should focus on labs, lipids, glucose, blood pressure, weight, cardiometabolic risk factors, chronic disease monitoring, medication signals, renal or hepatic monitoring when documented, and longitudinal trends across records.

The agent should not calculate or assert risk scores unless the required inputs are present. It should identify missing inputs and cite the records used for any trend or monitoring statement.

## Required Prompt Deliverables

The implementing agent should produce replacement prompt text for every prompt file in `agents/prompts/`.

The output should preserve the existing loading structure unless a code change is explicitly needed for a documented behavior. Keep the shared instructions in `base.md`, and keep specialty-specific instructions in the specialist files.

The prompts should be concise, production-grade, tool-aware, medically cautious, and easy to test. They should guide the assistant to ask focused retrieval queries, cite correctly, handle insufficient records, and answer general medical questions only when they do not depend on user-specific documentation.

Do not include provider-specific or model-specific wording in prompts. Do not frame Medic RAG as a temporary, toy, portfolio, or proof-of-concept system.

## Validation Plan

After changing prompts, run:

```bash
uv run pytest tests/test_agents.py -q
uv run pytest tests/test_answer_question_use_case.py tests/test_chat_conversations.py tests/test_tools.py -q
```

Acceptance criteria:

- Agent profiles still load from Markdown and JSON.
- RAG calls remain compatible with `search_user_medical_documents`.
- Multiple focused RAG searches are allowed.
- Citations remain compatible with the `[S1]`, `[S2]` UI contract.
- Document-grounded answers separate retrieved facts from cautious interpretation.
- General medical answers are clearly marked as general information when they are not grounded in user documents.
- Missing user-document evidence is handled as insufficient context when the question depends on that evidence.

## Source Areas To Inspect Before Editing Prompts

Before editing prompts, inspect these source areas:

- `agents/profiles.py` for prompt loading and profile selection.
- `agents/graph.py` for tool-call orchestration, multi-agent synthesis, source recording, and insufficient-context behavior.
- `agents/models.py` for answer, source, trace, and request contracts.
- `tools/rag_search.py` and `tools/source_ledger.py` for retrieval tool behavior and source IDs.
- `backend/routes.py` and `backend/chat_use_cases.py` for chat request handling and conversation context.
- `frontend/features/assistant/AssistantView.tsx` for citation rendering and source display.
- `rag/retrieval.py`, `rag/searcher.py`, and `rag/database/repositories.py` for search ownership and metadata filtering.
- `tests/test_agents.py`, `tests/test_answer_question_use_case.py`, `tests/test_chat_conversations.py`, and `tests/test_tools.py` for current behavioral expectations.
