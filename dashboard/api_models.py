from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class PipelineRunCreateRequest(BaseModel):
    document_ids: list[UUID] = Field(default_factory=list)


class ChatRunCreateRequest(BaseModel):
    question: str
    conversation_id: UUID | None = None
    limit: int = Field(default=5, ge=1, le=20)
    specialist: str | None = None


class ChatModelOptionDto(BaseModel):
    key: str
    label: str
    model_id: str


class ChatModelSelectionRequest(BaseModel):
    key: str


class ChatModelSettingsResponse(BaseModel):
    ok: bool = True
    options: list[ChatModelOptionDto]
    selected: str


class DocumentDeleteRequest(BaseModel):
    document_ids: list[UUID] = Field(min_length=1)


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1, le=20)


class DocumentDto(BaseModel):
    id: UUID | None
    relative_raw_path: str
    original_filename: str
    display_name: str
    byte_size: int | None
    raw_exists: bool
    parsed_markdown_path: str | None
    parsed_exists: bool
    content_hash: str | None
    processed_at: datetime | None
    indexed: bool | None
    status: str
    processing_error: str | None = None
    indexed_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DocumentPageResponse(BaseModel):
    ok: bool = True
    documents: list[DocumentDto]
    page: int
    page_size: int
    total: int
    pages: int
    status_counts: dict[str, int]
    qdrant_error: str | None


class DocumentResponse(BaseModel):
    ok: bool = True
    document: DocumentDto


class DocumentMarkdownResponse(DocumentResponse):
    markdown: str | None


class ChunkDto(BaseModel):
    index: int
    char_start: int | None
    char_end: int | None
    characters: int
    content: str


class DocumentChunksResponse(DocumentResponse):
    chunks: list[ChunkDto]
    page: int
    page_size: int
    total: int


class IndexPreviewDto(BaseModel):
    available: bool
    collection_name: str | None
    collection_exists: bool
    preview_limit: int
    points: list[dict[str, Any]] = Field(default_factory=list)
    shown_points: int = 0
    error: str | None = None


class DocumentIndexResponse(DocumentResponse):
    index: IndexPreviewDto


class UploadResultDto(BaseModel):
    file_name: str
    status: str
    document_id: UUID | None = None
    relative_raw_path: str | None = None
    bytes: int | None = None
    error: str | None = None


class DocumentUploadResponse(BaseModel):
    ok: bool
    uploads: list[dict[str, Any]]
    results: list[UploadResultDto]
    uploaded_count: int
    failed_count: int


class PipelineDocumentDto(BaseModel):
    document_id: UUID | None
    position: int
    document_name: str
    relative_raw_path: str
    status: str
    current_step: str | None
    error: str | None


class PipelineEventDto(BaseModel):
    sequence: int
    timestamp: datetime
    step: str
    status: str
    message: str
    counters: dict[str, Any]
    result: dict[str, Any]


class PipelineRunDto(BaseModel):
    id: UUID
    status: str
    summary: str | None
    error: str | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime
    documents: list[PipelineDocumentDto]
    events: list[PipelineEventDto]


class PipelineRunResponse(BaseModel):
    ok: bool = True
    run: PipelineRunDto


class PipelineRunListResponse(BaseModel):
    ok: bool = True
    runs: list[PipelineRunDto]


class SearchResultDto(BaseModel):
    score: float | None
    source: str | None
    content_hash: str | None
    document_name: str | None
    excerpt: str
    qdrant_point_id: str | None
    document_id: UUID | None
    relative_raw_path: str | None
    chunk_index: int | None
    char_start: int | None
    char_end: int | None


class SearchResponse(BaseModel):
    ok: bool
    query: str
    limit: int
    elapsed_ms: float
    results: list[SearchResultDto]
    error: str | None = None


class ChatTraceEventDto(BaseModel):
    id: UUID | None = None
    sequence: int
    event_type: str
    phase: str
    title: str
    status: str
    agent_name: str | None
    tool_name: str | None
    payload: dict[str, Any]
    duration_ms: int | None
    created_at: datetime | None = None


class ChatSourceDto(BaseModel):
    id: UUID
    source_id: str
    source: str | None
    content_hash: str | None
    document_id: UUID | None
    document_name: str | None
    relative_raw_path: str | None
    qdrant_point_id: str | None
    chunk_index: int | None
    char_start: int | None
    char_end: int | None
    retrieval_query: str | None
    score: float | None
    excerpt: str
    used: bool = False


class ChatMessageDto(BaseModel):
    id: UUID
    role: str
    content: str
    sequence: int
    insufficient_context: bool
    created_at: datetime
    sources: list[ChatSourceDto]
    trace_events: list[ChatTraceEventDto]


class ConversationDto(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[ChatMessageDto]


class ConversationSummaryDto(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


class ChatRunStartedDto(BaseModel):
    conversation_id: UUID
    run_id: UUID


class ChatRunStartResponse(BaseModel):
    ok: bool = True
    run: ChatRunStartedDto


class ChatRunDto(BaseModel):
    id: UUID
    conversation_id: UUID
    status: str
    question: str
    error: str | None
    started_at: datetime
    finished_at: datetime | None
    trace_events: list[ChatTraceEventDto]
    conversation: ConversationDto | None


class ChatRunResponse(BaseModel):
    ok: bool = True
    run: ChatRunDto


class QdrantStatusDto(BaseModel):
    available: bool
    collection_name: str | None
    collection_exists: bool
    points_count: int | None
    error: str | None


class DashboardStatusDto(BaseModel):
    raw_pdf_count: int
    parsed_markdown_count: int
    document_count: int
    last_processed_at: datetime | None
    qdrant: QdrantStatusDto


class PostgresStatusDto(BaseModel):
    available: bool
    error: str | None


class WorkspaceOverviewResponse(BaseModel):
    ok: bool = True
    status: DashboardStatusDto
    postgres: PostgresStatusDto
    latest_pipeline_run: PipelineRunDto | None
    latest_conversation: ConversationSummaryDto | None
