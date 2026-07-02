import type { components } from "./schema";


export type DocumentRecord = components["schemas"]["DocumentDto"];
export type DocumentPage = components["schemas"]["DocumentPageResponse"];
export type DocumentUploadResult = components["schemas"]["UploadResultDto"];
export type DashboardStatus = components["schemas"]["DashboardStatusDto"];
export type PipelineEvent = components["schemas"]["PipelineEventDto"];
export type PipelineRunDocument = components["schemas"]["PipelineDocumentDto"];
export type PipelineRun = components["schemas"]["PipelineRunDto"];
export type TraceEvent = components["schemas"]["ChatTraceEventDto"];
export type Source = components["schemas"]["ChatSourceDto"];
export type ChatMessage = components["schemas"]["ChatMessageDto"];
export type ConversationSummary = components["schemas"]["ConversationSummaryDto"];
export type Conversation = components["schemas"]["ConversationDto"];
export type ChatRun = components["schemas"]["ChatRunDto"];
export type SearchResult = components["schemas"]["SearchResultDto"];
export type WorkspaceOverview =
  components["schemas"]["WorkspaceOverviewResponse"];
export type Chunk = components["schemas"]["ChunkDto"];
export type LLMProviderStats =
  components["schemas"]["LLMProviderStatsResponse"];

export interface ChatModelOption {
  key: string;
  label: string;
  model_id: string;
}

export interface ChatModelSettings {
  options: ChatModelOption[];
  selected: string;
}

export interface IndexPoint {
  id: string;
  source: string | null;
  content_hash: string | null;
  char_start: number | null;
  char_end: number | null;
  content: string;
  embeddings: Array<{
    vector_name: string;
    kind: string;
    dimensions: number;
    rows: number;
    sample: number[];
    indices_sample?: number[];
  }>;
}
