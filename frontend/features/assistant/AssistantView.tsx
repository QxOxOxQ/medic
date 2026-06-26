import type { ComponentChildren, JSX } from "preact";
import { useCallback, useEffect, useRef, useState } from "preact/hooks";
import { navigate } from "../../app/router";
import { api, jsonRequest } from "../../shared/api/client";
import type {
  ChatMessage,
  ChatRun,
  Conversation,
  ConversationSummary,
  Source,
  TraceEvent,
} from "../../shared/api/types";
import {
  Alert,
  Button,
  Drawer,
  EmptyState,
  ErrorState,
  LoadingState,
  StatusBadge,
} from "../../shared/ui";
import styles from "../features.module.css";

export function AssistantView(): JSX.Element {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [question, setQuestion] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [answering, setAnswering] = useState(false);
  const [liveTrace, setLiveTrace] = useState<TraceEvent[]>([]);
  const [source, setSource] = useState<Source | null>(null);
  const [connectionWarning, setConnectionWarning] = useState("");
  const streamRef = useRef<EventSource | null>(null);

  useEffect(
    () => () => {
      streamRef.current?.close();
    },
    [],
  );

  const loadConversations = useCallback(async () => {
    try {
      const payload = await api<{ conversations: ConversationSummary[] }>(
        "/api/chat/conversations",
      );
      setConversations(payload.conversations);
      if (!conversation && payload.conversations[0]) {
        const detail = await api<{ conversation: Conversation }>(
          `/api/chat/conversations/${payload.conversations[0].id}`,
        );
        setConversation(detail.conversation);
      }
    } catch (caught) {
      setError(
        caught instanceof Error ? caught.message : "Could not load conversations",
      );
    } finally {
      setLoading(false);
    }
  }, [conversation]);

  useEffect(() => void loadConversations(), []);

  const openConversation = async (id: string): Promise<void> => {
    setError("");
    try {
      const payload = await api<{ conversation: Conversation }>(
        `/api/chat/conversations/${id}`,
      );
      setConversation(payload.conversation);
      setLiveTrace([]);
      setSource(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load conversation");
    }
  };

  const submit = async (): Promise<void> => {
    const normalized = question.trim();
    if (!normalized || answering) return;
    setAnswering(true);
    setError("");
    setLiveTrace([]);
    try {
      const payload = await api<{
        run: { conversation_id: string; run_id: string };
      }>(
        "/api/chat/runs",
        jsonRequest("POST", {
          question: normalized,
          conversation_id: conversation?.id ?? null,
          limit: 5,
        }),
      );
      setQuestion("");
      connectRun(payload.run.run_id, normalized);
    } catch (caught) {
      setAnswering(false);
      setError(caught instanceof Error ? caught.message : "Could not start agent");
    }
  };

  const connectRun = (runId: string, submittedQuestion: string): void => {
    streamRef.current?.close();
    const stream = new EventSource(`/api/chat/runs/${runId}/events`);
    streamRef.current = stream;
    stream.onopen = () => setConnectionWarning("");
    stream.addEventListener("trace", (event) => {
      const trace = JSON.parse((event as MessageEvent).data) as TraceEvent;
      setLiveTrace((current) =>
        current.some((item) => item.sequence === trace.sequence)
          ? current
          : [...current, trace].sort((a, b) => a.sequence - b.sequence),
      );
    });
    stream.addEventListener("done", (event) => {
      const run = JSON.parse((event as MessageEvent).data) as ChatRun;
      if (run.conversation) setConversation(run.conversation);
      if (run.error) {
        setError(run.error);
        setQuestion(submittedQuestion);
      }
      setAnswering(false);
      stream.close();
      streamRef.current = null;
      void loadConversations();
    });
    stream.onerror = () => {
      setConnectionWarning(
        "Live connection interrupted. Reconnecting automatically…",
      );
    };
  };

  if (loading) return <LoadingState rows={8} />;

  return (
    <div class={styles.stack}>
      <section class={styles.sectionHeader}>
        <div>
          <h2>Source-grounded assistant</h2>
          <p>Watch agent phases live, then verify every cited chunk.</p>
        </div>
        <Button
          variant="secondary"
          onClick={() => {
            setConversation(null);
            setLiveTrace([]);
            setSource(null);
          }}
        >
          New conversation
        </Button>
      </section>
      {connectionWarning ? (
        <Alert title="Reconnecting">{connectionWarning}</Alert>
      ) : null}
      {error ? (
        <ErrorState
          message={error}
          retry={question.trim() && !answering ? () => void submit() : undefined}
        />
      ) : null}
      <section class={styles.chatLayout}>
        <aside class={`${styles.card} ${styles.conversationList}`}>
          {!conversations.length ? (
            <p class={styles.muted}>No saved conversations.</p>
          ) : (
            conversations.map((item) => (
              <button
                type="button"
                class={`${styles.conversation} ${
                  conversation?.id === item.id ? styles.conversationActive : ""
                }`}
                onClick={() => void openConversation(item.id)}
                key={item.id}
              >
                <strong>{item.title}</strong>
                <span class={styles.muted}>{item.message_count} messages</span>
              </button>
            ))
          )}
        </aside>
        <div class={styles.stack}>
          <section class={`${styles.card} ${styles.messages}`} aria-live="polite">
            {!conversation?.messages.length ? (
              <EmptyState title="Ask from your indexed documents">
                The response will show selected specialists, retrieval phases,
                citations and source evidence.
              </EmptyState>
            ) : (
              conversation.messages.map((message) => (
                <Message
                  message={message}
                  openSource={setSource}
                  key={message.id}
                />
              ))
            )}
            {answering ? (
              <article class={styles.message}>
                <div class={styles.itemHeader}>
                  <strong>Agent execution</strong>
                  <StatusBadge status="running" />
                </div>
                <Trace events={liveTrace} empty="Waiting for coordinator…" />
              </article>
            ) : null}
          </section>
          <form
            class={styles.composer}
            onSubmit={(event) => {
              event.preventDefault();
              void submit();
            }}
          >
            <textarea
              aria-label="Question"
              value={question}
              placeholder="Ask a question based on the indexed documentation"
              disabled={answering}
              onInput={(event) => setQuestion(event.currentTarget.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void submit();
                }
              }}
            />
            <Button type="submit" disabled={answering || !question.trim()}>
              {answering ? "Working…" : "Ask"}
            </Button>
          </form>
        </div>
      </section>
      {source ? (
        <SourceDrawer source={source} close={() => setSource(null)} />
      ) : null}
    </div>
  );
}

function Message({
  message,
  openSource,
}: {
  message: ChatMessage;
  openSource: (source: Source) => void;
}): JSX.Element {
  const selectedAgents = message.trace_events
    .filter((event) => event.event_type === "coordinator")
    .flatMap((event) => {
      const agents = event.payload.selected_agents;
      return Array.isArray(agents) ? agents.map(String) : [];
    });
  const models = modelsByAgent(message.trace_events);
  const expanded = expandedSourceIds(message.trace_events);
  const agents = uniqueStrings([
    ...(models.has("professor") ? ["professor"] : []),
    ...selectedAgents,
  ]);
  const usedSources = message.sources.filter((source) => source.used);
  const visibleSources = usedSources.length ? usedSources : message.sources;
  const hiddenSources = usedSources.length
    ? message.sources.filter((source) => !source.used)
    : [];
  return (
    <article
      class={`${styles.message} ${
        message.role === "user" ? styles.messageUser : ""
      }`}
    >
      <div class={styles.itemHeader}>
        <strong>{message.role === "user" ? "You" : "Medical agent"}</strong>
        <time class={styles.muted}>
          {new Date(message.created_at).toLocaleTimeString()}
        </time>
      </div>
      {agents.length ? (
        <div class={styles.actions}>
          {agents.map((agent) => (
            <span class={styles.agentChip} key={agent}>
              <StatusBadge status={agent.replaceAll("_", " ")} />
              {models.get(agent) ? (
                <span class={styles.agentModel}>{models.get(agent)}</span>
              ) : null}
            </span>
          ))}
        </div>
      ) : null}
      {message.insufficient_context ? (
        <StatusBadge status="insufficient context" />
      ) : null}
      <p>{citationContent(message, openSource)}</p>
      {message.sources.length ? (
        <div class={styles.trace}>
          <strong>Sources</strong>
          {visibleSources.map((source) => (
            <SourceButton
              source={source}
              expanded={expanded.has(source.source_id)}
              openSource={openSource}
              key={source.id}
            />
          ))}
          {hiddenSources.length ? (
            <details>
              <summary>Checked but not used ({hiddenSources.length})</summary>
              {hiddenSources.map((source) => (
                <SourceButton
                  source={source}
                  expanded={expanded.has(source.source_id)}
                  openSource={openSource}
                  key={source.id}
                />
              ))}
            </details>
          ) : null}
        </div>
      ) : null}
      {message.trace_events.length ? (
        <details class={styles.trace}>
          <summary>Answer trace ({message.trace_events.length})</summary>
          <Trace events={message.trace_events} />
        </details>
      ) : null}
    </article>
  );
}

function SourceButton({
  source,
  expanded,
  openSource,
}: {
  source: Source;
  expanded: boolean;
  openSource: (source: Source) => void;
}): JSX.Element {
  return (
    <button
      type="button"
      class={styles.sourceButton}
      onClick={() => openSource(source)}
    >
      [{source.source_id}] {source.document_name ?? source.source} · score{" "}
      {source.score?.toFixed(3) ?? "—"}
      {expanded ? <span class={styles.readInFull}> · read in full</span> : null}
    </button>
  );
}

function modelsByAgent(events: TraceEvent[]): Map<string, string> {
  const models = new Map<string, string>();
  for (const event of events) {
    if (event.event_type !== "model_call" || !event.agent_name) continue;
    const model = payloadString(event.payload, "model");
    if (model && !models.has(event.agent_name)) {
      models.set(event.agent_name, model);
    }
  }
  return models;
}

function expandedSourceIds(events: TraceEvent[]): Set<string> {
  const event = events.find((item) => item.event_type === "source_expansion");
  const ids = event?.payload.expanded_source_ids;
  return new Set(Array.isArray(ids) ? ids.map(String) : []);
}

function payloadString(
  payload: Record<string, unknown>,
  key: string,
): string | null {
  const value = payload[key];
  return typeof value === "string" ? value : null;
}

function uniqueStrings(values: string[]): string[] {
  const seen: string[] = [];
  for (const value of values) {
    if (value && !seen.includes(value)) seen.push(value);
  }
  return seen;
}

function citationContent(
  message: ChatMessage,
  openSource: (source: Source) => void,
): ComponentChildren {
  return message.content
    .split(/(\[[^\]]*?S\d+[^\]]*?\])/g)
    .map((part, index) => {
      if (!/^\[[^\]]*?S\d+[^\]]*?\]$/.test(part)) return part;
      return (
        <span key={`cite-${index}`}>
          {part.split(/(S\d+)/g).map((token, tokenIndex) => {
            const source = /^S\d+$/.test(token)
              ? message.sources.find((item) => item.source_id === token)
              : null;
            if (!source) return token;
            return (
              <button
                type="button"
                class={styles.citation}
                onClick={() => openSource(source)}
                key={`cite-${index}-${tokenIndex}`}
              >
                {token}
              </button>
            );
          })}
        </span>
      );
    });
}

function Trace({
  events,
  empty = "No trace events.",
}: {
  events: TraceEvent[];
  empty?: string;
}): JSX.Element {
  if (!events.length) return <span class={styles.muted}>{empty}</span>;
  return (
    <ol class={styles.list}>
      {events.map((event) => (
        <li class={styles.listItem} key={event.sequence}>
          <div class={styles.itemHeader}>
            <strong>{event.title}</strong>
            <StatusBadge status={event.status} />
          </div>
          <span class={styles.muted}>
            {event.phase} · {event.agent_name ?? "system"}
            {payloadString(event.payload, "model")
              ? ` · ${payloadString(event.payload, "model")}`
              : ""}
            {event.tool_name ? ` · ${event.tool_name}` : ""}
          </span>
          {Object.keys(event.payload).length ? (
            <details>
              <summary>Payload</summary>
              <pre>{JSON.stringify(event.payload, null, 2)}</pre>
            </details>
          ) : null}
        </li>
      ))}
    </ol>
  );
}

function SourceDrawer({
  source,
  close,
}: {
  source: Source;
  close: () => void;
}): JSX.Element {
  return (
    <Drawer title={`Source ${source.source_id}`} close={close}>
      <div class={styles.details}>
        <dl class={styles.metadata}>
          <div>
            <dt>Document</dt>
            <dd>{source.document_name ?? source.source ?? "Unknown"}</dd>
          </div>
          <div>
            <dt>Retrieval query</dt>
            <dd>{source.retrieval_query ?? "—"}</dd>
          </div>
          <div>
            <dt>Score</dt>
            <dd>{source.score?.toFixed(4) ?? "—"}</dd>
          </div>
          <div>
            <dt>Chunk</dt>
            <dd>{source.chunk_index ?? "—"}</dd>
          </div>
          <div>
            <dt>Character range</dt>
            <dd>
              {source.char_start ?? "—"}–{source.char_end ?? "—"}
            </dd>
          </div>
          <div>
            <dt>Qdrant point</dt>
            <dd class={styles.mono}>{source.qdrant_point_id ?? "—"}</dd>
          </div>
        </dl>
        <article class={styles.card}>
          <p>{source.excerpt}</p>
        </article>
        {source.document_id ? (
          <Button
            onClick={() =>
              navigate(
                `/documents?document=${source.document_id}&tab=chunks${
                  source.chunk_index !== null
                    ? `&chunk=${source.chunk_index}`
                    : ""
                }`,
              )
            }
          >
            Open source document
          </Button>
        ) : null}
      </div>
    </Drawer>
  );
}
