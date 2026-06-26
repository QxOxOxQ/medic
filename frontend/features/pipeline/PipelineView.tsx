import type { JSX } from "preact";
import { useCallback, useEffect, useMemo, useState } from "preact/hooks";
import { api, jsonRequest } from "../../shared/api/client";
import type { PipelineEvent, PipelineRun } from "../../shared/api/types";
import {
  Alert,
  Button,
  EmptyState,
  ErrorState,
  LoadingState,
  StatusBadge,
} from "../../shared/ui";
import styles from "../features.module.css";

const steps = ["discover", "prepare", "chunk", "embed", "index", "pipeline"];

export function PipelineView(): JSX.Element {
  const [runs, setRuns] = useState<PipelineRun[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [connectionWarning, setConnectionWarning] = useState("");
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setError("");
    try {
      const payload = await api<{ runs: PipelineRun[] }>("/api/pipeline-runs");
      setRuns(payload.runs);
      setSelectedId((current) => current ?? payload.runs[0]?.id ?? null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load pipeline");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => void load(), [load]);

  const selected = runs.find((run) => run.id === selectedId) ?? null;

  useEffect(() => {
    if (!selected || !["queued", "running"].includes(selected.status)) return;
    const source = new EventSource(`/api/pipeline-runs/${selected.id}/events`);
    source.onopen = () => setConnectionWarning("");
    source.addEventListener("progress", (event) => {
      const payload = JSON.parse((event as MessageEvent).data) as PipelineEvent;
      setRuns((current) =>
        current.map((run) =>
          run.id === selected.id
            ? {
                ...run,
                status: "running",
                events: mergeEvents(run.events, payload),
              }
            : run,
        ),
      );
    });
    source.addEventListener("done", (event) => {
      const payload = JSON.parse((event as MessageEvent).data) as PipelineRun;
      setRuns((current) =>
        current.map((run) => (run.id === payload.id ? payload : run)),
      );
      source.close();
      void load();
    });
    source.onerror = () => {
      setConnectionWarning(
        "Live connection interrupted. Reconnecting automatically…",
      );
    };
    return () => source.close();
  }, [selected?.id, selected?.status, load]);

  const startAll = async (): Promise<void> => {
    setError("");
    try {
      const payload = await api<{ run: PipelineRun }>(
        "/api/pipeline-runs",
        jsonRequest("POST", { document_ids: [] }),
      );
      setRuns((current) => [payload.run, ...current]);
      setSelectedId(payload.run.id);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not start pipeline");
    }
  };

  if (loading) return <LoadingState rows={8} />;

  return (
    <div class={styles.stack}>
      <section class={styles.sectionHeader}>
        <div>
          <h2>Pipeline control center</h2>
          <p>Persistent runs, live events and per-document outcomes.</p>
        </div>
        <Button
          disabled={runs.some((run) => ["queued", "running"].includes(run.status))}
          onClick={() => void startAll()}
        >
          Run all eligible documents
        </Button>
      </section>
      {error ? <ErrorState message={error} retry={load} /> : null}
      {connectionWarning ? (
        <Alert title="Reconnecting">{connectionWarning}</Alert>
      ) : null}
      {!runs.length ? (
        <EmptyState
          title="No pipeline runs"
          action={<Button onClick={() => void startAll()}>Start first run</Button>}
        >
          Select documents in the document workspace or process all eligible files.
        </EmptyState>
      ) : (
        <section class={styles.grid2}>
          <article class={styles.card}>
            <div class={styles.sectionHeader}>
              <div>
                <h3>Run history</h3>
                <p>Stored in PostgreSQL and available after restart.</p>
              </div>
            </div>
            <div class={styles.list}>
              {runs.map((run) => (
                <button
                  type="button"
                  class={`${styles.listItem} ${
                    run.id === selectedId ? styles.conversationActive : ""
                  }`}
                  onClick={() => setSelectedId(run.id)}
                  key={run.id}
                >
                  <div class={styles.itemHeader}>
                    <StatusBadge status={run.status} />
                    <span class={styles.muted}>{formatDate(run.created_at)}</span>
                  </div>
                  <span>{run.summary ?? run.error ?? `${run.events.length} events`}</span>
                  <code class={styles.mono}>{run.id}</code>
                </button>
              ))}
            </div>
          </article>
          <article class={styles.card}>
            {selected ? <RunSummary run={selected} /> : null}
          </article>
        </section>
      )}
      {selected ? (
        <>
          <RunSteps run={selected} />
          <section class={styles.grid2}>
            <article class={styles.card}>
              <div class={styles.sectionHeader}>
                <div>
                  <h3>Documents in this run</h3>
                  <p>Snapshot and current processing outcome.</p>
                </div>
              </div>
              {!selected.documents.length ? (
                <Alert title="All eligible documents">
                  The run was started without an explicit document selection.
                </Alert>
              ) : (
                <ul class={styles.list}>
                  {selected.documents.map((document) => (
                    <li class={styles.listItem} key={document.position}>
                      <div class={styles.itemHeader}>
                        <strong>{document.document_name}</strong>
                        <StatusBadge status={document.status} />
                      </div>
                      <span class={styles.muted}>
                        {document.current_step ?? "Waiting"}
                      </span>
                      {document.error ? <span>{document.error}</span> : null}
                    </li>
                  ))}
                </ul>
              )}
            </article>
            <article class={styles.card}>
              <div class={styles.sectionHeader}>
                <div>
                  <h3>Event timeline</h3>
                  <p>Ordered, durable progress emitted by every pipeline stage.</p>
                </div>
              </div>
              <EventTimeline events={selected.events} />
            </article>
          </section>
        </>
      ) : null}
    </div>
  );
}

function RunSummary({ run }: { run: PipelineRun }): JSX.Element {
  const completedDocuments = run.documents.filter((document) =>
    ["succeeded", "failed", "skipped"].includes(document.status),
  ).length;
  const completedSteps = stepState(run.events);
  const progress = run.documents.length
    ? Math.round((completedDocuments / run.documents.length) * 100)
    : Math.round(
        ([...completedSteps.values()].filter((status) =>
          ["succeeded", "failed", "skipped"].includes(status),
        ).length /
          steps.length) *
          100,
      );
  const currentDocument = run.documents.find(
    (document) => document.status === "running",
  );
  const latestCounters = run.events.at(-1)?.counters ?? {};
  return (
    <div class={styles.stack}>
      <div class={styles.itemHeader}>
        <div>
          <span class={styles.metricLabel}>Selected run</span>
          <h3>{formatDate(run.created_at)}</h3>
        </div>
        <StatusBadge status={run.status} />
      </div>
      <dl class={styles.metadata}>
        <div>
          <dt>Progress</dt>
          <dd>{run.status === "succeeded" ? 100 : progress}%</dd>
        </div>
        <div>
          <dt>Current document</dt>
          <dd>{currentDocument?.document_name ?? "—"}</dd>
        </div>
        <div>
          <dt>Started</dt>
          <dd>{formatDate(run.started_at)}</dd>
        </div>
        <div>
          <dt>Finished</dt>
          <dd>{formatDate(run.finished_at)}</dd>
        </div>
        <div>
          <dt>Documents</dt>
          <dd>{run.documents.length || "All eligible"}</dd>
        </div>
        <div>
          <dt>Events</dt>
          <dd>{run.events.length}</dd>
        </div>
      </dl>
      <progress
        class={styles.progress}
        max={100}
        value={run.status === "succeeded" ? 100 : progress}
      />
      {Object.keys(latestCounters).length ? (
        <p class={styles.muted}>{formatPayload(latestCounters)}</p>
      ) : null}
      {run.summary ? <Alert title="Result">{run.summary}</Alert> : null}
      {run.error ? (
        <Alert title="Pipeline failed" error>
          {run.error}
        </Alert>
      ) : null}
    </div>
  );
}

function RunSteps({ run }: { run: PipelineRun }): JSX.Element {
  const state = useMemo(() => stepState(run.events), [run.events]);
  return (
    <section class={styles.card}>
      <div class={styles.sectionHeader}>
        <div>
          <h3>Live processing stages</h3>
          <p>The current stage is derived directly from persisted events.</p>
        </div>
        <StatusBadge status={run.status} />
      </div>
      <div class={styles.stepper}>
        {steps.map((step) => {
          const status = state.get(step) ?? "queued";
          return (
            <article
              class={`${styles.step} ${
                status === "running"
                  ? styles.stepActive
                  : status === "succeeded"
                    ? styles.stepDone
                    : status === "failed"
                      ? styles.stepFailed
                      : ""
              }`}
              key={step}
            >
              <strong>{step === "pipeline" ? "Complete" : title(step)}</strong>
              <StatusBadge status={status} />
            </article>
          );
        })}
      </div>
    </section>
  );
}

function EventTimeline({ events }: { events: PipelineEvent[] }): JSX.Element {
  const [filter, setFilter] = useState("all");
  const visibleEvents =
    filter === "all"
      ? events
      : events.filter(
          (event) => event.step === filter || event.status === filter,
        );
  if (!events.length) {
    return <p class={styles.muted}>Waiting for the first pipeline event.</p>;
  }
  return (
    <div class={styles.stack}>
      <label class={styles.field}>
        Filter timeline
        <select
          class={styles.select}
          value={filter}
          onChange={(event) => setFilter(event.currentTarget.value)}
        >
          <option value="all">All events</option>
          {[...new Set(events.flatMap((event) => [event.step, event.status]))].map(
            (value) => (
              <option value={value} key={value}>
                {value}
              </option>
            ),
          )}
        </select>
      </label>
      {!visibleEvents.length ? (
        <p class={styles.muted}>No events match this filter.</p>
      ) : (
        <div class={styles.eventTimeline}>
          {visibleEvents.map((event) => (
            <article class={styles.event} key={event.sequence}>
              <time>{new Date(event.timestamp).toLocaleTimeString()}</time>
              <StatusBadge status={event.status} />
              <div>
                <strong>{event.message}</strong>
                <div class={styles.muted}>
                  {event.step} · {formatPayload(event.counters)}
                </div>
                {Object.keys(event.result).length ? (
                  <details>
                    <summary>Technical details</summary>
                    <pre>{JSON.stringify(event.result, null, 2)}</pre>
                  </details>
                ) : null}
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}

function stepState(events: PipelineEvent[]): Map<string, string> {
  const result = new Map<string, string>();
  for (const event of events) {
    result.set(event.step === "job" ? "pipeline" : event.step, event.status);
  }
  return result;
}

function mergeEvents(events: PipelineEvent[], next: PipelineEvent): PipelineEvent[] {
  if (events.some((event) => event.sequence === next.sequence)) return events;
  return [...events, next].sort((left, right) => left.sequence - right.sequence);
}

function title(value: string): string {
  return `${value.charAt(0).toUpperCase()}${value.slice(1)}`;
}

function formatDate(value: string | null): string {
  return value ? new Date(value).toLocaleString() : "—";
}

function formatPayload(payload: Record<string, unknown>): string {
  const entries = Object.entries(payload);
  return entries.length
    ? entries.map(([key, value]) => `${key}: ${String(value)}`).join(" · ")
    : "No counters";
}
