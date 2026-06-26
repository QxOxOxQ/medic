import type { JSX } from "preact";
import { useCallback, useEffect, useState } from "preact/hooks";
import { navigate } from "../../app/router";
import { api } from "../../shared/api/client";
import type { WorkspaceOverview } from "../../shared/api/types";
import {
  Button,
  ErrorState,
  LoadingState,
  StatusBadge,
} from "../../shared/ui";
import styles from "../features.module.css";

const workflow = [
  ["Upload", "Add one or many source PDFs"],
  ["Prepare", "Extract readable markdown"],
  ["Index", "Create chunks and embeddings"],
  ["Ask", "Run retrieval-grounded agents"],
  ["Verify", "Inspect citations and source chunks"],
];

export function OverviewView(): JSX.Element {
  const [data, setData] = useState<WorkspaceOverview | null>(null);
  const [error, setError] = useState("");
  const load = useCallback(async () => {
    setError("");
    try {
      setData(await api<WorkspaceOverview>("/api/workspace/overview"));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load overview");
    }
  }, []);

  useEffect(() => void load(), [load]);

  if (error) return <ErrorState message={error} retry={load} />;
  if (!data) return <LoadingState rows={6} />;

  const indexed = data.status.qdrant.points_count ?? 0;
  return (
    <div class={styles.stack}>
      <section class={styles.sectionHeader}>
        <div>
          <h2>Operational workspace</h2>
          <p>Move from source documents to verifiable answers without losing context.</p>
        </div>
        <div class={styles.actions}>
          <Button onClick={() => navigate("/documents")}>Upload PDFs</Button>
          <Button variant="secondary" onClick={() => navigate("/assistant")}>
            Ask a question
          </Button>
        </div>
      </section>

      <section class={styles.grid4} aria-label="Workspace metrics">
        <Metric
          label="Documents"
          value={data.status.document_count}
          detail={`${data.status.raw_pdf_count} source PDFs`}
        />
        <Metric
          label="Prepared markdown"
          value={data.status.parsed_markdown_count}
          detail="Ready for chunk inspection"
        />
        <Metric
          label="Index points"
          value={indexed}
          detail={data.status.qdrant.collection_name ?? "No active collection"}
        />
        <article class={`${styles.card} ${styles.metric}`}>
          <span class={styles.metricLabel}>System readiness</span>
          <div class={styles.actions}>
            <StatusBadge
              status={data.postgres.available ? "succeeded" : "failed"}
            />
            <span>PostgreSQL</span>
          </div>
          <div class={styles.actions}>
            <StatusBadge
              status={data.status.qdrant.available ? "succeeded" : "failed"}
            />
            <span>Qdrant</span>
          </div>
        </article>
      </section>

      <section class={styles.card}>
        <div class={styles.sectionHeader}>
          <div>
            <h3>Transparent workflow</h3>
            <p>Every stage remains inspectable, including failures and source evidence.</p>
          </div>
        </div>
        <ol class={styles.workflow}>
          {workflow.map(([title, detail], index) => (
            <li class={styles.workflowStep} key={title}>
              <span class={styles.workflowNumber}>{index + 1}</span>
              <div>
                <strong>{title}</strong>
                <span>{detail}</span>
              </div>
            </li>
          ))}
        </ol>
      </section>

      <section class={styles.grid2}>
        <article class={styles.card}>
          <div class={styles.sectionHeader}>
            <div>
              <h3>Latest pipeline run</h3>
              <p>Persistent status and event history</p>
            </div>
            {data.latest_pipeline_run ? (
              <StatusBadge status={data.latest_pipeline_run.status} />
            ) : null}
          </div>
          {data.latest_pipeline_run ? (
            <>
              <p>{data.latest_pipeline_run.summary ?? "Run in progress"}</p>
              <Button variant="secondary" onClick={() => navigate("/pipeline")}>
                Open pipeline
              </Button>
            </>
          ) : (
            <p class={styles.muted}>No pipeline runs yet.</p>
          )}
        </article>
        <article class={styles.card}>
          <div class={styles.sectionHeader}>
            <div>
              <h3>Latest conversation</h3>
              <p>Source-grounded medical-documentation assistant</p>
            </div>
          </div>
          {data.latest_conversation ? (
            <>
              <p>{data.latest_conversation.title}</p>
              <Button variant="secondary" onClick={() => navigate("/assistant")}>
                Continue conversation
              </Button>
            </>
          ) : (
            <p class={styles.muted}>No saved conversations yet.</p>
          )}
        </article>
      </section>
    </div>
  );
}

function Metric({
  label,
  value,
  detail,
}: {
  label: string;
  value: number;
  detail: string;
}): JSX.Element {
  return (
    <article class={`${styles.card} ${styles.metric}`}>
      <span class={styles.metricLabel}>{label}</span>
      <strong class={styles.metricValue}>{value}</strong>
      <span class={styles.metricDetail}>{detail}</span>
    </article>
  );
}
