import type { JSX } from "preact";
import { useEffect, useState } from "preact/hooks";
import { navigate } from "../../app/router";
import { api, ApiError, jsonRequest } from "../../shared/api/client";
import type { DashboardStatus, SearchResult } from "../../shared/api/types";
import {
  Button,
  EmptyState,
  ErrorState,
  StatusBadge,
} from "../../shared/ui";
import styles from "../features.module.css";

interface SearchResponse {
  query: string;
  limit: number;
  elapsed_ms: number;
  results: SearchResult[];
}

export function RetrievalView(): JSX.Element {
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(10);
  const [data, setData] = useState<SearchResponse | null>(null);
  const [health, setHealth] = useState<DashboardStatus | null>(null);
  const [error, setError] = useState<{ title: string; message: string } | null>(
    null,
  );
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    void api<DashboardStatus>("/api/status")
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const search = async (): Promise<void> => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      setData(
        await api<SearchResponse>(
          "/api/search",
          jsonRequest("POST", { query, limit }),
        ),
      );
    } catch (caught) {
      const message =
        caught instanceof Error ? caught.message : "Retrieval request failed";
      setError({
        title:
          caught instanceof ApiError && caught.status === 503
            ? "Qdrant is unavailable"
            : "Retrieval failed",
        message,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div class={styles.stack}>
      <section class={styles.sectionHeader}>
        <div>
          <h2>Retrieval inspector</h2>
          <p>Test exactly what the agent can retrieve from your indexed documents.</p>
        </div>
      </section>
      <section class={styles.card}>
        <form
          class={styles.toolbar}
          onSubmit={(event) => {
            event.preventDefault();
            void search();
          }}
        >
          <label class={styles.field}>
            Search query
            <input
              class={styles.input}
              value={query}
              placeholder="e.g. ACL progression criteria"
              onInput={(event) => setQuery(event.currentTarget.value)}
            />
          </label>
          <label class={styles.field}>
            Result limit
            <select
              class={styles.select}
              value={limit}
              onChange={(event) => setLimit(Number(event.currentTarget.value))}
            >
              {[5, 10, 15, 20].map((value) => (
                <option value={value} key={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>
          <div class={styles.field}>
            <span>&nbsp;</span>
            <Button type="submit" disabled={loading || !query.trim()}>
              {loading ? "Searching…" : "Run retrieval"}
            </Button>
          </div>
        </form>
      </section>
      {health && !health.qdrant.available ? (
        <ErrorState
          title="Qdrant is offline"
          message={health.qdrant.error ?? "The vector index is unavailable."}
          retry={() => window.location.reload()}
        />
      ) : null}
      {error ? (
        <ErrorState
          title={error.title}
          message={error.message}
          retry={() => void search()}
        />
      ) : null}
      {data ? (
        <section class={styles.card}>
          <div class={styles.sectionHeader}>
            <div>
              <h3>Ranked results</h3>
              <p>
                {data.results.length} result(s) in {data.elapsed_ms.toFixed(2)} ms
              </p>
            </div>
            <StatusBadge status={data.results.length ? "succeeded" : "skipped"} />
          </div>
          {!data.results.length ? (
            health?.qdrant.points_count === 0 ? (
              <EmptyState title="The index is empty">
                Run the document pipeline before testing retrieval.
              </EmptyState>
            ) : (
              <EmptyState title="No matching chunks">
                The index is available, but this query returned no results.
              </EmptyState>
            )
          ) : (
            <ol class={styles.list}>
              {data.results.map((result, index) => (
                <li
                  class={styles.listItem}
                  key={result.qdrant_point_id ?? `${result.source}-${index}`}
                >
                  <div class={styles.itemHeader}>
                    <div>
                      <strong>
                        #{index + 1} {result.document_name ?? result.source ?? "Unknown source"}
                      </strong>
                      <div class={styles.muted}>
                        score {result.score?.toFixed(4) ?? "—"} · chunk{" "}
                        {result.chunk_index ?? "—"} · chars{" "}
                        {result.char_start ?? "—"}–{result.char_end ?? "—"}
                      </div>
                    </div>
                    {result.document_id ? (
                      <Button
                        variant="secondary"
                        onClick={() =>
                          navigate(
                            `/documents?document=${result.document_id}&tab=chunks${
                              result.chunk_index !== null
                                ? `&chunk=${result.chunk_index}`
                                : ""
                            }`,
                          )
                        }
                      >
                        Open chunk
                      </Button>
                    ) : null}
                  </div>
                  <p>{result.excerpt}</p>
                  <div class={styles.metadata}>
                    <div>
                      <dt>Qdrant point</dt>
                      <dd class={styles.mono}>{result.qdrant_point_id ?? "—"}</dd>
                    </div>
                    <div>
                      <dt>Content hash</dt>
                      <dd class={styles.mono}>{result.content_hash ?? "—"}</dd>
                    </div>
                  </div>
                </li>
              ))}
            </ol>
          )}
        </section>
      ) : (
        <EmptyState title="Run a retrieval query">
          Results expose ranking, ownership metadata and the exact source chunk.
        </EmptyState>
      )}
    </div>
  );
}
