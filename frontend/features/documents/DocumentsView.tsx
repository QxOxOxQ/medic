import type { JSX } from "preact";
import { useCallback, useEffect, useMemo, useState } from "preact/hooks";
import { navigate } from "../../app/router";
import { api, jsonRequest, uploadRequest } from "../../shared/api/client";
import type {
  Chunk,
  DocumentPage,
  DocumentRecord,
  DocumentUploadResult,
  IndexPoint,
} from "../../shared/api/types";
import {
  Alert,
  Button,
  Dialog,
  Drawer,
  EmptyState,
  ErrorState,
  LoadingState,
  StatusBadge,
  Tabs,
} from "../../shared/ui";
import styles from "../features.module.css";

type DetailTab = "overview" | "markdown" | "chunks" | "index";

function initialFilters(): {
  page: number;
  query: string;
  status: string;
  sort: string;
  direction: string;
} {
  const params = new URLSearchParams(window.location.search);
  return {
    page: Number(params.get("page") ?? 1),
    query: params.get("query") ?? "",
    status: params.get("status") ?? "",
    sort: params.get("sort") ?? "updated_at",
    direction: params.get("direction") ?? "desc",
  };
}

export function DocumentsView(): JSX.Element {
  const [filters, setFilters] = useState(initialFilters);
  const [data, setData] = useState<DocumentPage | null>(null);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [files, setFiles] = useState<File[]>([]);
  const [uploadResults, setUploadResults] = useState<DocumentUploadResult[]>([]);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState("");
  const [deleteOpen, setDeleteOpen] = useState(false);
  const params = new URLSearchParams(window.location.search);
  const [detailId, setDetailId] = useState<string | null>(params.get("document"));
  const [detailTab, setDetailTab] = useState<DetailTab>(
    (params.get("tab") as DetailTab | null) ?? "overview",
  );
  const [targetChunk, setTargetChunk] = useState<number | null>(() => {
    const value = Number(params.get("chunk"));
    return Number.isInteger(value) && value >= 0 ? value : null;
  });

  const load = useCallback(async () => {
    setError("");
    const search = new URLSearchParams({
      page: String(Math.max(1, filters.page)),
      page_size: "25",
      sort: filters.sort,
      direction: filters.direction,
    });
    if (filters.query) search.set("query", filters.query);
    if (filters.status) search.set("status", filters.status);
    try {
      const payload = await api<DocumentPage>(`/api/documents?${search}`);
      setData(payload);
      setSelected((current) => {
        const visible = new Set(payload.documents.map((document) => document.id));
        return new Set([...current].filter((id) => visible.has(id)));
      });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load documents");
    }
  }, [filters]);

  useEffect(() => void load(), [load]);

  useEffect(() => {
    const search = new URLSearchParams();
    if (filters.page > 1) search.set("page", String(filters.page));
    if (filters.query) search.set("query", filters.query);
    if (filters.status) search.set("status", filters.status);
    if (filters.sort !== "updated_at") search.set("sort", filters.sort);
    if (filters.direction !== "desc") search.set("direction", filters.direction);
    if (detailId) search.set("document", detailId);
    if (detailId && detailTab !== "overview") search.set("tab", detailTab);
    if (detailId && detailTab === "chunks" && targetChunk !== null) {
      search.set("chunk", String(targetChunk));
    }
    window.history.replaceState(
      {},
      "",
      `/documents${search.size ? `?${search}` : ""}`,
    );
  }, [filters, detailId, detailTab, targetChunk]);

  const documents = data?.documents ?? [];
  const selectedDocuments = useMemo(
    () => documents.filter((document) => document.id && selected.has(document.id)),
    [documents, selected],
  );

  const updateTextFilter = (
    key: "query" | "status",
    value: string,
  ): void => {
    setFilters((current) => ({ ...current, [key]: value, page: 1 }));
  };

  const updatePage = (page: number): void => {
    setFilters((current) => ({ ...current, page }));
  };

  const toggle = (document: DocumentRecord): void => {
    if (!document.id) return;
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(document.id as string)) next.delete(document.id as string);
      else next.add(document.id as string);
      return next;
    });
  };

  const upload = async (): Promise<void> => {
    if (!files.length) return;
    setUploading(true);
    setMessage("");
    setUploadResults([]);
    try {
      const payload = await api<{
        uploaded_count: number;
        failed_count: number;
        results: DocumentUploadResult[];
      }>(
        "/api/documents/upload",
        uploadRequest(files),
      );
      setUploadResults(payload.results);
      setMessage(
        `${payload.uploaded_count} uploaded, ${payload.failed_count} failed.`,
      );
      setFiles([]);
      await load();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const openDocument = (documentId: string | null): void => {
    setDetailId(documentId);
    setDetailTab("overview");
    setTargetChunk(null);
  };

  const removeSelected = async (): Promise<void> => {
    try {
      await api(
        "/api/documents/delete-by-id",
        jsonRequest("POST", { document_ids: [...selected] }),
      );
      setDeleteOpen(false);
      setSelected(new Set());
      setMessage("Selected documents were deleted.");
      await load();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Delete failed");
    }
  };

  const runPipeline = async (): Promise<void> => {
    try {
      await api(
        "/api/pipeline-runs",
        jsonRequest("POST", { document_ids: [...selected] }),
      );
      navigate("/pipeline");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not start pipeline");
    }
  };

  return (
    <div class={styles.stack}>
      <section class={styles.sectionHeader}>
        <div>
          <h2>Document workspace</h2>
          <p>Upload, process and inspect every artifact created by the RAG pipeline.</p>
        </div>
      </section>

      {error ? <ErrorState message={error} retry={load} /> : null}
      {message ? <Alert title="Operation completed">{message}</Alert> : null}
      {uploadResults.length ? (
        <section class={styles.card} aria-label="Upload results">
          <h3>Upload queue results</h3>
          <ul class={styles.list}>
            {uploadResults.map((result) => (
              <li class={styles.listItem} key={result.file_name}>
                <div class={styles.itemHeader}>
                  <strong>{result.file_name}</strong>
                  <StatusBadge status={result.status} />
                </div>
                {result.error ? <span>{result.error}</span> : null}
              </li>
            ))}
          </ul>
        </section>
      ) : null}
      {data?.qdrant_error ? (
        <Alert title="Index status unavailable" error>
          <p>{data.qdrant_error}</p>
        </Alert>
      ) : null}

      <section class={styles.card}>
        <label
          class={styles.dropzone}
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault();
            setFiles([...(event.dataTransfer?.files ?? [])].filter(isPdf));
          }}
        >
          <input
            type="file"
            accept="application/pdf,.pdf"
            multiple
            onChange={(event) =>
              setFiles(
                [...(event.currentTarget.files ?? [])].filter(isPdf),
              )
            }
          />
          <div>
            <strong>Drop PDF files here or choose files</strong>
            <span>Files are validated before they enter the document pipeline.</span>
          </div>
        </label>
        {files.length ? (
          <div class={styles.details}>
            <ul class={styles.list}>
              {files.map((file) => (
                <li class={styles.listItem} key={`${file.name}-${file.size}`}>
                  <div class={styles.itemHeader}>
                    <strong>{file.name}</strong>
                    <span class={styles.muted}>{formatBytes(file.size)}</span>
                  </div>
                </li>
              ))}
            </ul>
            <div class={styles.actions}>
              <Button disabled={uploading} onClick={() => void upload()}>
                {uploading ? "Uploading…" : `Upload ${files.length} file(s)`}
              </Button>
              <Button variant="ghost" onClick={() => setFiles([])}>
                Clear
              </Button>
            </div>
          </div>
        ) : null}
      </section>

      <section class={styles.card}>
        <div class={styles.toolbar}>
          <label class={styles.field}>
            Search documents
            <input
              class={styles.input}
              value={filters.query}
              placeholder="PDF name"
              onInput={(event) =>
                updateTextFilter("query", event.currentTarget.value)
              }
            />
          </label>
          <label class={styles.field}>
            Status
            <select
              class={styles.select}
              value={filters.status}
              onChange={(event) =>
                updateTextFilter("status", event.currentTarget.value)
              }
            >
              <option value="">All statuses</option>
              {["raw", "prepared", "indexed", "failed", "stale"].map((status) => (
                <option value={status} key={status}>
                  {status} ({data?.status_counts[status] ?? 0})
                </option>
              ))}
            </select>
          </label>
          <label class={styles.field}>
            Sort
            <select
              class={styles.select}
              value={`${filters.sort}:${filters.direction}`}
              onChange={(event) => {
                const [sort, direction] = event.currentTarget.value.split(":");
                setFilters((current) => ({
                  ...current,
                  sort: sort ?? "updated_at",
                  direction: direction ?? "desc",
                  page: 1,
                }));
              }}
            >
              <option value="updated_at:desc">Recently updated</option>
              <option value="name:asc">Name A–Z</option>
              <option value="status:asc">Status</option>
              <option value="processed_at:desc">Recently processed</option>
            </select>
          </label>
        </div>

        {!data ? (
          <LoadingState />
        ) : !documents.length ? (
          <EmptyState title="No documents found">
            Upload a PDF or change the current filters.
          </EmptyState>
        ) : (
          <>
            <div class={styles.tableWrap}>
              <table class={styles.table}>
                <thead>
                  <tr>
                    <th>Select</th>
                    <th>Document</th>
                    <th>Status</th>
                    <th>Artifacts</th>
                    <th>Processed</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {documents.map((document) => (
                    <DocumentRow
                      key={document.id ?? document.relative_raw_path}
                      document={document}
                      selected={Boolean(document.id && selected.has(document.id))}
                      toggle={() => toggle(document)}
                      open={() => openDocument(document.id)}
                    />
                  ))}
                </tbody>
              </table>
            </div>
            <div class={styles.mobileCards}>
              {documents.map((document) => (
                <DocumentCard
                  key={document.id ?? document.relative_raw_path}
                  document={document}
                  selected={Boolean(document.id && selected.has(document.id))}
                  toggle={() => toggle(document)}
                  open={() => openDocument(document.id)}
                />
              ))}
            </div>
            <div class={styles.pagination}>
              <span>
                Page {data.page} of {data.pages} · {data.total} documents
              </span>
              <div class={styles.actions}>
                <Button
                  variant="secondary"
                  disabled={data.page <= 1}
                  onClick={() => updatePage(data.page - 1)}
                >
                  Previous
                </Button>
                <Button
                  variant="secondary"
                  disabled={data.page >= data.pages}
                  onClick={() => updatePage(data.page + 1)}
                >
                  Next
                </Button>
              </div>
            </div>
          </>
        )}
      </section>

      {selected.size ? (
        <aside class={styles.selectionBar} aria-label="Selected document actions">
          <strong>{selected.size} selected</strong>
          <div class={styles.actions}>
            <Button onClick={() => void runPipeline()}>Run pipeline</Button>
            <Button variant="danger" onClick={() => setDeleteOpen(true)}>
              Delete
            </Button>
            <Button variant="ghost" onClick={() => setSelected(new Set())}>
              Clear
            </Button>
          </div>
        </aside>
      ) : null}

      {deleteOpen ? (
        <Dialog
          title={`Delete ${selected.size} document(s)?`}
          close={() => setDeleteOpen(false)}
          actions={
            <>
              <Button variant="secondary" onClick={() => setDeleteOpen(false)}>
                Cancel
              </Button>
              <Button variant="danger" onClick={() => void removeSelected()}>
                Delete documents
              </Button>
            </>
          }
        >
          <p>
            This removes the source PDF, generated markdown, database records,
            chunks and matching Qdrant points.
          </p>
          <ul>
            {selectedDocuments.map((document) => (
              <li key={document.id}>{document.display_name}</li>
            ))}
          </ul>
        </Dialog>
      ) : null}

      {detailId ? (
        <DocumentDrawer
          documentId={detailId}
          tab={detailTab}
          setTab={(nextTab) => {
            setDetailTab(nextTab);
            if (nextTab !== "chunks") setTargetChunk(null);
          }}
          targetChunk={targetChunk}
          close={() => setDetailId(null)}
        />
      ) : null}
    </div>
  );
}

function DocumentRow({
  document,
  selected,
  toggle,
  open,
}: DocumentItemProps): JSX.Element {
  return (
    <tr key={document.id ?? document.relative_raw_path}>
      <td>
        <input
          type="checkbox"
          checked={selected}
          aria-label={`Select ${document.display_name}`}
          onChange={toggle}
        />
      </td>
      <td>
        <div class={styles.documentName}>
          <strong>{document.display_name}</strong>
          <span>{formatBytes(document.byte_size)}</span>
        </div>
      </td>
      <td>
        <StatusBadge status={document.status} />
      </td>
      <td>
        {document.parsed_exists ? "Markdown" : "PDF only"} ·{" "}
        {document.indexed ? "Indexed" : "Not indexed"}
      </td>
      <td>{formatDate(document.processed_at)}</td>
      <td>
        <Button variant="secondary" onClick={open}>
          Inspect
        </Button>
      </td>
    </tr>
  );
}

interface DocumentItemProps {
  document: DocumentRecord;
  selected: boolean;
  toggle: () => void;
  open: () => void;
}

function DocumentCard(props: DocumentItemProps): JSX.Element {
  const { document, selected, toggle, open } = props;
  return (
    <article class={styles.listItem} key={document.id ?? document.relative_raw_path}>
      <div class={styles.itemHeader}>
        <label class={styles.actions}>
          <input type="checkbox" checked={selected} onChange={toggle} />
          <strong>{document.display_name}</strong>
        </label>
        <StatusBadge status={document.status} />
      </div>
      <span class={styles.muted}>
        {formatBytes(document.byte_size)} · {formatDate(document.processed_at)}
      </span>
      <Button variant="secondary" onClick={open}>
        Inspect document
      </Button>
    </article>
  );
}

function DocumentDrawer({
  documentId,
  tab,
  setTab,
  targetChunk,
  close,
}: {
  documentId: string;
  tab: DetailTab;
  setTab: (tab: DetailTab) => void;
  targetChunk: number | null;
  close: () => void;
}): JSX.Element {
  const [document, setDocument] = useState<DocumentRecord | null>(null);
  const [content, setContent] = useState<unknown>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    setError("");
    setContent(null);
    const path =
      tab === "overview"
        ? `/api/documents/${documentId}`
        : documentDetailPath(documentId, tab, targetChunk);
    void api<Record<string, unknown>>(path)
      .then((payload) => {
        setDocument(payload.document as DocumentRecord);
        setContent(payload);
      })
      .catch((caught) =>
        setError(caught instanceof Error ? caught.message : "Could not load details"),
      );
  }, [documentId, tab, targetChunk]);

  return (
    <Drawer title={document?.display_name ?? "Document details"} close={close}>
      <Tabs
        value={tab}
        onChange={setTab}
        tabs={[
          { value: "overview", label: "Overview" },
          { value: "markdown", label: "Markdown" },
          { value: "chunks", label: "Chunks" },
          { value: "index", label: "Index points" },
        ]}
      />
      <div class={styles.details}>
        {error ? <ErrorState message={error} /> : null}
        {!content ? <LoadingState /> : null}
        {document && tab === "overview" ? <DocumentOverview document={document} /> : null}
        {tab === "markdown" && content ? (
          <pre class={styles.codePanel}>
            {String((content as { markdown?: string | null }).markdown ?? "No markdown")}
          </pre>
        ) : null}
        {tab === "chunks" && content ? (
          <ChunkList
            chunks={(content as { chunks?: Chunk[] }).chunks ?? []}
            targetChunk={targetChunk}
          />
        ) : null}
        {tab === "index" && content ? (
          <IndexList
            points={
              (content as { index?: { points?: IndexPoint[] } }).index?.points ?? []
            }
          />
        ) : null}
      </div>
    </Drawer>
  );
}

function DocumentOverview({ document }: { document: DocumentRecord }): JSX.Element {
  const values = [
    ["Status", document.status],
    ["Source path", document.relative_raw_path],
    ["Markdown", document.parsed_markdown_path ?? "Not prepared"],
    ["Content hash", document.content_hash ?? "Not calculated"],
    ["Size", formatBytes(document.byte_size)],
    ["Processed", formatDate(document.processed_at)],
  ];
  return (
    <>
      {document.processing_error ? (
        <Alert title="Processing failed" error>
          <p>{document.processing_error}</p>
        </Alert>
      ) : null}
      <dl class={styles.metadata}>
        {values.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </>
  );
}

function ChunkList({
  chunks,
  targetChunk,
}: {
  chunks: Chunk[];
  targetChunk: number | null;
}): JSX.Element {
  useEffect(() => {
    if (targetChunk === null) return;
    document
      .getElementById(`chunk-${targetChunk}`)
      ?.scrollIntoView({ block: "center" });
  }, [chunks, targetChunk]);
  if (!chunks.length) return <EmptyState title="No chunks">Run the pipeline first.</EmptyState>;
  return (
    <ol class={styles.list}>
      {chunks.map((chunk) => (
        <li
          id={`chunk-${chunk.index}`}
          class={`${styles.listItem} ${
            chunk.index === targetChunk ? styles.highlightedItem : ""
          }`}
          key={chunk.index}
        >
          <div class={styles.itemHeader}>
            <strong>Chunk {chunk.index}</strong>
            <span class={styles.muted}>
              {chunk.char_start ?? "–"}–{chunk.char_end ?? "–"} · {chunk.characters} chars
            </span>
          </div>
          <pre>{chunk.content}</pre>
        </li>
      ))}
    </ol>
  );
}

function documentDetailPath(
  documentId: string,
  tab: Exclude<DetailTab, "overview">,
  targetChunk: number | null,
): string {
  if (tab === "index") {
    return `/api/documents/${documentId}/index-points`;
  }
  if (tab === "markdown") {
    return `/api/documents/${documentId}/markdown`;
  }
  const page =
    targetChunk === null
      ? 1
      : Math.floor(Math.max(0, targetChunk - 1) / 20) + 1;
  return `/api/documents/${documentId}/chunks?page=${page}&page_size=20`;
}

function IndexList({ points }: { points: IndexPoint[] }): JSX.Element {
  if (!points.length) {
    return <EmptyState title="No index points">No points were returned for this document.</EmptyState>;
  }
  return (
    <ol class={styles.list}>
      {points.map((point) => (
        <li class={styles.listItem} key={point.id}>
          <strong class={styles.mono}>{point.id}</strong>
          <span class={styles.muted}>
            chars {point.char_start ?? "–"}–{point.char_end ?? "–"}
          </span>
          <p>{point.content}</p>
          {point.embeddings.map((embedding) => (
            <code class={styles.mono} key={embedding.vector_name}>
              {embedding.vector_name}: {embedding.dimensions} dimensions · sample [
              {embedding.sample.join(", ")}]
            </code>
          ))}
        </li>
      ))}
    </ol>
  );
}

function isPdf(file: File): boolean {
  return file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
}

function formatBytes(value: number | null): string {
  if (value === null) return "Unknown size";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function formatDate(value: string | null | undefined): string {
  return value ? new Date(value).toLocaleString() : "Not processed";
}
