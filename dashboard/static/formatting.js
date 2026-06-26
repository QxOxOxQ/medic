window.MedicDashboard = window.MedicDashboard || {};

window.MedicDashboard.formatting = {
  escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  },

  escapeAttribute(value) {
    return String(value).replace(/[^a-zA-Z0-9_-]/g, "_");
  },

  formatIndexed(indexed) {
    if (indexed === true) {
      return this.pill("indexed");
    }
    if (indexed === false) {
      return this.pill("not_indexed");
    }
    return this.pill("unknown");
  },

  formatQdrant(qdrant) {
    const { i18n } = window.MedicDashboard;
    if (!qdrant || !qdrant.available) {
      return i18n.t("format.unavailable");
    }
    if (!qdrant.collection_exists) {
      return i18n.t("format.noCollection");
    }
    if (qdrant.points_count === null || qdrant.points_count === undefined) {
      return i18n.t("format.connected");
    }
    return i18n.count(
      "format.points.one",
      "format.points.other",
      qdrant.points_count,
    );
  },

  pill(value) {
    return `<span class="status-pill ${this.escapeAttribute(value)}">${this.escapeHtml(this.statusLabel(value))}</span>`;
  },

  statusLabel(value) {
    const { i18n } = window.MedicDashboard;
    const labels = {
      failed: "status.failed",
      indexed: "status.indexed",
      not_indexed: "status.notIndexed",
      prepared: "status.prepared",
      prepared_unverified: "status.preparedUnverified",
      queued: "status.queued",
      raw: "status.raw",
      running: "status.running",
      skipped: "status.skipped",
      stale: "status.stale",
      succeeded: "status.succeeded",
      unknown: "status.unknown",
    };
    const key = labels[value];
    if (!key) {
      return value || "-";
    }
    return i18n.t(key);
  },

  rangeLabel(start, end) {
    if (
      start === null ||
      start === undefined ||
      end === null ||
      end === undefined
    ) {
      return window.MedicDashboard.i18n.t("format.rangeEmpty");
    }
    return window.MedicDashboard.i18n.t("format.charsRange", {
      start: Number(start),
      end: Number(end),
    });
  },

  shortHash(value) {
    if (!value) {
      return "-";
    }
    const text = String(value);
    return text.length <= 16 ? text : `${text.slice(0, 16)}...`;
  },

  timeOnly(timestamp) {
    if (!timestamp) {
      return "";
    }
    const date = new Date(timestamp);
    if (Number.isNaN(date.getTime())) {
      return timestamp;
    }
    return date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  },

  vectorPreviewsHtml(embeddings) {
    const values = embeddings || [];
    if (!values.length) {
      return this.vectorPreviewHtml(null);
    }
    return values.map((embedding) => this.vectorPreviewHtml(embedding)).join("");
  },

  vectorPreviewHtml(embedding) {
    const { i18n } = window.MedicDashboard;
    if (!embedding) {
      return '<div class="vector-preview"><span>Embedding: -</span></div>';
    }
    const sample = (embedding.sample || [])
      .map((value) => Number(value).toFixed(6))
      .join(", ");
    const indices = (embedding.indices_sample || []).join(", ");
    return `
      <div class="vector-preview">
        <span>${i18n.t("format.vector")}: ${this.escapeHtml(embedding.vector_name || "-")}</span>
        <span>${i18n.t("format.type")}: ${this.escapeHtml(embedding.kind || "-")}</span>
        <span>${i18n.t("format.dimensions")}: ${this.escapeHtml(embedding.dimensions ?? "-")}</span>
        <span>${i18n.t("format.rows")}: ${this.escapeHtml(embedding.rows ?? "-")}</span>
        ${indices ? `<span>${i18n.t("format.indices")}: ${this.escapeHtml(indices)}</span>` : ""}
        <code>${this.escapeHtml(sample || "-")}</code>
      </div>
    `;
  },
};
