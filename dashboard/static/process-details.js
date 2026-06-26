window.MedicDashboard = window.MedicDashboard || {};

window.MedicDashboard.processDetails = (() => {
  const { api, formatting, i18n } = window.MedicDashboard;
  let currentPayload = null;

  async function load(relativeRawPath) {
    const { elements, state } = window.MedicDashboard;
    state.selectedDocumentPath = relativeRawPath;
    elements.processDocument.textContent = relativeRawPath;
    elements.processStatus.textContent = i18n.t("process.loading");

    try {
      const payload = await api.json(
        `/api/documents/process?relative_raw_path=${encodeURIComponent(relativeRawPath)}`,
      );
      render(payload);
    } catch (error) {
      elements.processStatus.textContent = formatting.statusLabel("failed");
      elements.processMeta.innerHTML = `<span>${formatting.escapeHtml(error.message)}</span>`;
    }
  }

  function clear() {
    const { elements, state } = window.MedicDashboard;
    currentPayload = null;
    state.selectedDocumentPath = null;
    elements.processDocument.textContent = i18n.t("process.selectDocument");
    elements.processStatus.textContent = i18n.t("process.noPreview");
    elements.processMeta.innerHTML = "";
    elements.processError.hidden = true;
    elements.processError.innerHTML = "";
    elements.markdownPreview.innerHTML = "";
    elements.chunksPreview.innerHTML = "";
    elements.indexPreview.innerHTML = "";
  }

  function showTab(tabName) {
    const { elements } = window.MedicDashboard;
    for (const tab of elements.processTabs) {
      const isActive = tab.dataset.processTab === tabName;
      tab.classList.toggle("active", isActive);
      tab.setAttribute("aria-selected", isActive ? "true" : "false");
    }
    for (const preview of [
      elements.markdownPreview,
      elements.chunksPreview,
      elements.indexPreview,
    ]) {
      const isActive = preview.id === `${tabName}-preview`;
      preview.classList.toggle("active", isActive);
      preview.hidden = !isActive;
    }
  }

  function render(payload) {
    currentPayload = payload;
    const { elements } = window.MedicDashboard;
    elements.processStatus.innerHTML = formatting.pill(payload.document.status);
    elements.processMeta.innerHTML = renderMeta(payload);
    renderProcessingError(payload.document.processing_error);
    elements.markdownPreview.innerHTML = renderMarkdown(payload.markdown);
    elements.chunksPreview.innerHTML = renderChunks(payload.chunks);
    elements.indexPreview.innerHTML = renderIndex(payload.index);
  }

  function renderProcessingError(errorMessage) {
    const { elements } = window.MedicDashboard;
    if (!errorMessage) {
      elements.processError.hidden = true;
      elements.processError.innerHTML = "";
      return;
    }
    elements.processError.hidden = false;
    elements.processError.innerHTML = `
      <strong>${formatting.escapeHtml(i18n.t("documents.processingError"))}</strong>
      <pre>${formatting.escapeHtml(errorMessage)}</pre>
    `;
  }

  function renderMeta(payload) {
    return [
      [i18n.t("process.document"), payload.document.display_name || payload.document.original_filename || payload.document.relative_raw_path],
      [i18n.t("common.pdf"), payload.document.relative_raw_path],
      [i18n.t("process.parsed"), payload.document.parsed_markdown_path || "-"],
      [i18n.t("process.chunks"), payload.chunk_count],
      [i18n.t("format.hash"), formatting.shortHash(payload.document.content_hash)],
    ]
      .map(([label, value]) => `<span><strong>${label}:</strong> ${formatting.escapeHtml(value)}</span>`)
      .join("");
  }

  function renderMarkdown(markdown) {
    if (!markdown) {
      return `<p class="muted">${formatting.escapeHtml(i18n.t("process.noMarkdown"))}</p>`;
    }
    return `<pre>${formatting.escapeHtml(markdown)}</pre>`;
  }

  function renderChunks(chunks) {
    if (!chunks.length) {
      return `<p class="muted">${formatting.escapeHtml(i18n.t("process.noChunks"))}</p>`;
    }
    return chunks.map(renderChunk).join("");
  }

  function renderChunk(chunk) {
    return `
      <article class="chunk-card">
        <div class="chunk-meta">
          <span><strong>#${chunk.index}</strong></span>
          <span>${formatting.rangeLabel(chunk.char_start, chunk.char_end)}</span>
          <span>${formatting.escapeHtml(i18n.count("process.chars.one", "process.chars.other", chunk.characters))}</span>
        </div>
        <pre>${formatting.escapeHtml(chunk.content)}</pre>
      </article>
    `;
  }

  function renderIndex(index) {
    if (!index || !index.available) {
      return `<p class="form-error">${formatting.escapeHtml(index?.error || i18n.t("process.indexUnavailable"))}</p>`;
    }
    const summary = `
      <div class="index-summary">
        <span><strong>Collection:</strong> ${formatting.escapeHtml(index.collection_name || "-")}</span>
        <span><strong>${formatting.escapeHtml(i18n.t("format.shown"))}:</strong> ${index.shown_points || 0}/${index.preview_limit}</span>
      </div>
    `;
    if (!index.collection_exists) {
      return `${summary}<p class="muted">${formatting.escapeHtml(i18n.t("format.collectionMissing"))}</p>`;
    }
    if (!index.points.length) {
      return `${summary}<p class="muted">${formatting.escapeHtml(i18n.t("format.noPointsForDocument"))}</p>`;
    }
    return `${summary}${index.points.map(renderPoint).join("")}`;
  }

  function renderPoint(point) {
    return `
      <article class="index-point">
        <div class="chunk-meta">
          <strong>${formatting.escapeHtml(point.id || "-")}</strong>
          <span>${formatting.rangeLabel(point.char_start, point.char_end)}</span>
        </div>
        <p>${formatting.escapeHtml(point.content || "")}</p>
        ${formatting.vectorPreviewsHtml(point.embeddings || [point.embedding])}
      </article>
    `;
  }

  function localize() {
    if (currentPayload) {
      render(currentPayload);
      return;
    }
    if (!window.MedicDashboard.state.selectedDocumentPath) {
      clear();
    }
  }

  return { clear, load, localize, showTab };
})();
