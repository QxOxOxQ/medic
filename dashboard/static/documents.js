window.MedicDashboard = window.MedicDashboard || {};

window.MedicDashboard.documents = (() => {
  const { api, formatting, i18n, processDetails } = window.MedicDashboard;

  async function loadStatus() {
    const { elements, setMessage } = window.MedicDashboard;
    try {
      const status = await api.json("/api/status");
      elements.rawPdfCount.textContent = status.raw_pdf_count;
      elements.parsedMarkdownCount.textContent = status.parsed_markdown_count;
      elements.documentCount.textContent = status.document_count;
      elements.lastProcessed.textContent =
        status.last_processed_at || i18n.t("documents.lastProcessedEmpty");
      elements.qdrantStatus.textContent = formatting.formatQdrant(status.qdrant);
    } catch (error) {
      elements.qdrantStatus.textContent = i18n.t("documents.statusError");
      setMessage(error.message, "failed");
    }
  }

  async function loadDocuments() {
    const { elements, setMessage, state } = window.MedicDashboard;
    try {
      const payload = await api.json("/api/documents");
      state.documentPaths = payload.documents.map(
        (documentRecord) => documentRecord.relative_raw_path,
      );
      pruneSelection();
      elements.documentsTable.innerHTML = "";
      for (const documentRecord of payload.documents) {
        elements.documentsTable.appendChild(documentRow(documentRecord));
      }
      updateSelectionControls();
      await refreshSelectedDocument(payload.documents, state.selectedDocumentPath);
      if (payload.qdrant_error) {
        setMessage(`Qdrant index check: ${payload.qdrant_error}`, "skipped");
      }
    } catch (error) {
      setMessage(error.message, "failed");
    }
  }

  async function uploadPdf() {
    const { elements, refreshDashboard, setMessage } = window.MedicDashboard;
    const formData = new FormData(elements.uploadForm);
    try {
      const payload = await api.json("/api/documents/upload", {
        method: "POST",
        body: formData,
        headers: { "X-CSRF-Token": api.csrfToken },
      });
      elements.uploadForm.reset();
      const uploadedCount = payload.uploaded_count || 1;
      setMessage(
        i18n.count("documents.uploaded.one", "documents.uploaded.other", uploadedCount),
        "succeeded",
      );
      await refreshDashboard();
    } catch (error) {
      setMessage(error.message, "failed");
    }
  }

  async function deletePdf(relativeRawPath) {
    if (!window.confirm(i18n.t("documents.deleteConfirm", { path: relativeRawPath }))) {
      return;
    }

    const { refreshDashboard, setMessage, state } = window.MedicDashboard;
    const formData = new FormData();
    formData.append("relative_raw_path", relativeRawPath);
    formData.append("csrf_token", api.csrfToken);

    try {
      const payload = await api.json("/api/documents/delete", {
        method: "POST",
        body: formData,
        headers: { "X-CSRF-Token": api.csrfToken },
      });
      showDeleteMessage(payload.qdrant_cleanup);
      state.selectedDocumentPaths.delete(relativeRawPath);
      if (state.selectedDocumentPath === relativeRawPath) {
        processDetails.clear();
      }
      await refreshDashboard();
    } catch (error) {
      setMessage(error.message, "failed");
    }
  }

  function documentRow(documentRecord) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td class="selection-cell"></td>
      <td class="path-cell">
        <span>${formatting.escapeHtml(documentName(documentRecord))}</span>
        ${processingErrorSummary(documentRecord)}
      </td>
      <td>${formatting.pill(documentRecord.status)}</td>
      <td>${i18n.t(documentRecord.parsed_exists ? "common.yes" : "common.no")}</td>
      <td>${formatting.formatIndexed(documentRecord.indexed)}</td>
      <td>${formatting.escapeHtml(documentRecord.processed_at || "-")}</td>
      <td><div class="row-actions"></div></td>
    `;
    const selectionCell = row.querySelector(".selection-cell");
    selectionCell.appendChild(selectionCheckbox(documentRecord.relative_raw_path));
    const actions = row.querySelector(".row-actions");
    actions.appendChild(detailsButton(documentRecord.relative_raw_path));
    actions.appendChild(deleteButton(documentRecord.relative_raw_path));
    return row;
  }

  function processingErrorSummary(documentRecord) {
    if (!documentRecord.processing_error) {
      return "";
    }
    const firstLine = String(documentRecord.processing_error).split("\n")[0];
    return `<p class="document-error-summary">${formatting.escapeHtml(
      i18n.t("documents.errorPrefix", { message: firstLine }),
    )}</p>`;
  }

  function documentName(documentRecord) {
    return (
      documentRecord.display_name ||
      documentRecord.original_filename ||
      documentRecord.relative_raw_path
    );
  }

  async function refreshSelectedDocument(documents, selectedPath) {
    if (!selectedPath) {
      return;
    }
    if (documents.some((record) => record.relative_raw_path === selectedPath)) {
      await processDetails.load(selectedPath);
      return;
    }
    processDetails.clear();
  }

  function detailsButton(relativeRawPath) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "secondary-button";
    button.textContent = i18n.t("documents.details");
    button.addEventListener("click", () => processDetails.load(relativeRawPath));
    return button;
  }

  function deleteButton(relativeRawPath) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "danger-button";
    button.textContent = i18n.t("documents.delete");
    button.addEventListener("click", () => deletePdf(relativeRawPath));
    return button;
  }

  function selectionCheckbox(relativeRawPath) {
    const { state } = window.MedicDashboard;
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "document-checkbox";
    checkbox.checked = state.selectedDocumentPaths.has(relativeRawPath);
    checkbox.setAttribute(
      "aria-label",
      i18n.t("documents.selectDocument", { path: relativeRawPath }),
    );
    checkbox.addEventListener("change", () =>
      toggleDocument(relativeRawPath, checkbox.checked),
    );
    return checkbox;
  }

  function toggleDocument(relativeRawPath, selected) {
    const { state } = window.MedicDashboard;
    if (selected) {
      state.selectedDocumentPaths.add(relativeRawPath);
    } else {
      state.selectedDocumentPaths.delete(relativeRawPath);
    }
    updateSelectionControls();
  }

  function setAllSelected(selected) {
    const { elements, state } = window.MedicDashboard;
    if (selected) {
      for (const relativeRawPath of state.documentPaths) {
        state.selectedDocumentPaths.add(relativeRawPath);
      }
    } else {
      state.selectedDocumentPaths.clear();
    }
    for (const checkbox of elements.documentsTable.querySelectorAll(
      ".document-checkbox",
    )) {
      checkbox.checked = selected;
    }
    updateSelectionControls();
  }

  function selectedPaths() {
    const { state } = window.MedicDashboard;
    return state.documentPaths.filter((relativeRawPath) =>
      state.selectedDocumentPaths.has(relativeRawPath),
    );
  }

  function pruneSelection() {
    const { state } = window.MedicDashboard;
    const currentPaths = new Set(state.documentPaths);
    for (const selectedPath of [...state.selectedDocumentPaths]) {
      if (!currentPaths.has(selectedPath)) {
        state.selectedDocumentPaths.delete(selectedPath);
      }
    }
  }

  function updateSelectionControls() {
    const { elements, state } = window.MedicDashboard;
    const selectedCount = selectedPaths().length;
    const documentCount = state.documentPaths.length;
    elements.selectedCount.textContent = i18n.count(
      "documents.selectedCount.one",
      "documents.selectedCount.other",
      selectedCount,
    );
    elements.deleteSelected.disabled = selectedCount === 0;
    elements.selectAllDocuments.checked =
      documentCount > 0 && selectedCount === documentCount;
    elements.selectAllDocuments.indeterminate =
      selectedCount > 0 && selectedCount < documentCount;
  }

  async function deleteSelected() {
    const { refreshDashboard, setMessage, state } = window.MedicDashboard;
    const paths = selectedPaths();
    if (!paths.length) {
      setMessage(i18n.t("documents.warningNoSelectionForDelete"), "failed");
      return;
    }
    if (
      !window.confirm(
        i18n.t("documents.deleteSelectedConfirm", { count: paths.length }),
      )
    ) {
      return;
    }

    try {
      const payload = await api.json("/api/documents/delete-selected", {
        method: "POST",
        body: JSON.stringify({ relative_raw_paths: paths }),
        headers: {
          "Content-Type": "application/json",
          "X-CSRF-Token": api.csrfToken,
        },
      });
      for (const relativeRawPath of paths) {
        state.selectedDocumentPaths.delete(relativeRawPath);
      }
      if (paths.includes(state.selectedDocumentPath)) {
        processDetails.clear();
      }
      showBulkDeleteMessage(payload);
      await refreshDashboard();
    } catch (error) {
      setMessage(error.message, "failed");
    }
  }

  function showBulkDeleteMessage(payload) {
    const { setMessage } = window.MedicDashboard;
    const cleanupErrors = (payload.qdrant_cleanups || []).filter(
      (cleanup) => cleanup.error,
    ).length;
    if (cleanupErrors) {
      setMessage(
        i18n.t("documents.deletedWithCleanupErrors", {
          count: payload.deleted_count,
          errors: cleanupErrors,
        }),
        "skipped",
      );
      return;
    }
    setMessage(
      i18n.t("documents.deleted", { count: payload.deleted_count }),
      "succeeded",
    );
  }

  function showDeleteMessage(cleanup) {
    const { setMessage } = window.MedicDashboard;
    if (cleanup && cleanup.error) {
      setMessage(
        i18n.t("documents.pdfDeletedWithCleanupError", { error: cleanup.error }),
        "skipped",
      );
      return;
    }
    setMessage(i18n.t("documents.pdfDeleted"), "succeeded");
  }

  return {
    deleteSelected,
    loadDocuments,
    loadStatus,
    selectedPaths,
    setAllSelected,
    uploadPdf,
  };
})();
