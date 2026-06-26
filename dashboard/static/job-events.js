window.MedicDashboard = window.MedicDashboard || {};

window.MedicDashboard.jobEvents = (() => {
  const { formatting, i18n } = window.MedicDashboard;
  const currentEvents = [];

  function render(event) {
    currentEvents.push(event);
    appendEvent(event);
  }

  function appendEvent(event) {
    const { elements } = window.MedicDashboard;
    const row = document.createElement("div");
    row.className = "event-row";
    row.innerHTML = `
      <div class="event-main">
        <code>${formatting.escapeHtml(formatting.timeOnly(event.timestamp))}</code>
        <span>${formatting.pill(event.status)}</span>
        <span>${formatting.escapeHtml(event.message)}</span>
      </div>
    `;
    const details = eventDetails(event.result || {});
    if (details) {
      row.appendChild(details);
    }
    elements.eventLog.appendChild(row);
    elements.eventLog.scrollTop = elements.eventLog.scrollHeight;
  }

  function updateStep(step, status) {
    const stepName = step === "job" ? "pipeline" : step;
    const item = document.querySelector(`[data-step="${stepName}"]`);
    if (!item) {
      return;
    }
    item.className = status;
    item.querySelector("strong").textContent = formatting.statusLabel(status);
  }

  function resetSteps() {
    currentEvents.length = 0;
    for (const item of document.querySelectorAll(".stepper li")) {
      item.className = "";
      item.querySelector("strong").textContent = i18n.t("pipeline.step.waiting");
    }
  }

  function localize() {
    const { elements } = window.MedicDashboard;
    for (const item of document.querySelectorAll(".stepper li")) {
      const status = item.className;
      item.querySelector("strong").textContent = status
        ? formatting.statusLabel(status)
        : i18n.t("pipeline.step.waiting");
    }
    elements.eventLog.innerHTML = "";
    for (const event of currentEvents) {
      appendEvent(event);
    }
  }

  function eventDetails(result) {
    if (result.chunks && result.chunks.length) {
      return detailsBlock(result.chunks.map(chunkPreview).join(""));
    }
    if (result.embedding) {
      return detailsBlock(formatting.vectorPreviewHtml(result.embedding));
    }
    if (result.points && result.points.length) {
      return detailsBlock(result.points.map(pointPreview).join(""));
    }
    if (result.error) {
      return detailsBlock(errorPreview(result));
    }
    return null;
  }

  function detailsBlock(html) {
    const details = document.createElement("div");
    details.className = "event-details";
    details.innerHTML = html;
    return details;
  }

  function chunkPreview(chunk) {
    return `
      <div>
        <strong>Chunk ${Number(chunk.index)}</strong>
        <span>${formatting.rangeLabel(chunk.char_start, chunk.char_end)}</span>
        <p>${formatting.escapeHtml(chunk.content || "")}</p>
      </div>
    `;
  }

  function pointPreview(point) {
    return `
      <div>
        <strong>${formatting.escapeHtml(point.id || "-")}</strong>
        <span>${formatting.rangeLabel(point.char_start, point.char_end)}</span>
        ${formatting.vectorPreviewHtml(point.embedding)}
      </div>
    `;
  }

  function errorPreview(result) {
    return `
      <div class="event-error">
        ${result.path ? `<strong>${formatting.escapeHtml(result.path)}</strong>` : ""}
        <p>${formatting.escapeHtml(result.error)}</p>
      </div>
    `;
  }

  return { localize, render, resetSteps, updateStep };
})();
