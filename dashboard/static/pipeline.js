window.MedicDashboard = window.MedicDashboard || {};

window.MedicDashboard.pipeline = (() => {
  const { api, formatting, i18n, jobEvents, jobHistory } = window.MedicDashboard;

  async function startIngest() {
    const { elements, setMessage, state } = window.MedicDashboard;
    const selectedPaths = window.MedicDashboard.documents.selectedPaths();
    if (!selectedPaths.length) {
      setMessage(i18n.t("pipeline.noSelection"), "failed");
      return;
    }
    elements.runIngest.disabled = true;
    jobEvents.resetSteps();
    elements.eventLog.innerHTML = "";
    setMessage(
      i18n.t("pipeline.started", { count: selectedPaths.length }),
      "running",
    );

    try {
      const payload = await api.json("/api/jobs/ingest", {
        method: "POST",
        body: JSON.stringify({ relative_raw_paths: selectedPaths }),
        headers: {
          "Content-Type": "application/json",
          "X-CSRF-Token": api.csrfToken,
        },
      });
      elements.jobState.textContent = formatting.statusLabel(payload.job.status);
      connectEvents(payload.job.id);
    } catch (error) {
      elements.runIngest.disabled = false;
      setMessage(error.message, "failed");
      closeEventSource(state);
    }
  }

  function connectEvents(jobId) {
    const { state } = window.MedicDashboard;
    closeEventSource(state);
    state.eventSource = new EventSource(
      `/api/jobs/${encodeURIComponent(jobId)}/events`,
    );
    state.eventSource.addEventListener("progress", handleProgress);
    state.eventSource.addEventListener("done", handleDone);
    state.eventSource.onerror = handleError;
  }

  function handleProgress(event) {
    const { elements } = window.MedicDashboard;
    const payload = JSON.parse(event.data);
    jobEvents.render(payload);
    jobEvents.updateStep(payload.step, payload.status);
    elements.jobState.textContent = formatting.statusLabel(payload.status);
  }

  async function handleDone(event) {
    const { elements, refreshDashboard, state } = window.MedicDashboard;
    const payload = JSON.parse(event.data);
    elements.jobState.textContent = formatting.statusLabel(payload.status);
    elements.runIngest.disabled = false;
    closeEventSource(state);
    await refreshDashboard();
    await jobHistory.load();
  }

  function handleError() {
    const { elements, setMessage, state } = window.MedicDashboard;
    elements.runIngest.disabled = false;
    setMessage(i18n.t("pipeline.connectionInterrupted"), "failed");
    closeEventSource(state);
  }

  function closeEventSource(state) {
    if (state.eventSource) {
      state.eventSource.close();
      state.eventSource = null;
    }
  }

  return { startIngest };
})();
