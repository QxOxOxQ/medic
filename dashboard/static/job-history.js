window.MedicDashboard = window.MedicDashboard || {};

window.MedicDashboard.jobHistory = (() => {
  const { api, formatting, i18n } = window.MedicDashboard;
  let currentJobs = [];

  async function load() {
    const { elements } = window.MedicDashboard;
    if (!elements.jobHistoryList) {
      return;
    }
    try {
      const payload = await api.json("/api/jobs");
      render(payload.jobs || []);
    } catch (error) {
      elements.jobHistoryList.innerHTML = `<p class="form-error">${formatting.escapeHtml(error.message)}</p>`;
    }
  }

  function render(jobs) {
    currentJobs = jobs;
    const { elements } = window.MedicDashboard;
    if (!jobs.length) {
      elements.jobHistoryList.innerHTML =
        `<p class="muted">${formatting.escapeHtml(i18n.t("history.empty"))}</p>`;
      return;
    }
    elements.jobHistoryList.innerHTML = "";
    for (const job of jobs) {
      elements.jobHistoryList.appendChild(jobItem(job));
    }
  }

  function jobItem(job) {
    const item = document.createElement("article");
    item.className = "job-history-item";
    item.innerHTML = `
      <div>
        <strong>${formatting.escapeHtml(formatting.statusLabel(job.status))}</strong>
        <span>${formatting.escapeHtml(formatting.timeOnly(job.finished_at || job.started_at))}</span>
      </div>
      <p>${formatting.escapeHtml(jobSummary(job))}</p>
    `;
    return item;
  }

  function jobSummary(job) {
    const jobEvent = [...(job.events || [])]
      .reverse()
      .find((event) => event.step === "job" || event.step === "pipeline");
    const summary = jobEvent?.result?.summary;
    if (summary) {
      return summary;
    }
    if (job.error) {
      return job.error;
    }
    return i18n.count(
      "history.events.one",
      "history.events.other",
      job.events?.length || 0,
    );
  }

  function localize() {
    render(currentJobs);
  }

  return { load, localize };
})();
