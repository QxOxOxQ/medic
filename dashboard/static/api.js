window.MedicDashboard = window.MedicDashboard || {};

window.MedicDashboard.api = {
  csrfToken: document.querySelector('meta[name="csrf-token"]').content,

  async json(path, options = {}) {
    const response = await fetch(path, {
      credentials: "same-origin",
      ...options,
    });
    const contentType = response.headers.get("content-type") || "";
    const data = contentType.includes("application/json")
      ? await response.json()
      : { detail: await response.text() };
    if (!response.ok) {
      throw new Error(data.detail || data.error || `HTTP ${response.status}`);
    }
    return data;
  },
};

