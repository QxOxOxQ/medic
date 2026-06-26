window.MedicDashboard = window.MedicDashboard || {};
Object.assign(window.MedicDashboard, {
  elements: {
    rawPdfCount: document.getElementById("raw-pdf-count"),
    parsedMarkdownCount: document.getElementById("parsed-markdown-count"),
    documentCount: document.getElementById("document-count"),
    qdrantStatus: document.getElementById("qdrant-status"),
    lastProcessed: document.getElementById("last-processed"),
    actionMessage: document.getElementById("action-message"),
    documentsTable: document.getElementById("documents-table"),
    eventLog: document.getElementById("event-log"),
    jobState: document.getElementById("job-state"),
    jobHistoryList: document.getElementById("job-history-list"),
    uploadForm: document.getElementById("upload-form"),
    runIngest: document.getElementById("run-ingest"),
    chatForm: document.getElementById("chat-form"),
    chatQuestion: document.getElementById("chat-question"),
    chatSubmit: document.querySelector("#chat-form button[type='submit']"),
    chatHistory: document.getElementById("chat-history"),
    chatSources: document.getElementById("chat-sources"),
    chatState: document.getElementById("chat-state"),
    conversationList: document.getElementById("conversation-list"),
    newChat: document.getElementById("new-chat"),
    sourceDrawer: document.getElementById("source-drawer"),
    sourceDrawerTitle: document.getElementById("source-drawer-title"),
    sourceDrawerBody: document.getElementById("source-drawer-body"),
    sourceDrawerClose: document.getElementById("source-drawer-close"),
    processDocument: document.getElementById("process-document"),
    processStatus: document.getElementById("process-status"),
    processMeta: document.getElementById("process-meta"),
    processError: document.getElementById("process-error"),
    markdownPreview: document.getElementById("markdown-preview"),
    chunksPreview: document.getElementById("chunks-preview"),
    indexPreview: document.getElementById("index-preview"),
    processTabs: document.querySelectorAll("[data-process-tab]"),
    selectAllDocuments: document.getElementById("select-all-documents"),
    selectedCount: document.getElementById("selected-count"),
    deleteSelected: document.getElementById("delete-selected"),
  },
  state: {
    eventSource: null,
    selectedDocumentPath: null,
    selectedDocumentPaths: new Set(),
    documentPaths: [],
  },

  async refreshDashboard() {
    const app = window.MedicDashboard;
    await Promise.all([
      app.documents.loadStatus(),
      app.documents.loadDocuments(),
      app.jobHistory.load(),
      app.chat.loadConversations(),
    ]);
  },

  setMessage(message, status) {
    if (!this.elements?.actionMessage) {
      return;
    }
    this.elements.actionMessage.textContent = message || "";
    this.elements.actionMessage.className = status || "";
  },
});

document.addEventListener("DOMContentLoaded", () => {
  const app = window.MedicDashboard;
  app.i18n.apply();
  app.refreshDashboard();

  app.elements.uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await app.documents.uploadPdf();
  });

  app.elements.runIngest.addEventListener("click", async () => {
    await app.pipeline.startIngest();
  });

  app.elements.selectAllDocuments.addEventListener("change", (event) => {
    app.documents.setAllSelected(event.target.checked);
  });

  app.elements.deleteSelected.addEventListener("click", async () => {
    await app.documents.deleteSelected();
  });

  app.elements.chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await app.chat.ask();
  });

  app.elements.newChat.addEventListener("click", () => {
    app.chat.startNewConversation();
  });

  app.elements.sourceDrawerClose.addEventListener("click", () => {
    app.chat.closeSourceDrawer();
  });

  for (const tab of app.elements.processTabs) {
    tab.addEventListener("click", () =>
      app.processDetails.showTab(tab.dataset.processTab),
    );
  }
});
