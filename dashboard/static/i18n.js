window.MedicDashboard = window.MedicDashboard || {};

window.MedicDashboard.i18n = (() => {
  const translations = {
    "auth.logout": "Log out {username}",
    "chat.agent": "Agent",
    "chat.answerReady": "answer ready",
    "chat.answering": "answering",
    "chat.ask": "Ask",
    "chat.closeSource": "Close",
    "chat.conversation": "Conversation",
    "chat.conversations": "Conversations",
    "chat.empty": "Ask a question after documents are indexed.",
    "chat.enterQuestion": "enter a question",
    "chat.error": "error",
    "chat.fetchError": "Could not fetch the answer.",
    "chat.insufficientContext": "Not enough data in the documentation.",
    "chat.loadingConversation": "loading conversation",
    "chat.messageCount.one": "{count} message",
    "chat.messageCount.other": "{count} messages",
    "chat.new": "New",
    "chat.newConversation": "new conversation",
    "chat.noContext": "missing context",
    "chat.noSavedConversations": "No saved conversations.",
    "chat.noSources": "No sources to display.",
    "chat.readInFull": "read in full",
    "chat.unusedSources": "Checked but not used ({count})",
    "chat.placeholder": "Ask a question based on the documentation",
    "chat.ready": "ready",
    "chat.source": "Source",
    "chat.sourceAnswer": "Answer source",
    "chat.sourceNotFound": "source not found",
    "chat.sourceState": "source {sourceId}",
    "chat.sources": "Sources",
    "chat.sourcesCount.one": "{count} fragment",
    "chat.sourcesCount.other": "{count} fragments",
    "chat.showDocument": "Show document",
    "chat.title": "Medical agent",
    "chat.traceAgent": "agent",
    "chat.traceModel": "model",
    "chat.traceSummary": "Answer trace ({count})",
    "chat.traceTool": "tool",
    "chat.unknownSource": "unknown source",
    "chat.user": "You",
    "common.actions": "Actions",
    "common.close": "Close",
    "common.index": "Index",
    "common.markdown": "Markdown",
    "common.no": "no",
    "common.pdf": "PDF",
    "common.yes": "yes",
    "dashboard.documentCount": "Documents in database",
    "dashboard.rawPdfCount": "Source PDFs",
    "documents.delete": "Delete",
    "documents.deleteConfirm": "Delete {path}?",
    "documents.deleteSelected": "Delete selected",
    "documents.deleteSelectedConfirm": "Delete {count} selected documents?",
    "documents.deleted": "Deleted {count} PDF.",
    "documents.deletedWithCleanupErrors": "Deleted {count} PDF. Qdrant cleanup: {errors} errors",
    "documents.details": "Details",
    "documents.errorPrefix": "Error: {message}",
    "documents.lastProcessedEmpty": "no processing yet",
    "documents.pdfDeleted": "PDF deleted.",
    "documents.pdfDeletedWithCleanupError": "PDF deleted. Qdrant cleanup: {error}",
    "documents.processingError": "Processing error",
    "documents.select": "Select",
    "documents.selectAll": "Mark all",
    "documents.selectDocument": "Select {path}",
    "documents.selectedCount.one": "{count} selected",
    "documents.selectedCount.other": "{count} selected",
    "documents.status": "Status",
    "documents.statusError": "status error",
    "documents.title": "Documents",
    "documents.upload": "Add PDF",
    "documents.uploaded.one": "PDF added.",
    "documents.uploaded.other": "PDFs added: {count}.",
    "documents.warningNoSelectionForDelete": "Select documents to delete.",
    "format.charsRange": "chars {start}-{end}",
    "format.collectionMissing": "Collection does not exist.",
    "format.connected": "connected",
    "format.dimensions": "Dimensions",
    "format.hash": "Hash",
    "format.indices": "Indices",
    "format.noCollection": "no collection",
    "format.noPointsForDocument": "No points for this document.",
    "format.points.one": "{count} point",
    "format.points.other": "{count} points",
    "format.rangeEmpty": "range -",
    "format.rows": "Rows",
    "format.shown": "Shown",
    "format.type": "Type",
    "format.unavailable": "unavailable",
    "format.vector": "Vector",
    "history.empty": "No pipeline runs yet.",
    "history.events.one": "{count} event",
    "history.events.other": "{count} events",
    "history.subtitle": "recent runs",
    "history.title": "Pipeline history",
    "metric.qdrantChecking": "checking",
    "pipeline.connectionInterrupted": "Live status connection was interrupted.",
    "pipeline.noActiveJob": "no active job",
    "pipeline.noSelection": "Select documents before running the pipeline.",
    "pipeline.run": "Run pipeline",
    "pipeline.started": "Pipeline started for {count} documents.",
    "pipeline.step.chunks": "Chunks",
    "pipeline.step.embed": "Embedding",
    "pipeline.step.index": "Index",
    "pipeline.step.result": "Result",
    "pipeline.step.waiting": "waiting",
    "process.chars.one": "{count} char",
    "process.chars.other": "{count} chars",
    "process.chunks": "Chunks",
    "process.document": "Document",
    "process.indexUnavailable": "Index unavailable.",
    "process.loading": "loading",
    "process.noChunks": "No chunks.",
    "process.noMarkdown": "No markdown.",
    "process.noPreview": "no preview",
    "process.parsed": "Parsed",
    "process.processed": "Processed",
    "process.range": "Range",
    "process.selectDocument": "select a document",
    "process.title": "Process details",
    "status.failed": "error",
    "status.indexed": "indexed",
    "status.notIndexed": "no",
    "status.prepared": "ready",
    "status.preparedUnverified": "ready, index unchecked",
    "status.queued": "queued",
    "status.raw": "PDF without markdown",
    "status.running": "running",
    "status.skipped": "skipped",
    "status.stale": "stale",
    "status.succeeded": "OK",
    "status.unknown": "unknown",
  };

  function language() {
    return "en";
  }

  function t(key, params = {}) {
    return interpolate(translations[key] || key, params);
  }

  function count(singularKey, pluralKey, amount) {
    const key = Number(amount) === 1 ? singularKey : pluralKey;
    return t(key, { count: amount });
  }

  function apply(root = document) {
    if (!root.querySelectorAll) {
      return;
    }
    for (const element of root.querySelectorAll("[data-i18n]")) {
      element.textContent = t(element.dataset.i18n, paramsFrom(element));
    }
    for (const element of root.querySelectorAll("[data-i18n-placeholder]")) {
      element.setAttribute(
        "placeholder",
        t(element.dataset.i18nPlaceholder, paramsFrom(element)),
      );
    }
    for (const element of root.querySelectorAll("[data-i18n-aria-label]")) {
      element.setAttribute(
        "aria-label",
        t(element.dataset.i18nAriaLabel, paramsFrom(element)),
      );
    }
  }

  function paramsFrom(element) {
    const params = {};
    for (const [name, value] of Object.entries(element.dataset)) {
      if (!name.startsWith("i18nParam")) {
        continue;
      }
      params[paramName(name)] = value;
    }
    return params;
  }

  function paramName(datasetKey) {
    const rawName = datasetKey.slice("i18nParam".length);
    return `${rawName.charAt(0).toLowerCase()}${rawName.slice(1)}`;
  }

  function interpolate(template, params) {
    return Object.entries(params).reduce(
      (text, [name, value]) => text.replaceAll(`{${name}}`, String(value)),
      template,
    );
  }

  return {
    apply,
    count,
    language,
    t,
  };
})();
