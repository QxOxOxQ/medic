window.MedicDashboard = window.MedicDashboard || {};

window.MedicDashboard.chat = (() => {
  const { api, formatting, i18n } = window.MedicDashboard;
  const DEFAULT_CONTEXT_LIMIT = 5;
  const state = {
    currentConversation: null,
    currentConversationId: null,
    currentSourceId: null,
    currentSourceRecordId: null,
    currentStateKey: "chat.ready",
    currentStateParams: {},
    handlersAttached: false,
  };

  async function loadConversations({ selectFirst = true } = {}) {
    attachHandlers();
    const { elements } = window.MedicDashboard;
    try {
      const payload = await api.json("/api/chat/conversations");
      renderConversationList(payload.conversations || []);
      if (state.currentConversationId || !selectFirst) {
        return;
      }
      const first = (payload.conversations || [])[0];
      if (first) {
        await loadConversation(first.id);
        return;
      }
      renderEmptyConversation();
    } catch (error) {
      elements.conversationList.innerHTML = `<p class="form-error">${formatting.escapeHtml(error.message)}</p>`;
    }
  }

  async function loadConversation(conversationId) {
    attachHandlers();
    setState("chat.loadingConversation");
    const payload = await api.json(`/api/chat/conversations/${conversationId}`);
    renderConversation(payload.conversation);
    setState("chat.ready");
  }

  function startNewConversation() {
    attachHandlers();
    state.currentConversation = null;
    state.currentConversationId = null;
    renderEmptyConversation();
    renderConversationListSelection();
    closeSourceDrawer();
    setState("chat.newConversation");
  }

  async function ask() {
    attachHandlers();
    const { elements } = window.MedicDashboard;
    const question = elements.chatQuestion.value.trim();
    if (!question) {
      setState("chat.enterQuestion");
      return;
    }

    clearEmptyState();
    appendTransientUserMessage(question);
    setLoading(true);
    setState("chat.answering");
    elements.chatSources.innerHTML = "";

    try {
      const payload = await api.json(askPath(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          limit: DEFAULT_CONTEXT_LIMIT,
        }),
      });
      renderConversation(payload.conversation);
      await loadConversations({ selectFirst: false });
      elements.chatQuestion.value = "";
      setState(conversationStateKey(payload.conversation));
    } catch (error) {
      appendError(error.message);
      setState("chat.error");
    } finally {
      setLoading(false);
    }
  }

  function askPath() {
    if (!state.currentConversationId) {
      return "/api/chat/conversations";
    }
    return `/api/chat/conversations/${state.currentConversationId}/messages`;
  }

  function renderConversationList(conversations) {
    const { elements } = window.MedicDashboard;
    if (!conversations.length) {
      elements.conversationList.innerHTML =
        `<p class="muted">${formatting.escapeHtml(i18n.t("chat.noSavedConversations"))}</p>`;
      return;
    }
    elements.conversationList.innerHTML = conversations
      .map(
        (conversation) => `
          <button
            class="conversation-item"
            type="button"
            data-conversation-id="${formatting.escapeHtml(conversation.id)}"
          >
            <strong>${formatting.escapeHtml(conversation.title || i18n.t("chat.conversation"))}</strong>
            <span>${formatting.escapeHtml(messageCountLabel(conversation.message_count || 0))}</span>
          </button>
        `,
      )
      .join("");
    renderConversationListSelection();
  }

  function renderConversationListSelection() {
    const { elements } = window.MedicDashboard;
    for (const item of elements.conversationList.querySelectorAll(
      "[data-conversation-id]",
    )) {
      item.classList.toggle(
        "active",
        item.dataset.conversationId === state.currentConversationId,
      );
    }
  }

  function renderConversation(conversation, { preserveSource = false } = {}) {
    const { elements } = window.MedicDashboard;
    const sourceId = preserveSource ? state.currentSourceId : null;
    const sourceRecordId = preserveSource ? state.currentSourceRecordId : null;
    state.currentConversation = conversation || null;
    state.currentConversationId = conversation?.id || null;
    elements.chatHistory.innerHTML = "";
    closeSourceDrawer();

    const messages = conversation?.messages || [];
    if (!messages.length) {
      renderEmptyConversation();
      return;
    }
    for (const message of messages) {
      appendMessage(message);
    }
    const lastAssistant = lastAssistantMessage(messages);
    renderSources(
      lastAssistant ? lastAssistant.sources || [] : [],
      expandedSourceIdSet(lastAssistant),
    );
    renderConversationListSelection();
    if (sourceId) {
      openSourceDrawer(sourceId, sourceRecordId);
    }
  }

  function renderEmptyConversation() {
    const { elements } = window.MedicDashboard;
    elements.chatHistory.innerHTML =
      `<p class="chat-empty">${formatting.escapeHtml(i18n.t("chat.empty"))}</p>`;
    elements.chatSources.innerHTML = "";
  }

  function appendMessage(message) {
    const { elements } = window.MedicDashboard;
    const item = document.createElement("article");
    item.className = `chat-message ${message.role}`;
    item.innerHTML = `
      <span>${message.role === "user" ? i18n.t("chat.user") : i18n.t("chat.agent")}</span>
      <div>${messageContentHtml(message)}</div>
    `;
    elements.chatHistory.appendChild(item);
    elements.chatHistory.scrollTop = elements.chatHistory.scrollHeight;
  }

  function appendTransientUserMessage(question) {
    appendMessage({
      role: "user",
      content: question,
      sources: [],
      trace_events: [],
      insufficient_context: false,
    });
  }

  function messageContentHtml(message) {
    if (message.role === "user") {
      return `<p>${formatting.escapeHtml(message.content || "")}</p>`;
    }
    const sources = message.sources || [];
    const contextWarning = message.insufficient_context
      ? `<p class="chat-warning">${formatting.escapeHtml(i18n.t("chat.insufficientContext"))}</p>`
      : "";
    return `
      ${agentLabels(message.trace_events || [])}
      ${contextWarning}
      <p>${linkifyCitations(message.content || "", sources)}</p>
      ${traceSummary(message.trace_events || [])}
    `;
  }

  function agentLabels(traceEvents) {
    const modelByAgent = modelByAgentMap(traceEvents);
    const coordinator = traceEvents.find(
      (event) => event.event_type === "coordinator",
    );
    const specialists = coordinator?.payload?.selected_agents || [];
    const agents = uniqueAgents([
      ...(modelByAgent.professor ? ["professor"] : []),
      ...specialists,
    ]);
    if (!agents.length) {
      return "";
    }
    const labels = agents
      .map((agent) => agentChip(agent, modelByAgent))
      .join("");
    return `<div class="agent-labels">${labels}</div>`;
  }

  function uniqueAgents(agents) {
    const seen = [];
    for (const agent of agents) {
      if (agent && !seen.includes(agent)) {
        seen.push(agent);
      }
    }
    return seen;
  }

  function modelByAgentMap(traceEvents) {
    const map = {};
    for (const event of traceEvents) {
      if (event.event_type !== "model_call" || !event.agent_name) {
        continue;
      }
      const model = event.payload && event.payload.model;
      if (model && !map[event.agent_name]) {
        map[event.agent_name] = model;
      }
    }
    return map;
  }

  function agentChip(agent, modelByAgent) {
    const model = modelByAgent[agent];
    const modelHtml = model
      ? `<span class="agent-model">${formatting.escapeHtml(model)}</span>`
      : "";
    return `<span class="agent-label">${formatting.escapeHtml(agentLabel(agent))}${modelHtml}</span>`;
  }

  function agentLabel(agent) {
    return String(agent || "-").replaceAll("_", " ");
  }

  function traceSummary(events) {
    if (!events.length) {
      return "";
    }
    return `
      <details class="trace-details">
        <summary>${formatting.escapeHtml(i18n.t("chat.traceSummary", { count: events.length }))}</summary>
        ${traceDetails(events)}
      </details>
    `;
  }

  function traceDetails(events) {
    if (!events.length) {
      return "";
    }
    const rows = events.map(traceRow).join("");
    return `<ol>${rows}</ol>`;
  }

  function traceRow(event) {
    return `
      <li>
        <div class="trace-main">
          <strong>${formatting.escapeHtml(event.title || event.event_type || "-")}</strong>
          <span>${formatting.escapeHtml(formatting.statusLabel(event.status || "-"))}</span>
        </div>
        <div class="trace-meta">
          ${traceMeta(i18n.t("chat.traceAgent"), event.agent_name)}
          ${traceMeta(i18n.t("chat.traceModel"), event.payload && event.payload.model)}
          ${traceMeta(i18n.t("chat.traceTool"), event.tool_name)}
          ${tracePayload(event.payload || {})}
        </div>
      </li>
    `;
  }

  function traceMeta(label, value) {
    if (!value) {
      return "";
    }
    return `<span><strong>${label}:</strong> ${formatting.escapeHtml(value)}</span>`;
  }

  function tracePayload(payload) {
    const compact = JSON.stringify(payload);
    if (!compact || compact === "{}") {
      return "";
    }
    return `<code>${formatting.escapeHtml(compact)}</code>`;
  }

  function linkifyCitations(text, sources) {
    const escaped = formatting.escapeHtml(text);
    return escaped.replace(/\[[^\]]*?\]/g, (group) => {
      if (!/S\d+/.test(group)) {
        return group;
      }
      return group.replace(/S\d+/g, (sourceId) => {
        const source = findSourceInList(sourceId, sources);
        if (!source) {
          return sourceId;
        }
        const escapedSourceId = formatting.escapeHtml(sourceId);
        const escapedRecordId = formatting.escapeHtml(sourceRecordKey(source));
        return `<button class="citation-button" type="button" data-source-id="${escapedSourceId}" data-source-record-id="${escapedRecordId}">${escapedSourceId}</button>`;
      });
    });
  }

  function renderSources(sources, expandedIds = new Set()) {
    const { elements } = window.MedicDashboard;
    if (!sources.length) {
      elements.chatSources.innerHTML =
        `<p class="muted">${formatting.escapeHtml(i18n.t("chat.noSources"))}</p>`;
      return;
    }

    const used = sources.filter((source) => source.used);
    const visible = used.length ? used : sources;
    const hidden = used.length ? sources.filter((source) => !source.used) : [];
    const items = visible
      .map((source) => sourceItem(source, expandedIds))
      .join("");
    elements.chatSources.innerHTML = `
      <div class="subpanel-header">
        <h3>${formatting.escapeHtml(i18n.t("chat.sources"))}</h3>
        <span>${formatting.escapeHtml(sourceCountLabel(visible.length))}</span>
      </div>
      <div class="source-list">${items}</div>
      ${unusedSourcesSection(hidden, expandedIds)}
    `;
  }

  function unusedSourcesSection(sources, expandedIds) {
    if (!sources.length) {
      return "";
    }
    const items = sources
      .map((source) => sourceItem(source, expandedIds))
      .join("");
    return `
      <details class="trace-details unused-sources">
        <summary>${formatting.escapeHtml(i18n.t("chat.unusedSources", { count: sources.length }))}</summary>
        <div class="source-list">${items}</div>
      </details>
    `;
  }

  function sourceItem(source, expandedIds = new Set()) {
    const id = sourceKey(source);
    const readInFull = expandedIds.has(id)
      ? `<span class="source-flag">${formatting.escapeHtml(i18n.t("chat.readInFull"))}</span>`
      : "";
    return `
      <article class="source-row">
        <div class="source-meta">
          <button
            class="source-open"
            type="button"
            data-source-id="${formatting.escapeHtml(id)}"
            data-source-record-id="${formatting.escapeHtml(sourceRecordKey(source))}"
          >${formatting.escapeHtml(id || "-")}</button>
          <span>${formatting.escapeHtml(source.document_name || source.source || i18n.t("chat.unknownSource"))}</span>
          <span>${scoreLabel(source.score)}</span>
          ${readInFull}
          <code>${formatting.escapeHtml(formatting.shortHash(source.content_hash))}</code>
        </div>
        <p>${formatting.escapeHtml(source.excerpt || "")}</p>
      </article>
    `;
  }

  function lastAssistantMessage(messages) {
    for (const message of [...messages].reverse()) {
      if (message.role === "assistant") {
        return message;
      }
    }
    return null;
  }

  function expandedSourceIdSet(message) {
    const events = (message && message.trace_events) || [];
    const event = events.find(
      (item) => item.event_type === "source_expansion",
    );
    return new Set(
      (event && event.payload && event.payload.expanded_source_ids) || [],
    );
  }

  function conversationStateKey(conversation) {
    const messages = conversation?.messages || [];
    for (const message of [...messages].reverse()) {
      if (message.role === "assistant") {
        return message.insufficient_context ? "chat.noContext" : "chat.answerReady";
      }
    }
    return "chat.ready";
  }

  function openSourceDrawer(sourceId, sourceRecordId = "") {
    const normalizedSourceId = normalizeSourceId(sourceId);
    const normalizedSourceRecordId = normalizeSourceRecordId(sourceRecordId);
    const source = findSource(normalizedSourceId, normalizedSourceRecordId);
    if (!source) {
      setState("chat.sourceNotFound");
      return;
    }
    const { elements } = window.MedicDashboard;
    state.currentSourceId = sourceKey(source);
    state.currentSourceRecordId = sourceRecordKey(source);
    elements.sourceDrawer.hidden = false;
    elements.sourceDrawerTitle.textContent = sourceKey(source);
    elements.sourceDrawerBody.innerHTML = `
      <div class="source-drawer-meta">
        ${sourceMeta(i18n.t("process.document"), source.document_name || source.source || "-")}
        ${sourceMeta("Query", source.retrieval_query || "-")}
        ${sourceMeta("Score", scoreLabel(source.score))}
        ${sourceMeta("Chunk", source.chunk_index ?? "-")}
        ${sourceMeta(i18n.t("process.range"), formatting.rangeLabel(source.char_start, source.char_end))}
        ${sourceMeta(i18n.t("format.hash"), formatting.shortHash(source.content_hash))}
      </div>
      <p>${formatting.escapeHtml(source.excerpt || "")}</p>
      <button
        class="secondary-button"
        type="button"
        data-jump-source="${formatting.escapeHtml(source.relative_raw_path || "")}"
        ${source.relative_raw_path ? "" : "disabled"}
      >
        ${formatting.escapeHtml(i18n.t("chat.showDocument"))}
      </button>
    `;
    focusSourceDrawer();
  }

  function sourceMeta(label, value) {
    return `<span><strong>${label}:</strong> ${formatting.escapeHtml(String(value))}</span>`;
  }

  function closeSourceDrawer() {
    const { elements } = window.MedicDashboard;
    state.currentSourceId = null;
    state.currentSourceRecordId = null;
    elements.sourceDrawer.hidden = true;
    elements.sourceDrawerBody.innerHTML = "";
  }

  function focusSourceDrawer() {
    const { elements } = window.MedicDashboard;
    elements.sourceDrawer.setAttribute("tabindex", "-1");
    if (elements.sourceDrawer.scrollIntoView) {
      elements.sourceDrawer.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
    if (elements.sourceDrawer.focus) {
      elements.sourceDrawer.focus({ preventScroll: true });
    }
    setState("chat.sourceState", {
      sourceId: elements.sourceDrawerTitle.textContent || "",
    });
  }

  async function jumpToSource(relativeRawPath) {
    if (!relativeRawPath) {
      return;
    }
    await window.MedicDashboard.processDetails.load(relativeRawPath);
    window.MedicDashboard.processDetails.showTab("chunks");
    const panel = document.querySelector(".process-panel");
    if (panel?.scrollIntoView) {
      panel.scrollIntoView({ block: "start", behavior: "smooth" });
    }
  }

  function findSource(sourceId, sourceRecordId = "") {
    if (sourceRecordId) {
      return findSourceByRecordId(sourceRecordId);
    }
    return findSourceByDisplayId(sourceId);
  }

  function findSourceByRecordId(sourceRecordId) {
    const messages = state.currentConversation?.messages || [];
    for (const message of messages) {
      for (const source of message.sources || []) {
        if (normalizeSourceRecordId(sourceRecordKey(source)) === sourceRecordId) {
          return source;
        }
      }
    }
    return null;
  }

  function findSourceByDisplayId(sourceId) {
    const messages = state.currentConversation?.messages || [];
    for (const message of messages) {
      const source = findSourceInList(sourceId, message.sources || []);
      if (source) {
        return source;
      }
    }
    return null;
  }

  function findSourceInList(sourceId, sources) {
    const normalizedSourceId = normalizeSourceId(sourceId);
    return sources.find(
      (source) => normalizeSourceId(sourceKey(source)) === normalizedSourceId,
    ) || null;
  }

  function sourceKey(source) {
    return String(source?.source_id || source?.id || "");
  }

  function sourceRecordKey(source) {
    return String(source?.id || "");
  }

  function normalizeSourceId(sourceId) {
    return String(sourceId || "").trim().replace(/^\[/, "").replace(/\]$/, "");
  }

  function normalizeSourceRecordId(sourceRecordId) {
    return String(sourceRecordId || "").trim();
  }

  function appendError(message) {
    appendMessage({
      role: "assistant",
      content: message || i18n.t("chat.fetchError"),
      sources: [],
      trace_events: [],
      insufficient_context: false,
    });
  }

  function clearEmptyState() {
    const { elements } = window.MedicDashboard;
    const empty = elements.chatHistory.querySelector(".chat-empty");
    if (empty) {
      empty.remove();
    }
  }

  function setLoading(isLoading) {
    const { elements } = window.MedicDashboard;
    elements.chatQuestion.disabled = isLoading;
    elements.chatSubmit.disabled = isLoading;
  }

  function setState(key, params = {}) {
    const element = window.MedicDashboard.elements.chatState;
    state.currentStateKey = key;
    state.currentStateParams = params;
    setI18nMetadata(element, key, params);
    element.textContent = i18n.t(key, params);
  }

  function localize() {
    const openSourceId = state.currentSourceId;
    if (state.currentConversation) {
      renderConversation(state.currentConversation, { preserveSource: true });
    } else {
      renderEmptyConversation();
    }
    if (!openSourceId) {
      setState(state.currentStateKey, state.currentStateParams);
    }
  }

  function messageCountLabel(count) {
    return i18n.count("chat.messageCount.one", "chat.messageCount.other", count);
  }

  function sourceCountLabel(count) {
    return i18n.count("chat.sourcesCount.one", "chat.sourcesCount.other", count);
  }

  function setI18nMetadata(element, key, params) {
    if (!element.dataset) {
      return;
    }
    element.dataset.i18n = key;
    for (const name of Object.keys(element.dataset)) {
      if (name.startsWith("i18nParam")) {
        delete element.dataset[name];
      }
    }
    for (const [name, value] of Object.entries(params)) {
      const dataKey = `i18nParam${name.charAt(0).toUpperCase()}${name.slice(1)}`;
      element.dataset[dataKey] = String(value);
    }
  }

  function scoreLabel(score) {
    if (score === null || score === undefined) {
      return "score -";
    }
    const value = Number(score);
    return Number.isFinite(value) ? `score ${value.toFixed(3)}` : "score -";
  }

  function attachHandlers() {
    if (state.handlersAttached) {
      return;
    }
    const { elements } = window.MedicDashboard;
    elements.conversationList.addEventListener("click", async (event) => {
      const button = event.target.closest("[data-conversation-id]");
      if (!button) {
        return;
      }
      await loadConversation(button.dataset.conversationId);
    });
    elements.chatHistory.addEventListener("click", (event) => {
      const button = event.target.closest("[data-source-id]");
      if (!button) {
        return;
      }
      event.preventDefault();
      openSourceDrawer(
        button.dataset.sourceId || button.textContent,
        button.dataset.sourceRecordId || "",
      );
    });
    elements.chatSources.addEventListener("click", (event) => {
      const button = event.target.closest("[data-source-id]");
      if (!button) {
        return;
      }
      event.preventDefault();
      openSourceDrawer(
        button.dataset.sourceId || button.textContent,
        button.dataset.sourceRecordId || "",
      );
    });
    elements.sourceDrawerBody.addEventListener("click", async (event) => {
      const button = event.target.closest("[data-jump-source]");
      if (!button) {
        return;
      }
      await jumpToSource(button.dataset.jumpSource);
    });
    state.handlersAttached = true;
  }

  return {
    ask,
    closeSourceDrawer,
    localize,
    loadConversation,
    loadConversations,
    startNewConversation,
  };
})();
