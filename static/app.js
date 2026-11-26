const savedProvider = localStorage.getItem("rag_provider");

const state = {
  provider: savedProvider || "ollama",
  hasSavedProvider: Boolean(savedProvider),
  style: localStorage.getItem("rag_style") || "simple",
  customInstruction: localStorage.getItem("rag_custom_instruction") || "",
  apiKey: localStorage.getItem("rag_gemini_key") || "",
  model: "",
  documents: [],
  selectedDocIds: new Set(),
  history: [],
  health: null,
  busy: false,
};

const els = {
  sidebar: document.getElementById("sidebar"),
  menuBtn: document.getElementById("menuBtn"),
  closeSidebarBtn: document.getElementById("closeSidebarBtn"),
  uploadTriggerBtn: document.getElementById("uploadTriggerBtn"),
  attachBtn: document.getElementById("attachBtn"),
  fileInput: document.getElementById("fileInput"),
  uploadStatus: document.getElementById("uploadStatus"),
  toggleAllDocsBtn: document.getElementById("toggleAllDocsBtn"),
  docList: document.getElementById("docList"),
  noFilesMsg: document.getElementById("noFilesMsg"),
  pillBtns: document.querySelectorAll(".pill-btn"),
  styleSelect: document.getElementById("styleSelect"),
  settingsBtn: document.getElementById("settingsBtn"),
  statusDot: document.getElementById("statusDot"),
  statusLabel: document.getElementById("statusLabel"),
  clearChatBtn: document.getElementById("clearChatBtn"),
  messages: document.getElementById("messages"),
  emptyState: document.getElementById("emptyState"),
  chatForm: document.getElementById("chatForm"),
  questionInput: document.getElementById("questionInput"),
  sendBtn: document.getElementById("sendBtn"),
  settingsModal: document.getElementById("settingsModal"),
  closeSettingsBtn: document.getElementById("closeSettingsBtn"),
  doneSettingsBtn: document.getElementById("doneSettingsBtn"),
  statusDetail: document.getElementById("statusDetail"),
  apiKeyGroup: document.getElementById("apiKeyGroup"),
  apiKeyInput: document.getElementById("apiKeyInput"),
  apiKeyHint: document.getElementById("apiKeyHint"),
  modelSelect: document.getElementById("modelSelect"),
  customInstructionInput: document.getElementById("customInstructionInput"),
};

function authHeaders() {
  const headers = {};
  if (state.apiKey.trim()) {
    headers["X-Gemini-Api-Key"] = state.apiKey.trim();
  }
  return headers;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatInline(text) {
  return text
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*\n]+)\*/g, "<em>$1</em>")
    .replace(
      /\[([^\]|]+ \| Page [0-9, ]+)\]/g,
      '<span class="cite-pill">[$1]</span>'
    );
}

function renderMarkdown(rawText) {
  let html = escapeHtml(rawText || "");
  const codeBlocks = [];

  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, _lang, code) => {
    const idx = codeBlocks.length;
    codeBlocks.push(`<pre><code>${code}</code></pre>`);
    return `@@CODEBLOCK_${idx}@@`;
  });

  const lines = html.split("\n");
  const out = [];
  let listType = null;

  const closeList = () => {
    if (listType) {
      out.push(`</${listType}>`);
      listType = null;
    }
  };

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      closeList();
      continue;
    }
    if (trimmed.startsWith("@@CODEBLOCK_")) {
      closeList();
      out.push(trimmed);
      continue;
    }
    const headingMatch = trimmed.match(/^(#{1,3})\s+(.*)$/);
    if (headingMatch) {
      closeList();
      const level = Math.min(4, headingMatch[1].length + 1);
      out.push(`<h${level}>${formatInline(headingMatch[2])}</h${level}>`);
      continue;
    }
    const ulMatch = trimmed.match(/^[-*]\s+(.*)$/);
    if (ulMatch) {
      if (listType !== "ul") {
        closeList();
        listType = "ul";
        out.push("<ul>");
      }
      out.push(`<li>${formatInline(ulMatch[1])}</li>`);
      continue;
    }
    const olMatch = trimmed.match(/^\d+\.\s+(.*)$/);
    if (olMatch) {
      if (listType !== "ol") {
        closeList();
        listType = "ol";
        out.push("<ol>");
      }
      out.push(`<li>${formatInline(olMatch[1])}</li>`);
      continue;
    }
    closeList();
    out.push(`<p>${formatInline(trimmed)}</p>`);
  }
  closeList();

  let rendered = out.join("");
  codeBlocks.forEach((block, idx) => {
    rendered = rendered.replace(`@@CODEBLOCK_${idx}@@`, block);
  });
  return rendered || "<p></p>";
}

const MODEL_LABELS = {
  "gemini-4-hydrogen": "gemini-4-hydrogen (Recommended • Free tier)",
  "gemini-3.5-flash-lite": "gemini-3.5-flash-lite (Fastest • Free tier)",
  "gemini-4-neon": "gemini-4-neon (Paid tier)",
  "gemini-4-carbon": "gemini-4-carbon (Paid tier)",
};

function populateModelSelect(models, preferred) {
  const prev = state.model;
  els.modelSelect.innerHTML = "";
  models.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = MODEL_LABELS[m] || m;
    els.modelSelect.appendChild(opt);
  });

  if (models.includes(prev)) state.model = prev;
  else if (models.includes(preferred)) state.model = preferred;
  else state.model = models[0] || "";
  els.modelSelect.value = state.model;
}

function renderProviderUI() {
  els.pillBtns.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.provider === state.provider);
  });

  const isGemini = state.provider === "gemini";
  els.apiKeyGroup.hidden = !isGemini;

  const health = state.health;
  if (!health) return;

  if (isGemini) {
    const gem = health.gemini;
    const models = gem.chat_models || ["gemini-4-hydrogen"];
    populateModelSelect(models, health.defaults.gemini_chat_model);

    if (gem.configured) {
      els.statusDot.className = "dot ok";
      els.statusLabel.textContent = state.model || "Gemini";
      els.statusDetail.textContent = "Online mode is ready with Google Gemini.";
    } else {
      els.statusDot.className = "dot err";
      els.statusLabel.textContent = "Add API Key";
      els.statusDetail.textContent = "Paste your Gemini API key below to chat online.";
    }
    els.apiKeyHint.textContent = gem.has_server_key
      ? "Server key detected. Optional: override above."
      : "Saved only in your browser.";
  } else {
    const oll = health.ollama;
    const models = oll.models && oll.models.length > 0
      ? oll.models
      : [health.defaults.ollama_chat_model];
    populateModelSelect(models, health.defaults.ollama_chat_model);

    if (oll.reachable) {
      els.statusDot.className = "dot ok";
      els.statusLabel.textContent = state.model || "Ollama";
      els.statusDetail.textContent = `Offline mode ready (${oll.models.length} local models found).`;
    } else {
      els.statusDot.className = "dot err";
      els.statusLabel.textContent = "Ollama Off";
      els.statusDetail.textContent =
        "Offline mode needs Ollama running on your own PC. On Railway/cloud, switch to Online mode at the top.";
    }
  }
}

async function refreshHealth() {
  try {
    const resp = await fetch("/health", { headers: authHeaders() });
    state.health = await resp.json();
    // On cloud deployments where Ollama isn't running, default first-time visitors to Online.
    if (!state.hasSavedProvider && !state.health.ollama.reachable) {
      state.provider = "gemini";
    }
    renderProviderUI();
  } catch (err) {
    els.statusDot.className = "dot err";
    els.statusLabel.textContent = "Offline";
    els.statusDetail.textContent = "Could not reach the local backend server.";
  }
}

async function refreshDocuments() {
  try {
    const resp = await fetch("/documents");
    const docs = await resp.json();
    const prevAll =
      state.documents.length === 0 ||
      state.selectedDocIds.size === state.documents.length;

    state.documents = docs;
    if (prevAll) {
      state.selectedDocIds = new Set(docs.map((d) => d.doc_id));
    } else {
      const valid = new Set(docs.map((d) => d.doc_id));
      state.selectedDocIds = new Set(
        [...state.selectedDocIds].filter((id) => valid.has(id))
      );
    }
    renderDocuments();
  } catch (err) {
    console.error("Failed to fetch documents", err);
  }
}

function renderDocuments() {
  const hasDocs = state.documents.length > 0;
  els.noFilesMsg.hidden = hasDocs;
  els.toggleAllDocsBtn.hidden = state.documents.length < 2;
  els.toggleAllDocsBtn.textContent =
    state.selectedDocIds.size === state.documents.length ? "Unselect all" : "Select all";

  els.docList.innerHTML = "";
  state.documents.forEach((doc) => {
    const li = document.createElement("li");
    li.className = "file-item";

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = state.selectedDocIds.has(doc.doc_id);
    cb.addEventListener("change", () => {
      if (cb.checked) state.selectedDocIds.add(doc.doc_id);
      else state.selectedDocIds.delete(doc.doc_id);
      renderDocuments();
    });

    const info = document.createElement("div");
    info.className = "file-info";
    const pageWord = doc.page_count === 1 ? "1 page" : `${doc.page_count} pages`;
    info.innerHTML = `
      <div class="file-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div>
      <div class="file-pages">${pageWord}</div>
    `;

    const del = document.createElement("button");
    del.type = "button";
    del.className = "file-del";
    del.title = "Remove file";
    del.innerHTML = "&times;";
    del.addEventListener("click", () => deleteDocument(doc.doc_id));

    li.appendChild(cb);
    li.appendChild(info);
    li.appendChild(del);
    els.docList.appendChild(li);
  });
}

async function deleteDocument(docId) {
  await fetch(`/documents/${encodeURIComponent(docId)}`, { method: "DELETE" });
  state.selectedDocIds.delete(docId);
  await refreshDocuments();
}

async function uploadFiles(fileList) {
  if (!fileList || fileList.length === 0) return;

  els.uploadStatus.className = "upload-feedback";
  els.uploadStatus.textContent = `Adding ${fileList.length} file(s)...`;

  const formData = new FormData();
  for (const file of fileList) {
    formData.append("files", file);
  }

  try {
    const resp = await fetch(`/documents/upload?provider=${encodeURIComponent(state.provider)}`, {
      method: "POST",
      headers: authHeaders(),
      body: formData,
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "Could not upload file");

    data.forEach((d) => state.selectedDocIds.add(d.doc_id));
    await refreshDocuments();
    els.uploadStatus.className = "upload-feedback ok";
    els.uploadStatus.textContent = "Ready! Ask anything.";
    setTimeout(() => {
      els.uploadStatus.textContent = "";
    }, 3000);
  } catch (err) {
    els.uploadStatus.className = "upload-feedback error";
    els.uploadStatus.textContent = err.message;
  } finally {
    els.fileInput.value = "";
  }
}

function appendMessage(role, initialText = "") {
  if (els.emptyState) els.emptyState.hidden = true;

  const row = document.createElement("div");
  row.className = `msg-row ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = renderMarkdown(initialText);

  row.appendChild(bubble);
  els.messages.appendChild(row);
  els.messages.scrollTop = els.messages.scrollHeight;
  return { row, bubble };
}

function attachSources(row, sources) {
  if (!sources || sources.length === 0) return;

  const seen = new Set();
  const uniquePages = [];
  for (const s of sources) {
    const key = `${s.filename}:${s.page_number}`;
    if (!seen.has(key)) {
      seen.add(key);
      uniquePages.push(s);
    }
  }

  const details = document.createElement("details");
  details.className = "sources-toggle";

  const summary = document.createElement("summary");
  summary.textContent = `View sources (${uniquePages.length} ${uniquePages.length === 1 ? "page" : "pages"})`;
  details.appendChild(summary);

  const list = document.createElement("div");
  list.className = "sources-list";

  uniquePages.forEach((src) => {
    const item = document.createElement("div");
    item.className = "source-item";
    item.innerHTML = `
      <div class="source-title">${escapeHtml(src.filename)} &bull; Page ${src.page_number}</div>
      <div class="source-excerpt">${escapeHtml(src.text)}</div>
    `;
    list.appendChild(item);
  });

  details.appendChild(list);
  row.appendChild(details);
  els.messages.scrollTop = els.messages.scrollHeight;
}

async function sendQuestion(questionText, reuseRows = null) {
  const question = questionText.trim();
  if (!question || state.busy) return;

  state.busy = true;
  els.sendBtn.disabled = true;

  let userRow;
  let botRefs;
  if (reuseRows) {
    userRow = reuseRows.userRow;
    botRefs = reuseRows.botRefs;
    const oldRetry = botRefs.row.querySelector(".retry-btn");
    if (oldRetry) oldRetry.remove();
    botRefs.bubble.innerHTML = renderMarkdown("Retrying...");
  } else {
    els.questionInput.value = "";
    els.questionInput.style.height = "auto";
    userRow = appendMessage("user", question).row;
    const thinkingLabel =
      state.selectedDocIds.size > 0 ? "Reading your files..." : "Thinking...";
    botRefs = appendMessage("assistant", thinkingLabel);
  }

  const docIds = [...state.selectedDocIds];

  let fullAnswer = "";
  let retrievedSources = [];

  try {
    const resp = await fetch("/chat/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...authHeaders(),
      },
      body: JSON.stringify({
        question,
        provider: state.provider,
        model: state.model || null,
        style: state.style,
        custom_instruction: state.customInstruction.trim() || null,
        doc_ids: docIds,
        history: state.history.slice(-6),
      }),
    });

    if (!resp.ok || !resp.body) {
      const errData = await resp.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        const evt = JSON.parse(line.slice(5).trim());

        if (evt.type === "sources") {
          retrievedSources = evt.sources || [];
        } else if (evt.type === "token") {
          fullAnswer += evt.content;
          botRefs.bubble.innerHTML = renderMarkdown(fullAnswer);
          els.messages.scrollTop = els.messages.scrollHeight;
        } else if (evt.type === "error") {
          throw new Error(evt.detail || "Something went wrong");
        }
      }
    }

    if (!fullAnswer) {
      throw new Error("No answer received. Check your model connection in Settings.");
    }

    attachSources(botRefs.row, retrievedSources);
    state.history.push({ role: "user", content: question });
    state.history.push({ role: "assistant", content: fullAnswer });
  } catch (err) {
    botRefs.bubble.innerHTML = renderMarkdown(`**Could not answer:** ${err.message}`);
    const retryBtn = document.createElement("button");
    retryBtn.type = "button";
    retryBtn.className = "retry-btn";
    retryBtn.textContent = "Retry";
    retryBtn.addEventListener("click", () => {
      sendQuestion(question, { userRow, botRefs });
    });
    botRefs.row.appendChild(retryBtn);
  } finally {
    state.busy = false;
    els.sendBtn.disabled = false;
    els.questionInput.focus();
  }
}

// Event listeners
els.pillBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    state.provider = btn.dataset.provider;
    state.hasSavedProvider = true;
    localStorage.setItem("rag_provider", state.provider);
    renderProviderUI();
    if (state.provider === "gemini" && state.health && !state.health.gemini.configured) {
      els.settingsModal.hidden = false;
    }
  });
});

els.styleSelect.value = state.style;
els.styleSelect.addEventListener("change", () => {
  state.style = els.styleSelect.value;
  localStorage.setItem("rag_style", state.style);
});

els.customInstructionInput.value = state.customInstruction;
els.customInstructionInput.addEventListener("input", () => {
  state.customInstruction = els.customInstructionInput.value;
  localStorage.setItem("rag_custom_instruction", state.customInstruction);
});

els.settingsBtn.addEventListener("click", () => {
  els.settingsModal.hidden = false;
});
els.closeSettingsBtn.addEventListener("click", () => {
  els.settingsModal.hidden = true;
});
els.doneSettingsBtn.addEventListener("click", () => {
  els.settingsModal.hidden = true;
});
els.settingsModal.addEventListener("click", (e) => {
  if (e.target === els.settingsModal) els.settingsModal.hidden = true;
});

els.apiKeyInput.value = state.apiKey;
els.apiKeyInput.addEventListener("input", () => {
  state.apiKey = els.apiKeyInput.value;
  localStorage.setItem("rag_gemini_key", state.apiKey);
  refreshHealth();
});

els.modelSelect.addEventListener("change", () => {
  state.model = els.modelSelect.value;
  renderProviderUI();
});

els.uploadTriggerBtn.addEventListener("click", () => els.fileInput.click());
els.attachBtn.addEventListener("click", () => els.fileInput.click());
els.fileInput.addEventListener("change", () => uploadFiles(els.fileInput.files));

els.uploadTriggerBtn.addEventListener("dragover", (e) => {
  e.preventDefault();
  els.uploadTriggerBtn.classList.add("dragover");
});
els.uploadTriggerBtn.addEventListener("dragleave", () => {
  els.uploadTriggerBtn.classList.remove("dragover");
});
els.uploadTriggerBtn.addEventListener("drop", (e) => {
  e.preventDefault();
  els.uploadTriggerBtn.classList.remove("dragover");
  uploadFiles(e.dataTransfer.files);
});

els.toggleAllDocsBtn.addEventListener("click", () => {
  if (state.selectedDocIds.size === state.documents.length) {
    state.selectedDocIds.clear();
  } else {
    state.selectedDocIds = new Set(state.documents.map((d) => d.doc_id));
  }
  renderDocuments();
});

els.clearChatBtn.addEventListener("click", () => {
  state.history = [];
  els.messages.innerHTML = "";
  if (els.emptyState) {
    els.emptyState.hidden = false;
    els.messages.appendChild(els.emptyState);
  }
});

document.querySelectorAll(".suggestion-btn").forEach((btn) => {
  btn.addEventListener("click", () => sendQuestion(btn.textContent.trim()));
});

els.questionInput.addEventListener("input", () => {
  els.questionInput.style.height = "auto";
  els.questionInput.style.height = `${Math.min(els.questionInput.scrollHeight, 150)}px`;
});

els.questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendQuestion(els.questionInput.value);
  }
});

els.chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  sendQuestion(els.questionInput.value);
});

els.menuBtn.addEventListener("click", () => els.sidebar.classList.add("open"));
els.closeSidebarBtn.addEventListener("click", () => els.sidebar.classList.remove("open"));

refreshHealth();
refreshDocuments();
