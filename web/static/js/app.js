// State management
const state = {
  isStreaming: false,
  sessionId: "default",
  currentController: null,
  config: {
    retrievalMode: "balanced",
    modelMode: "online",
  },
  messages: [],
  sessionCounter: 1,
};

// Generate a new session ID
function generateSessionId() {
  state.sessionCounter++;
  return "session_" + Date.now() + "_" + state.sessionCounter;
}

// Switch to a different session
async function switchSession(sessionId) {
  if (state.isStreaming) return;

  state.sessionId = sessionId;
  state.messages = [];

  chatContainer.innerHTML = "";
  chatContainer.appendChild(welcomeScreen);
  welcomeScreen.style.display = "flex";

  await loadHistory();
  updateSessionListUI();

  if (sessionId === "default") {
    document.getElementById("headerSubtitle").textContent =
      "Powered by RF-Mem Memory Retrieval";
  }
}

// Create a new chat session
function newChat() {
  if (state.isStreaming) return;

  const newId = generateSessionId();
  state.sessionId = newId;
  state.messages = [];

  chatContainer.innerHTML = "";
  chatContainer.appendChild(welcomeScreen);
  welcomeScreen.style.display = "flex";

  loadSessionList();
  updateSessionListUI();

  messageInput.focus();
}

// Load session list from server
async function loadSessionList() {
  try {
    const response = await fetch("/api/history/sessions");
    const data = await response.json();

    const listEl = document.getElementById("sessionList");
    if (!listEl) return;

    let html = `
      <div class="session-list-item ${state.sessionId === "default" ? "active" : ""}" 
           data-session="default" onclick="switchSession('default')">
        <div class="session-item-content">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
          <span class="session-title">Current Chat</span>
        </div>
      </div>
    `;

    if (data.sessions && data.sessions.length > 0) {
      for (const session of data.sessions) {
        const sid = session.session_id;
        const title =
          session.title || session.summary?.substring(0, 50) || "Untitled Chat";
        const count = session.message_count || 0;
        const updated = session.updated_at || "";
        const isActive = sid === state.sessionId;

        html += `
          <div class="session-list-item ${isActive ? "active" : ""}" 
               data-session="${sid}" onclick="switchSession('${sid}')">
            <div class="session-item-content">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
              </svg>
              <span class="session-title" title="${title}">${title}</span>
              <span class="session-meta">${count} messages</span>
            </div>
            <button class="session-rename-btn" onclick="event.stopPropagation(); renameSession('${sid}')" title="Rename">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z" />
              </svg>
            </button>
            <button class="session-delete-btn" onclick="event.stopPropagation(); deleteSession('${sid}')" title="Delete">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
        `;
      }
    }

    listEl.innerHTML = html;
  } catch (error) {
    console.error("Failed to load session list:", error);
  }
}

// Delete a session
async function deleteSession(sessionId) {
  if (
    !confirm(
      "Are you sure you want to delete this conversation? This action cannot be undone.",
    )
  ) {
    return;
  }

  try {
    const response = await fetch(`/api/history/${sessionId}`, {
      method: "DELETE",
    });
    const data = await response.json();

    if (data.success) {
      if (state.sessionId === sessionId) {
        state.sessionId = "default";
        state.messages = [];
        chatContainer.innerHTML = "";
        chatContainer.appendChild(welcomeScreen);
        welcomeScreen.style.display = "flex";
      }
      loadSessionList();
    }
  } catch (error) {
    console.error("Failed to delete session:", error);
  }
}

// Rename a session
async function renameSession(sessionId) {
  const item = document.querySelector(
    `.session-list-item[data-session="${sessionId}"]`,
  );
  if (!item) return;

  const titleEl = item.querySelector(".session-title");
  const currentTitle = titleEl.textContent.trim();

  const newTitle = prompt(
    "Enter a new name for this conversation:",
    currentTitle,
  );
  if (newTitle === null || newTitle.trim() === "") return;

  try {
    const response = await fetch(`/api/history/${sessionId}/rename`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, title: newTitle.trim() }),
    });
    const data = await response.json();

    if (data.success) {
      loadSessionList();
    }
  } catch (error) {
    console.error("Failed to rename session:", error);
  }
}

// Update session list UI to reflect active state
function updateSessionListUI() {
  document.querySelectorAll(".session-list-item").forEach((item) => {
    const sid = item.getAttribute("data-session");
    item.classList.toggle("active", sid === state.sessionId);
  });
}

// DOM Elements
const chatContainer = document.getElementById("chatContainer");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const welcomeScreen = document.getElementById("welcomeScreen");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const inputInfo = document.getElementById("inputInfo");

// Initialize marked.js
marked.setOptions({
  highlight: function (code, lang) {
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(code, { language: lang }).value;
    }
    return hljs.highlightAuto(code).value;
  },
  breaks: true,
  gfm: true,
});

// Check server health
async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();

    statusDot.classList.add("connected");
    statusDot.classList.remove("error");
    statusText.textContent = "Connected";

    if (data.retrieval_mode) {
      document.getElementById("retrievalMode").value = data.retrieval_mode;
    }
    if (data.llm_mode) {
      document.getElementById("modelMode").value = data.llm_mode;
    }
  } catch (error) {
    statusDot.classList.add("error");
    statusDot.classList.remove("connected");
    statusText.textContent = "Disconnected";
    console.error("Health check failed:", error);
  }
}

// Auto-resize textarea
function autoResize(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 150) + "px";
  sendBtn.disabled = !textarea.value.trim();
}

// Handle Enter key
function handleKeyDown(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    if (messageInput.value.trim() && !state.isStreaming) {
      sendMessage();
    }
  }
}

// Insert query from suggestions
function insertQuery(query) {
  messageInput.value = query;
  autoResize(messageInput);
  messageInput.focus();
}

// Toggle sidebar
function toggleSidebar() {
  document.getElementById("sidebar").classList.toggle("open");
}

// Set theme
function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  document.querySelectorAll(".theme-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.theme === theme);
  });
  localStorage.setItem("theme", theme);
}

// Update config
async function updateConfig() {
  const mode = document.getElementById("retrievalMode").value;
  const model = document.getElementById("modelMode").value;

  try {
    await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode, model }),
    });

    state.config.retrievalMode = mode;
    state.config.modelMode = model;

    const modeNames = {
      fast: "Fast",
      balanced: "Balanced",
      accurate: "Accurate",
    };
    const modelNames = { online: "Online", local: "Local" };
    const subtitle = document.getElementById("headerSubtitle");
    subtitle.textContent = `Mode: ${modeNames[mode] || mode} | Model: ${modelNames[model] || model}`;
  } catch (error) {
    console.error("Failed to update config:", error);
  }
}

// Clear chat
function clearChat() {
  if (state.isStreaming) return;

  chatContainer.innerHTML = "";
  chatContainer.appendChild(welcomeScreen);
  welcomeScreen.style.display = "flex";
  state.messages = [];

  fetch("/api/session/clear", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: state.sessionId }),
  }).catch(console.error);
}

// Get current time string
function getTimeString() {
  return new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

// Add message to chat
function addMessage(role, content, sources = null) {
  welcomeScreen.style.display = "none";

  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${role}`;

  const avatar = role === "user" ? "U" : "AI";
  const sender = role === "user" ? "You" : "Research Assistant";

  let sourcesHTML = "";
  if (sources && sources.length > 0) {
    sourcesHTML = `
            <div class="message-sources">
                <div class="sources-header" onclick="showSources(${JSON.stringify(sources).replace(/"/g, "&quot;")})">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
                        <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
                    </svg>
                    Cited ${sources.length} paper(s)
                </div>
                <div class="source-tags">
                    ${sources
                      .map(
                        (s) => `
                        <span class="source-tag" onclick="showSources([${JSON.stringify(s).replace(/"/g, "&quot;")}])">
                            ${s.title || "Unknown"} (${s.year || "?"})
                        </span>
                    `,
                      )
                      .join("")}
                </div>
            </div>
        `;
  }

  messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-header">
                <span class="message-sender">${sender}</span>
                <span class="message-time">${getTimeString()}</span>
            </div>
            <div class="message-bubble">${role === "ai" ? marked.parse(content) : escapeHtml(content)}</div>
            ${sourcesHTML}
            <div class="message-actions">
                <button class="msg-action-btn" onclick="copyMessage(this)">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                    </svg>
                    Copy
                </button>
            </div>
        </div>
    `;

  chatContainer.appendChild(messageDiv);
  scrollToBottom();

  // Apply syntax highlighting
  messageDiv.querySelectorAll("pre code").forEach((block) => {
    hljs.highlightElement(block);
  });

  return messageDiv;
}

// Add streaming message placeholder
function addStreamingMessage() {
  welcomeScreen.style.display = "none";

  const messageDiv = document.createElement("div");
  messageDiv.className = "message ai";
  messageDiv.id = "streaming-message";

  messageDiv.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-content">
            <div class="message-header">
                <span class="message-sender">Research Assistant</span>
                <span class="message-time">${getTimeString()}</span>
            </div>
            <div class="message-bubble">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        </div>
    `;

  chatContainer.appendChild(messageDiv);
  scrollToBottom();
  return messageDiv;
}

// Add status message
function addStatusMessage(text) {
  const statusDiv = document.createElement("div");
  statusDiv.className = "status-message";
  statusDiv.id = "status-message";
  statusDiv.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
        </svg>
        <span>${text}</span>
    `;

  const existing = document.getElementById("status-message");
  if (existing) existing.remove();

  chatContainer.appendChild(statusDiv);
  scrollToBottom();
}

// Remove status message
function removeStatusMessage() {
  const existing = document.getElementById("status-message");
  if (existing) existing.remove();
}

// Escape HTML
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// Scroll to bottom
function scrollToBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Copy message
function copyMessage(btn) {
  const bubble = btn
    .closest(".message-content")
    .querySelector(".message-bubble");
  const text = bubble.innerText;
  navigator.clipboard.writeText(text).then(() => {
    const original = btn.innerHTML;
    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> Copied`;
    setTimeout(() => (btn.innerHTML = original), 2000);
  });
}

// Show sources modal
function showSources(sources) {
  const modal = document.getElementById("sourceModal");
  const body = document.getElementById("sourceModalBody");

  body.innerHTML = sources
    .map(
      (s) => `
        <div class="source-item">
            <h4>${s.title || "Unknown Document"}</h4>
            <div class="source-meta">
                <span>Year: ${s.year || "?"}</span>
                ${s.section ? `<span>Section: ${s.section}</span>` : ""}
                ${s.doc_id ? `<span>ID: ${s.doc_id}</span>` : ""}
            </div>
        </div>
    `,
    )
    .join("");

  modal.classList.add("active");
}

// Close sources modal
function closeSourceModal() {
  document.getElementById("sourceModal").classList.remove("active");
}

// Send message with SSE streaming
async function sendMessage() {
  const message = messageInput.value.trim();
  if (!message || state.isStreaming) return;

  state.isStreaming = true;
  sendBtn.disabled = true;
  messageInput.value = "";
  autoResize(messageInput);

  addMessage("user", message);
  const streamingMsg = addStreamingMessage();

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: message,
        session_id: state.sessionId,
      }),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let fullResponse = "";
    let sources = null;
    let eventType = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("event:")) {
          eventType = line.slice(6).trim();
          continue;
        }

        if (line.startsWith("data:")) {
          const dataStr = line.slice(5).trim();
          if (!dataStr) continue;

          try {
            const data = JSON.parse(dataStr);

            if (eventType === "status") {
              addStatusMessage(data.message);
            } else if (eventType === "token") {
              removeStatusMessage();
              fullResponse += data.token;

              const bubble = streamingMsg.querySelector(".message-bubble");
              bubble.innerHTML = marked.parse(fullResponse);

              bubble.querySelectorAll("pre code").forEach((block) => {
                hljs.highlightElement(block);
              });

              scrollToBottom();
            } else if (eventType === "complete") {
              removeStatusMessage();
              sources = data.sources;
              inputInfo.textContent = `Time: ${data.elapsed}s | References: ${data.doc_count} paper(s)`;
            } else if (eventType === "error") {
              removeStatusMessage();
              fullResponse = `Error: ${data.error}`;
            }
          } catch (e) {
            console.error("Parse error:", e);
          }
        }
      }
    }

    streamingMsg.id = "";

    if (sources) {
      const sourcesDiv = document.createElement("div");
      sourcesDiv.className = "message-sources";
      sourcesDiv.innerHTML = `
                <div class="sources-header" onclick='showSources(${JSON.stringify(sources)})'>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
                        <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
                    </svg>
                    Cited ${sources.length} paper(s)
                </div>
                <div class="source-tags">
                    ${sources
                      .map(
                        (s) => `
                        <span class="source-tag" onclick='showSources([${JSON.stringify(s)}])'>
                            ${s.title || "Unknown"} (${s.year || "?"})
                        </span>
                    `,
                      )
                      .join("")}
                </div>
            `;
      streamingMsg.querySelector(".message-content").appendChild(sourcesDiv);
    }
  } catch (error) {
    removeStatusMessage();
    console.error("Stream error:", error);

    const bubble = streamingMsg.querySelector(".message-bubble");
    bubble.innerHTML = `<p style="color: var(--error)">Connection error, please retry.</p>`;
    streamingMsg.id = "";
  } finally {
    state.isStreaming = false;
    sendBtn.disabled = !messageInput.value.trim();
  }
}

// Load conversation history from server
async function loadHistory() {
  try {
    const response = await fetch(`/api/history?session_id=${state.sessionId}`);
    const data = await response.json();

    if (data.success && data.messages && data.messages.length > 0) {
      state.messages = data.messages;

      for (const msg of data.messages) {
        if (msg.role === "user") {
          addMessage("user", msg.content);
        } else if (msg.role === "assistant") {
          addMessage("ai", msg.content);
        }
      }

      console.log(`Loaded ${data.messages.length} messages from history`);
    }
  } catch (error) {
    console.error("Failed to load history:", error);
  }
}

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  const savedTheme = localStorage.getItem("theme") || "dark";
  setTheme(savedTheme);

  checkHealth();
  setInterval(checkHealth, 30000);

  document.getElementById("newChatBtn").addEventListener("click", newChat);

  loadSessionList();

  messageInput.focus();

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeSourceModal();
    }
  });
});

