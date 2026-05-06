# -*- coding: utf-8 -*-
"""Fix HTML file to use only English text with proper UTF-8 encoding"""

html_content = r'''<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research RAG - Academic Paper Assistant</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>book</text></svg>">
    <link rel="stylesheet" href="/static/css/main.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/styles/github-dark.min.css">
    <script src="https://cdn.jsdelivr.net/npm/marked@11.0.0/marked.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/highlight.min.js"></script>
</head>

<body>
    <div class="app-container">
        <aside class="sidebar" id="sidebar">
            <div class="sidebar-header">
                <div class="logo">
                    <svg class="logo-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                        <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
                        <line x1="9" y1="7" x2="16" y2="7" />
                        <line x1="9" y1="11" x2="14" y2="11" />
                    </svg>
                    <span class="logo-text">Research RAG</span>
                </div>
                <button class="btn-icon" id="newChatBtn" title="New Chat">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="12" y1="5" x2="12" y2="19" />
                        <line x1="5" y1="12" x2="19" y2="12" />
                    </svg>
                </button>
            </div>

            <div class="sidebar-section">
                <div class="section-title">Quick Actions</div>
                <div class="quick-actions">
                    <button class="action-btn" onclick="insertQuery('Summarize the core contributions of VGGT-based methods')">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                            <polyline points="14 2 14 8 20 8" />
                        </svg>
                        <span>Paper Summary</span>
                    </button>
                    <button class="action-btn" onclick="insertQuery('Compare methodologies and experimental results in the literature')">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="18" y1="20" x2="18" y2="10" />
                            <line x1="12" y1="20" x2="12" y2="4" />
                            <line x1="6" y1="20" x2="6" y2="14" />
                        </svg>
                        <span>Method Comparison</span>
                    </button>
                    <button class="action-btn" onclick="insertQuery('What are the main limitations and future directions mentioned in these papers?')">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="10" />
                            <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
                            <line x1="12" y1="17" x2="12.01" y2="17" />
                        </svg>
                        <span>Find Limitations</span>
                    </button>
                </div>
            </div>

            <div class="sidebar-section">
                <div class="section-title">Settings</div>
                <div class="setting-item">
                    <label>Retrieval Mode</label>
                    <select id="retrievalMode" onchange="updateConfig()">
                        <option value="fast">Fast</option>
                        <option value="balanced" selected>Balanced</option>
                        <option value="accurate">Accurate</option>
                    </select>
                </div>
                <div class="setting-item">
                    <label>Model</label>
                    <select id="modelMode" onchange="updateConfig()">
                        <option value="online" selected>Online Model (GLM)</option>
                        <option value="local">Local Model (Ollama)</option>
                    </select>
                </div>
                <div class="setting-item">
                    <label>Theme</label>
                    <div class="theme-toggle">
                        <button class="theme-btn active" data-theme="dark" onclick="setTheme('dark')">Dark</button>
                        <button class="theme-btn" data-theme="light" onclick="setTheme('light')">Light</button>
                    </div>
                </div>
            </div>

            <div class="sidebar-footer">
                <div class="status-indicator">
                    <span class="status-dot" id="statusDot"></span>
                    <span id="statusText">Connecting...</span>
                </div>
            </div>
        </aside>

        <main class="main-content">
            <header class="chat-header">
                <button class="btn-icon" id="sidebarToggle" onclick="toggleSidebar()">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="3" y1="12" x2="21" y2="12" />
                        <line x1="3" y1="6" x2="21" y2="6" />
                        <line x1="3" y1="18" x2="21" y2="18" />
                    </svg>
                </button>
                <div class="header-title">
                    <h1>Research Assistant</h1>
                    <span class="header-subtitle" id="headerSubtitle">Powered by RF-Mem Memory Retrieval</span>
                </div>
                <div class="header-actions">
                    <button class="btn-icon" onclick="clearChat()" title="Clear Chat">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="3 6 5 6 21 6" />
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                        </svg>
                    </button>
                </div>
            </header>

            <div class="chat-container" id="chatContainer">
                <div class="welcome-screen" id="welcomeScreen">
                    <div class="welcome-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                            <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                            <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
                        </svg>
                    </div>
                    <h2>How can I help with your research?</h2>
                    <p>Ask questions about academic papers, compare research methods, or explore research trends.</p>
                    <div class="suggestion-grid">
                        <button class="suggestion-card" onclick="insertQuery('What are the main contributions of VGGT and its variants?')">
                            <div class="suggestion-icon">summary</div>
                            <div class="suggestion-text">Summarize VGGT Contributions</div>
                        </button>
                        <button class="suggestion-card" onclick="insertQuery('Compare low-light image enhancement methods: Retinex-based vs Transformer-based')">
                            <div class="suggestion-icon">compare</div>
                            <div class="suggestion-text">Compare Enhancement Methods</div>
                        </button>
                        <button class="suggestion-card" onclick="insertQuery('What are the common datasets and evaluation metrics for low-light image enhancement?')">
                            <div class="suggestion-icon">data</div>
                            <div class="suggestion-text">Explore Datasets and Metrics</div>
                        </button>
                        <button class="suggestion-card" onclick="insertQuery('What are the main challenges and future research directions in this field?')">
                            <div class="suggestion-icon">future</div>
                            <div class="suggestion-text">Future Research Directions</div>
                        </button>
                    </div>
                </div>
            </div>

            <div class="input-area">
                <div class="input-container">
                    <div class="input-wrapper">
                        <textarea id="messageInput" placeholder="Enter your research question..." rows="1" onkeydown="handleKeyDown(event)"
                            oninput="autoResize(this)"></textarea>
                        <button class="btn-send" id="sendBtn" onclick="sendMessage()" disabled>
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <line x1="22" y1="2" x2="11" y2="13" />
                                <polygon points="22 2 15 22 11 13 2 9 22 2" />
                            </svg>
                        </button>
                    </div>
                    <div class="input-footer">
                        <span class="input-hint">Press Enter to send, Shift+Enter for new line</span>
                        <span class="input-info" id="inputInfo"></span>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <div class="source-modal" id="sourceModal">
        <div class="modal-overlay" onclick="closeSourceModal()"></div>
        <div class="modal-content">
            <div class="modal-header">
                <h3>Source Documents</h3>
                <button class="btn-close" onclick="closeSourceModal()">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18" />
                        <line x1="6" y1="6" x2="18" y2="18" />
                    </svg>
                </button>
            </div>
            <div class="modal-body" id="sourceModalBody"></div>
        </div>
    </div>

    <script src="/static/js/app.js"></script>
</body>

</html>'''

if __name__ == "__main__":
    file_path = r"d:\Code\Python\research_rag\web\index.html"
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"File written: {file_path}")
    
    with open(file_path, "rb") as f:
        data = f.read()
        try:
            data.decode("utf-8")
            print("UTF-8 verification: PASSED")
        except UnicodeDecodeError as e:
            print(f"UTF-8 verification: FAILED - {e}")
