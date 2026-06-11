export class UIManager {
    constructor() {
        this.chatHistory = document.getElementById('chat-history');
        this.promptInput = document.getElementById('prompt-input');
        this.consoleOutput = document.getElementById('console-output');

        this.perceptionPanelExpanded = true;
        this.autortPanelExpanded = false;

        this.initTheme();
        this.initCameraRetry();
        this.restoreChat();
    }

    /* --- THEME MANAGEMENT --- */
    initTheme() {
        const saved = localStorage.getItem('acrl-theme');
        if (saved) document.body.setAttribute('data-theme', saved);
        this.updateThemeIndicator();
    }

    toggleTheme() {
        const current = document.body.getAttribute('data-theme');
        const isCurrentlyLight = current === 'light' ||
            (!current && window.matchMedia('(prefers-color-scheme: light)').matches);
        const next = isCurrentlyLight ? 'dark' : 'light';
        document.body.setAttribute('data-theme', next);
        localStorage.setItem('acrl-theme', next);
        this.updateThemeIndicator();
    }

    updateThemeIndicator() {
        const btn = document.getElementById('btn-theme');
        if (!btn) return;
        const current = document.body.getAttribute('data-theme');
        const isLight = current === 'light' ||
            (!current && window.matchMedia('(prefers-color-scheme: light)').matches);
        btn.title = `Theme: ${isLight ? 'Light' : 'Dark'} (click to switch)`;
        btn.classList.toggle('is-active', isLight);
    }

    /* --- TOAST NOTIFICATIONS --- */
    toast(message, level = 'info', durationMs = 4000) {
        const container = document.getElementById('toast-container');
        if (!container) return;
        const el = document.createElement('div');
        el.className = `toast toast-${level}`;
        el.textContent = message;
        container.appendChild(el);
        // force reflow then animate in
        requestAnimationFrame(() => el.classList.add('toast-show'));
        const remove = () => {
            el.classList.remove('toast-show');
            el.addEventListener('transitionend', () => el.remove(), { once: true });
        };
        el.addEventListener('click', remove);
        setTimeout(remove, durationMs);
    }

    /* --- CAMERA STREAM RETRY --- */
    initCameraRetry() {
        const img = document.getElementById('stream-rgb');
        if (!img) return;
        const url = '/api/stream/rgb';
        const placeholder = img.nextElementSibling;

        const showStream = () => {
            img.style.display = 'block';
            if (placeholder) placeholder.style.display = 'none';
        };

        const showOffline = () => {
            img.style.display = 'none';
            if (placeholder && placeholder.classList.contains('feed-placeholder')) {
                const icon = placeholder.querySelector('i');
                if (icon) icon.className = 'fa-solid fa-video-slash';
                const label = placeholder.querySelector('span');
                if (label) label.textContent = 'Camera (Offline)';
                placeholder.style.display = 'flex';
            }
        };

        // MJPEG streams never fire 'load' reliably — poll naturalWidth which is set
        // once the browser decodes the first frame from the multipart stream.
        let pollTimer = null;
        const pollStream = () => {
            if (img.naturalWidth > 0) { showStream(); return; }
            pollTimer = setTimeout(pollStream, 500);
        };

        img.addEventListener('error', () => {
            clearTimeout(pollTimer);
            showOffline();
            setTimeout(() => {
                img.src = `${url}?_t=${Date.now()}`;
                pollStream();
            }, 5000);
        });

        img.src = `${url}?_t=${Date.now()}`;
        pollStream();
    }

    /* --- LOGGING & CHAT UI --- */
    logToConsole(msg, level = 'info') {
        const div = document.createElement('div');
        div.className = `log-line ${level}`;
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
        div.textContent = `[${time}] ${msg}`;

        const { scrollHeight, clientHeight, scrollTop } = this.consoleOutput;
        const isScrolledToBottom = Math.abs(scrollHeight - clientHeight - scrollTop) < 5;

        this.consoleOutput.appendChild(div);

        if (isScrolledToBottom) {
            this.consoleOutput.scrollTop = this.consoleOutput.scrollHeight;
        }
    }

    downloadLogs() {
        const lines = Array.from(this.consoleOutput.querySelectorAll('.log-line'))
            .map(el => el.textContent)
            .join('\n');
        const blob = new Blob([lines], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `acrl-logs-${new Date().toISOString().replace(/[:.]/g, '-')}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    }

    /* --- CHAT (with refresh-persistent history) --- */
    appendChatMessage(text, sender) {
        this._renderChatEntry({ kind: 'message', sender, text });
        this._saveChatEntry({ kind: 'message', sender, text });
    }

    handleSequenceResult(data) {
        if (!data || !data.parsed_commands) return;
        this._renderChatEntry({ kind: 'result', data });
        this._saveChatEntry({ kind: 'result', data });
    }

    _renderChatEntry(entry) {
        const div = document.createElement('div');
        if (entry.kind === 'message') {
            const icon = entry.sender === 'user' ? 'fa-user' : 'fa-robot';
            div.className = `message ${entry.sender}`;
            div.innerHTML = `
                <div class="msg-icon"><i class="fa-solid ${icon}"></i></div>
                <div class="msg-content">${this._esc(entry.text)}</div>
            `;
        } else {
            const data = entry.data;
            let html = `<strong>Plan Generated (${data.parsed_commands.length} steps):</strong><br><ol class="plan-steps">`;
            data.parsed_commands.forEach(cmd => {
                html += `<li><span class="plan-op">${this._esc(cmd.operation)}</span>`;
                if (cmd.params) {
                    const paramsStr = Object.entries(cmd.params)
                        .map(([k, v]) => `${k}: ${v}`)
                        .join(', ');
                    html += ` <em>(${this._esc(paramsStr)})</em>`;
                }
                html += `</li>`;
            });
            html += `</ol>`;
            div.className = `message system`;
            div.innerHTML = `
                <div class="msg-icon"><i class="fa-solid fa-code-branch"></i></div>
                <div class="msg-content">${html}</div>
            `;
        }
        this.chatHistory.appendChild(div);
        this.chatHistory.scrollTop = this.chatHistory.scrollHeight;
    }

    _saveChatEntry(entry) {
        try {
            const log = JSON.parse(localStorage.getItem('acrl-chat') || '[]');
            log.push(entry);
            // cap to last 200 entries to bound storage
            localStorage.setItem('acrl-chat', JSON.stringify(log.slice(-200)));
        } catch { /* storage full / disabled — non-fatal */ }
    }

    restoreChat() {
        let log;
        try {
            log = JSON.parse(localStorage.getItem('acrl-chat') || '[]');
        } catch { return; }
        if (!Array.isArray(log) || log.length === 0) return;
        log.forEach(entry => this._renderChatEntry(entry));
    }

    clearChat() {
        localStorage.removeItem('acrl-chat');
        if (this.chatHistory) this.chatHistory.innerHTML = '';
        this.toast('Chat history cleared', 'info');
    }

    /* Export the full session (chat + logs) as a Markdown file. */
    exportMission() {
        const stamp = new Date().toISOString();
        const chat = (() => {
            try { return JSON.parse(localStorage.getItem('acrl-chat') || '[]'); }
            catch { return []; }
        })();
        let md = `# ACRL Mission Transcript\n\n_Exported ${stamp}_\n\n## Conversation\n\n`;
        chat.forEach(e => {
            if (e.kind === 'message') {
                md += `**${e.sender}:** ${e.text}\n\n`;
            } else if (e.kind === 'result') {
                const ops = (e.data.parsed_commands || [])
                    .map((c, i) => `  ${i + 1}. ${c.operation}`).join('\n');
                md += `**plan (${e.data.parsed_commands.length} steps):**\n${ops}\n\n`;
            }
        });
        md += `## Server Log\n\n\`\`\`\n`;
        md += Array.from(this.consoleOutput.querySelectorAll('.log-line'))
            .map(el => el.textContent).join('\n');
        md += `\n\`\`\`\n`;

        const blob = new Blob([md], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `acrl-mission-${stamp.replace(/[:.]/g, '-')}.md`;
        a.click();
        URL.revokeObjectURL(url);
        this.toast('Mission exported', 'success');
    }

    _esc(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    hideThinkingIndicator() {
        const indicator = document.getElementById('thinking-indicator');
        const sendBtn = document.getElementById('btn-send');
        if (indicator) indicator.style.display = 'none';
        if (sendBtn) sendBtn.disabled = false;
    }

    showThinkingIndicator() {
        const indicator = document.getElementById('thinking-indicator');
        const sendBtn = document.getElementById('btn-send');
        if (indicator) indicator.style.display = 'flex';
        if (sendBtn) sendBtn.disabled = true;
    }

    /* --- PANELS TOGGLES --- */
    togglePerceptionPanel() {
        const panel = document.getElementById('perception-panel');
        const btn = document.getElementById('btn-perception-collapse');
        if (!panel) return;
        this.perceptionPanelExpanded = !this.perceptionPanelExpanded;
        panel.classList.toggle('expanded', this.perceptionPanelExpanded);
        if (btn) {
            const icon = btn.querySelector('i');
            if (icon) icon.className = this.perceptionPanelExpanded ? 'fa-solid fa-chevron-up' : 'fa-solid fa-chevron-down';
        }
    }

    toggleAutortPanel() {
        const panel = document.getElementById('autort-panel');
        const btn = document.getElementById('btn-autort-collapse');
        if (!panel) return;
        this.autortPanelExpanded = !this.autortPanelExpanded;
        panel.classList.toggle('expanded', this.autortPanelExpanded);
        if (btn) {
            const icon = btn.querySelector('i');
            if (icon) icon.className = this.autortPanelExpanded ? 'fa-solid fa-chevron-up' : 'fa-solid fa-chevron-down';
        }
    }
}
