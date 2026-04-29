export class UIManager {
    constructor() {
        this.chatHistory = document.getElementById('chat-history');
        this.promptInput = document.getElementById('prompt-input');
        this.consoleOutput = document.getElementById('console-output');
        
        this.perceptionPanelExpanded = true;
        this.teleopPanelExpanded = false;
        this.autortPanelExpanded = false;

        this.initTheme();
        this.initPanels();
        this.initCameraRetry();
    }

    /* --- PANELS DEFAULT --- */
    initPanels() {
        const teleopPanel = document.getElementById('teleop-panel');
        if (teleopPanel && !this.teleopPanelExpanded) {
            teleopPanel.classList.remove('expanded');
            const icon = document.querySelector('#btn-teleop-collapse i');
            if (icon) icon.className = 'fa-solid fa-chevron-down';
        }
    }

    /* --- THEME MANAGEMENT --- */
    initTheme() {
        const saved = localStorage.getItem('acrl-theme');
        if (saved) document.body.setAttribute('data-theme', saved);
    }

    toggleTheme() {
        const current = document.body.getAttribute('data-theme');
        const isCurrentlyLight = current === 'light' ||
            (!current && window.matchMedia('(prefers-color-scheme: light)').matches);
        const next = isCurrentlyLight ? 'dark' : 'light';
        document.body.setAttribute('data-theme', next);
        localStorage.setItem('acrl-theme', next);
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

    appendChatMessage(text, sender) {
        const div = document.createElement('div');
        div.className = `message ${sender}`;
        const icon = sender === 'user' ? 'fa-user' : 'fa-robot';

        div.innerHTML = `
            <div class="msg-icon"><i class="fa-solid ${icon}"></i></div>
            <div class="msg-content">${text}</div>
        `;

        this.chatHistory.appendChild(div);
        this.chatHistory.scrollTop = this.chatHistory.scrollHeight;
    }

    handleSequenceResult(data) {
        if (!data || !data.parsed_commands) return;

        let html = `<strong>Plan Generated (${data.parsed_commands.length} steps):</strong><br><ol style="margin-top: 0.5rem; margin-left: 1.5rem; font-size: 0.85rem; color: var(--text-muted);">`;

        data.parsed_commands.forEach(cmd => {
            html += `<li><span style="color: var(--accent);">${cmd.operation}</span>`;
            if (cmd.params) {
                const paramsStr = Object.entries(cmd.params)
                    .map(([k, v]) => `${k}: ${v}`)
                    .join(', ');
                html += ` <em>(${paramsStr})</em>`;
            }
            html += `</li>`;
        });
        html += `</ol>`;

        const div = document.createElement('div');
        div.className = `message system`;
        div.innerHTML = `
            <div class="msg-icon"><i class="fa-solid fa-code-branch"></i></div>
            <div class="msg-content">${html}</div>
        `;

        this.chatHistory.appendChild(div);
        this.chatHistory.scrollTop = this.chatHistory.scrollHeight;
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

    toggleTeleopPanel() {
        const panel = document.getElementById('teleop-panel');
        const btn = document.getElementById('btn-teleop-collapse');
        if (!panel) return;
        this.teleopPanelExpanded = !this.teleopPanelExpanded;
        panel.classList.toggle('expanded', this.teleopPanelExpanded);
        if (btn) {
            const icon = btn.querySelector('i');
            if (icon) icon.className = this.teleopPanelExpanded ? 'fa-solid fa-chevron-down' : 'fa-solid fa-chevron-up';
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
