/**
 * Fetch JSON with explicit HTTP-status checking.
 *
 * Throws on a non-2xx response or a network failure so callers can surface a
 * real error instead of silently treating a 500 body as a successful payload.
 */
export async function fetchJson(url, opts) {
    const resp = await fetch(url, opts);
    if (!resp.ok) {
        throw new Error(`${resp.status} ${resp.statusText}`);
    }
    return resp.json();
}

export class NetworkManager {
    constructor(ui, renderer, autort) {
        this.ui = ui;
        this.renderer = renderer;
        this.autort = autort;

        this.ws = null;
        this.reconnectAttempts = 0;
        this.pollIntervalId = null;

        this.connectWebSocket();
        this.startStatusPolling();
    }

    /* --- REST API / POLLING --- */
    startStatusPolling() {
        if (this.pollIntervalId !== null) clearInterval(this.pollIntervalId);
        const poll = () => this.pollStatus();
        poll();
        this.pollIntervalId = setInterval(poll, 3000);
    }

    async pollStatus() {
        try {
            const status = await fetchJson('/api/status');
            this.updateStatusBadges(status);
        } catch {
            this.updateStatusBadges({ ros: false, llm: false, unity: false });
        }
    }

    updateStatusBadges(status) {
        const wsConnected = this.ws && this.ws.readyState === WebSocket.OPEN;
        const effective = { backend: wsConnected, ...status };

        const mapping = {
            backend: 'badge-backend',
            ros: 'badge-ros',
            llm: 'badge-llm',
            unity: 'badge-unity',
        };
        for (const [key, elemId] of Object.entries(mapping)) {
            const el = document.getElementById(elemId);
            if (!el) continue;
            const online = !!effective[key];
            el.classList.toggle('badge-online', online);
            el.classList.toggle('badge-offline', !online);
            el.title = `${key}: ${online ? 'Connected' : 'Disconnected'}`;
        }
    }

    /* --- WEBSOCKET CONNECTION --- */
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        this.ui.logToConsole(`Connecting to ${wsUrl}...`, 'info');

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                this.ui.logToConsole('WebSocket Connected to Backend.', 'success');
                if (this.reconnectAttempts > 0) this.ui.toast('Reconnected to backend', 'success');
                this.reconnectAttempts = 0;
                const banner = document.getElementById('disconnect-banner');
                if (banner) banner.style.display = 'none';
                this.updateStatusBadges({});
                this.pollStatus();
                this.autort.fetchPendingTasks();
            };

            this.ws.onmessage = (event) => {
                let msg;
                try {
                    msg = JSON.parse(event.data);
                } catch (e) {
                    this.ui.logToConsole(`Dropped malformed WS message: ${e}`, 'error');
                    return;
                }
                this.handleMessage(msg);
            };

            this.ws.onclose = () => {
                this.ui.logToConsole('WebSocket Disconnected. Reconnecting...', 'warning');
                if (this.reconnectAttempts === 0) this.ui.toast('Backend disconnected', 'error');
                const banner = document.getElementById('disconnect-banner');
                if (banner) banner.style.display = 'flex';
                this.updateStatusBadges({});

                const timeout = Math.min(10000, 1000 * Math.pow(1.5, this.reconnectAttempts));
                this.reconnectAttempts++;
                setTimeout(() => this.connectWebSocket(), timeout);
            };

            this.ws.onerror = () => {
                this.ui.logToConsole('WebSocket Error occurred.', 'error');
            };
        } catch (e) {
            this.ui.logToConsole(`Failed to create WebSocket: ${e}`, 'error');
        }
    }

    handleMessage(msg) {
        if (!msg || !msg.type) return;

        switch (msg.type) {
            case 'log':
                this.ui.logToConsole(msg.message, msg.level);
                if (msg.level === 'error') this.ui.hideThinkingIndicator();
                break;
            case 'world_state':
                this.renderer.updateWorldState(msg.data);
                break;
            case 'sequence_result':
                this.ui.hideThinkingIndicator();
                this.ui.handleSequenceResult(msg.data);
                break;
            case 'autort_tasks':
                this.autort.handleAutortTasks(msg);
                break;
            case 'stereo_pointcloud':
                if (this.renderer) this.renderer.updateStereoPointCloud(msg.data);
                break;
            case 'vgn_debug':
                // Backend-only debug stream; no dashboard visualization yet. Ignore silently.
                break;
            default:
                this.ui.logToConsole(`Unknown message type: ${msg.type}`, 'warning');
        }
    }

    /* --- API ACTIONS --- */
    sendPrompt(text) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'sequence_prompt',
                prompt: text
            }));
            this.ui.logToConsole(`Sent prompt: ${text}`, 'info');
            this.ui.showThinkingIndicator();
        } else {
            this.ui.logToConsole('Cannot send prompt: Disconnected', 'error');
            this.ui.toast('Cannot send — backend disconnected', 'error');
        }
    }

    async triggerReset() {
        const btn = document.getElementById('btn-reset');
        if (btn) btn.disabled = true;
        this.ui.logToConsole('Resetting simulation...', 'info');
        try {
            const data = await fetchJson('/api/reset', { method: 'POST' });
            if (data.success) {
                this.ui.logToConsole('Simulation reset complete.', 'success');
                this.ui.toast('Simulation reset', 'success');
            } else {
                this.ui.logToConsole(`Reset failed: ${data.error}`, 'error');
                this.ui.toast(`Reset failed: ${data.error}`, 'error');
            }
        } catch (err) {
            this.ui.logToConsole(`Reset API failed: ${err}`, 'error');
            this.ui.toast(`Reset failed: ${err.message}`, 'error');
        } finally {
            if (btn) btn.disabled = false;
        }
    }

    async triggerEStop() {
        const btn = document.getElementById('btn-estop');
        if (btn) btn.disabled = true;
        this.ui.logToConsole('E-STOP TRIGGERED! Sending HALT to all modules...', 'error');
        this.ui.toast('E-STOP sent', 'error');
        try {
            await fetchJson('/api/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type: 'estop', action: 'halt_all' })
            });
        } catch (err) {
            this.ui.logToConsole(`E-Stop API failed: ${err}`, 'error');
            this.ui.toast(`E-Stop failed to send: ${err.message}`, 'error');
        } finally {
            if (btn) btn.disabled = false;
        }
    }
}
