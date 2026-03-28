export class NetworkManager {
    constructor(ui, renderer, autort) {
        this.ui = ui;
        this.renderer = renderer;
        this.autort = autort;
        
        this.ws = null;
        this.reconnectAttempts = 0;
        
        this.connectWebSocket();
        this.startStatusPolling();
    }

    /* --- REST API / POLLING --- */
    startStatusPolling() {
        const poll = () => this.pollStatus();
        poll();
        setInterval(poll, 3000);
    }

    pollStatus() {
        fetch('/api/status')
            .then(r => r.json())
            .then(status => {
                this.updateStatusBadges(status);
            })
            .catch(() => this.updateStatusBadges({ ros: false, llm: false, unity: false }));
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
                this.reconnectAttempts = 0;
                const banner = document.getElementById('disconnect-banner');
                if (banner) banner.style.display = 'none';
                this.updateStatusBadges({});
                this.pollStatus();
                this.autort.fetchPendingTasks();
            };

            this.ws.onmessage = (event) => {
                this.handleMessage(JSON.parse(event.data));
            };

            this.ws.onclose = () => {
                this.ui.logToConsole('WebSocket Disconnected. Reconnecting...', 'warning');
                const banner = document.getElementById('disconnect-banner');
                if (banner) banner.style.display = 'flex';
                this.updateStatusBadges({});

                const timeout = Math.min(10000, 1000 * Math.pow(1.5, this.reconnectAttempts));
                this.reconnectAttempts++;
                setTimeout(() => this.connectWebSocket(), timeout);
            };

            this.ws.onerror = (error) => {
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
            case 'vgn_debug':
                if (this.renderer) this.renderer.updateVGNDebug(msg.data);
                break;
            default:
                console.log("Unknown msg:", msg);
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
        }
    }

    jogRobot(direction) {
        const stepSize = 0.05; 
        let offset = { x: 0, y: 0, z: 0 };

        switch (direction) {
            case 'X+': offset.x = stepSize; break;
            case 'X-': offset.x = -stepSize; break;
            case 'Y+': offset.y = stepSize; break;
            case 'Y-': offset.y = -stepSize; break;
            case 'Z+': offset.z = stepSize; break;
            case 'Z-': offset.z = -stepSize; break;
        }

        const robot_id = document.getElementById('teleop-robot-id').value;
        this.ui.logToConsole(`Jogging ${robot_id} ${direction}`, 'info');

        fetch('/api/command', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                robot_id: robot_id,
                command: { type: 'move_relative', offset: offset }
            })
        }).catch(err => this.ui.logToConsole(`Jog API failed: ${err}`, 'error'));
    }

    triggerEStop() {
        this.ui.logToConsole('E-STOP TRIGGERED! Sending HALT to all modules...', 'error');
        fetch('/api/command', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                command: { type: 'estop', action: 'halt_all' }
            })
        }).catch(err => this.ui.logToConsole(`E-Stop API failed: ${err}`, 'error'));
    }

    sendGripperCmd(action) {
        const robot_id = document.getElementById('teleop-robot-id').value;
        fetch('/api/command', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                robot_id: robot_id,
                command: { type: 'gripper', action: action }
            })
        }).then(r => r.json()).then(data => console.log(data))
          .catch(err => console.error('Gripper cmd failed:', err));
    }
}
