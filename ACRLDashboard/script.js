/**
 * script.js - Frontend logic for ACRL Mission Control
 * Handles WebSocket connections, REST APIs, and Three.js rendering.
 */

class DashboardController {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;

        // Element references
        this.chatHistory = document.getElementById('chat-history');
        this.promptInput = document.getElementById('prompt-input');
        this.consoleOutput = document.getElementById('console-output');

        this.initEventListeners();
        this.initThreeJS();
        this.connectWebSocket();
    }

    initEventListeners() {
        document.getElementById('btn-send').addEventListener('click', () => this.sendPrompt());

        document.getElementById('prompt-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendPrompt();
            }
        });

        document.getElementById('btn-clear').addEventListener('click', () => {
            this.promptInput.value = '';
        });

        document.getElementById('btn-estop').addEventListener('click', () => {
            this.triggerEStop();
        });

        const autoRtBtn = document.querySelector('.panel-actions .btn-icon[title="Toggle AutoRT"]');
        if (autoRtBtn) {
            autoRtBtn.addEventListener('click', () => this.toggleAutoRT(autoRtBtn));
        }

        // Setup Teleop Jog Buttons
        const jogBtns = document.querySelectorAll('.xyz-controls .btn-icon-small');
        jogBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const title = e.currentTarget.getAttribute('title');
                if (title) this.jogRobot(title);
            });
        });
    }

    /* --- WEBSOCKET CONNECTION --- */

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        this.logToConsole(`Connecting to ${wsUrl}...`, 'info');

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                this.logToConsole('WebSocket Connected to Backend.', 'success');
                this.reconnectAttempts = 0;
                document.querySelector('.dot.online').style.boxShadow = '0 0 15px var(--success)';
            };

            this.ws.onmessage = (event) => {
                this.handleMessage(JSON.parse(event.data));
            };

            this.ws.onclose = () => {
                this.logToConsole('WebSocket Disconnected. Reconnecting...', 'warning');
                document.querySelector('.dot.online').style.boxShadow = 'none';

                // Exponential backoff
                const timeout = Math.min(10000, 1000 * Math.pow(1.5, this.reconnectAttempts));
                this.reconnectAttempts++;
                setTimeout(() => this.connectWebSocket(), timeout);
            };

            this.ws.onerror = (error) => {
                this.logToConsole('WebSocket Error occurred.', 'error');
            };

        } catch (e) {
            this.logToConsole(`Failed to create WebSocket: ${e}`, 'error');
        }
    }

    handleMessage(msg) {
        if (!msg || !msg.type) return;

        switch (msg.type) {
            case 'log':
                this.logToConsole(msg.message, msg.level);
                break;
            case 'world_state':
                this.updateWorldState(msg.data);
                break;
            case 'sequence_result':
                this.handleSequenceResult(msg.data);
                break;
            default:
                console.log("Unknown msg:", msg);
        }
    }

    /* --- API ACTIONS --- */

    toggleAutoRT(btnElement) {
        this.autoRtActive = !this.autoRtActive;
        const action = this.autoRtActive ? 'start' : 'stop';

        if (this.autoRtActive) {
            btnElement.classList.add('active');
            btnElement.style.color = 'var(--success)';
            btnElement.style.borderColor = 'var(--success)';
        } else {
            btnElement.classList.remove('active');
            btnElement.style.color = '';
            btnElement.style.borderColor = '';
        }

        fetch('/api/command', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ type: 'autort', action: action })
        })
            .then(r => r.json())
            .then(res => {
                this.logToConsole(`AutoRT ${action} response: ${res.success}`, res.success ? 'success' : 'error');
            })
            .catch(err => this.logToConsole(`AutoRT API error: ${err}`, 'error'));
    }

    sendPrompt() {
        const text = this.promptInput.value.trim();
        if (!text) return;

        // Add to UI
        this.appendChatMessage(text, 'user');
        this.promptInput.value = '';

        // Send to backend
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'sequence_prompt',
                prompt: text
            }));
            this.logToConsole(`Sent prompt: ${text}`, 'info');
        } else {
            this.logToConsole('Cannot send prompt: Disconnected', 'error');
        }
    }

    jogRobot(direction) {
        const stepSize = 0.05; // 5cm
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
        this.logToConsole(`Jogging ${robot_id} ${direction}`, 'info');

        fetch('/api/command', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                robot_id: robot_id,
                command: {
                    type: 'move_relative',
                    offset: offset
                }
            })
        }).catch(err => this.logToConsole(`Jog API failed: ${err}`, 'error'));
    }

    triggerEStop() {
        this.logToConsole('E-STOP TRIGGERED! Sending HALT to all modules...', 'error');
        // Flash animation is handled by CSS, we just need to send the API request
        fetch('/api/command', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                command: { type: 'estop', action: 'halt_all' }
            })
        }).catch(err => this.logToConsole(`E-Stop API failed: ${err}`, 'error'));
    }

    /* --- UI UPDATES --- */

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

    logToConsole(msg, level = 'info') {
        const div = document.createElement('div');
        div.className = `log-line ${level}`;
        const time = new Date().toLocaleTimeString();
        div.textContent = `[${time}] ${msg}`;

        this.consoleOutput.appendChild(div);

        // Autoscroll logic (only if currently near bottom)
        const isScrolledToBottom = this.consoleOutput.scrollHeight - this.consoleOutput.clientHeight <= this.consoleOutput.scrollTop + 10;
        if (isScrolledToBottom) {
            this.consoleOutput.scrollTop = this.consoleOutput.scrollHeight;
        }
    }

    /* --- THREE.JS VISUALIZATION --- */

    initThreeJS() {
        const container = document.getElementById('threejs-container');

        // Scene bounds
        const width = container.clientWidth;
        const height = container.clientHeight;

        // Scene, Camera, Renderer
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a0a);

        this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
        this.camera.position.set(0, 2, 3);

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(this.renderer.domElement);

        // Controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;

        // Grid & Lights
        const gridHelper = new THREE.GridHelper(5, 20, 0x4361ee, 0x222222);
        this.scene.add(gridHelper);

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(2, 5, 2);
        this.scene.add(dirLight);

        // Resize handler
        window.addEventListener('resize', () => {
            if (!container) return;
            const newWidth = container.clientWidth;
            const newHeight = container.clientHeight;
            this.renderer.setSize(newWidth, newHeight);
            this.camera.aspect = newWidth / newHeight;
            this.camera.updateProjectionMatrix();
        });

        // Object cache
        this.meshCache = {};

        // Render loop
        const animate = () => {
            requestAnimationFrame(animate);
            this.controls.update();
            this.renderer.render(this.scene, this.camera);
        };
        animate();
    }

    updateWorldState(data) {
        if (!this.scene) return;

        // Keep track of what we saw this frame
        const seenIds = new Set();

        // Update Objects
        if (data.objects) {
            data.objects.forEach(obj => {
                const id = obj.object_id;
                seenIds.add(id);
                this.updateOrCreateMesh(id, obj, 'object');
            });
        }

        // Update Robots (represented as simple blocks for now)
        if (data.robots) {
            data.robots.forEach(robot => {
                const id = robot.robot_id;
                seenIds.add(id);
                this.updateOrCreateMesh(id, robot, 'robot');
            });
        }

        // Remove things that disappeared
        Object.keys(this.meshCache).forEach(id => {
            if (!seenIds.has(id)) {
                this.scene.remove(this.meshCache[id]);
                delete this.meshCache[id];
            }
        });
    }

    updateOrCreateMesh(id, data, type) {
        let mesh = this.meshCache[id];

        if (!mesh) {
            // Create new mesh
            if (type === 'robot') {
                const geo = new THREE.BoxGeometry(0.2, 0.5, 0.2);
                const mat = new THREE.MeshStandardMaterial({ color: 0x4cc9f0, wireframe: true });
                mesh = new THREE.Mesh(geo, mat);
            } else {
                // Determine color based on data if available, default red
                let colorHex = 0xe71d36; // red default
                if (data.color) {
                    if (data.color.toLowerCase() === 'blue') colorHex = 0x4361ee;
                    else if (data.color.toLowerCase() === 'green') colorHex = 0x2ec4b6;
                    else if (data.color.toLowerCase() === 'yellow') colorHex = 0xff9f1c;
                }
                const geo = new THREE.BoxGeometry(0.1, 0.1, 0.1);
                const mat = new THREE.MeshStandardMaterial({ color: colorHex });
                mesh = new THREE.Mesh(geo, mat);
            }

            this.scene.add(mesh);
            this.meshCache[id] = mesh;
        }

        // Update position if available
        if (data.position && Array.isArray(data.position) && data.position.length >= 3) {
            mesh.position.set(data.position[0], data.position[1], data.position[2]);
        }
    }
}

// Global scope initialization
window.sendCmd = function (action) {
    const robot_id = document.getElementById('teleop-robot-id').value;
    fetch('/api/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            robot_id: robot_id,
            command: { type: 'gripper', action: action }
        })
    }).then(r => r.json()).then(data => console.log(data));
};

document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new DashboardController();
});
