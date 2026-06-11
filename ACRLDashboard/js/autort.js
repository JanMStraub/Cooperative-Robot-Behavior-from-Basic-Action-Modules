import { fetchJson } from './network.js?v=20260611_0001';

export class AutoRTManager {
    constructor(ui) {
        this.ui = ui;
        this.network = null; // Injected later
        this.autortTasks = new Map();
    }

    setNetwork(network) {
        this.network = network;
    }

    async fetchPendingTasks() {
        if (!this.network) return;
        try {
            const data = await fetchJson('/api/autort/tasks');
            if (data.success) this.handleAutortTasks(data);
        } catch (err) {
            this.ui.logToConsole(`AutoRT tasks fetch failed: ${err}`, 'error');
        }
    }

    handleAutortTasks(payload) {
        const tasks = payload.tasks || [];
        if (payload.loop_running !== undefined) {
            this._updateLoopBadge(payload.loop_running);
        }
        if (tasks.length === 0) return;

        if (!this.ui.autortPanelExpanded) this.ui.toggleAutortPanel();

        tasks.forEach(task => {
            if (!this.autortTasks.has(task.task_id)) {
                this.autortTasks.set(task.task_id, task);
                this._renderTaskCard(task);
            }
        });
        this._updateTaskCountBadge();
    }

    async generateTasks() {
        const btn = document.getElementById('btn-autort-generate');
        if (btn) btn.disabled = true;
        try {
            const data = await fetchJson('/api/autort/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ strategy: 'balanced' }),
            });
            if (!data.success) {
                this.ui.logToConsole(`AutoRT generate failed: ${data.error}`, 'error');
                this.ui.toast(`Generate failed: ${data.error}`, 'error');
            } else {
                const n = (data.tasks || []).length;
                this.ui.logToConsole(`AutoRT generated ${n} task(s)`, 'info');
                this.ui.toast(n > 0 ? `Generated ${n} task(s)` : 'No tasks proposed', n > 0 ? 'success' : 'info');
                if (n > 0) this.handleAutortTasks(data);
            }
        } catch (err) {
            this.ui.logToConsole(`AutoRT generate error: ${err}`, 'error');
            this.ui.toast(`Generate error: ${err.message}`, 'error');
        } finally {
            if (btn) btn.disabled = false;
        }
    }

    async approveTask(taskId) {
        this._removeTaskCard(taskId);
        this.autortTasks.delete(taskId);
        this._updateTaskCountBadge();
        try {
            const data = await fetchJson('/api/autort/execute', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_id: taskId }),
            });
            const level = data.success ? 'info' : 'error';
            this.ui.logToConsole(`Task ${taskId} execute: ${data.success ? 'started' : data.error}`, level);
            if (!data.success) this.ui.toast(`Task failed: ${data.error}`, 'error');
        } catch (err) {
            this.ui.logToConsole(`Task execute error: ${err}`, 'error');
            this.ui.toast(`Task execute error: ${err.message}`, 'error');
        }
    }

    rejectTask(taskId) {
        this._removeTaskCard(taskId);
        this.autortTasks.delete(taskId);
        this._updateTaskCountBadge();
        this.ui.logToConsole(`Task ${taskId} rejected (local only)`, 'info');
    }

    _renderTaskCard(task) {
        const list = document.getElementById('autort-task-list');
        const empty = document.getElementById('autort-empty');
        if (!list) return;

        if (empty) empty.style.display = 'none';

        const tier = this._complexityTier(task.estimated_complexity);
        const opsHtml = (task.operations || [])
            .map(op => `${this._escHtml(op.robot_id || '')}: ${this._escHtml(op.type)}`)
            .join('<br>');

        const card = document.createElement('div');
        card.className = 'autort-task-card';
        card.dataset.taskId = task.task_id;
        card.dataset.complexity = tier;

        // We do *NOT* use inline onclicks in the template string anymore. We'll attach listeners directly.
        card.innerHTML = `
            <div class="autort-task-header">
                <span class="autort-task-desc">${this._escHtml(task.description)}</span>
                <span class="autort-complexity-badge ${tier}">${tier}</span>
            </div>
            <div class="autort-ops-list">${opsHtml || '<em>No operations listed</em>'}</div>
            <div class="autort-task-actions">
                <button class="btn-autort-reject" data-action="reject">
                    Reject
                </button>
                <button class="btn-autort-approve" data-action="approve">
                    Approve &amp; Execute
                </button>
            </div>
        `;

        const rejectBtn = card.querySelector('.btn-autort-reject');
        rejectBtn.addEventListener('click', () => this.rejectTask(task.task_id));

        const approveBtn = card.querySelector('.btn-autort-approve');
        approveBtn.addEventListener('click', () => this.approveTask(task.task_id));

        list.appendChild(card);
    }

    _removeTaskCard(taskId) {
        const list = document.getElementById('autort-task-list');
        if (!list) return;

        const card = list.querySelector(`[data-task-id="${CSS.escape(taskId)}"]`);
        if (!card) return;

        card.classList.add('removing');
        card.addEventListener('animationend', () => {
            card.remove();
            const remaining = list.querySelectorAll('.autort-task-card').length;
            const empty = document.getElementById('autort-empty');
            if (empty) empty.style.display = remaining === 0 ? 'flex' : 'none';
        }, { once: true });
    }

    _updateTaskCountBadge() {
        const badge = document.getElementById('autort-task-count');
        if (!badge) return;
        const count = this.autortTasks.size;
        badge.textContent = count;
        badge.style.display = count > 0 ? 'inline-flex' : 'none';
    }

    _updateLoopBadge(running) {
        const badge = document.getElementById('autort-loop-badge');
        if (!badge) return;
        badge.textContent = running ? 'RUNNING' : 'STOPPED';
        badge.classList.toggle('running', !!running);
    }

    _complexityTier(n) {
        if (n <= 2) return 'low';
        if (n <= 4) return 'medium';
        return 'high';
    }

    _escHtml(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }
}
