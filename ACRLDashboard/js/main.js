import { UIManager } from './ui.js?v=20260611_0001';
import { Renderer } from './renderer.js?v=20260420_0002';
import { AutoRTManager } from './autort.js?v=20260611_0002';
import { NetworkManager } from './network.js?v=20260611_0001';

document.addEventListener('DOMContentLoaded', () => {
    const ui = new UIManager();
    const renderer = new Renderer();
    const autort = new AutoRTManager(ui);
    const network = new NetworkManager(ui, renderer, autort);
    autort.setNetwork(network);

    const wsTimeout = setTimeout(() => {
        const robotList = document.getElementById('ws-robot-list');
        if (robotList) {
            const empty = robotList.querySelector('.ws-empty');
            if (empty) {
                empty.innerHTML = `
                    <i class="fa-solid fa-circle-exclamation"></i>
                    <span>No data from Unity after 30s.<br>Check backend is running.</span>
                `;
            }
        }
    }, 30000);
    renderer.wsConnectionTimeout = wsTimeout;

    // Event Wiring
    document.getElementById('btn-send').addEventListener('click', () => {
        const text = ui.promptInput.value.trim();
        if (!text) return;
        ui.appendChatMessage(text, 'user');
        ui.promptInput.value = '';
        network.sendPrompt(text);
    });

    document.getElementById('prompt-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            document.getElementById('btn-send').click();
        }
    });

    document.getElementById('btn-clear').addEventListener('click', () => {
        ui.promptInput.value = '';
    });

    const exportBtn = document.getElementById('btn-export-mission');
    if (exportBtn) {
        exportBtn.addEventListener('click', () => ui.exportMission());
    }

    const clearChatBtn = document.getElementById('btn-clear-chat');
    if (clearChatBtn) {
        clearChatBtn.addEventListener('click', () => ui.clearChat());
    }

    document.getElementById('btn-reset').addEventListener('click', () => {
        network.triggerReset();
    });

    document.getElementById('btn-estop').addEventListener('click', () => {
        network.triggerEStop();
    });

    document.getElementById('btn-theme').addEventListener('click', () => {
        ui.toggleTheme();
    });

    document.getElementById('btn-download-logs').addEventListener('click', () => {
        ui.downloadLogs();
    });

    const generateBtn = document.getElementById('btn-autort-generate');
    if (generateBtn) {
        generateBtn.addEventListener('click', () => autort.generateTasks());
    }

    const collapseBtn = document.getElementById('btn-autort-collapse');
    if (collapseBtn) {
        collapseBtn.addEventListener('click', () => ui.toggleAutortPanel());
    }

    const perceptionCollapseBtn = document.getElementById('btn-perception-collapse');
    if (perceptionCollapseBtn) {
        perceptionCollapseBtn.addEventListener('click', () => ui.togglePerceptionPanel());
    }
});
