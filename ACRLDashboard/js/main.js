import { UIManager } from './ui.js?v=20260328_0001';
import { Renderer } from './renderer.js?v=20260328_0005';
import { AutoRTManager } from './autort.js?v=20260328_0001';
import { NetworkManager } from './network.js?v=20260328_0001';

document.addEventListener('DOMContentLoaded', () => {
    const ui = new UIManager();
    const renderer = new Renderer();
    const autort = new AutoRTManager(ui);
    const network = new NetworkManager(ui, renderer, autort);
    autort.setNetwork(network);

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

    document.getElementById('btn-estop').addEventListener('click', () => {
        network.triggerEStop();
    });

    document.getElementById('btn-theme').addEventListener('click', () => {
        ui.toggleTheme();
    });

    document.getElementById('btn-download-logs').addEventListener('click', () => {
        ui.downloadLogs();
    });

    const autoRtBtn = document.querySelector('.panel-actions .btn-icon[title="Toggle AutoRT"]');
    if (autoRtBtn) {
        autoRtBtn.addEventListener('click', () => autort.toggleAutoRT(autoRtBtn));
    }

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

    const teleopCollapseBtn = document.getElementById('btn-teleop-collapse');
    if (teleopCollapseBtn) {
        teleopCollapseBtn.addEventListener('click', () => ui.toggleTeleopPanel());
    }

    const jogBtns = document.querySelectorAll('.xyz-controls .btn-icon-small');
    jogBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const title = e.currentTarget.getAttribute('title');
            if (title) network.jogRobot(title);
        });
    });

    // We replace the inline html onclicks for gripper with event listeners here
    const gripperBtns = document.querySelectorAll('.gripper-controls .btn');
    if (gripperBtns && gripperBtns.length >= 2) {
        // Assume first is open, second is close based on UI layout
        gripperBtns[0].addEventListener('click', () => network.sendGripperCmd('open'));
        gripperBtns[1].addEventListener('click', () => network.sendGripperCmd('close'));
    }
});
