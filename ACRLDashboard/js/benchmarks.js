/**
 * benchmarks.js
 * Logic for the Benchmark Analytics Dashboard
 */

let stepChartInstance = null;

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    setupEventListeners();
    fetchBenchmarkList();
});

function setupEventListeners() {
    document.getElementById('btn-refresh').addEventListener('click', fetchBenchmarkList);
    
    // Theme toggle
    const themeBtn = document.getElementById('btn-theme');
    if (themeBtn) {
        themeBtn.addEventListener('click', () => {
            document.body.classList.toggle('dark-theme');
            document.body.classList.toggle('light-theme');
            updateChartTheme();
        });
    }
}

function relativeTime(mtimeSeconds) {
    const diffMs = Date.now() - mtimeSeconds * 1000;
    const diffS = Math.floor(diffMs / 1000);
    if (diffS < 60) return 'just now';
    const diffM = Math.floor(diffS / 60);
    if (diffM < 60) return `${diffM} min ago`;
    const diffH = Math.floor(diffM / 60);
    if (diffH < 24) return `${diffH} h ago`;
    return `${Math.floor(diffH / 24)} d ago`;
}

function initTheme() {
    if (!document.body.classList.contains('light-theme')) {
        document.body.classList.add('dark-theme');
    }

    const isDark = !document.body.classList.contains('light-theme');
    Chart.defaults.color = isDark ? '#7d8590' : '#656d76';
    Chart.defaults.font.family = "'IBM Plex Mono', monospace";
    Chart.defaults.font.size = 11;
}

function updateChartTheme() {
    const isDark = !document.body.classList.contains('light-theme');
    Chart.defaults.color = isDark ? '#7d8590' : '#656d76';
    Chart.defaults.font.family = "'IBM Plex Mono', monospace";
    if (stepChartInstance) stepChartInstance.update();
}

async function fetchBenchmarkList() {
    const listContainer = document.getElementById('history-list');
    listContainer.innerHTML = `
        <div class="ws-empty" id="history-loading">
            <i class="fa-solid fa-circle-notch fa-spin"></i>
            <span>Loading benchmarks...</span>
        </div>`;

    try {
        const response = await fetch('/api/benchmarks');
        const result = await response.json();
        
        if (!result.success) throw new Error(result.error);
        
        const files = result.files;
        if (files.length === 0) {
            listContainer.innerHTML = `
                <div class="ws-empty">
                    <i class="fa-solid fa-inbox"></i>
                    <span>No benchmarks found.</span>
                </div>`;
            return;
        }

        listContainer.innerHTML = '';
        files.forEach(file => {
            const statusClass = file.success ? 'status-pass' : 'status-fail';
            const statusText = file.success ? 'PASS' : 'FAIL';
            const duration = (file.total_duration_ms / 1000).toFixed(1) + 's';
            const ago = relativeTime(file.mtime);
            const opsText = file.ops_executed != null ? `${file.ops_executed} ops` : '';

            const successRate = file.success_rate != null ? file.success_rate : (file.success ? 1 : 0);
            const fillPct = Math.round(successRate * 100);
            const fillClass = successRate >= 1 ? 'progress-pass' : successRate > 0 ? 'progress-warn' : 'progress-fail';

            const item = document.createElement('div');
            item.className = 'history-item';
            item.dataset.filename = file.filename;

            item.innerHTML = `
                <div class="history-item-top">
                    <span class="history-item-badge">B${file.benchmark_id}</span>
                    <span class="history-item-name">${file.benchmark_name || 'Benchmark'}</span>
                    <span class="history-item-status ${statusClass}">${statusText}</span>
                </div>
                <div class="history-item-meta">
                    <span>${opsText}${opsText ? ' · ' : ''}${duration} · ${ago}</span>
                </div>
                <div class="history-item-progress">
                    <div class="history-item-progress-fill ${fillClass}" style="width: ${fillPct}%"></div>
                </div>
            `;

            item.addEventListener('click', () => {
                document.querySelectorAll('.history-item').forEach(i => i.classList.remove('active'));
                item.classList.add('active');
                loadBenchmarkDetails(file.filename);
            });

            listContainer.appendChild(item);
        });
        
    } catch (e) {
        listContainer.innerHTML = `
            <div class="ws-empty">
                <i class="fa-solid fa-triangle-exclamation" style="color: #e74c3c"></i>
                <span>Failed to load history: ${e.message}</span>
            </div>`;
    }
}

async function loadBenchmarkDetails(filename) {
    document.getElementById('no-selection').style.display = 'none';
    const detailsPanel = document.getElementById('benchmark-details');
    detailsPanel.style.display = 'flex'; // show but might be loading
    
    try {
        const response = await fetch(`/api/benchmarks/${filename}`);
        const result = await response.json();
        
        if (!result.success) throw new Error(result.error);
        
        renderDetails(result.data);
    } catch (e) {
        alert("Failed to load details: " + e.message);
    }
}

function renderDetails(data) {
    // 1. KPIs
    document.getElementById('kpi-duration').textContent = (data.total_duration_ms / 1000).toFixed(2) + 's';
    document.getElementById('kpi-success-rate').textContent = (data.success_rate * 100).toFixed(1) + '%';
    document.getElementById('kpi-ops').textContent = `${data.ops_succeeded} / ${data.ops_executed}`;
    document.getElementById('kpi-avg-step').textContent = data.avg_step_duration_ms.toFixed(0) + 'ms';
    
    // 2. Populate Table
    const tbody = document.getElementById('step-table-body');
    tbody.innerHTML = '';
    
    const labels = [];
    const durations = [];
    const colors = [];

    (data.steps || []).forEach(step => {
        const tr = document.createElement('tr');
        
        const statusIcon = step.success 
            ? '<i class="fa-solid fa-check" style="color:#2ecc71;"></i>' 
            : '<i class="fa-solid fa-xmark" style="color:#e74c3c;"></i>';
            
        tr.innerHTML = `
            <td>${step.index}</td>
            <td class="step-op">${step.operation}</td>
            <td>${statusIcon}</td>
            <td>${step.duration_ms.toFixed(0)}</td>
            <td class="step-error">${step.error_code || ''} ${step.error_message ? '- ' + step.error_message : ''}</td>
        `;
        tbody.appendChild(tr);
        
        // Data for chart
        labels.push(`Step ${step.index}`);
        durations.push(step.duration_ms);
        colors.push(step.success ? 'rgba(47, 158, 140, 0.7)' : 'rgba(248, 81, 73, 0.65)');
    });
    
    // 3. Render Chart
    renderChart(labels, durations, colors);
}

function renderChart(labels, data, colors) {
    const ctx = document.getElementById('stepDurationChart').getContext('2d');
    
    if (stepChartInstance) {
        stepChartInstance.destroy();
    }
    
    stepChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Step Duration (ms)',
                data: data,
                backgroundColor: colors,
                borderWidth: 1,
                borderColor: colors.map(c => c.replace(/[\d.]+\)$/, '1.0)')),
                borderRadius: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            plugins: {
                legend: { display: false },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(13, 17, 23, 0.96)',
                    borderColor: 'rgba(255, 255, 255, 0.12)',
                    borderWidth: 1,
                    titleColor: '#e6edf3',
                    bodyColor: '#7d8590',
                    titleFont: { family: "'IBM Plex Sans', sans-serif", size: 12, weight: '600' },
                    bodyFont:  { family: "'IBM Plex Mono', monospace", size: 11 },
                    padding: 10,
                    cornerRadius: 2,
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.05)', drawBorder: false },
                    border: { color: 'rgba(255, 255, 255, 0.08)' },
                    ticks: {
                        color: '#7d8590',
                        font: { family: "'IBM Plex Mono', monospace", size: 10 },
                    },
                    title: {
                        display: true,
                        text: 'DURATION (ms)',
                        color: '#484f58',
                        font: { family: "'IBM Plex Mono', monospace", size: 10 },
                        padding: { bottom: 6 },
                    }
                },
                x: {
                    grid: { display: false },
                    border: { color: 'rgba(255, 255, 255, 0.08)' },
                    ticks: {
                        color: '#7d8590',
                        font: { family: "'IBM Plex Mono', monospace", size: 10 },
                    }
                }
            }
        }
    });
}
