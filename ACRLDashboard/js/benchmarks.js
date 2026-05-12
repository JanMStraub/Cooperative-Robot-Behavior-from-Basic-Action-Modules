/**
 * benchmarks.js
 * Logic for the Benchmark Analytics Dashboard — Run Details + Analysis tabs.
 */

// ── Palette (Okabe-Ito colorblind-safe) ──────────────────────────────────────
const PALETTE = {
    blue:        '#0072B2',
    orange:      '#E69F00',
    teal:        '#009E73',
    vermillion:  '#D55E00',
    skyblue:     '#56B4E9',
    yellow:      '#F0E442',
    purple:      '#CC79A7',
};

const PASS_COLOR  = PALETTE.teal;
const FAIL_COLOR  = PALETTE.vermillion;
const WARN_COLOR  = PALETTE.orange;

// ── State ─────────────────────────────────────────────────────────────────────
let stepChartInstance = null;
let analysisChartInstances = {};
let aggregateData = null;   // cached from /api/benchmarks/aggregate
let thesisMode = false;
let currentTab = 'details';

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initTabs();
    setupEventListeners();
    fetchBenchmarkList();
});

function setupEventListeners() {
    document.getElementById('btn-refresh').addEventListener('click', fetchBenchmarkList);

    const themeBtn = document.getElementById('btn-theme');
    if (themeBtn) {
        themeBtn.addEventListener('click', () => {
            document.body.classList.toggle('dark-theme');
            document.body.classList.toggle('light-theme');
            updateChartTheme();
        });
    }

    const thesisBtn = document.getElementById('btn-thesis-mode');
    if (thesisBtn) {
        thesisBtn.addEventListener('click', toggleThesisMode);
    }

    const ablationSelect = document.getElementById('ablation-metric');
    if (ablationSelect) {
        ablationSelect.addEventListener('change', () => {
            if (aggregateData) renderAblationChart(aggregateData);
        });
    }
}

// ── Tabs ──────────────────────────────────────────────────────────────────────
function initTabs() {
    document.querySelectorAll('.bm-tab').forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });
}

function switchTab(tab) {
    currentTab = tab;

    document.querySelectorAll('.bm-tab').forEach(btn => {
        btn.classList.toggle('bm-tab--active', btn.dataset.tab === tab);
    });
    document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.toggle('tab-pane--active', pane.id === `tab-${tab}`);
    });

    // Thesis button only visible in analysis tab
    const thesisBtn = document.getElementById('btn-thesis-mode');
    if (thesisBtn) thesisBtn.style.display = tab === 'analysis' ? '' : 'none';

    if (tab === 'analysis') loadAnalysis();
}

// ── Theme ─────────────────────────────────────────────────────────────────────
function initTheme() {
    if (!document.body.classList.contains('light-theme')) {
        document.body.classList.add('dark-theme');
    }
    applyChartDefaults();
}

function updateChartTheme() {
    applyChartDefaults();
    if (stepChartInstance) stepChartInstance.update();
    Object.values(analysisChartInstances).forEach(c => c && c.update());
}

function applyChartDefaults() {
    const isDark = document.body.classList.contains('dark-theme');
    const isThesis = thesisMode;

    Chart.defaults.color = isThesis ? '#333333' : (isDark ? '#7d8590' : '#656d76');
    Chart.defaults.font.family = "'IBM Plex Mono', monospace";
    Chart.defaults.font.size = 11;
}

// ── Thesis Mode ───────────────────────────────────────────────────────────────
function toggleThesisMode() {
    thesisMode = !thesisMode;
    document.body.classList.toggle('thesis-mode', thesisMode);

    const btn = document.getElementById('btn-thesis-mode');
    if (btn) btn.classList.toggle('bm-tab--active', thesisMode);

    applyChartDefaults();
    if (aggregateData) renderAllAnalysisCharts(aggregateData);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
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

function successColor(rate) {
    if (rate >= 0.8) return PASS_COLOR;
    if (rate >= 0.5) return WARN_COLOR;
    return FAIL_COLOR;
}

function isDarkMode() {
    return document.body.classList.contains('dark-theme') && !thesisMode;
}

function chartGridColor() {
    return isDarkMode() ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.08)';
}

function chartBorderColor() {
    return isDarkMode() ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.12)';
}

function chartTextColor() {
    if (thesisMode) return '#333333';
    return isDarkMode() ? '#7d8590' : '#656d76';
}

function tooltipDefaults() {
    if (thesisMode) {
        return {
            backgroundColor: 'rgba(255,255,255,0.98)',
            borderColor: '#cccccc',
            borderWidth: 1,
            titleColor: '#111111',
            bodyColor: '#444444',
        };
    }
    return {
        backgroundColor: 'rgba(13,17,23,0.96)',
        borderColor: 'rgba(255,255,255,0.12)',
        borderWidth: 1,
        titleColor: '#e6edf3',
        bodyColor: '#7d8590',
    };
}

function destroyChart(id) {
    if (analysisChartInstances[id]) {
        analysisChartInstances[id].destroy();
        analysisChartInstances[id] = null;
    }
}

// ── Run History ───────────────────────────────────────────────────────────────
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

// ── Run Details Tab ───────────────────────────────────────────────────────────
async function loadBenchmarkDetails(filename) {
    document.getElementById('no-selection').style.display = 'none';
    const detailsPanel = document.getElementById('benchmark-details');
    detailsPanel.style.display = 'flex';

    try {
        const response = await fetch(`/api/benchmarks/${filename}`);
        const result = await response.json();

        if (!result.success) throw new Error(result.error);

        renderDetails(result.data);
    } catch (e) {
        alert('Failed to load details: ' + e.message);
    }
}

function renderDetails(data) {
    document.getElementById('kpi-duration').textContent = (data.total_duration_ms / 1000).toFixed(2) + 's';
    document.getElementById('kpi-success-rate').textContent = (data.success_rate * 100).toFixed(1) + '%';
    document.getElementById('kpi-ops').textContent = `${data.ops_succeeded} / ${data.ops_executed}`;
    document.getElementById('kpi-avg-step').textContent = data.avg_step_duration_ms.toFixed(0) + 'ms';

    renderRunInfo(data);
    renderOpStats(data.per_op_stats);

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

        const robotId  = step.robot_id          != null ? `<span class="robot-badge robot-badge--${step.robot_id}">${step.robot_id}</span>` : '<span class="step-na">—</span>';
        const pgId     = step.parallel_group_id != null ? `<span class="pg-badge">pg${step.parallel_group_id}</span>` : '<span class="step-na">—</span>';

        tr.innerHTML = `
            <td>${step.index}</td>
            <td class="step-op">${step.operation}</td>
            <td>${statusIcon}</td>
            <td>${step.duration_ms.toFixed(0)}</td>
            <td>${robotId}</td>
            <td>${pgId}</td>
            <td class="step-error">${step.error_code || ''} ${step.error_message ? '- ' + step.error_message : ''}</td>
        `;
        tbody.appendChild(tr);

        labels.push(`Step ${step.index}`);
        durations.push(step.duration_ms);
        colors.push(step.success ? 'rgba(47, 158, 140, 0.7)' : 'rgba(248, 81, 73, 0.65)');
    });

    renderStepChart(labels, durations, colors);
}

function renderRunInfo(data) {
    const panel = document.getElementById('run-info-panel');
    if (!panel) return;

    const flags = data.feature_flags || {};
    const parsedPlan = data.parsed_plan || [];
    const execMode = data.execution_mode || null;

    if (!execMode && Object.keys(flags).length === 0 && parsedPlan.length === 0) {
        panel.style.display = 'none';
        return;
    }

    const flagMeta = {
        use_rag:             { label: 'RAG',          desc: 'Retrieval-Augmented Generation active — LLM has access to operation docs and prior examples. Tests whether retrieval improves plan quality.' },
        use_vgn:             { label: 'VGN',          desc: 'Volumetric Grasp Network enabled — replaces heuristic grasp candidates with learned 6-DOF grasp poses. Key ablation for grasp success rate.' },
        use_knowledge_graph: { label: 'KG',           desc: 'Knowledge Graph active — spatial reasoning layer provides reachability, proximity, and handoff queries to the LLM. Ablation for planning quality.' },
        use_ros_movement:    { label: 'ROS/MoveIt',   desc: 'MoveIt trajectory planning active (vs. Unity IK). Tests whether motion planning improves collision avoidance and trajectory quality.' },
        reflexion_enabled:   { label: 'Reflexion',    desc: 'Reflexion self-correction loop enabled — LLM retries failed steps with error context. Directly measured via reflexion_recoveries metric.' },
        dry_run:             { label: 'Dry Run',      desc: 'No real Unity execution — operations are simulated. Used for testing plan generation without robot hardware. Results not comparable to live runs.' },
        use_negotiation:     { label: 'Negotiation',  desc: 'Multi-robot LLM negotiation active — robots negotiate task allocation before execution. Core ablation for dual-arm coordination benchmarks.' },
    };

    const modeMeta = {
        offline: 'Offline — operations executed against mock/simulated responses. Use for plan generation tests; timing and success rates are not representative of real robot performance.',
        live:    'Live — operations dispatched to Unity simulation with real physics. Results are representative; compare directly with other live runs.',
    };

    let html = '<div class="run-info-header"><i class="fa-solid fa-sliders"></i> Run Configuration</div><div class="run-info-body">';

    if (execMode) {
        const modeDesc = modeMeta[execMode] || execMode;
        html += `<div class="run-info-row">
            <span class="run-info-label">Execution Mode <i class="fa-solid fa-circle-info kpi-info" title="${modeDesc}"></i></span>
            <span class="run-mode-badge run-mode-badge--${execMode}">${execMode.toUpperCase()}</span>
        </div>`;
    }

    const flagKeys = Object.keys(flagMeta).filter(k => k in flags);
    if (flagKeys.length > 0) {
        const pillsHtml = flagKeys.map(k => {
            const on = !!flags[k];
            const meta = flagMeta[k];
            return `<span class="flag-pill flag-pill--${on ? 'on' : 'off'}" title="${meta.desc}">${meta.label}: ${on ? 'ON' : 'OFF'}</span>`;
        }).join('');
        html += `<div class="run-info-row run-info-row--flags">
            <span class="run-info-label">Feature Flags <i class="fa-solid fa-circle-info kpi-info" title="Active system capabilities for this run. Enables comparing ablation conditions — each flag corresponds to a benchmark series (B9–B14). Hover each pill for details."></i></span>
            <div class="flag-pills">${pillsHtml}</div>
        </div>`;
    }

    if (parsedPlan.length > 0) {
        const opsHtml = parsedPlan.map((op, i) =>
            `<span class="plan-op"><span class="plan-op-idx">${i + 1}</span>${op}</span>`
        ).join('');
        html += `<div class="run-info-row run-info-row--plan">
            <span class="run-info-label">Parsed Plan <i class="fa-solid fa-circle-info kpi-info" title="The sequence of operation names the LLM generated before execution began. Comparing this to the executed steps reveals hallucinated ops (in plan but not in registry), skipped steps, or mid-run replanning."></i></span>
            <div class="plan-ops">${opsHtml}</div>
        </div>`;
    }

    html += '</div>';
    panel.innerHTML = html;
    panel.style.display = '';
}

function renderOpStats(perOpStats) {
    const panel = document.getElementById('op-stats-panel');
    if (!panel) return;
    if (!perOpStats || Object.keys(perOpStats).length === 0) {
        panel.style.display = 'none';
        return;
    }

    const rows = Object.entries(perOpStats)
        .sort((a, b) => (b[1].fail_count || 0) - (a[1].fail_count || 0));

    let html = `<div class="run-info-header">
        <i class="fa-solid fa-chart-simple"></i> Per-Operation Stats
        <i class="fa-solid fa-circle-info kpi-info" style="margin-left:0.4rem" title="Aggregated stats per operation type across all steps in this run. fail_count directly identifies the most error-prone operations. avg_duration_ms reveals which ops dominate latency. Sort order: highest fail_count first."></i>
    </div>
    <div class="table-wrapper">
    <table class="benchmark-table op-stats-table">
        <thead><tr>
            <th title="Registered operation name">Operation</th>
            <th title="Total times this operation was dispatched in the run">Count <i class="fa-solid fa-circle-info kpi-info"></i></th>
            <th title="Number of times this operation returned a failure result. Non-zero values are the primary signal for identifying unreliable operations.">Failures</th>
            <th title="Failure rate for this operation type (fail_count / count). 100% means the operation always fails — likely a configuration or environment issue.">Fail Rate</th>
            <th title="Mean execution time for this operation in milliseconds. High values indicate expensive operations (e.g. VGN inference, grasp planning) that dominate total run duration.">Avg Duration</th>
        </tr></thead>
        <tbody>`;

    rows.forEach(([op, stats]) => {
        const count    = stats.count || 0;
        const fails    = stats.fail_count || 0;
        const avgMs    = stats.avg_duration_ms != null ? stats.avg_duration_ms.toFixed(0) : '—';
        const failRate = count > 0 ? (fails / count * 100).toFixed(0) : '0';
        const rateClass = fails === 0 ? 'rate-ok' : (fails / count >= 0.5 ? 'rate-bad' : 'rate-warn');
        html += `<tr>
            <td class="step-op">${op}</td>
            <td>${count}</td>
            <td>${fails > 0 ? `<span style="color:var(--danger)">${fails}</span>` : '0'}</td>
            <td><span class="fail-rate ${rateClass}">${failRate}%</span></td>
            <td>${avgMs}ms</td>
        </tr>`;
    });

    html += '</tbody></table></div>';
    panel.innerHTML = html;
    panel.style.display = '';
}

function renderStepChart(labels, data, colors) {
    const ctx = document.getElementById('stepDurationChart').getContext('2d');

    if (stepChartInstance) stepChartInstance.destroy();

    const tt = tooltipDefaults();
    stepChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Step Duration (ms)',
                data,
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
                    ...tt,
                    padding: 10,
                    cornerRadius: 2,
                    titleFont: { family: "'IBM Plex Sans', sans-serif", size: 12, weight: '600' },
                    bodyFont:  { family: "'IBM Plex Mono', monospace", size: 11 },
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: chartGridColor(), drawBorder: false },
                    border: { color: chartBorderColor() },
                    ticks: { color: chartTextColor(), font: { family: "'IBM Plex Mono', monospace", size: 10 } },
                    title: {
                        display: true,
                        text: 'DURATION (ms)',
                        color: chartTextColor(),
                        font: { family: "'IBM Plex Mono', monospace", size: 10 },
                        padding: { bottom: 6 },
                    }
                },
                x: {
                    grid: { display: false },
                    border: { color: chartBorderColor() },
                    ticks: { color: chartTextColor(), font: { family: "'IBM Plex Mono', monospace", size: 10 } }
                }
            }
        }
    });
}

// ── Analysis Tab ──────────────────────────────────────────────────────────────
async function loadAnalysis() {
    if (aggregateData) {
        // already loaded — just re-render in case theme changed
        renderAllAnalysisCharts(aggregateData);
        return;
    }

    document.getElementById('analysis-loading').style.display = 'flex';
    document.getElementById('analysis-charts').style.display = 'none';

    try {
        const resp = await fetch('/api/benchmarks/aggregate');
        const result = await resp.json();

        if (!result.success) throw new Error(result.error);

        aggregateData = result.data;
        document.getElementById('analysis-loading').style.display = 'none';
        document.getElementById('analysis-charts').style.display = 'flex';
        renderAllAnalysisCharts(aggregateData);
    } catch (e) {
        document.getElementById('analysis-loading').innerHTML = `
            <i class="fa-solid fa-triangle-exclamation" style="color:#e74c3c"></i>
            <span>Failed: ${e.message}</span>`;
    }
}

function renderAllAnalysisCharts(data) {
    const sorted = Object.values(data).sort((a, b) => a.benchmark_id - b.benchmark_id);
    renderMainResultsChart(sorted);
    renderAblationChart(data);
    renderDurationChart(sorted);

    const hasMultiRun = sorted.some(d => d.run_count >= 2);
    const panelD = document.getElementById('chart-d-panel');
    if (hasMultiRun) {
        panelD.style.display = '';
        renderStabilityChart(sorted);
    } else {
        panelD.style.display = 'none';
    }

    renderLatencyDecomposition(sorted);
    renderOperationHeatmap(sorted);
    renderComplexityScaling(sorted);
    renderRobotBreakdown(sorted);
    renderCoverageMatrix(sorted);
}

// Chart A — Main Results: success rate for B1–B8 (horizontal bar)
function renderMainResultsChart(sortedEntries) {
    destroyChart('A');

    const coreEntries = sortedEntries.filter(d => d.benchmark_id <= 8);
    if (coreEntries.length === 0) return;

    const labels = coreEntries.map(d => `B${d.benchmark_id}: ${d.benchmark_name}`);
    const rates  = coreEntries.map(d => +(d.mean_success_rate * 100).toFixed(1));
    const colors = coreEntries.map(d => successColor(d.mean_success_rate));
    const runCounts = coreEntries.map(d => d.run_count);

    const ctx = document.getElementById('chartA').getContext('2d');
    const tt = tooltipDefaults();

    analysisChartInstances['A'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Success Rate (%)',
                data: rates,
                backgroundColor: colors.map(c => c + 'cc'),
                borderColor: colors,
                borderWidth: 1.5,
                borderRadius: 3,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 200 },
            plugins: {
                legend: { display: false },
                tooltip: {
                    ...tt,
                    padding: 10,
                    cornerRadius: 2,
                    titleFont: { family: "'IBM Plex Sans', sans-serif", size: 12, weight: '600' },
                    bodyFont:  { family: "'IBM Plex Mono', monospace", size: 11 },
                    callbacks: {
                        label: (ctx) => {
                            const idx = ctx.dataIndex;
                            return [
                                `Success Rate: ${ctx.parsed.x}%`,
                                `Runs: ${runCounts[idx]}`,
                            ];
                        }
                    }
                }
            },
            scales: {
                x: {
                    min: 0,
                    max: 100,
                    grid: { color: chartGridColor() },
                    border: { color: chartBorderColor() },
                    ticks: {
                        color: chartTextColor(),
                        font: { family: "'IBM Plex Mono', monospace", size: 10 },
                        callback: v => v + '%',
                    },
                    title: {
                        display: true,
                        text: 'SUCCESS RATE (%)',
                        color: chartTextColor(),
                        font: { family: "'IBM Plex Mono', monospace", size: 10 },
                    }
                },
                y: {
                    grid: { display: false },
                    border: { color: chartBorderColor() },
                    ticks: { color: chartTextColor(), font: { family: "'IBM Plex Mono', monospace", size: 10 } }
                }
            }
        }
    });
}

// Chart B — Ablation: grouped bars enabled vs disabled for B9–B14
function renderAblationChart(data) {
    destroyChart('B');

    const ablationBenchmarks = Object.values(data)
        .filter(d => d.benchmark_id >= 9 && d.ablation)
        .sort((a, b) => a.benchmark_id - b.benchmark_id);

    if (ablationBenchmarks.length === 0) return;

    const metric = document.getElementById('ablation-metric').value;
    const metricLabels = {
        mean_success_rate:          'Success Rate',
        mean_hallucinated_ops:      'Hallucinated Ops',
        mean_reflexion_recoveries:  'Reflexion Recoveries',
        mean_negotiation_rounds:    'Negotiation Rounds',
    };

    const labels = ablationBenchmarks.map(d => `B${d.benchmark_id}: ${d.benchmark_name}`);

    // Collect all condition keys across benchmarks
    const conditionKeys = [...new Set(
        ablationBenchmarks.flatMap(d => Object.keys(d.ablation))
    )].sort();

    const condColors = [PALETTE.teal, PALETTE.orange, PALETTE.blue, PALETTE.vermillion];

    const datasets = conditionKeys.map((cond, i) => ({
        label: cond,
        data: ablationBenchmarks.map(d => {
            const ab = d.ablation[cond];
            if (!ab) return null;
            let val = ab[metric] ?? 0;
            if (metric === 'mean_success_rate') val = +(val * 100).toFixed(1);
            else val = +val.toFixed(2);
            return val;
        }),
        backgroundColor: condColors[i % condColors.length] + 'cc',
        borderColor: condColors[i % condColors.length],
        borderWidth: 1.5,
        borderRadius: 3,
    }));

    const isRate = metric === 'mean_success_rate';
    const ctx = document.getElementById('chartB').getContext('2d');
    const tt = tooltipDefaults();

    analysisChartInstances['B'] = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 200 },
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: chartTextColor(),
                        font: { family: "'IBM Plex Mono', monospace", size: 10 },
                        boxWidth: 12,
                    }
                },
                tooltip: {
                    ...tt,
                    padding: 10,
                    cornerRadius: 2,
                    titleFont: { family: "'IBM Plex Sans', sans-serif", size: 12, weight: '600' },
                    bodyFont:  { family: "'IBM Plex Mono', monospace", size: 11 },
                    callbacks: {
                        label: (ctx) => {
                            const suffix = isRate ? '%' : '';
                            return `${ctx.dataset.label}: ${ctx.parsed.y}${suffix}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ...(isRate ? { min: 0, max: 100 } : {}),
                    grid: { color: chartGridColor() },
                    border: { color: chartBorderColor() },
                    ticks: {
                        color: chartTextColor(),
                        font: { family: "'IBM Plex Mono', monospace", size: 10 },
                        callback: v => isRate ? v + '%' : v,
                    },
                    title: {
                        display: true,
                        text: metricLabels[metric].toUpperCase(),
                        color: chartTextColor(),
                        font: { family: "'IBM Plex Mono', monospace", size: 10 },
                    }
                },
                x: {
                    grid: { display: false },
                    border: { color: chartBorderColor() },
                    ticks: { color: chartTextColor(), font: { family: "'IBM Plex Mono', monospace", size: 10 } }
                }
            }
        }
    });
}

// Chart C — Duration: avg task duration across all benchmarks
function renderDurationChart(sortedEntries) {
    destroyChart('C');

    const labels = sortedEntries.map(d => `B${d.benchmark_id}`);
    const values = sortedEntries.map(d => +(d.mean_duration_ms / 1000).toFixed(2));
    const maxVal = Math.max(...values);
    const colors = values.map(v => {
        const t = maxVal > 0 ? v / maxVal : 0;
        // interpolate teal → orange → vermillion
        if (t < 0.5) return PALETTE.teal;
        if (t < 0.8) return PALETTE.orange;
        return PALETTE.vermillion;
    });

    const ctx = document.getElementById('chartC').getContext('2d');
    const tt = tooltipDefaults();

    analysisChartInstances['C'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Avg Duration (s)',
                data: values,
                backgroundColor: colors.map(c => c + 'cc'),
                borderColor: colors,
                borderWidth: 1.5,
                borderRadius: 3,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 200 },
            plugins: {
                legend: { display: false },
                tooltip: {
                    ...tt,
                    padding: 10,
                    cornerRadius: 2,
                    titleFont: { family: "'IBM Plex Sans', sans-serif", size: 12, weight: '600' },
                    bodyFont:  { family: "'IBM Plex Mono', monospace", size: 11 },
                    callbacks: {
                        label: ctx => `Duration: ${ctx.parsed.y}s`,
                        afterLabel: (ctx) => {
                            const entry = sortedEntries[ctx.dataIndex];
                            return `Runs: ${entry.run_count}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: chartGridColor() },
                    border: { color: chartBorderColor() },
                    ticks: {
                        color: chartTextColor(),
                        font: { family: "'IBM Plex Mono', monospace", size: 10 },
                        callback: v => v + 's',
                    },
                    title: {
                        display: true,
                        text: 'AVG DURATION (s)',
                        color: chartTextColor(),
                        font: { family: "'IBM Plex Mono', monospace", size: 10 },
                    }
                },
                x: {
                    grid: { display: false },
                    border: { color: chartBorderColor() },
                    ticks: { color: chartTextColor(), font: { family: "'IBM Plex Mono', monospace", size: 10 } }
                }
            }
        }
    });
}

// Chart D — Stability: mean ± std across runs (custom error-bar rendering)
function renderStabilityChart(sortedEntries) {
    destroyChart('D');

    const entries = sortedEntries.filter(d => d.run_count >= 2);
    if (entries.length === 0) return;

    const labels = entries.map(d => `B${d.benchmark_id}`);
    const means  = entries.map(d => +(d.mean_success_rate * 100).toFixed(1));
    const stds   = entries.map(d => +(d.std_success_rate  * 100).toFixed(1));
    const colors = means.map(m => successColor(m / 100));

    // Custom plugin to draw ± error bars
    const errorBarPlugin = {
        id: 'errorBars',
        afterDatasetsDraw(chart) {
            const { ctx, scales: { x, y } } = chart;
            const dataset = chart.data.datasets[0];
            chart.getDatasetMeta(0).data.forEach((bar, i) => {
                const mean = dataset.data[i];
                const std  = stds[i];
                if (std === 0) return;

                const xCenter = bar.x;
                const yTop    = y.getPixelForValue(mean + std);
                const yBot    = y.getPixelForValue(mean - std);
                const capW    = 6;

                ctx.save();
                ctx.strokeStyle = colors[i];
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.moveTo(xCenter, yTop);
                ctx.lineTo(xCenter, yBot);
                // caps
                ctx.moveTo(xCenter - capW, yTop);
                ctx.lineTo(xCenter + capW, yTop);
                ctx.moveTo(xCenter - capW, yBot);
                ctx.lineTo(xCenter + capW, yBot);
                ctx.stroke();
                ctx.restore();
            });
        }
    };

    const ctx = document.getElementById('chartD').getContext('2d');
    const tt = tooltipDefaults();

    analysisChartInstances['D'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Mean Success Rate (%)',
                data: means,
                backgroundColor: colors.map(c => c + 'cc'),
                borderColor: colors,
                borderWidth: 1.5,
                borderRadius: 3,
            }]
        },
        plugins: [errorBarPlugin],
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 200 },
            plugins: {
                legend: { display: false },
                tooltip: {
                    ...tt,
                    padding: 10,
                    cornerRadius: 2,
                    titleFont: { family: "'IBM Plex Sans', sans-serif", size: 12, weight: '600' },
                    bodyFont:  { family: "'IBM Plex Mono', monospace", size: 11 },
                    callbacks: {
                        label: (ctx) => {
                            const i = ctx.dataIndex;
                            return [
                                `Mean: ${means[i]}%`,
                                `σ: ±${stds[i]}%`,
                                `Runs: ${entries[i].run_count}`,
                            ];
                        }
                    }
                }
            },
            scales: {
                y: {
                    min: 0,
                    max: 100,
                    grid: { color: chartGridColor() },
                    border: { color: chartBorderColor() },
                    ticks: {
                        color: chartTextColor(),
                        font: { family: "'IBM Plex Mono', monospace", size: 10 },
                        callback: v => v + '%',
                    },
                    title: {
                        display: true,
                        text: 'MEAN SUCCESS RATE (%)',
                        color: chartTextColor(),
                        font: { family: "'IBM Plex Mono', monospace", size: 10 },
                    }
                },
                x: {
                    grid: { display: false },
                    border: { color: chartBorderColor() },
                    ticks: { color: chartTextColor(), font: { family: "'IBM Plex Mono', monospace", size: 10 } }
                }
            }
        }
    });
}

// ── Operation Categories ──────────────────────────────────────────────────────
const OP_CATEGORIES = {
    perception:   { ops: ['detect_object_stereo','detect_field','analyze_scene','generate_point_cloud','detect_all_fields'], color: PALETTE.blue },
    motion:       { ops: ['move_to_coordinate','adjust_end_effector_orientation','return_to_start_position','pick_object_at_coordinate','move_relative_to_object'], color: PALETTE.teal },
    grasp:        { ops: ['grasp_object','place_object','receive_handoff'], color: PALETTE.orange },
    coordination: { ops: ['signal','wait_for_signal','wait','detect_other_robot','mirror_movement','stabilize_object'], color: PALETTE.purple },
    gripper:      { ops: ['control_gripper','release_object'], color: PALETTE.skyblue },
};

function opCategory(opName) {
    for (const [cat, meta] of Object.entries(OP_CATEGORIES)) {
        if (meta.ops.includes(opName)) return cat;
    }
    return 'other';
}

// Chart E — Latency Decomposition: stacked bar, time split by op category per benchmark
function renderLatencyDecomposition(sortedEntries) {
    destroyChart('E');

    const entries = sortedEntries.filter(d => d.op_stats && Object.keys(d.op_stats).length > 0);
    if (entries.length === 0) return;

    const labels = entries.map(d => `B${d.benchmark_id}`);

    const catTotals = {};
    for (const [cat, meta] of Object.entries(OP_CATEGORIES)) {
        catTotals[cat] = entries.map(d => {
            let sum = 0;
            for (const op of meta.ops) {
                const stat = d.op_stats[op];
                if (stat) sum += stat.mean_duration_ms * stat.call_count;
            }
            return +(sum / 1000).toFixed(2);
        });
    }

    const datasets = Object.entries(OP_CATEGORIES).map(([cat, meta]) => ({
        label: cat,
        data: catTotals[cat],
        backgroundColor: meta.color + 'cc',
        borderColor: meta.color,
        borderWidth: 1,
    }));

    const ctx = document.getElementById('chartE').getContext('2d');
    const tt = tooltipDefaults();

    analysisChartInstances['E'] = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 200 },
            plugins: {
                legend: {
                    display: true,
                    labels: { color: chartTextColor(), font: { family: "'IBM Plex Mono', monospace", size: 10 }, boxWidth: 12 }
                },
                tooltip: {
                    ...tt, padding: 10, cornerRadius: 2,
                    titleFont: { family: "'IBM Plex Sans', sans-serif", size: 12, weight: '600' },
                    bodyFont:  { family: "'IBM Plex Mono', monospace", size: 11 },
                    callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y}s` }
                }
            },
            scales: {
                x: { stacked: true, grid: { display: false }, border: { color: chartBorderColor() }, ticks: { color: chartTextColor(), font: { family: "'IBM Plex Mono', monospace", size: 10 } } },
                y: {
                    stacked: true, beginAtZero: true,
                    grid: { color: chartGridColor() }, border: { color: chartBorderColor() },
                    ticks: { color: chartTextColor(), font: { family: "'IBM Plex Mono', monospace", size: 10 }, callback: v => v + 's' },
                    title: { display: true, text: 'TOTAL TIME (s)', color: chartTextColor(), font: { family: "'IBM Plex Mono', monospace", size: 10 } }
                }
            }
        }
    });
}

// Chart F — Operation Timing Heatmap: HTML table, cells = avg ms per op per benchmark
function renderOperationHeatmap(sortedEntries) {
    const container = document.getElementById('chartF');
    if (!container) return;

    const entries = sortedEntries.filter(d => d.op_stats && Object.keys(d.op_stats).length > 0);
    if (entries.length === 0) { container.innerHTML = '<p style="color:var(--text-dim);padding:1rem">No step data available.</p>'; return; }

    // Collect all ops across all benchmarks
    const allOps = [...new Set(entries.flatMap(d => Object.keys(d.op_stats)))].sort();
    // Column max for per-column color scaling
    const colMax = {};
    allOps.forEach(op => {
        colMax[op] = Math.max(...entries.map(d => d.op_stats[op]?.mean_duration_ms ?? 0));
    });

    const isDark = isDarkMode();
    let html = '<div class="heatmap-scroll"><table class="heatmap-table"><thead><tr><th>Benchmark</th>';
    allOps.forEach(op => { html += `<th title="${op}">${op.replace(/_/g, '_<wbr>')}</th>`; });
    html += '</tr></thead><tbody>';

    entries.forEach(d => {
        html += `<tr><td class="heatmap-bm-label">B${d.benchmark_id}</td>`;
        allOps.forEach(op => {
            const stat = d.op_stats[op];
            if (!stat) { html += '<td class="heatmap-cell heatmap-cell--empty">—</td>'; return; }
            const ms = stat.mean_duration_ms;
            const t = colMax[op] > 0 ? ms / colMax[op] : 0;
            const alpha = 0.1 + t * 0.85;
            const bg = isDark
                ? `rgba(230,159,0,${alpha})`
                : `rgba(180,100,0,${alpha})`;
            const textColor = t > 0.6 ? (isDark ? '#fff' : '#fff') : (isDark ? '#e6edf3' : '#333');
            const label = ms >= 1000 ? `${(ms/1000).toFixed(1)}s` : `${ms.toFixed(0)}ms`;
            html += `<td class="heatmap-cell" style="background:${bg};color:${textColor}" title="${op}: ${ms.toFixed(0)}ms avg">${label}</td>`;
        });
        html += '</tr>';
    });

    html += '</tbody></table></div>';
    container.innerHTML = html;
}

// Chart G — Complexity Scaling: scatter plot, plan length vs total duration
function renderComplexityScaling(sortedEntries) {
    destroyChart('G');

    const entries = sortedEntries.filter(d => d.mean_plan_length > 0);
    if (entries.length === 0) return;

    const points = entries.map(d => ({ x: d.mean_plan_length, y: +(d.mean_duration_ms / 1000).toFixed(2) }));
    const pointLabels = entries.map(d => `B${d.benchmark_id}: ${d.benchmark_name}`);

    const ctx = document.getElementById('chartG').getContext('2d');
    const tt = tooltipDefaults();

    analysisChartInstances['G'] = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Benchmarks',
                data: points,
                backgroundColor: PALETTE.blue + 'cc',
                borderColor: PALETTE.blue,
                borderWidth: 1.5,
                pointRadius: 7,
                pointHoverRadius: 9,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 200 },
            plugins: {
                legend: { display: false },
                tooltip: {
                    ...tt, padding: 10, cornerRadius: 2,
                    titleFont: { family: "'IBM Plex Sans', sans-serif", size: 12, weight: '600' },
                    bodyFont:  { family: "'IBM Plex Mono', monospace", size: 11 },
                    callbacks: {
                        title: ctx => pointLabels[ctx[0].dataIndex],
                        label: ctx => [`Plan length: ${ctx.parsed.x} ops`, `Avg duration: ${ctx.parsed.y}s`],
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: chartGridColor() }, border: { color: chartBorderColor() },
                    ticks: { color: chartTextColor(), font: { family: "'IBM Plex Mono', monospace", size: 10 } },
                    title: { display: true, text: 'PLAN LENGTH (ops)', color: chartTextColor(), font: { family: "'IBM Plex Mono', monospace", size: 10 } }
                },
                y: {
                    beginAtZero: true,
                    grid: { color: chartGridColor() }, border: { color: chartBorderColor() },
                    ticks: { color: chartTextColor(), font: { family: "'IBM Plex Mono', monospace", size: 10 }, callback: v => v + 's' },
                    title: { display: true, text: 'AVG DURATION (s)', color: chartTextColor(), font: { family: "'IBM Plex Mono', monospace", size: 10 } }
                }
            }
        }
    });
}

// Chart H — Per-Robot Step Breakdown: grouped bars, total step time per robot per benchmark
function renderRobotBreakdown(sortedEntries) {
    destroyChart('H');

    const entries = sortedEntries.filter(d =>
        d.per_robot_stats && Object.keys(d.per_robot_stats).length > 0
    );
    const panelH = document.getElementById('chart-h-panel');
    if (!entries.length) { if (panelH) panelH.style.display = 'none'; return; }
    if (panelH) panelH.style.display = '';

    const labels = entries.map(d => `B${d.benchmark_id}`);
    const robots = [...new Set(entries.flatMap(d => Object.keys(d.per_robot_stats)))].sort();
    const robotColors = [PALETTE.teal, PALETTE.orange, PALETTE.blue, PALETTE.vermillion];

    const datasets = robots.map((rid, i) => ({
        label: rid,
        data: entries.map(d => {
            const rs = d.per_robot_stats[rid];
            return rs ? +(rs.total_duration_ms / 1000).toFixed(2) : 0;
        }),
        backgroundColor: robotColors[i % robotColors.length] + 'cc',
        borderColor: robotColors[i % robotColors.length],
        borderWidth: 1.5,
        borderRadius: 3,
    }));

    const ctx = document.getElementById('chartH').getContext('2d');
    const tt = tooltipDefaults();

    analysisChartInstances['H'] = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 200 },
            plugins: {
                legend: {
                    display: true,
                    labels: { color: chartTextColor(), font: { family: "'IBM Plex Mono', monospace", size: 10 }, boxWidth: 12 }
                },
                tooltip: {
                    ...tt, padding: 10, cornerRadius: 2,
                    titleFont: { family: "'IBM Plex Sans', sans-serif", size: 12, weight: '600' },
                    bodyFont:  { family: "'IBM Plex Mono', monospace", size: 11 },
                    callbacks: {
                        label: ctx => {
                            const rid = ctx.dataset.label;
                            const entry = entries[ctx.dataIndex];
                            const rs = entry?.per_robot_stats[rid];
                            const steps = rs?.step_count ?? 0;
                            return [`${rid}: ${ctx.parsed.y}s total`, `${steps} steps`];
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: chartGridColor() }, border: { color: chartBorderColor() },
                    ticks: { color: chartTextColor(), font: { family: "'IBM Plex Mono', monospace", size: 10 }, callback: v => v + 's' },
                    title: { display: true, text: 'TOTAL STEP TIME (s)', color: chartTextColor(), font: { family: "'IBM Plex Mono', monospace", size: 10 } }
                },
                x: {
                    grid: { display: false }, border: { color: chartBorderColor() },
                    ticks: { color: chartTextColor(), font: { family: "'IBM Plex Mono', monospace", size: 10 } }
                }
            }
        }
    });
}

// Chart I — Coverage Matrix: binary HTML table, op present in benchmark steps
function renderCoverageMatrix(sortedEntries) {
    const container = document.getElementById('chartI');
    if (!container) return;

    const entries = sortedEntries.filter(d => d.op_stats && Object.keys(d.op_stats).length > 0);
    if (entries.length === 0) { container.innerHTML = '<p style="color:var(--text-dim);padding:1rem">No step data available.</p>'; return; }

    const allOps = [...new Set(entries.flatMap(d => Object.keys(d.op_stats)))].sort();

    let html = '<div class="heatmap-scroll"><table class="heatmap-table coverage-table"><thead><tr><th>Operation</th>';
    entries.forEach(d => { html += `<th>B${d.benchmark_id}</th>`; });
    html += '</tr></thead><tbody>';

    allOps.forEach(op => {
        const cat = opCategory(op);
        const catColor = OP_CATEGORIES[cat]?.color ?? '#888';
        html += `<tr><td class="heatmap-bm-label" style="border-left:3px solid ${catColor}">${op}</td>`;
        entries.forEach(d => {
            const present = !!d.op_stats[op];
            html += present
                ? `<td class="heatmap-cell coverage-cell--present" title="B${d.benchmark_id} uses ${op}">✓</td>`
                : `<td class="heatmap-cell heatmap-cell--empty">—</td>`;
        });
        html += '</tr>';
    });

    html += '</tbody></table></div>';
    container.innerHTML = html;
}

// ── Export ────────────────────────────────────────────────────────────────────
function exportChartPng(chartKey, filename) {
    const instance = analysisChartInstances[chartKey];
    if (!instance) { alert('Chart not rendered yet.'); return; }

    const canvas = instance.canvas;

    if (thesisMode) {
        // Compose on a white background for print
        const offscreen = document.createElement('canvas');
        offscreen.width  = canvas.width;
        offscreen.height = canvas.height;
        const offCtx = offscreen.getContext('2d');
        offCtx.fillStyle = '#ffffff';
        offCtx.fillRect(0, 0, offscreen.width, offscreen.height);
        offCtx.drawImage(canvas, 0, 0);
        triggerDownload(offscreen.toDataURL('image/png'), filename);
    } else {
        canvas.toBlob(blob => {
            triggerDownload(URL.createObjectURL(blob), filename);
        }, 'image/png');
    }
}

function triggerDownload(href, filename) {
    const a = document.createElement('a');
    a.href = href;
    a.download = filename;
    a.click();
}

async function exportAllCharts() {
    const pairs = [
        ['A', 'main-results.png'],
        ['B', 'ablation.png'],
        ['C', 'duration.png'],
        ['D', 'stability.png'],
        ['E', 'latency-decomposition.png'],
        ['G', 'complexity-scaling.png'],
        ['H', 'robot-breakdown.png'],
    ];
    for (const [key, name] of pairs) {
        if (analysisChartInstances[key]) {
            exportChartPng(key, name);
            // small stagger so browser handles multiple downloads
            await new Promise(r => setTimeout(r, 300));
        }
    }
}

async function downloadAggregateJson() {
    if (!aggregateData) { alert('No data loaded.'); return; }
    const blob = new Blob([JSON.stringify(aggregateData, null, 2)], { type: 'application/json' });
    triggerDownload(URL.createObjectURL(blob), 'benchmark-aggregate.json');
}
