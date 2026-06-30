const PALETTE = {
    blue: '#0072B2',
    orange: '#E69F00',
    teal: '#009E73',
    vermillion: '#D55E00',
    skyblue: '#56B4E9',
    yellow: '#F0E442',
    purple: '#CC79A7',
};

const PASS_COLOR = PALETTE.teal;
const FAIL_COLOR = PALETTE.vermillion;
const WARN_COLOR = PALETTE.orange;

// Escape server-provided strings (model names, benchmark names) before
// interpolating into innerHTML/title attributes.
function escHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

let stepChartInstance = null;
let analysisChartInstances = {};
let compareStepChartInstance = null;
let aggregateData = null;   // cached from /api/benchmarks/aggregate

// Analysis-tab cleanup rules, mirroring tools/PlotBenchmarks.py so the dashboard
// agrees with the static thesis figures.
const ABLATION_BIDS = [11, 12, 13, 14, 15, 16]; // b17 is AutoRT, not a paired ablation
const OP_MIN_BENCH_PRESENCE = 2;  // op must appear in >=2 benchmarks
const OP_MIN_DURATION_MS = 100;   // ...and exceed 100ms somewhere

// Models excluded from the Model × Task matrix (mirrors PlotBenchmarks.py
// EXCLUDE_MODELS). The nomic model is the RAG embedder, not a generation model,
// but gets tagged on some runs.
const EXCLUDED_MODELS = new Set([
    'text-embedding-nomic-embed-text-v1.5',
]);

// Ops worth charting: present in >= OP_MIN_BENCH_PRESENCE benchmarks AND with a
// max mean duration over OP_MIN_DURATION_MS (mirrors PlotBenchmarks.plot_op_latency).
function interestingOps(entries) {
    const presence = {};   // op -> count of benchmarks it appears in
    const maxDur = {};     // op -> max mean_duration_ms across benchmarks
    for (const d of entries) {
        for (const [op, stat] of Object.entries(d.op_stats || {})) {
            presence[op] = (presence[op] || 0) + 1;
            maxDur[op] = Math.max(maxDur[op] || 0, stat.mean_duration_ms ?? 0);
        }
    }
    return new Set(
        Object.keys(presence).filter(
            op => presence[op] >= OP_MIN_BENCH_PRESENCE && maxDur[op] > OP_MIN_DURATION_MS
        )
    );
}
// Some benchmarks record no per-step data (B9 fails before any op; B11/B14/B17
// are ablation/offline runs that log only aggregate metrics). They can't appear
// in op- or robot-level charts — flag them explicitly instead of dropping them
// silently so the gaps are self-explanatory.
function renderExclusionNote(elId, sortedEntries, shownBids, reason) {
    const el = document.getElementById(elId);
    if (!el) return;
    const shown = new Set(shownBids);
    const missing = sortedEntries
        .map(d => d.benchmark_id)
        .filter(b => !shown.has(b))
        .sort((a, b) => a - b);
    if (missing.length === 0) {
        el.style.display = 'none';
        el.innerHTML = '';
        return;
    }
    el.style.display = '';
    el.innerHTML = `<i class="fa-solid fa-circle-info"></i>Not shown — ${reason}: `
        + missing.map(b => 'B' + b).join(', ');
}

let thesisMode = false;
let currentTab = 'details';
let compareSet = new Set();       // filenames selected for comparison
let compareDataCache = {};        // { filename: runData }
let folderCheckboxes = [];        // [{ checkbox, filepaths }] for folder-level selection

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
    if (tab === 'compare') loadCompareView();
}

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

function toggleThesisMode() {
    thesisMode = !thesisMode;
    document.body.classList.toggle('thesis-mode', thesisMode);

    const btn = document.getElementById('btn-thesis-mode');
    if (btn) btn.classList.toggle('bm-tab--active', thesisMode);

    applyChartDefaults();
    if (aggregateData) renderAllAnalysisCharts(aggregateData);
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

function syncAllCheckboxes() {
    document.querySelectorAll('.compare-checkbox').forEach(cb => {
        const fn = cb.dataset.filename;
        cb.checked = compareSet.has(fn);
        const item = cb.closest('.history-item');
        if (item) item.classList.toggle('history-item--in-compare', compareSet.has(fn));
    });
    folderCheckboxes.forEach(({ checkbox, filepaths }) => {
        const selected = filepaths.filter(fp => compareSet.has(fp)).length;
        checkbox.checked = selected === filepaths.length && filepaths.length > 0;
        checkbox.indeterminate = selected > 0 && selected < filepaths.length;
    });
}

function toggleCompareItem(filename, checked) {
    if (checked) {
        compareSet.add(filename);
    } else {
        compareSet.delete(filename);
        delete compareDataCache[filename];
    }
    updateCompareTab();
    syncAllCheckboxes();
    refreshCompareIfActive();
}

// Re-render the compare view live when it's the active tab and still has enough
// runs to show. updateCompareTab() already bounces to 'details' when the set
// drops below 2, so we only refresh while a valid comparison is on screen.
function refreshCompareIfActive() {
    if (currentTab === 'compare' && compareSet.size >= 2) {
        loadCompareView();
    }
}

function updateCompareTab() {
    const btn = document.getElementById('btn-compare-tab');
    if (!btn) return;
    if (compareSet.size >= 2) {
        btn.style.display = '';
        btn.innerHTML = `<i class="fa-solid fa-code-compare"></i> Compare (${compareSet.size})`;
    } else {
        btn.style.display = 'none';
        if (currentTab === 'compare') switchTab('details');
    }
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
        folderCheckboxes = [];

        // Build two-level map: benchmarkKey -> { name, models: Map<modelKey, files[]>, files: [] }
        const benchmarkMap = new Map();
        files.forEach(file => {
            const parts = file.folder ? file.folder.split('/') : [];
            const benchmarkKey = parts[0] || '';
            const modelKey = parts[1] || '';
            if (!benchmarkMap.has(benchmarkKey)) {
                benchmarkMap.set(benchmarkKey, { name: file.benchmark_name || '', models: new Map(), files: [] });
            }
            const entry = benchmarkMap.get(benchmarkKey);
            if (!entry.name && file.benchmark_name) entry.name = file.benchmark_name;
            if (modelKey) {
                if (!entry.models.has(modelKey)) entry.models.set(modelKey, []);
                entry.models.get(modelKey).push(file);
            } else {
                entry.files.push(file);
            }
        });

        const numOf = k => parseInt(k.replace(/\D/g, ''), 10);
        const sortedBenchmarks = [...benchmarkMap.keys()].sort((a, b) => {
            if (a === '' && b !== '') return 1;
            if (b === '' && a !== '') return -1;
            const na = numOf(a), nb = numOf(b);
            if (!isNaN(na) && !isNaN(nb)) return na - nb;
            return a.localeCompare(b);
        });

        const makeHistoryItem = (file, container) => {
            const filepath = file.filepath;
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
            item.dataset.filename = filepath;

            const isChecked = compareSet.has(filepath);
            item.innerHTML = `
                <div style="display:flex;align-items:flex-start;gap:0">
                    <input type="checkbox" class="compare-checkbox" data-filename="${filepath}" ${isChecked ? 'checked' : ''} title="Add to compare">
                    <div style="flex:1;min-width:0">
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
                    </div>
                </div>
            `;

            const checkbox = item.querySelector('.compare-checkbox');
            checkbox.addEventListener('click', (e) => {
                e.stopPropagation();
                toggleCompareItem(filepath, checkbox.checked);
            });
            item.addEventListener('click', (e) => {
                if (e.target === checkbox) return;
                document.querySelectorAll('.history-item').forEach(i => i.classList.remove('active'));
                item.classList.add('active');
                loadBenchmarkDetails(filepath);
            });
            container.appendChild(item);
        };

        const makeFolderHeader = (labelText, allFilepaths, isModel = false) => {
            const header = document.createElement('div');
            header.className = 'folder-header';

            const left = document.createElement('div');
            left.className = 'folder-header-left';

            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.className = 'folder-compare-checkbox';
            cb.title = 'Select all for comparison';
            cb.addEventListener('click', (e) => {
                e.stopPropagation();
                allFilepaths.forEach(fp => {
                    if (cb.checked) {
                        compareSet.add(fp);
                    } else {
                        compareSet.delete(fp);
                        delete compareDataCache[fp];
                    }
                });
                updateCompareTab();
                syncAllCheckboxes();
                refreshCompareIfActive();
            });
            left.appendChild(cb);

            const labelSpan = document.createElement('span');
            labelSpan.textContent = labelText;
            left.appendChild(labelSpan);

            const chevron = document.createElement('i');
            chevron.className = 'fa-solid fa-chevron-down folder-chevron';

            header.appendChild(left);
            header.appendChild(chevron);

            folderCheckboxes.push({ checkbox: cb, filepaths: allFilepaths });
            return header;
        };

        sortedBenchmarks.forEach(benchmarkKey => {
            const { name, models, files: flatFiles } = benchmarkMap.get(benchmarkKey);

            // Collect all filepaths in this benchmark for the top-level checkbox
            const allModelFiles = [...models.values()].flat();
            const allBenchmarkFilepaths = [...flatFiles, ...allModelFiles].map(f => f.filepath);

            const group = document.createElement('div');
            group.className = 'folder-group';

            const labelText = benchmarkKey === ''
                ? 'Other'
                : (name ? `${benchmarkKey.toUpperCase()} — ${name}` : benchmarkKey.toUpperCase());
            const header = makeFolderHeader(labelText, allBenchmarkFilepaths, false);
            header.addEventListener('click', (e) => {
                if (e.target.classList.contains('folder-compare-checkbox')) return;
                group.classList.toggle('expanded');
            });
            group.appendChild(header);

            const itemsContainer = document.createElement('div');
            itemsContainer.className = 'folder-items';
            group.appendChild(itemsContainer);
            listContainer.appendChild(group);

            // Flat files (no model sub-folder) go directly in outer items
            flatFiles.forEach(file => makeHistoryItem(file, itemsContainer));

            // Model sub-groups, sorted alphabetically
            [...models.keys()].sort((a, b) => a.localeCompare(b)).forEach(modelKey => {
                const modelFiles = models.get(modelKey);
                const modelFilepaths = modelFiles.map(f => f.filepath);

                const modelGroup = document.createElement('div');
                modelGroup.className = 'folder-group folder-group--model';

                const modelHeader = makeFolderHeader(modelKey, modelFilepaths, true);
                modelHeader.addEventListener('click', (e) => {
                    if (e.target.classList.contains('folder-compare-checkbox')) return;
                    modelGroup.classList.toggle('expanded');
                });
                modelGroup.appendChild(modelHeader);

                const modelItems = document.createElement('div');
                modelItems.className = 'folder-items';
                modelGroup.appendChild(modelItems);
                itemsContainer.appendChild(modelGroup);

                modelFiles.forEach(file => makeHistoryItem(file, modelItems));
            });
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
    renderChainMetrics(data);
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

        const robotId = step.robot_id != null ? `<span class="robot-badge robot-badge--${step.robot_id}">${step.robot_id}</span>` : '<span class="step-na">—</span>';
        const pgId = step.parallel_group_id != null ? `<span class="pg-badge">pg${step.parallel_group_id}</span>` : '<span class="step-na">—</span>';
        const retryCount = step.retry_count ?? 0;
        const retryCell = retryCount > 0
            ? `<span style="color:var(--warn,#E69F00);font-weight:600">${retryCount}</span>`
            : `<span class="step-na">—</span>`;

        tr.innerHTML = `
            <td>${step.index}</td>
            <td class="step-op">${step.operation}</td>
            <td>${statusIcon}</td>
            <td>${step.duration_ms.toFixed(0)}</td>
            <td>${robotId}</td>
            <td>${pgId}</td>
            <td>${retryCell}</td>
            <td class="step-error">${step.error_code || ''} ${step.error_message ? '- ' + step.error_message : ''}</td>
        `;

        if (data.first_failure_step != null && step.index === data.first_failure_step) {
            tr.classList.add('step-row--first-failure');
        }

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
        use_rag: { label: 'RAG', desc: 'Retrieval-Augmented Generation active — LLM has access to operation docs and prior examples. Tests whether retrieval improves plan quality.' },
        use_vgn: { label: 'VGN', desc: 'Volumetric Grasp Network enabled — replaces heuristic grasp candidates with learned 6-DOF grasp poses. Key ablation for grasp success rate.' },
        use_knowledge_graph: { label: 'KG', desc: 'Knowledge Graph active — spatial reasoning layer provides reachability, proximity, and handoff queries to the LLM. Ablation for planning quality.' },
        use_ros_movement: { label: 'ROS/MoveIt', desc: 'MoveIt trajectory planning active (vs. Unity IK). Tests whether motion planning improves collision avoidance and trajectory quality.' },
        reflection_enabled: { label: 'Reflection', desc: 'Reflection self-correction loop enabled — LLM retries failed steps with error context. Directly measured via reflection_recoveries metric.' },
        dry_run: { label: 'Dry Run', desc: 'No real Unity execution — operations are simulated. Used for testing plan generation without robot hardware. Results not comparable to live runs.' },
        use_negotiation: { label: 'Negotiation', desc: 'Multi-robot LLM negotiation active — robots negotiate task allocation before execution. Core ablation for dual-arm coordination benchmarks.' },
    };

    const modeMeta = {
        offline: 'Offline — operations executed against mock/simulated responses. Use for plan generation tests; timing and success rates are not representative of real robot performance.',
        live: 'Live — operations dispatched to Unity simulation with real physics. Results are representative; compare directly with other live runs.',
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
            <span class="run-info-label">Feature Flags <i class="fa-solid fa-circle-info kpi-info" title="Active system capabilities for this run. Enables comparing ablation conditions — each flag corresponds to a benchmark series (B11–B16). Hover each pill for details."></i></span>
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

    const qm = [
        { label: 'Hallucinated Ops',     value: data.hallucinated_ops     ?? 0, warn: v => v > 0,
          tip: 'Operations the LLM generated that do not exist in the operation registry. Always fail. Higher values indicate poor LLM adherence to the available operation set.' },
        { label: 'Reflection Recoveries', value: data.reflection_recoveries ?? data.reflexion_recoveries ?? 0, warn: () => false,
          tip: 'Number of times the Reflection self-correction loop successfully recovered a failed step by re-prompting with error context. Higher is better when reflection is enabled.' },
        { label: 'Negotiation Rounds',   value: data.negotiation_rounds    ?? 0, warn: () => false,
          tip: 'LLM negotiation rounds completed between robots before execution. Relevant for dual-arm benchmarks with use_negotiation enabled.' },
        { label: 'Total Retries',        value: data.retry_count           ?? 0, warn: v => v > 0,
          tip: 'Total number of step-level retries across the entire run (includes Reflection retries and policy retries). Non-zero values indicate at least one step needed recovery.' },
    ];
    const qmHtml = qm.map(m =>
        `<span class="qm-chip${m.warn(m.value) ? ' qm-chip--warn' : ''}" title="${m.tip}">${m.label}: <strong>${m.value}</strong></span>`
    ).join('');
    html += `<div class="run-info-row run-info-row--qm">
        <span class="run-info-label">Quality Metrics <i class="fa-solid fa-circle-info kpi-info" title="Run-level quality signals: plan correctness (hallucinations), self-correction effectiveness (reflection), coordination overhead (negotiation), and overall retry burden."></i></span>
        <div class="qm-chips">${qmHtml}</div>
    </div>`;

    html += '</div>';
    panel.innerHTML = html;
    panel.style.display = '';
}

function renderChainMetrics(data) {
    const panel = document.getElementById('chain-metrics-panel');
    if (!panel) return;
    const cm = data.chain_metrics;
    if (!cm) { panel.style.display = 'none'; return; }

    const phases = Object.entries(cm.per_phase_success || {});
    const phaseBars = phases.map(([phase, rate]) => {
        const pct = Math.round(rate * 100);
        const cls = rate >= 1 ? 'progress-pass' : rate > 0 ? 'progress-warn' : 'progress-fail';
        return `<div class="chain-phase">
            <span class="chain-phase-label">${phase}</span>
            <div class="history-item-progress" style="flex:1">
                <div class="history-item-progress-fill ${cls}" style="width:${pct}%"></div>
            </div>
            <span class="chain-phase-pct">${pct}%</span>
        </div>`;
    }).join('');

    panel.innerHTML = `
        <div class="run-info-header"><i class="fa-solid fa-list-check"></i> Chain Metrics</div>
        <div class="run-info-body">
            <div class="run-info-row">
                <span class="run-info-label">Tasks Completed <i class="fa-solid fa-circle-info kpi-info" title="Number of discrete sub-tasks completed out of the total planned. Each task is a self-contained pick/place/transport sequence within the chain benchmark."></i></span>
                <span>${cm.completed_tasks} / ${cm.total_tasks}</span>
            </div>
            <div class="run-info-row">
                <span class="run-info-label">Error Rate <i class="fa-solid fa-circle-info kpi-info" title="Fraction of sub-tasks that failed (error_rate = 1 - completed/total). Directly comparable across chain benchmarks."></i></span>
                <span>${(cm.error_rate * 100).toFixed(1)}%</span>
            </div>
            <div class="run-info-row">
                <span class="run-info-label">Recoveries <i class="fa-solid fa-circle-info kpi-info" title="Number of sub-task-level recovery attempts (distinct from step-level Reflection retries). Positive values mean the chain self-healed at least once."></i></span>
                <span>${cm.recovery_count}</span>
            </div>
            ${phases.length ? `<div class="run-info-row run-info-row--phases">
                <span class="run-info-label">Phase Success <i class="fa-solid fa-circle-info kpi-info" title="Pass rate per pipeline phase (A=detect, B=grasp, C=transport, D=place). Identifies which phase of the chain degrades first."></i></span>
                <div class="chain-phases">${phaseBars}</div>
            </div>` : ''}
        </div>`;
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
        const count = stats.count || 0;
        const fails = stats.fail_count || 0;
        const avgMs = stats.avg_duration_ms != null ? stats.avg_duration_ms.toFixed(0) : '—';
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
                    bodyFont: { family: "'IBM Plex Mono', monospace", size: 11 },
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
    renderAblationChart(data);

    const hasMultiRun = sorted.some(d => d.run_count >= 2);
    const panelD = document.getElementById('chart-d-panel');
    if (hasMultiRun) {
        panelD.style.display = '';
        renderStabilityChart(sorted);
    } else {
        panelD.style.display = 'none';
    }

    renderModelMatrix(sorted);
    renderModelOpMatrix(sorted);
    renderLatencyDecomposition(sorted);
    renderOperationHeatmap(sorted);
    renderComplexityScaling(sorted);
    renderRobotBreakdown(sorted);
}

// Model × Task success matrix — all benchmarks (B1–B17) broken out by model.
// b1-b10 are run across several LLMs; the top-level success rate pools them, so
// this matrix is the trustworthy per-model view. Ablations (B11+) appear too
// when they carry per-model data; benchmarks with no by_model breakout are
// dropped by the filter below.
function renderModelMatrix(sortedEntries) {
    const container = document.getElementById('chartModelMatrix');
    if (!container) return;

    const entries = sortedEntries.filter(
        d => d.by_model && Object.keys(d.by_model).length > 0
    );
    if (entries.length === 0) {
        container.innerHTML = '<p style="color:var(--text-dim);padding:1rem">No model data available.</p>';
        return;
    }

    const models = [...new Set(entries.flatMap(d => Object.keys(d.by_model)))]
        .filter(m => !EXCLUDED_MODELS.has(m))
        .sort((a, b) => a.localeCompare(b));

    let html = '<div class="heatmap-scroll"><table class="heatmap-table model-matrix-table"><thead><tr><th>Model</th>';
    entries.forEach(d => {
        html += `<th title="B${d.benchmark_id}: ${escHtml(d.benchmark_name)}">B${d.benchmark_id}</th>`;
    });
    html += '</tr></thead><tbody>';

    models.forEach(model => {
        const safeModel = escHtml(model);
        html += `<tr><td class="heatmap-bm-label" title="${safeModel}">${safeModel}</td>`;
        entries.forEach(d => {
            const m = d.by_model[model];
            if (!m) { html += '<td class="heatmap-cell heatmap-cell--empty">—</td>'; return; }
            const rate = m.mean_success_rate;
            const bg = successColor(rate);
            const textColor = (rate > 0.3 && rate < 0.8) ? '#1a1a1a' : '#fff';
            html += `<td class="heatmap-cell" style="background:${bg};color:${textColor}" `
                + `title="${safeModel} · B${d.benchmark_id}: ${(rate * 100).toFixed(0)}% over ${m.run_count} runs">`
                + `${(rate * 100).toFixed(0)}%</td>`;
        });
        html += '</tr>';
    });

    html += '</tbody></table></div>';
    container.innerHTML = html;
}

// Model × Operation success matrix — per-model success rate for each operation
// type, pooled across all benchmarks. Surfaces which ops a given model is
// unreliable at (e.g. grasp_object) independent of which task invoked them.
function renderModelOpMatrix(sortedEntries) {
    const container = document.getElementById('chartModelOpMatrix');
    if (!container) return;

    // Pool op stats per model across every benchmark that carries a by_model
    // op breakout. ok = call_count - fail_count (call_count is total steps).
    const perModel = {};   // model -> op -> {ok, total}
    sortedEntries.forEach(d => {
        if (!d.by_model) return;
        Object.entries(d.by_model).forEach(([model, mv]) => {
            if (EXCLUDED_MODELS.has(model) || !mv.op_stats) return;
            const bucket = perModel[model] || (perModel[model] = {});
            Object.entries(mv.op_stats).forEach(([op, os]) => {
                const total = os.call_count || 0;
                const fails = os.fail_count || 0;
                const b = bucket[op] || (bucket[op] = { ok: 0, total: 0 });
                b.ok += total - fails;
                b.total += total;
            });
        });
    });

    const models = Object.keys(perModel).sort((a, b) => a.localeCompare(b));
    if (models.length === 0) {
        container.innerHTML = '<p style="color:var(--text-dim);padding:1rem">No per-model operation data available.</p>';
        return;
    }

    // Column order: ops sorted by total call volume (most-exercised first).
    const opTotals = {};
    models.forEach(m => Object.entries(perModel[m]).forEach(([op, b]) => {
        opTotals[op] = (opTotals[op] || 0) + b.total;
    }));
    const ops = Object.keys(opTotals).sort((a, b) => opTotals[b] - opTotals[a]);

    let html = '<div class="heatmap-scroll"><table class="heatmap-table model-matrix-table"><thead><tr><th>Model</th>';
    ops.forEach(op => {
        html += `<th title="${escHtml(op)}">${escHtml(op)}</th>`;
    });
    html += '</tr></thead><tbody>';

    models.forEach(model => {
        const safeModel = escHtml(model);
        html += `<tr><td class="heatmap-bm-label" title="${safeModel}">${safeModel}</td>`;
        ops.forEach(op => {
            const b = perModel[model][op];
            if (!b || b.total === 0) { html += '<td class="heatmap-cell heatmap-cell--empty">—</td>'; return; }
            const rate = b.ok / b.total;
            const bg = successColor(rate);
            const textColor = (rate > 0.3 && rate < 0.8) ? '#1a1a1a' : '#fff';
            html += `<td class="heatmap-cell" style="background:${bg};color:${textColor}" `
                + `title="${safeModel} · ${escHtml(op)}: ${(rate * 100).toFixed(0)}% over ${b.total} calls">`
                + `${(rate * 100).toFixed(0)}%</td>`;
        });
        html += '</tr>';
    });

    html += '</tbody></table></div>';
    container.innerHTML = html;
}

// Chart B — Ablation: grouped bars enabled vs disabled for B11–B16
function renderAblationChart(data) {
    destroyChart('B');

    const ablationBenchmarks = Object.values(data)
        .filter(d => ABLATION_BIDS.includes(d.benchmark_id) && d.ablation)
        .sort((a, b) => a.benchmark_id - b.benchmark_id);

    if (ablationBenchmarks.length === 0) return;

    const metric = document.getElementById('ablation-metric').value;
    const metricLabels = {
        mean_success_rate: 'Success Rate',
        mean_hallucinated_ops: 'Hallucinated Ops',
        mean_reflection_recoveries: 'Reflection Recoveries',
        mean_negotiation_rounds: 'Negotiation Rounds',
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
                    bodyFont: { family: "'IBM Plex Mono', monospace", size: 11 },
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

// Chart D — Stability: mean ± std across runs (custom error-bar rendering)
function renderStabilityChart(sortedEntries) {
    destroyChart('D');

    const entries = sortedEntries.filter(d => d.run_count >= 2);
    if (entries.length === 0) return;

    const labels = entries.map(d => `B${d.benchmark_id}`);
    const means = entries.map(d => +(d.mean_success_rate * 100).toFixed(1));
    const stds = entries.map(d => +(d.std_success_rate * 100).toFixed(1));
    const colors = means.map(m => successColor(m / 100));

    // Custom plugin to draw ± error bars
    const errorBarPlugin = {
        id: 'errorBars',
        afterDatasetsDraw(chart) {
            const { ctx, scales: { x, y } } = chart;
            const dataset = chart.data.datasets[0];
            chart.getDatasetMeta(0).data.forEach((bar, i) => {
                const mean = dataset.data[i];
                const std = stds[i];
                if (std === 0) return;

                const xCenter = bar.x;
                const yTop = y.getPixelForValue(mean + std);
                const yBot = y.getPixelForValue(mean - std);
                const capW = 6;

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
                    bodyFont: { family: "'IBM Plex Mono', monospace", size: 11 },
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

const OP_CATEGORIES = {
    perception: { ops: ['detect_object_stereo', 'detect_field', 'analyze_scene', 'generate_point_cloud', 'detect_all_fields'], color: PALETTE.blue },
    motion: { ops: ['move_to_coordinate', 'adjust_end_effector_orientation', 'return_to_start_position', 'pick_object_at_coordinate', 'move_relative_to_object'], color: PALETTE.teal },
    grasp: { ops: ['grasp_object', 'place_object', 'receive_handoff'], color: PALETTE.orange },
    coordination: { ops: ['signal', 'wait_for_signal', 'wait', 'detect_other_robot', 'mirror_movement', 'stabilize_object'], color: PALETTE.purple },
    gripper: { ops: ['control_gripper', 'release_object'], color: PALETTE.skyblue },
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
    renderExclusionNote('noteE', sortedEntries, entries.map(d => d.benchmark_id), 'no per-step timing data');
    if (entries.length === 0) return;

    const keepOps = interestingOps(entries);
    const labels = entries.map(d => `B${d.benchmark_id}`);

    const catTotals = {};
    for (const [cat, meta] of Object.entries(OP_CATEGORIES)) {
        catTotals[cat] = entries.map(d => {
            let sum = 0;
            for (const op of meta.ops) {
                if (!keepOps.has(op)) continue;
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
                    bodyFont: { family: "'IBM Plex Mono', monospace", size: 11 },
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
    renderExclusionNote('noteF', sortedEntries, entries.map(d => d.benchmark_id), 'no per-step timing data');
    if (entries.length === 0) { container.innerHTML = '<p style="color:var(--text-dim);padding:1rem">No step data available.</p>'; return; }

    // Show every op that appears in any benchmark — fast atomic ops are still
    // timed and belong in the timing heatmap (not filtered via interestingOps).
    const allOps = [...new Set(entries.flatMap(d => Object.keys(d.op_stats)))]
        .sort();
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
            const textColor = t > 0.6 ? '#fff' : (isDark ? '#e6edf3' : '#333');
            const label = ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms.toFixed(0)}ms`;
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
    const pointLabels = entries.map(d => `B${d.benchmark_id}: ${d.benchmark_name || 'Unknown'}`);

    // Draw the benchmark name beside each point. Labels are placed to the right
    // (or left, near the right edge) and nudged down when they would overlap a
    // label already drawn, so dense clusters stay legible.
    const pointLabelPlugin = {
        id: 'complexityPointLabels',
        afterDatasetsDraw(chart) {
            const { ctx, chartArea } = chart;
            const meta = chart.getDatasetMeta(0);
            if (!meta || !meta.data.length) return;
            ctx.save();
            ctx.font = "9px 'IBM Plex Mono', monospace";
            ctx.textBaseline = 'middle';
            ctx.fillStyle = chartTextColor();
            const lineH = 12;
            const placed = [];
            // Place top-to-bottom so the nudge-down stacking is deterministic.
            const order = meta.data
                .map((pt, i) => ({ i, px: pt.x, py: pt.y }))
                .sort((a, b) => a.py - b.py);
            for (const o of order) {
                const text = pointLabels[o.i];
                const w = ctx.measureText(text).width;
                const toLeft = o.px > chartArea.right - w - 24;
                const lx = toLeft ? o.px - 9 : o.px + 9;
                let ly = o.py;
                const left0 = toLeft ? lx - w : lx;
                for (const p of placed) {
                    if (Math.abs(ly - p.y) < lineH && left0 < p.right && left0 + w > p.left) {
                        ly = p.y + lineH;
                    }
                }
                ctx.textAlign = toLeft ? 'right' : 'left';
                ctx.fillText(text, lx, ly);
                const left = toLeft ? lx - w : lx;
                placed.push({ left, right: left + w, y: ly });
            }
            ctx.restore();
        }
    };

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
                pointRadius: 6,
                pointHoverRadius: 8,
            }]
        },
        plugins: [pointLabelPlugin],
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 200 },
            layout: { padding: { top: 14, right: 80, left: 10, bottom: 4 } },
            plugins: {
                legend: { display: false },
                tooltip: {
                    ...tt, padding: 10, cornerRadius: 2,
                    titleFont: { family: "'IBM Plex Sans', sans-serif", size: 12, weight: '600' },
                    bodyFont: { family: "'IBM Plex Mono', monospace", size: 11 },
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
    renderExclusionNote('noteH', sortedEntries, entries.map(d => d.benchmark_id), 'no per-robot step data');

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
                    bodyFont: { family: "'IBM Plex Mono', monospace", size: 11 },
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

async function loadCompareView() {
    const loading = document.getElementById('compare-loading');
    const content = document.getElementById('compare-content');
    loading.style.display = 'flex';
    content.style.display = 'none';

    const filenames = [...compareSet];
    const runsData = [];

    try {
        await Promise.all(filenames.map(async (fn) => {
            if (!compareDataCache[fn]) {
                const resp = await fetch(`/api/benchmarks/${fn}`);
                const result = await resp.json();
                if (!result.success) throw new Error(result.error);
                compareDataCache[fn] = result.data;
            }
            runsData.push({ filename: fn, data: compareDataCache[fn] });
        }));

        // Sort by run timestamp (run_id is YYYYMMDDTHHmmss_xxx)
        runsData.sort((a, b) => a.data.run_id.localeCompare(b.data.run_id));

        loading.style.display = 'none';
        content.style.display = 'flex';
        renderCompareView(runsData);
    } catch (e) {
        loading.innerHTML = `
            <i class="fa-solid fa-triangle-exclamation" style="color:#e74c3c"></i>
            <span>Failed to load: ${e.message}</span>`;
    }
}

function renderCompareView(runsData) {
    renderCompareHeader(runsData);
    renderCompareKPIMatrix(runsData);  // Configuration rows now live in here
    renderCompareStepChart(runsData);
    renderCompareOpStats(runsData);
}

function renderCompareHeader(runsData) {
    const el = document.getElementById('compare-header');
    const bids = [...new Set(runsData.map(r => r.data.benchmark_id))];
    const names = [...new Set(runsData.map(r => r.data.benchmark_name || `B${r.data.benchmark_id}`))];
    const mixed = bids.length > 1;

    let html = '';
    if (mixed) {
        html += `<span class="compare-header-badge">MIXED</span>`;
        html += `<span class="compare-header-title">${names.join(' + ')}</span>`;
        html += `<span class="compare-mixed-warning"><i class="fa-solid fa-triangle-exclamation"></i> Mixed benchmark IDs — results may not be directly comparable</span>`;
    } else {
        html += `<span class="compare-header-badge">B${bids[0]}</span>`;
        html += `<span class="compare-header-title">${names[0]}</span>`;
    }
    const models = [...new Set(runsData.map(compareRunModel))];
    const modelSuffix = models.length > 1 ? ` · ${models.length} models` : (models[0] ? ` · ${escHtml(models[0])}` : '');
    html += `<span class="compare-header-count">${runsData.length} runs selected${modelSuffix}</span>`;
    el.innerHTML = html;
}

// Metrics shown in the compare KPI tables (shared by the per-model blocks and
// the cross-model summary). `higher` drives best/worst highlighting.
const COMPARE_METRICS = [
    { key: 'success_rate',       label: 'Success Rate',        fmt: v => (v * 100).toFixed(1) + '%',  higher: true  },
    { key: 'total_duration_ms',  label: 'Total Duration',       fmt: v => (v / 1000).toFixed(2) + 's', higher: false },
    { key: 'ops_ratio',          label: 'Ops (succ/total)',     fmt: (v, r) => `${r.data.ops_succeeded}/${r.data.ops_executed}`, avgFmt: v => (v * 100).toFixed(1) + '%', higher: true, val: r => r.data.ops_succeeded / Math.max(r.data.ops_executed, 1) },
    { key: 'avg_step_duration_ms', label: 'Avg Step Time',     fmt: v => v.toFixed(0) + 'ms',         higher: false },
    { key: 'slowest_step', label: 'Slowest Step',
      val: r => { const s = r.data.steps || []; return s.length ? Math.max(...s.map(x => x.duration_ms)) : null; },
      fmt: (v, r) => { const step = (r.data.steps || []).find(x => x.duration_ms === v); return step ? `${step.operation} <span class="compare-kpi-sub">${v.toFixed(0)}ms</span>` : '—'; },
      avgFmt: v => v.toFixed(0) + 'ms avg',
      higher: false },
    { key: 'hallucinated_ops',     label: 'Hallucinated Ops',     fmt: v => v, higher: false },
    { key: 'reflection_recoveries', label: 'Reflection Recoveries', fmt: v => v ?? 0, higher: true,
      val: r => r.data.reflection_recoveries ?? r.data.reflexion_recoveries ?? null },
    { key: 'negotiation_rounds',   label: 'Negotiation Rounds',   fmt: v => v ?? 0, higher: false },
];

// Resolve a run's model: the directory model (bN/<model>/file) wins, else the
// run's model field; normalise publisher/model -> model so it matches the plots.
function compareRunModel(run) {
    const parts = (run.filename || '').split('/');
    const raw = parts.length >= 3
        ? parts[parts.length - 2]
        : ((run.data && run.data.model) || '');
    return raw ? raw.split('/').pop() : 'unknown';
}

// One stable color per model in the compare set (runs of a model share it).
const COMPARE_PALETTE = [
    PALETTE.blue, PALETTE.orange, PALETTE.teal, PALETTE.vermillion,
    PALETTE.purple, PALETTE.skyblue, PALETTE.yellow,
];
function compareModelColorMap(runsData) {
    const map = new Map();
    let i = 0;
    runsData.forEach(r => {
        const model = compareRunModel(r);
        if (!map.has(model)) map.set(model, COMPARE_PALETTE[i++ % COMPARE_PALETTE.length]);
    });
    return map;
}

// Feature flags surfaced in the compare table's Configuration rows.
const COMPARE_FLAG_META = {
    use_rag: 'RAG',
    use_vgn: 'VGN',
    use_knowledge_graph: 'Knowledge Graph',
    use_ros_movement: 'ROS/MoveIt',
    reflection_enabled: 'Reflection',
    dry_run: 'Dry Run',
    use_negotiation: 'Negotiation',
};

// Config entries (flags + execution mode) whose value differs across the runs.
function compareDiffConfig(runsData) {
    const rows = [];
    Object.keys(COMPARE_FLAG_META).forEach(k => {
        const vals = runsData.map(r => r.data.feature_flags?.[k] ?? null);
        if ([...new Set(vals.filter(v => v !== null))].length > 1) {
            rows.push({ key: k, label: COMPARE_FLAG_META[k], isMode: false });
        }
    });
    const modes = runsData.map(r => r.data.execution_mode || null);
    if ([...new Set(modes.filter(v => v !== null))].length > 1) {
        rows.push({ key: 'execution_mode', label: 'Execution Mode', isMode: true });
    }
    return rows;
}

function compareConfigCell(run, cfg) {
    if (cfg.isMode) {
        const v = run.data.execution_mode || null;
        return v ? `<span class="run-mode-badge run-mode-badge--${v}">${String(v).toUpperCase()}</span>` : '—';
    }
    const v = run.data.feature_flags?.[cfg.key];
    if (v === undefined || v === null) return '<span class="flag-pill" style="opacity:0.4">—</span>';
    const on = !!v;
    return `<span class="flag-pill flag-pill--${on ? 'on' : 'off'}">${on ? 'ON' : 'OFF'}</span>`;
}

function compareMetricVals(metric, runs) {
    return runs.map(r => (metric.val ? metric.val(r) : (r.data[metric.key] ?? null)));
}

function compareMetricAvg(metric, runs) {
    const nums = compareMetricVals(metric, runs).filter(v => v !== null && !isNaN(v));
    if (!nums.length) return null;
    return nums.reduce((a, b) => a + b, 0) / nums.length;
}

function compareMetricDisplay(metric, avg) {
    if (avg === null) return '—';
    const fmtFn = metric.avgFmt ?? (metric.fmt.length > 1 ? null : metric.fmt);
    return fmtFn ? fmtFn(avg) : avg.toFixed(2);
}

// One KPI table (metrics × runs, + an Avg column) for a single model's runs.
// `diffConfig` (config entries that differ across the whole comparison) is
// appended as Configuration rows so the run config sits beside its metrics.
function buildKpiTable(runs, diffConfig = []) {
    const showAvg = runs.length > 1;
    const colHeaders = runs.map(r => {
        const rid = r.data.run_id || r.filename;
        const short = rid.slice(-6);
        let timeStr = '';
        const m = rid.match(/(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})/);
        if (m) timeStr = `${m[4]}:${m[5]}`;
        return { short, timeStr };
    });

    let html = `<table class="compare-kpi-table"><thead><tr><th>Metric</th>`;
    colHeaders.forEach((h, i) => {
        html += `<th>Run ${i + 1}<span class="compare-run-header">${h.timeStr || h.short}</span></th>`;
    });
    if (showAvg) html += `<th class="compare-kpi-avg-header">Avg</th>`;
    html += `</tr></thead><tbody>`;

    COMPARE_METRICS.forEach(metric => {
        const vals = compareMetricVals(metric, runs);
        const numericVals = vals.filter(v => v !== null && !isNaN(v));
        const allSame = numericVals.length > 1 && numericVals.every(v => v === numericVals[0]);
        const best = (!numericVals.length || allSame) ? null : (metric.higher ? Math.max(...numericVals) : Math.min(...numericVals));
        const worst = (!numericVals.length || allSame) ? null : (metric.higher ? Math.min(...numericVals) : Math.max(...numericVals));

        html += `<tr><td class="compare-kpi-label">${metric.label}</td>`;
        runs.forEach((r, i) => {
            const raw = vals[i];
            if (raw === null) { html += `<td>—</td>`; return; }
            const display = typeof metric.fmt === 'function'
                ? (metric.fmt.length > 1 ? metric.fmt(raw, r) : metric.fmt(raw))
                : raw;
            let cls = '';
            if (!allSame && numericVals.length > 1) {
                if (raw === best && raw !== worst) cls = 'compare-kpi-best';
                else if (raw === worst && raw !== best) cls = 'compare-kpi-worst';
            }
            html += `<td${cls ? ` class="${cls}"` : ''}>${display}</td>`;
        });
        if (showAvg) {
            html += `<td class="compare-kpi-avg">${compareMetricDisplay(metric, compareMetricAvg(metric, runs))}</td>`;
        }
        html += `</tr>`;
    });

    if (diffConfig.length) {
        const span = runs.length + 1 + (showAvg ? 1 : 0);
        html += `<tr class="compare-config-sep"><td colspan="${span}">Configuration</td></tr>`;
        diffConfig.forEach(cfg => {
            html += `<tr><td class="compare-kpi-label">${cfg.label}</td>`;
            runs.forEach(r => { html += `<td>${compareConfigCell(r, cfg)}</td>`; });
            if (showAvg) html += `<td class="compare-kpi-avg">—</td>`;
            html += `</tr>`;
        });
    }

    html += `</tbody></table>`;
    return html;
}

// Cross-model summary: each model's per-metric average, one model per row, with
// best/worst highlighting across models.
function buildModelSummary(modelGroups) {
    const entries = [...modelGroups.entries()].map(([model, runs]) => ({
        model, runs, avgs: COMPARE_METRICS.map(m => compareMetricAvg(m, runs)),
    }));
    const colExtremes = COMPARE_METRICS.map((m, ci) => {
        const nums = entries.map(e => e.avgs[ci]).filter(v => v !== null && !isNaN(v));
        if (nums.length < 2 || nums.every(v => v === nums[0])) return { best: null, worst: null };
        return {
            best: m.higher ? Math.max(...nums) : Math.min(...nums),
            worst: m.higher ? Math.min(...nums) : Math.max(...nums),
        };
    });

    let html = `<div class="compare-section-title">Model averages</div>`;
    html += `<table class="compare-kpi-table compare-summary-table"><thead><tr><th>Model</th>`;
    COMPARE_METRICS.forEach(m => { html += `<th>${m.label}</th>`; });
    html += `</tr></thead><tbody>`;
    entries.forEach(({ model, runs, avgs }) => {
        html += `<tr><td class="compare-kpi-label" title="${escHtml(model)}">${escHtml(model)}`
            + ` <span class="compare-kpi-sub">${runs.length} run${runs.length > 1 ? 's' : ''}</span></td>`;
        avgs.forEach((avg, ci) => {
            const { best, worst } = colExtremes[ci];
            let cls = '';
            if (avg !== null && best !== null) {
                if (avg === best && avg !== worst) cls = 'compare-kpi-best';
                else if (avg === worst && avg !== best) cls = 'compare-kpi-worst';
            }
            html += `<td${cls ? ` class="${cls}"` : ''}>${compareMetricDisplay(COMPARE_METRICS[ci], avg)}</td>`;
        });
        html += `</tr>`;
    });
    html += `</tbody></table>`;
    return html;
}

function renderCompareKPIMatrix(runsData) {
    const el = document.getElementById('compare-kpi-matrix');

    // Group runs by model, preserving first-seen order.
    const modelGroups = new Map();
    runsData.forEach(r => {
        const model = compareRunModel(r);
        if (!modelGroups.has(model)) modelGroups.set(model, []);
        modelGroups.get(model).push(r);
    });

    // Config entries that differ across the whole comparison, shown as rows in
    // each model block (replaces the separate Configuration Differences panel).
    const diffConfig = compareDiffConfig(runsData);

    let html = '';
    // Cross-model summary only adds value when more than one model is present.
    if (modelGroups.size > 1) html += buildModelSummary(modelGroups);

    modelGroups.forEach((runs, model) => {
        html += `<div class="compare-section-title compare-model-title">${escHtml(model)}`
            + ` <span class="compare-kpi-sub">${runs.length} run${runs.length > 1 ? 's' : ''}</span></div>`;
        html += buildKpiTable(runs, diffConfig);
    });

    el.innerHTML = html;
}

function renderCompareStepChart(runsData) {
    if (compareStepChartInstance) {
        compareStepChartInstance.destroy();
        compareStepChartInstance = null;
    }

    // Build union of op names in order of first appearance across all runs
    const opOrder = [];
    const opSeen = new Set();
    runsData.forEach(r => {
        (r.data.steps || []).forEach(s => {
            if (!opSeen.has(s.operation)) {
                opSeen.add(s.operation);
                opOrder.push(s.operation);
            }
        });
    });

    if (opOrder.length === 0) return;

    // One color per model so all runs of a model share it (not per-run colors).
    const modelColors = compareModelColorMap(runsData);

    // Group runs by model so each model is a single dataset (one legend entry).
    const modelGroups = new Map();
    runsData.forEach(r => {
        const model = compareRunModel(r);
        if (!modelGroups.has(model)) modelGroups.set(model, []);
        modelGroups.get(model).push(r);
    });

    const datasets = Array.from(modelGroups.entries()).map(([model, runs]) => {
        // Mean op duration across the model's runs (each run sums its own repeated ops first).
        const opTotals = {};
        const opCounts = {};
        runs.forEach(r => {
            const perRun = {};
            (r.data.steps || []).forEach(s => {
                perRun[s.operation] = (perRun[s.operation] || 0) + s.duration_ms;
            });
            Object.entries(perRun).forEach(([op, ms]) => {
                opTotals[op] = (opTotals[op] || 0) + ms;
                opCounts[op] = (opCounts[op] || 0) + 1;
            });
        });

        const color = modelColors.get(model);
        const label = runs.length > 1 ? `${model} (${runs.length} runs, avg)` : model;

        return {
            label,
            data: opOrder.map(op => opCounts[op] ? opTotals[op] / opCounts[op] : null),
            backgroundColor: color + 'cc',
            borderColor: color,
            borderWidth: 1.5,
            borderRadius: 2,
        };
    });

    const ctx = document.getElementById('compareStepChart').getContext('2d');
    const tt = tooltipDefaults();

    compareStepChartInstance = new Chart(ctx, {
        type: 'bar',
        data: { labels: opOrder, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
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
                    ...tt, padding: 10, cornerRadius: 2,
                    titleFont: { family: "'IBM Plex Sans', sans-serif", size: 12, weight: '600' },
                    bodyFont: { family: "'IBM Plex Mono', monospace", size: 11 },
                    callbacks: {
                        label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y != null ? ctx.parsed.y.toFixed(0) + 'ms' : '—'}`
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    border: { color: chartBorderColor() },
                    ticks: {
                        color: chartTextColor(),
                        font: { family: "'IBM Plex Mono', monospace", size: 9 },
                        maxRotation: 90,
                        minRotation: 0,
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: { color: chartGridColor() },
                    border: { color: chartBorderColor() },
                    ticks: {
                        color: chartTextColor(),
                        font: { family: "'IBM Plex Mono', monospace", size: 10 },
                        callback: v => v >= 1000 ? (v / 1000).toFixed(1) + 's' : v + 'ms',
                    },
                    title: {
                        display: true,
                        text: 'DURATION (ms, summed per op, avg across runs)',
                        color: chartTextColor(),
                        font: { family: "'IBM Plex Mono', monospace", size: 10 },
                    }
                }
            }
        }
    });
}

function renderCompareOpStats(runsData) {
    const el = document.getElementById('compare-op-stats');

    // Union of all ops across all runs
    const allOps = [...new Set(runsData.flatMap(r => Object.keys(r.data.per_op_stats || {})))].sort();
    if (allOps.length === 0) { el.style.display = 'none'; return; }
    el.style.display = '';

    // Per-run max avg_duration_ms per op for heat coloring
    const globalMax = {};
    allOps.forEach(op => {
        globalMax[op] = Math.max(...runsData.map(r => r.data.per_op_stats?.[op]?.avg_duration_ms ?? 0));
    });

    const isDark = isDarkMode();
    const runColors = [PALETTE.blue, PALETTE.orange, PALETTE.teal, PALETTE.vermillion, PALETTE.purple];

    let html = `<div class="compare-op-stats-header"><i class="fa-solid fa-chart-simple"></i> Per-Operation Timing</div>`;
    html += `<div class="table-wrapper"><table class="compare-op-stats-table"><thead><tr><th>Operation</th>`;

    runsData.forEach((r, i) => {
        const rid = r.data.run_id || r.filename;
        const m = rid.match(/(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})/);
        const label = m ? `Run ${i + 1} (${m[4]}:${m[5]})` : `Run ${i + 1}`;
        const color = runColors[i % runColors.length];
        html += `<th style="color:${color}">${label}</th>`;
    });
    html += `</tr></thead><tbody>`;

    allOps.forEach(op => {
        html += `<tr><td class="step-op">${op}</td>`;
        runsData.forEach(r => {
            const stat = r.data.per_op_stats?.[op];
            if (!stat) { html += `<td class="heatmap-cell heatmap-cell--empty">—</td>`; return; }
            const ms = stat.avg_duration_ms;
            const t = globalMax[op] > 0 ? ms / globalMax[op] : 0;
            const alpha = 0.1 + t * 0.75;
            const bg = isDark ? `rgba(230,159,0,${alpha})` : `rgba(180,100,0,${alpha})`;
            const textColor = t > 0.55 ? '#fff' : (isDark ? '#e6edf3' : '#333');
            const label = ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms.toFixed(0)}ms`;
            const failSub = (stat.fail_count ?? 0) > 0
                ? `<span class="compare-fail-sub">${stat.fail_count}✕</span>`
                : '';
            html += `<td style="background:${bg};color:${textColor}">${label}${failSub}</td>`;
        });
        html += `</tr>`;
    });

    html += `</tbody></table></div>`;
    el.innerHTML = html;
}

function exportHeatmapCsv() {
    if (!aggregateData) { alert('No data loaded.'); return; }
    const sorted = Object.values(aggregateData).sort((a, b) => a.benchmark_id - b.benchmark_id);
    const entries = sorted.filter(d => d.op_stats && Object.keys(d.op_stats).length > 0);
    if (!entries.length) { alert('No operation timing data available.'); return; }

    const allOps = [...new Set(entries.flatMap(d => Object.keys(d.op_stats)))].sort();

    const rows = [];
    rows.push(['Benchmark', ...allOps]);
    entries.forEach(d => {
        const row = [`B${d.benchmark_id}: ${d.benchmark_name}`];
        allOps.forEach(op => {
            const stat = d.op_stats[op];
            row.push(stat ? stat.mean_duration_ms.toFixed(2) : '');
        });
        rows.push(row);
    });

    const csv = rows.map(r => r.map(v => `"${String(v).replace(/"/g, '""')}"`).join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    triggerDownload(URL.createObjectURL(blob), 'operation-timing.csv');
}

function exportChartPng(chartKey, filename) {
    // Normalise key: strip leading 'chart' prefix so both 'chartA' and 'A' work
    const key = chartKey.replace(/^chart/i, '');
    const instance = analysisChartInstances[key];
    if (!instance) { alert('Chart not rendered yet.'); return; }

    const canvas = instance.canvas;

    if (thesisMode) {
        // Compose on a white background for print
        const offscreen = document.createElement('canvas');
        offscreen.width = canvas.width;
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
        ['B', 'ablation.png'],
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

function exportCompareChartPng() {
    if (!compareStepChartInstance) { alert('Chart not rendered yet.'); return; }
    const canvas = compareStepChartInstance.canvas;
    canvas.toBlob(blob => {
        triggerDownload(URL.createObjectURL(blob), 'compare-step-duration.png');
    }, 'image/png');
}

function downloadCompareJson() {
    if (Object.keys(compareDataCache).length === 0) { alert('No compare data loaded.'); return; }
    const runs = [...compareSet].map(fn => ({ filename: fn, data: compareDataCache[fn] })).filter(r => r.data);
    if (runs.length === 0) { alert('No compare data loaded.'); return; }
    const blob = new Blob([JSON.stringify(runs, null, 2)], { type: 'application/json' });
    const names = runs.map(r => `B${r.data.benchmark_id}`).join('-');
    triggerDownload(URL.createObjectURL(blob), `compare-${names}.json`);
}

// ── Global exports (required: file loaded as ES module, onclick= can't reach module scope) ──
window.exportChartPng         = exportChartPng;
window.exportAllCharts        = exportAllCharts;
window.downloadAggregateJson  = downloadAggregateJson;
window.exportCompareChartPng  = exportCompareChartPng;
window.downloadCompareJson    = downloadCompareJson;
window.exportHeatmapCsv       = exportHeatmapCsv;
