#!/usr/bin/env python3
"""Generate the thesis benchmark figures from benchmark_results/.

Writes images 01-10 into Thesis/images/. The model name comes from the
directory path (benchmark_results/bN/<model>/...), falling back to the JSON
"model" field for the flat ablation layout.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "benchmark_results"
PLOTS_DIR = Path(__file__).parent.parent.parent / "Thesis" / "images"

SINGLE_ROBOT_COLOR = "#4C72B0"
DUAL_ROBOT_COLOR = "#DD8452"
DUAL_ROBOT_IDS = {6, 7}
ABLATION_IDS = {12, 13, 14, 15, 16}
MAIN_MAX_ID = 11  # b1-b11 are the model x task capability benchmarks
# Default backend. The single-model figures (ablation, timeline, AutoRT) use it
# so they match the prose; the per-model comparison lives in figures 01-04.
MAIN_MODEL = "magistral-small-2509"
ABLATION_B11_MODEL = MAIN_MODEL

# Shared font sizes so every figure matches.
FS_TITLE = 16
FS_SUPTITLE = 18
FS_AXIS = 13
FS_TICK = 12
FS_LEGEND = 11
FS_ANNOT = 12
FS_CELL = 9  # dense heatmap cells

BAR_EDGE = "white"
BAR_EDGE_W = 0.5
CAPSIZE = 3
ERR_KW = {"elinewidth": 1.4, "ecolor": "#333333"}
REF_LINE = {"color": "gray", "linestyle": "--", "linewidth": 0.8, "alpha": 0.5}


def _despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _apply_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.titlesize": FS_TITLE,
            "axes.titleweight": "bold",
            "axes.labelsize": FS_AXIS,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "legend.fontsize": FS_LEGEND,
            "legend.framealpha": 0.9,
            "figure.titlesize": FS_SUPTITLE,
            "figure.titleweight": "bold",
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# Known models get a fixed colour; others cycle through the fallback palette.
MODEL_COLORS = {
    "qwen3-vl-30b": "#4C72B0",
    "qwen3-vl-8b": "#55A868",
    "magistral-small-2509": "#DD8452",
    "ministral-3-14b-reasoning": "#8172B2",
    "gemma-4-e4b": "#C44E52",
    "gemma-4-e2b": "#937860",
}
_FALLBACK_COLORS = ["#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]

OP_COLORS = {
    "detect_object_stereo": "#4C72B0",
    "detect_field": "#64B5CD",
    "detect_all_fields": "#4C9BD4",
    "generate_point_cloud": "#3A7CA5",
    "analyze_scene": "#2E5E8A",
    "move_to_coordinate": "#55A868",
    "move_relative_to_object": "#6FBF73",
    "pick_object_at_coordinate": "#3E7D4E",
    "return_to_start_position": "#C44E52",
    "adjust_end_effector_orientation": "#937860",
    "grasp_object": "#DD8452",
    "place_object": "#E8A87C",
    "receive_handoff": "#8172B2",
    "stabilize_object": "#A088C9",
    "signal": "#DA8BC3",
    "wait_for_signal": "#8C8C8C",
    "control_gripper": "#CCB974",
    "release_object": "#B8A55E",
    "other": "#BBBBBB",
}


def model_from_path(rel_path: Path) -> str:
    # b1/qwen3-vl-8b/foo.json -> qwen3-vl-8b; flat layout -> "default".
    parts = rel_path.parts
    if len(parts) >= 3:
        return parts[1]
    return "default"


def load_results() -> dict[int, list[dict]]:
    # Tag each run with _model. Prefer the directory name; the JSON "model"
    # field is sometimes wrong (the embedding id leaks in) so only trust it for
    # the flat layout.
    groups: dict[int, list[dict]] = defaultdict(list)
    for path in sorted(RESULTS_DIR.rglob("benchmark*.json")):
        with open(path) as f:
            data = json.load(f)
        path_model = model_from_path(path.relative_to(RESULTS_DIR))
        raw_model = (
            path_model if path_model != "default" else (data.get("model") or "default")
        )
        data["_model"] = _normalize_model(raw_model)
        groups[data["benchmark_id"]].append(data)
    return dict(sorted(groups.items()))


def op_color(op: str) -> str:
    return OP_COLORS.get(op, OP_COLORS["other"])


# Zero-duration ops; drawn as labelled event markers on the timeline, not bars.
EVENT_OPS = {"signal", "wait_for_signal", "release_object"}


# The nomic embedding model is the RAG retriever, not a benchmark subject.
EXCLUDE_MODELS = {
    "text-embedding-nomic-embed-text-v1.5",
}


def _normalize_model(name: str) -> str:
    # Drop a publisher prefix (google/gemma-4-e2b) so it matches the dir name.
    return name.split("/")[-1] if name else name


def models_in(groups: dict[int, list[dict]], bids: list[int]) -> list[str]:
    seen = {r["_model"] for bid in bids for r in groups.get(bid, [])} - EXCLUDE_MODELS
    known = [m for m in MODEL_COLORS if m in seen]
    extra = sorted(seen - set(MODEL_COLORS))
    return known + extra


def model_color(model: str, ordered_models: list[str]) -> str:
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    idx = [m for m in ordered_models if m not in MODEL_COLORS].index(model)
    return _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)]


def _main_bids(groups: dict[int, list[dict]]) -> list[int]:
    # B11 is the RAG ablation, not a capability task.
    return [b for b in sorted(groups) if b <= MAIN_MAX_ID and b != 11]


OP_GAP_EPS = 0.02  # task/op rates within this are treated as equal

# Each bar is split into three bands: [0, task] task complete (solid colour),
# [task, op] ops ok but task incomplete (hatched), [op, 1] op failure (grey).
GAP_ALPHA = 0.32
GAP_HATCH = "////"
BAND_FAIL_COLOR = "0.88"
BAND_LEGEND = [
    (mpatches.Patch(facecolor="0.30", edgecolor="white"), "task complete"),
    (
        mpatches.Patch(facecolor="0.62", edgecolor="0.35", hatch=GAP_HATCH),
        "ops ok, task incomplete",
    ),
    (mpatches.Patch(facecolor=BAND_FAIL_COLOR, edgecolor="0.7"), "operation failure"),
]


def draw_outcome_bands(ax, lo, span, task, op, color, *, horizontal=False):
    # Stack the three outcome bands at position lo (width span). horizontal=True
    # runs the stack along x (leaderboard), otherwise up y.
    if np.isnan(task) or np.isnan(op):
        return
    segs = [
        (0.0, task, color, None, 1.0),
        (task, op - task, color, GAP_HATCH, GAP_ALPHA),
        (op, 1.0 - op, BAND_FAIL_COLOR, None, 1.0),
    ]
    for base, length, fc, hatch, alpha in segs:
        if length <= 0.001:
            continue
        kw = dict(
            facecolor=fc,
            alpha=alpha,
            hatch=hatch,
            edgecolor=BAR_EDGE if hatch is None else "0.35",
            linewidth=BAR_EDGE_W,
        )
        if horizontal:
            ax.barh(lo, length, left=base, height=span, **kw)
        else:
            ax.bar(lo, length, bottom=base, width=span, **kw)


def cell_rates(runs: list[dict]) -> tuple[float, float]:
    # task rate = fraction of runs with success=True; op rate = mean per-run
    # success_rate. They diverge on truncated plans (ops pass, task fails).
    if not runs:
        return (float("nan"), float("nan"))
    task = float(np.mean([1.0 if r.get("success") else 0.0 for r in runs]))
    op = float(np.mean([r.get("success_rate", 0.0) for r in runs]))
    return (task, op)


# --------------------------------------------------------------------------- #
# Model x task capability figures (b1-b11)
# --------------------------------------------------------------------------- #
def plot_success_rate_by_model(groups: dict[int, list[dict]]) -> None:
    bids = _main_bids(groups)
    if not bids:
        return
    ordered = models_in(groups, bids)

    fig, ax = plt.subplots(
        figsize=(max(12, 1.2 * len(bids)), 6.5), layout="constrained"
    )
    x = np.arange(len(bids))
    n = len(ordered)
    width = 0.8 / max(n, 1)

    # One full-height outcome stack per model; hue = model, band = outcome.
    model_handles = []
    for i, model in enumerate(ordered):
        color = model_color(model, ordered)
        offset = (i - (n - 1) / 2) * width
        for j, bid in enumerate(bids):
            task, op = cell_rates([r for r in groups[bid] if r["_model"] == model])
            draw_outcome_bands(ax, x[j] + offset, width, task, op, color)
        model_handles.append((mpatches.Patch(facecolor=color), model))

    labels = [f"B{bid}\n{groups[bid][0]['benchmark_name']}" for bid in bids]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS_TICK, rotation=35, ha="right")
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of runs", fontsize=FS_AXIS)
    ax.set_title("Task Outcome by Model", fontsize=FS_TITLE, fontweight="bold")

    handles = [h for h, _ in model_handles] + [h for h, _ in BAND_LEGEND]
    leg_labels = [m for _, m in model_handles] + [t for _, t in BAND_LEGEND]
    fig.legend(
        handles,
        leg_labels,
        fontsize=FS_LEGEND,
        ncol=5,
        loc="outside lower center",
        framealpha=0.9,
    )
    _despine(ax)

    fig.savefig(
        PLOTS_DIR / "01_success_rate_by_model.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print("  [1] success rate by model")


def plot_model_task_heatmap(groups: dict[int, list[dict]]) -> None:
    bids = _main_bids(groups)
    if not bids:
        return
    ordered = models_in(groups, bids)

    # Colour by task rate; keep the op rate to annotate where they diverge.
    matrix = np.full((len(ordered), len(bids)), np.nan)
    op_matrix = np.full((len(ordered), len(bids)), np.nan)
    for i, model in enumerate(ordered):
        for j, bid in enumerate(bids):
            runs = [r for r in groups[bid] if r["_model"] == model]
            if runs:
                task, op = cell_rates(runs)
                matrix[i, j] = task
                op_matrix[i, j] = op

    fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(bids)), 0.7 * len(ordered) + 2))
    cmap = plt.get_cmap("RdYlGn")
    cmap.set_bad("0.9")
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, label="Mean Task Success Rate", shrink=0.8)
    cbar.ax.yaxis.label.set_fontsize(FS_AXIS)
    cbar.ax.tick_params(labelsize=FS_TICK)

    ax.set_xticks(range(len(bids)))
    ax.set_xticklabels(
        [f"B{bid}  {groups[bid][0]['benchmark_name']}" for bid in bids],
        fontsize=FS_TICK,
        rotation=35,
        ha="right",
    )
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered, fontsize=FS_TICK)

    for i in range(len(ordered)):
        for j in range(len(bids)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            op = op_matrix[i, j]
            txt_color = "black" if 0.3 < val < 0.8 else "white"
            gap = (not np.isnan(op)) and (op - val > 0.05)
            # Task rate large, op rate as a small subscript when they diverge.
            ax.text(
                j,
                i - (0.12 if gap else 0.0),
                f"{val:.0%}",
                ha="center",
                va="center",
                fontsize=FS_CELL,
                color=txt_color,
            )
            if gap:
                ax.text(
                    j,
                    i + 0.24,
                    f"op {op:.0%}",
                    ha="center",
                    va="center",
                    fontsize=FS_CELL - 2,
                    color=txt_color,
                    alpha=0.85,
                )

    ax.set_title("Model x Task Success Matrix", fontsize=FS_TITLE, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_model_task_heatmap.png", dpi=150)
    plt.close(fig)
    print("  [2] model x task heatmap")


def plot_model_leaderboard(groups: dict[int, list[dict]]) -> None:
    bids = _main_bids(groups)
    if not bids:
        return
    ordered = models_in(groups, bids)

    # Average the per-task means so every task weighs equally; rank by task rate.
    task_scores, op_scores = {}, {}
    for model in ordered:
        t_means, o_means = [], []
        for bid in bids:
            runs = [r for r in groups[bid] if r["_model"] == model]
            if runs:
                task, op = cell_rates(runs)
                t_means.append(task)
                o_means.append(op)
        task_scores[model] = np.mean(t_means) if t_means else 0.0
        op_scores[model] = np.mean(o_means) if o_means else 0.0

    ranked = sorted(task_scores.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(9.5, max(3, 0.7 * len(ranked) + 1.2)))
    y = np.arange(len(ranked))
    for yi, (m, v) in zip(y, ranked):
        draw_outcome_bands(
            ax, yi, 0.62, v, op_scores[m], model_color(m, ordered), horizontal=True
        )
    ax.set_yticks(y)
    ax.set_yticklabels([m for m, _ in ranked], fontsize=FS_TICK)
    ax.set_xlim(0, 1.32)
    ax.set_xlabel("Fraction of runs (task-averaged, B1-B10)", fontsize=FS_AXIS)
    ax.set_title(
        "Model Leaderboard - Task Outcome", fontsize=FS_TITLE, fontweight="bold"
    )
    for yi, (m, v) in zip(y, ranked):
        op = op_scores[m]
        annot = f"{v:.0%}" + (f"   op {op:.0%}" if op - v > OP_GAP_EPS else "")
        ax.text(1.01, yi, annot, va="center", fontsize=FS_ANNOT, fontweight="bold")
    ax.legend(
        [h for h, _ in BAND_LEGEND],
        [t for _, t in BAND_LEGEND],
        fontsize=FS_LEGEND,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        framealpha=0.9,
        ncol=3,
    )
    _despine(ax)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_model_leaderboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [3] model leaderboard")


def plot_duration_by_model(groups: dict[int, list[dict]]) -> None:
    bids = _main_bids(groups)
    if not bids:
        return
    ordered = models_in(groups, bids)

    matrix = np.full((len(ordered), len(bids)), np.nan)
    for i, model in enumerate(ordered):
        for j, bid in enumerate(bids):
            durs = [
                r["total_duration_ms"] / 1000
                for r in groups[bid]
                if r["_model"] == model
            ]
            if durs:
                matrix[i, j] = np.mean(durs)

    fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(bids)), 0.7 * len(ordered) + 2))
    cmap = plt.get_cmap("YlOrRd")
    cmap.set_bad("0.9")
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, label="Mean Total Duration (s)", shrink=0.8)
    cbar.ax.yaxis.label.set_fontsize(FS_AXIS)
    cbar.ax.tick_params(labelsize=FS_TICK)

    ax.set_xticks(range(len(bids)))
    ax.set_xticklabels([f"B{bid}" for bid in bids], fontsize=FS_TICK)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered, fontsize=FS_TICK)

    vmax = np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 1.0
    for i in range(len(ordered)):
        for j in range(len(bids)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.0f}",
                    ha="center",
                    va="center",
                    fontsize=FS_CELL,
                    color="white" if val > 0.55 * vmax else "black",
                )

    ax.set_title("Model x Task Mean Duration", fontsize=FS_TITLE, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_duration_by_model.png", dpi=150)
    plt.close(fig)
    print("  [4] duration by model")


# --------------------------------------------------------------------------- #
# Operation-level figures (pooled across models)
# --------------------------------------------------------------------------- #
def plot_op_latency(groups: dict[int, list[dict]]) -> None:
    # All steps decide op inclusion, but only successful-step durations are
    # plotted (a failed op aborts in ~0ms and would skew the distribution).
    op_bid_durations: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    op_bid_ok: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for bid, runs in groups.items():
        for run in runs:
            for step in run["steps"]:
                op = step["operation"]
                op_bid_durations[op][bid].append(step["duration_ms"])
                if step.get("success"):
                    op_bid_ok[op][bid].append(step["duration_ms"])

    # Ops appearing in 2+ benchmarks whose *successful* execution takes real
    # time (>100ms). Filtering on the plotted (successful) data, not all steps,
    # drops synchronization primitives (signal/wait_for_signal) whose successful
    # duration is sub-millisecond bookkeeping rather than motor execution.
    interesting_ops = sorted(
        op
        for op, bid_data in op_bid_ok.items()
        if len(bid_data) >= 2 and max(max(v) for v in bid_data.values()) > 100
    )
    if not interesting_ops:
        return

    # Square-ish grid; durations shown in seconds.
    ncols = math.ceil(math.sqrt(len(interesting_ops)))
    nrows = math.ceil(len(interesting_ops) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 4.2 * nrows),
        sharey=False,
    )
    axes = axes.flatten() if nrows * ncols > 1 else [axes]

    for ax, op in zip(axes, interesting_ops):
        bids_present = [bid for bid in sorted(op_bid_ok[op]) if op_bid_ok[op][bid]]
        if not bids_present:
            ax.set_visible(False)
            continue
        bp = ax.boxplot(
            [[v / 1000.0 for v in op_bid_ok[op][bid]] for bid in bids_present],
            tick_labels=[f"B{bid}" for bid in bids_present],
            patch_artist=True,
            showfliers=False,  # outliers hidden so boxes fill the panel
            medianprops=dict(color="white", linewidth=2),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(op_color(op))
            patch.set_alpha(0.85)
        ax.set_title(op.replace("_", " "), fontsize=FS_TITLE - 1, fontweight="bold")
        ax.set_ylabel("Duration (s)", fontsize=FS_AXIS)
        ax.set_yscale("log")  # durations span ~0.1s to >100s
        ax.grid(axis="y", which="both", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        _despine(ax)
        ax.tick_params(axis="both", labelsize=FS_TICK)
        if len(bids_present) > 6:  # rotate so dense benchmark ticks don't collide
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for ax in axes[len(interesting_ops) :]:
        ax.set_visible(False)

    fig.suptitle(
        "Per-Operation Step Duration by Benchmark\n"
        "(all models pooled; successful steps only, log scale, outliers hidden)",
        fontsize=FS_SUPTITLE,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.5, w_pad=2.0)
    fig.savefig(PLOTS_DIR / "05_op_latency.png", dpi=150)
    plt.close(fig)
    print("  [5] op latency")


def plot_reliability_heatmap(groups: dict[int, list[dict]]) -> None:
    op_bid_success: dict[str, dict[int, list[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for bid, runs in groups.items():
        for run in runs:
            for step in run["steps"]:
                op_bid_success[step["operation"]][bid].append(step["success"])

    all_ops = sorted(op_bid_success.keys())
    bids = sorted(groups)
    if not all_ops:
        return

    matrix = np.full((len(all_ops), len(bids)), np.nan)
    for i, op in enumerate(all_ops):
        for j, bid in enumerate(bids):
            if bid in op_bid_success[op]:
                matrix[i, j] = np.mean(op_bid_success[op][bid])

    row_means = np.nanmean(matrix, axis=1)
    sort_idx = np.argsort(row_means)
    matrix = matrix[sort_idx]
    sorted_ops = [all_ops[i] for i in sort_idx]

    fig, ax = plt.subplots(
        figsize=(max(12, 0.85 * len(bids)), max(4, 0.5 * len(sorted_ops) + 2))
    )
    cmap = plt.get_cmap("RdYlGn")
    cmap.set_bad("0.9")
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, label="Success Rate", shrink=0.8)
    cbar.ax.yaxis.label.set_fontsize(FS_AXIS)
    cbar.ax.tick_params(labelsize=FS_TICK)

    ax.set_xticks(range(len(bids)))
    ax.set_xticklabels([f"B{bid}" for bid in bids], fontsize=FS_TICK)
    ax.set_yticks(range(len(sorted_ops)))
    ax.set_yticklabels([op.replace("_", " ") for op in sorted_ops], fontsize=FS_TICK)

    for i in range(len(sorted_ops)):
        for j in range(len(bids)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.0%}",
                    ha="center",
                    va="center",
                    fontsize=FS_CELL,
                    color="black" if 0.3 < val < 0.8 else "white",
                )

    ax.set_title(
        "Operation Reliability by Benchmark", fontsize=FS_TITLE, fontweight="bold"
    )
    ax.set_xlabel("Benchmark", fontsize=FS_AXIS)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "06_reliability_heatmap.png", dpi=150)
    plt.close(fig)
    print("  [6] reliability heatmap")


def plot_coordination_timeline(groups: dict[int, list[dict]]) -> None:
    dual_bids = [bid for bid in sorted(groups) if bid in DUAL_ROBOT_IDS]
    if not dual_bids:
        return

    fig, axes = plt.subplots(
        len(dual_bids), 1, figsize=(14, 4 * len(dual_bids)), layout="constrained"
    )
    if len(dual_bids) == 1:
        axes = [axes]

    robot_y = {"Robot1": 1, "Robot2": 0}
    robot_labels = ["Robot2", "Robot1"]
    legend_handles: dict[str, mpatches.Patch] = {}

    for ax, bid in zip(axes, dual_bids):
        # Main model, preferring a successful run so B6 shows a clean handoff.
        candidates = [
            r for r in groups[bid] if r["_model"] == MAIN_MODEL and r.get("steps")
        ]
        if not candidates:  # fall back to any run with steps
            candidates = [r for r in groups[bid] if r.get("steps")] or groups[bid]
        run = next((r for r in candidates if r.get("success")), candidates[0])
        name = run["benchmark_name"]

        seen_groups: set[int] = set()
        group_steps: dict[int | None, list[dict]] = defaultdict(list)
        for step in run["steps"]:
            group_steps[step["parallel_group_id"]].append(step)

        cursor: dict[str, float] = defaultdict(float)
        # Collected here and laid out after the bars (see below).
        events_here: list[tuple[float, float, str]] = []  # (start, lane_y, op)

        for step in run["steps"]:
            robot = step.get("robot_id") or "Robot1"
            dur = step["duration_ms"] / 1000
            op = step["operation"]
            color = op_color(op)

            gid = step["parallel_group_id"]
            if gid is not None and gid not in seen_groups:
                seen_groups.add(gid)
                robots_in_group = {
                    s["robot_id"] for s in group_steps[gid] if s.get("robot_id")
                }
                group_start = max(cursor[r] for r in (robots_in_group or ["Robot1"]))
                for s in group_steps[gid]:
                    cursor[s.get("robot_id") or "Robot1"] = group_start

            start = cursor[robot]
            y = robot_y.get(robot, 0)
            if op in EVENT_OPS:
                events_here.append((start, y, op))
            else:
                rect = mpatches.FancyBboxPatch(
                    (start, y - 0.35),
                    dur,
                    0.7,
                    boxstyle="round,pad=0.02",
                    facecolor=color,
                    edgecolor=BAR_EDGE,
                    linewidth=BAR_EDGE_W,
                    alpha=0.85,
                )
                ax.add_patch(rect)
                if op not in legend_handles:
                    legend_handles[op] = mpatches.Patch(
                        color=color, label=op.replace("_", " ")
                    )
            cursor[robot] += dur

        # Group near-simultaneous events onto one vertical guide, names stacked
        # above-right on short connectors so no leader lines overlap.
        total = max(cursor.values(), default=1.0)
        dx = total * 0.012
        if events_here:
            events_here.sort(key=lambda e: e[0])
            tol = max(total * 0.02, 1.0)
            clusters: list[list[tuple[float, float, str]]] = []
            for ev in events_here:
                if clusters and ev[0] - clusters[-1][0][0] <= tol:
                    clusters[-1].append(ev)
                else:
                    clusters.append([ev])
            base, step = 1.80, 0.52
            for grp in clusters:
                gx = grp[0][0]
                lo = min(lane for _, lane, _ in grp) + 0.36
                top = base + (len(grp) - 1) * step
                ax.plot([gx, gx], [lo, top], color="0.45", lw=1.1, zorder=6)
                for i, (_, _, ev_op) in enumerate(grp):
                    ly = base + i * step
                    ax.plot([gx, gx + dx], [ly, ly], color="0.45", lw=1.1, zorder=6)
                    ax.text(
                        gx + dx * 1.4,
                        ly,
                        ev_op.replace("_", " "),
                        ha="left",
                        va="center",
                        fontsize=FS_TICK,
                        fontweight="bold",
                        color="0.15",
                        zorder=7,
                    )

        ax.set_xlim(0, max(cursor.values(), default=1.0) * 1.02)
        ax.set_ylim(-0.7, 3.3)  # headroom for the event-label band
        ax.set_yticks([0, 1])
        ax.set_yticklabels(robot_labels, fontsize=FS_TICK + 4)
        ax.tick_params(axis="x", labelsize=FS_TICK + 4)
        ax.set_xlabel("Time (s)", fontsize=FS_AXIS + 4)
        title_suffix = " [FAILED]" if not run["success"] else ""
        ax.set_title(
            f"B{bid}: {name}{title_suffix}", fontsize=FS_TITLE + 3, fontweight="bold"
        )
        _despine(ax)
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    fig.legend(
        handles=list(legend_handles.values()),
        loc="outside lower center",
        fontsize=FS_LEGEND + 5,
        framealpha=0.9,
        ncol=4,
    )
    fig.suptitle(
        "Dual-Robot Coordination Timeline", fontsize=FS_SUPTITLE + 2, fontweight="bold"
    )
    fig.savefig(
        PLOTS_DIR / "07_coordination_timeline.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print("  [7] coordination timeline")


# --------------------------------------------------------------------------- #
# Ablation figure (b12-b16)
# --------------------------------------------------------------------------- #
def plot_ablation(groups: dict[int, list[dict]]) -> None:
    # Success rate by condition; conditions come from each run's ablation block,
    # and the enabled-vs-disabled delta is annotated where both exist.
    abl_bids = [bid for bid in sorted(groups) if bid in ABLATION_IDS or bid == 11]
    if not abl_bids:
        return

    # bid -> condition -> list of success rates
    per_bid: dict[int, dict[str, list[float]]] = {}
    missing_baseline: list[int] = []
    for bid in abl_bids:
        cond_sr: dict[str, list[float]] = defaultdict(list)
        has_baseline = False
        for r in groups[bid]:
            # Main model only, so the deltas match the headline numbers.
            if r["_model"] != MAIN_MODEL:
                continue
            ab = r.get("ablation")
            if ab:
                cond_sr[ab["condition"]].append(ab.get("success_rate", 0.0))
            base = r.get("ablation_baseline")
            if base:
                cond_sr[base["condition"]].append(base.get("success_rate", 0.0))
                has_baseline = True
        if not cond_sr:  # fall back to run-level success_rate
            cond_sr["run"] = [r["success_rate"] for r in groups[bid]]
        per_bid[bid] = cond_sr
        if not has_baseline and len(cond_sr) < 2:
            missing_baseline.append(bid)

    # Stable condition ordering / colors.
    cond_color = {
        "enabled": "#55A868",
        "ros": "#64B5CD",
        "unity": "#DD8452",
        "disabled": "#C44E52",
        "run": "#8C8C8C",
    }
    all_conds = []
    for cond_sr in per_bid.values():
        for c in cond_sr:
            if c not in all_conds:
                all_conds.append(c)

    fig, ax = plt.subplots(
        figsize=(max(9, 1.6 * len(abl_bids)), 5.5), layout="constrained"
    )
    x = np.arange(len(abl_bids))
    n = len(all_conds)
    width = 0.8 / max(n, 1)

    for i, cond in enumerate(all_conds):
        means = [
            np.mean(per_bid[bid][cond]) if cond in per_bid[bid] else np.nan
            for bid in abl_bids
        ]
        stds = [
            (
                np.std(per_bid[bid][cond], ddof=1)
                if cond in per_bid[bid] and len(per_bid[bid][cond]) > 1
                else 0.0
            )
            for bid in abl_bids
        ]
        offset = (i - (n - 1) / 2) * width
        means_arr = np.nan_to_num(np.array(means, dtype=float))
        stds_arr = np.array(stds, dtype=float)
        # Success rate is bounded [0, 1]; clip the upper whisker so it never
        # implies impossible >1.0 values (asymmetric error bars).
        yerr = np.vstack([stds_arr, np.minimum(stds_arr, 1.0 - means_arr)])
        ax.bar(
            x + offset,
            means_arr,
            width=width,
            yerr=yerr,
            capsize=CAPSIZE,
            error_kw=ERR_KW,
            color=cond_color.get(cond, "#BBBBBB"),
            edgecolor=BAR_EDGE,
            linewidth=BAR_EDGE_W,
            label=cond,
        )

    # Annotate the treatment-vs-control delta (on/off, or ros/unity for B16).
    for j, bid in enumerate(abl_bids):
        cs = per_bid[bid]
        treat = next((c for c in ("enabled", "ros") if c in cs), None)
        control = (
            "disabled" if "disabled" in cs else ("unity" if "unity" in cs else None)
        )
        if treat and control and treat != control:
            delta = np.mean(cs[treat]) - np.mean(cs[control])
            top = max(
                np.mean(cs[treat])
                + (np.std(cs[treat], ddof=1) if len(cs[treat]) > 1 else 0.0),
                np.mean(cs[control])
                + (np.std(cs[control], ddof=1) if len(cs[control]) > 1 else 0.0),
            )
            top = min(top, 1.0)  # whiskers are clipped at the 1.0 ceiling
            ax.text(
                j,
                float(min(top + 0.03, 1.04)),
                f"Δ {delta:+.0%}",
                ha="center",
                fontsize=FS_ANNOT,
                fontweight="bold",
                color="#333333",
            )

    labels = [f"B{bid}\n{groups[bid][0]['benchmark_name']}" for bid in abl_bids]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS_TICK, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Success Rate", fontsize=FS_AXIS)
    title = "Ablation: Success Rate by Condition"
    if missing_baseline:
        miss = ", ".join(f"B{b}" for b in missing_baseline)
        title += f"\n(no paired baseline in data for {miss} - re-run paired ablation)"
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold")
    ax.axhline(1.0, **REF_LINE)
    fig.legend(
        *ax.get_legend_handles_labels(),
        fontsize=FS_LEGEND,
        loc="outside lower center",
        ncol=4,
        framealpha=0.9,
    )
    _despine(ax)

    fig.savefig(PLOTS_DIR / "08_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [8] ablation")


# --------------------------------------------------------------------------- #
# AutoRT safety-gate + generation figure (b17)
# --------------------------------------------------------------------------- #
AUTORT_ID = 17


def plot_autort_safety(groups: dict[int, list[dict]]) -> None:
    # B17 safety-gate rates (left group) and generation quality (right group),
    # main model only so it matches the prose; per-model data is in the appendix.
    runs = [r for r in groups.get(AUTORT_ID, []) if r["_model"] == MAIN_MODEL]
    if not runs:
        runs = groups.get(AUTORT_ID, [])
    if not runs:
        return

    gates = [r.get("per_op_stats", {}).get("safety_gate", {}) for r in runs]
    gens = [r.get("per_op_stats", {}).get("generation", {}) for r in runs]

    def _ms(dicts: list[dict], key: str) -> tuple[float, float]:
        vals = [d.get(key, 0.0) for d in dicts]
        if not vals:
            return (0.0, 0.0)
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        return (float(np.mean(vals)), std)

    gate_keys = ["accuracy", "false_accept_rate", "false_reject_rate"]
    gen_keys = ["slot_success_rate", "first_attempt_rate"]
    # All five metrics on one axis, with an extra gap between the two groups.
    labels = [
        "accuracy",
        "false-accept",
        "false-reject",
        "slot success\n(pre-dedup)",
        "first-attempt\nvalid",
    ]
    colors = ["#937860", "#C44E52", "#CCB974", "#4C72B0", "#55A868"]
    means_stds = [_ms(gates, k) for k in gate_keys] + [_ms(gens, k) for k in gen_keys]
    means = [m for m, _ in means_stds]
    stds = [s for _, s in means_stds]

    # Unit spacing within each group, +0.6 gap between the two groups.
    x = np.array([0, 1, 2, 3.6, 4.6])
    BAR_W = 0.8  # b8-style: bars nearly fill their slot, so they sit close

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(
        x,
        means,
        BAR_W,
        yerr=stds,
        color=colors,
        edgecolor=BAR_EDGE,
        linewidth=BAR_EDGE_W,
        capsize=CAPSIZE,
        error_kw=ERR_KW,
    )
    for xi, m, s in zip(x, means, stds):
        ax.text(
            xi,
            m + s + 0.03,
            f"{m:.2f}",
            ha="center",
            va="bottom",
            fontsize=FS_ANNOT,
            fontweight="bold",
        )

    # Group labels under each cluster.
    ax.text(
        1.0,
        -0.22,
        "safety gate",
        ha="center",
        va="top",
        fontsize=FS_AXIS,
        fontweight="bold",
    )
    ax.text(
        4.1,
        -0.22,
        "generation",
        ha="center",
        va="top",
        fontsize=FS_AXIS,
        fontweight="bold",
    )

    ax.set_title(
        "AutoRT Safety & Generation (B17)", fontsize=FS_TITLE, fontweight="bold"
    )
    ax.set_ylabel("Rate", fontsize=FS_AXIS)
    ax.set_ylim(0, 1.12)
    ax.set_xlim(-0.6, 5.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS_TICK)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    _despine(ax)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "10_autort_safety.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [10] autort safety")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    groups = load_results()
    print(
        f"Loaded {sum(len(v) for v in groups.values())} runs "
        f"across {len(groups)} benchmarks"
    )
    print("Generating plots...")

    _apply_rc()

    plot_success_rate_by_model(groups)
    plot_model_task_heatmap(groups)
    plot_model_leaderboard(groups)
    plot_duration_by_model(groups)
    plot_op_latency(groups)
    plot_reliability_heatmap(groups)
    plot_coordination_timeline(groups)
    plot_ablation(groups)
    plot_autort_safety(groups)

    print(f"Saved plots to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
