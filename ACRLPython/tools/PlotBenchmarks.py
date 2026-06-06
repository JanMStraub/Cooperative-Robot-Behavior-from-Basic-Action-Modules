#!/usr/bin/env python3
"""Generate benchmark result plots from JSON files in benchmark_results/."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "benchmark_results"
PLOTS_DIR = RESULTS_DIR / "plots"

SINGLE_ROBOT_COLOR = "#4C72B0"
DUAL_ROBOT_COLOR = "#DD8452"
DUAL_ROBOT_IDS = {6, 7}

OP_COLORS = {
    "detect_object_stereo": "#4C72B0",
    "grasp_object": "#DD8452",
    "move_to_coordinate": "#55A868",
    "return_to_start_position": "#C44E52",
    "receive_handoff": "#8172B2",
    "adjust_end_effector_orientation": "#937860",
    "signal": "#DA8BC3",
    "wait_for_signal": "#8C8C8C",
    "release_object": "#CCB974",
    "other": "#BBBBBB",
}


def load_results() -> dict[int, list[dict]]:
    """Load all JSON result files grouped by benchmark_id."""
    groups: dict[int, list[dict]] = defaultdict(list)
    for path in sorted(RESULTS_DIR.glob("benchmark*.json")):
        with open(path) as f:
            data = json.load(f)
        groups[data["benchmark_id"]].append(data)
    return dict(sorted(groups.items()))


def benchmark_label(bid: int, name: str) -> str:
    return f"B{bid}\n{name}"


def op_color(op: str) -> str:
    return OP_COLORS.get(op, OP_COLORS["other"])


def plot_success_rate(groups: dict[int, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    bids = sorted(groups)
    names = [groups[bid][0]["benchmark_name"] for bid in bids]
    means = [np.mean([r["success_rate"] for r in groups[bid]]) for bid in bids]
    stds = [np.std([r["success_rate"] for r in groups[bid]]) for bid in bids]
    colors = [
        DUAL_ROBOT_COLOR if bid in DUAL_ROBOT_IDS else SINGLE_ROBOT_COLOR
        for bid in bids
    ]
    labels = [f"B{bid}\n{name}" for bid, name in zip(bids, names)]

    bars = ax.bar(
        labels,
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        width=0.6,
    )

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("Task Success Rate by Benchmark", fontsize=14, fontweight="bold")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{mean:.0%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    legend_patches = [
        mpatches.Patch(color=SINGLE_ROBOT_COLOR, label="Single-robot"),
        mpatches.Patch(color=DUAL_ROBOT_COLOR, label="Dual-robot"),
    ]
    ax.legend(handles=legend_patches, loc="lower left", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_success_rate.png", dpi=150)
    plt.close(fig)
    print("  [1] success rate")


def plot_duration_boxplots(groups: dict[int, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))

    bids = sorted(groups)
    data = [[r["total_duration_ms"] / 1000 for r in groups[bid]] for bid in bids]
    labels = [f"B{bid}\n{groups[bid][0]['benchmark_name']}" for bid in bids]
    colors = [
        DUAL_ROBOT_COLOR if bid in DUAL_ROBOT_IDS else SINGLE_ROBOT_COLOR
        for bid in bids
    ]

    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color="white", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for element in ("whiskers", "caps", "fliers"):
        for item in bp[element]:
            item.set_color("#555555")

    ax.set_ylabel("Total Duration (s)", fontsize=12)
    ax.set_title(
        "Benchmark Execution Time Distribution (n=5 runs each)",
        fontsize=14,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_duration_boxplots.png", dpi=150)
    plt.close(fig)
    print("  [2] duration boxplots")


def plot_time_composition(groups: dict[int, list[dict]]) -> None:
    all_ops = list(OP_COLORS.keys())

    bids = sorted(groups)
    labels = [f"B{bid}\n{groups[bid][0]['benchmark_name']}" for bid in bids]

    # avg per op type per benchmark (aggregate steps across runs)
    op_times: dict[str, list[float]] = {op: [] for op in all_ops}

    for bid in bids:
        totals: dict[str, float] = defaultdict(float)
        n_runs = len(groups[bid])
        for run in groups[bid]:
            for step in run["steps"]:
                op = step["operation"] if step["operation"] in OP_COLORS else "other"
                totals[op] += step["duration_ms"] / 1000
        for op in all_ops:
            op_times[op].append(totals.get(op, 0.0) / n_runs)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(bids))
    bottom = np.zeros(len(bids))

    for op in all_ops:
        vals = np.array(op_times[op])
        if vals.sum() == 0:
            continue
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=op_color(op),
            label=op.replace("_", " "),
            width=0.6,
        )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Avg Duration (s)", fontsize=12)
    ax.set_title(
        "Operation Time Composition per Benchmark", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_time_composition.png", dpi=150)
    plt.close(fig)
    print("  [3] time composition")


def plot_op_latency(groups: dict[int, list[dict]]) -> None:
    # Collect step durations per (op, bid)
    op_bid_durations: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for bid, runs in groups.items():
        for run in runs:
            for step in run["steps"]:
                op = step["operation"]
                op_bid_durations[op][bid].append(step["duration_ms"])

    # Focus on ops that appear in 2+ benchmarks and have meaningful duration
    interesting_ops = [
        op
        for op, bid_data in op_bid_durations.items()
        if len(bid_data) >= 1 and max(max(v) for v in bid_data.values()) > 100
    ]
    interesting_ops = sorted(interesting_ops)

    if not interesting_ops:
        return

    fig, axes = plt.subplots(
        1,
        len(interesting_ops),
        figsize=(max(12, 2.5 * len(interesting_ops)), 5),
        sharey=False,
    )
    if len(interesting_ops) == 1:
        axes = [axes]

    for ax, op in zip(axes, interesting_ops):
        bid_data = op_bid_durations[op]
        bids_present = sorted(bid_data)
        data = [bid_data[bid] for bid in bids_present]
        xlabels = [f"B{bid}" for bid in bids_present]

        bp = ax.boxplot(
            data,
            tick_labels=xlabels,
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
        )
        color = op_color(op)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        ax.set_title(op.replace("_", "\n"), fontsize=8, fontweight="bold")
        ax.set_ylabel("Duration (ms)" if op == interesting_ops[0] else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle(
        "Per-Operation Step Duration by Benchmark", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_op_latency.png", dpi=150)
    plt.close(fig)
    print("  [4] op latency")


def plot_coordination_timeline(groups: dict[int, list[dict]]) -> None:
    dual_bids = [bid for bid in sorted(groups) if bid in DUAL_ROBOT_IDS]
    if not dual_bids:
        return

    fig, axes = plt.subplots(len(dual_bids), 1, figsize=(14, 4 * len(dual_bids)))
    if len(dual_bids) == 1:
        axes = [axes]

    robot_y = {"Robot1": 1, "Robot2": 0}
    robot_labels = ["Robot2", "Robot1"]

    for ax, bid in zip(axes, dual_bids):
        # pick first run
        run = groups[bid][0]
        name = run["benchmark_name"]

        seen_groups: set[int] = set()

        # group steps by parallel_group_id to find concurrent timing
        group_steps: dict[int | None, list[dict]] = defaultdict(list)
        for step in run["steps"]:
            group_steps[step["parallel_group_id"]].append(step)

        # For sequential (group=None), assign time by order
        # For parallel groups, all steps in group start at same cursor max
        cursor: dict[str, float] = defaultdict(float)  # per-robot cursor in seconds

        patches = []
        legend_ops: set[str] = set()

        for step in run["steps"]:
            robot = step.get("robot_id") or "Robot1"
            dur = step["duration_ms"] / 1000
            op = step["operation"]
            color = op_color(op)

            gid = step["parallel_group_id"]
            if gid is not None and gid not in seen_groups:
                seen_groups.add(gid)
                # find max cursor across robots in this group
                robots_in_group = {
                    s["robot_id"] for s in group_steps[gid] if s.get("robot_id")
                }
                group_start = max(cursor[r] for r in (robots_in_group or ["Robot1"]))
                for s in group_steps[gid]:
                    r = s.get("robot_id") or "Robot1"
                    cursor[r] = group_start

            start = cursor[robot]
            y = robot_y.get(robot, 0)

            rect = mpatches.FancyBboxPatch(
                (start, y - 0.35),
                dur,
                0.7,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.85 if step["success"] else 0.35,
            )
            ax.add_patch(rect)

            # label if wide enough
            if dur > 0.5:
                ax.text(
                    start + dur / 2,
                    y,
                    op.replace("_", "\n"),
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white",
                    fontweight="bold",
                )

            cursor[robot] += dur

            if op not in legend_ops:
                legend_ops.add(op)
                patches.append(mpatches.Patch(color=color, label=op.replace("_", " ")))

        ax.set_xlim(0, max(cursor.values()) * 1.02)
        ax.set_ylim(-0.7, 1.7)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(robot_labels, fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=10)
        title_suffix = " [FAILED]" if not run["success"] else ""
        ax.set_title(f"B{bid}: {name}{title_suffix}", fontsize=12, fontweight="bold")
        ax.legend(
            handles=patches, loc="upper right", fontsize=7, framealpha=0.9, ncol=2
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    fig.suptitle("Dual-Robot Coordination Timeline", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "05_coordination_timeline.png", dpi=150)
    plt.close(fig)
    print("  [5] coordination timeline")


def plot_reliability_heatmap(groups: dict[int, list[dict]]) -> None:
    # Collect per (op, bid): success count, total count
    op_bid_counts: dict[str, dict[int, list[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for bid, runs in groups.items():
        for run in runs:
            for step in run["steps"]:
                op = step["operation"]
                op_bid_counts[op][bid].append(step["success"])

    all_ops = sorted(op_bid_counts.keys())
    bids = sorted(groups)

    # build matrix: rows=ops, cols=bids; NaN = not used
    matrix = np.full((len(all_ops), len(bids)), np.nan)
    for i, op in enumerate(all_ops):
        for j, bid in enumerate(bids):
            if bid in op_bid_counts[op]:
                successes = op_bid_counts[op][bid]
                matrix[i, j] = np.mean(successes)

    # sort rows: ops with failures first
    row_means = np.nanmean(matrix, axis=1)
    sort_idx = np.argsort(row_means)
    matrix = matrix[sort_idx]
    sorted_ops = [all_ops[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(sorted_ops) + 2)))

    cmap = plt.get_cmap("RdYlGn")
    cmap.set_bad("0.9")  # grey for NaN

    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Success Rate", shrink=0.8)

    ax.set_xticks(range(len(bids)))
    ax.set_xticklabels([f"B{bid}" for bid in bids], fontsize=10)
    ax.set_yticks(range(len(sorted_ops)))
    ax.set_yticklabels([op.replace("_", " ") for op in sorted_ops], fontsize=9)

    # annotate cells
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
                    fontsize=8,
                    color="black" if 0.3 < val < 0.8 else "white",
                )

    ax.set_title("Operation Reliability by Benchmark", fontsize=14, fontweight="bold")
    ax.set_xlabel("Benchmark", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "06_reliability_heatmap.png", dpi=150)
    plt.close(fig)
    print("  [6] reliability heatmap")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    groups = load_results()
    print(
        f"Loaded {sum(len(v) for v in groups.values())} runs across {len(groups)} benchmarks"
    )
    print("Generating plots...")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "figure.dpi": 100,
        }
    )

    plot_success_rate(groups)
    plot_duration_boxplots(groups)
    plot_time_composition(groups)
    plot_op_latency(groups)
    plot_coordination_timeline(groups)
    plot_reliability_heatmap(groups)

    print(f"Saved 6 plots to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
