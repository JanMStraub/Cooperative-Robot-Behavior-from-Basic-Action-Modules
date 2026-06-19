#!/usr/bin/env python3
"""Generate benchmark result plots from JSON files in benchmark_results/.

The benchmark suite has a model x task design:
  * b1-b11  : tasks run across several LLM models (one subdir per model).
  * b12-b16 : single-feature ablations (flat layout, one condition per run).

The model name is recorded in each result's ``model`` field for new runs and is
otherwise recovered from the directory path (``benchmark_results/bN/<model>/...``).
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
PLOTS_DIR = RESULTS_DIR / "plots"

SINGLE_ROBOT_COLOR = "#4C72B0"
DUAL_ROBOT_COLOR = "#DD8452"
DUAL_ROBOT_IDS = {6, 7}
ABLATION_IDS = {12, 13, 14, 15, 16}
MAIN_MAX_ID = 11  # b1-b11 are the model x task capability benchmarks

# Stable colors for known models; unknown models cycle through the fallback palette.
MODEL_COLORS = {
    "qwen3-vl-30b": "#4C72B0",
    "qwen3-vl-8b": "#55A868",
    "magistral-small-2509": "#DD8452",
    "gemma-4-e4b": "#8172B2",
    "gemma-4-e2b": "#C44E52",
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
    """Recover the model name from a result path relative to RESULTS_DIR.

    ``b1/qwen3-vl-8b/benchmark1_x.json`` -> ``qwen3-vl-8b``.
    ``b12/benchmark12_x.json`` (flat, no model subdir) -> ``"default"``.
    """
    parts = rel_path.parts
    if len(parts) >= 3:
        return parts[1]
    return "default"


def load_results() -> dict[int, list[dict]]:
    """Load all JSON result files grouped by benchmark_id.

    Each run dict is tagged with ``_model`` (from the JSON ``model`` field when
    present, otherwise recovered from the directory path).
    """
    groups: dict[int, list[dict]] = defaultdict(list)
    for path in sorted(RESULTS_DIR.rglob("benchmark*.json")):
        with open(path) as f:
            data = json.load(f)
        data["_model"] = data.get("model") or model_from_path(
            path.relative_to(RESULTS_DIR)
        )
        groups[data["benchmark_id"]].append(data)
    return dict(sorted(groups.items()))


def op_color(op: str) -> str:
    return OP_COLORS.get(op, OP_COLORS["other"])


def models_in(groups: dict[int, list[dict]], bids: list[int]) -> list[str]:
    """Sorted, deduplicated model list across the given benchmarks (known first)."""
    seen = {r["_model"] for bid in bids for r in groups.get(bid, [])}
    known = [m for m in MODEL_COLORS if m in seen]
    extra = sorted(seen - set(MODEL_COLORS))
    return known + extra


def model_color(model: str, ordered_models: list[str]) -> str:
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    idx = [m for m in ordered_models if m not in MODEL_COLORS].index(model)
    return _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)]


def _main_bids(groups: dict[int, list[dict]]) -> list[int]:
    return [b for b in sorted(groups) if b <= MAIN_MAX_ID]


# --------------------------------------------------------------------------- #
# Model x task capability figures (b1-b11)
# --------------------------------------------------------------------------- #
def plot_success_rate_by_model(groups: dict[int, list[dict]]) -> None:
    bids = _main_bids(groups)
    if not bids:
        return
    ordered = models_in(groups, bids)

    fig, ax = plt.subplots(figsize=(max(12, 1.2 * len(bids)), 6.5))
    x = np.arange(len(bids))
    n = len(ordered)
    width = 0.8 / max(n, 1)

    for i, model in enumerate(ordered):
        means, stds = [], []
        for bid in bids:
            srs = [r["success_rate"] for r in groups[bid] if r["_model"] == model]
            means.append(np.mean(srs) if srs else np.nan)
            stds.append(np.std(srs) if srs else 0.0)
        offset = (i - (n - 1) / 2) * width
        ax.bar(
            x + offset,
            np.nan_to_num(np.array(means, dtype=float)),
            width=width,
            yerr=stds,
            capsize=2,
            color=model_color(model, ordered),
            edgecolor="white",
            linewidth=0.5,
            label=model,
        )

    labels = [f"B{bid}\n{groups[bid][0]['benchmark_name']}" for bid in bids]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, rotation=35, ha="right")
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Success Rate", fontsize=15)
    ax.set_title("Task Success Rate by Model", fontsize=17, fontweight="bold")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=11, ncol=min(n, 3), loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_success_rate_by_model.png", dpi=150)
    plt.close(fig)
    print("  [1] success rate by model")


def plot_model_task_heatmap(groups: dict[int, list[dict]]) -> None:
    bids = _main_bids(groups)
    if not bids:
        return
    ordered = models_in(groups, bids)

    matrix = np.full((len(ordered), len(bids)), np.nan)
    for i, model in enumerate(ordered):
        for j, bid in enumerate(bids):
            srs = [r["success_rate"] for r in groups[bid] if r["_model"] == model]
            if srs:
                matrix[i, j] = np.mean(srs)

    fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(bids)), 0.7 * len(ordered) + 2))
    cmap = plt.get_cmap("RdYlGn")
    cmap.set_bad("0.9")
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean Success Rate", shrink=0.8)

    ax.set_xticks(range(len(bids)))
    ax.set_xticklabels(
        [f"B{bid}  {groups[bid][0]['benchmark_name']}" for bid in bids],
        fontsize=8,
        rotation=35,
        ha="right",
    )
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered, fontsize=10)

    for i in range(len(ordered)):
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

    ax.set_title("Model x Task Success Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_model_task_heatmap.png", dpi=150)
    plt.close(fig)
    print("  [2] model x task heatmap")


def plot_model_leaderboard(groups: dict[int, list[dict]]) -> None:
    bids = _main_bids(groups)
    if not bids:
        return
    ordered = models_in(groups, bids)

    # Mean over per-task means so every task weighs equally regardless of run count.
    scores = {}
    for model in ordered:
        task_means = []
        for bid in bids:
            srs = [r["success_rate"] for r in groups[bid] if r["_model"] == model]
            if srs:
                task_means.append(np.mean(srs))
        scores[model] = np.mean(task_means) if task_means else 0.0

    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(9, max(3, 0.6 * len(ranked) + 1)))
    y = np.arange(len(ranked))
    ax.barh(
        y,
        [v for _, v in ranked],
        color=[model_color(m, ordered) for m, _ in ranked],
        edgecolor="white",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([m for m, _ in ranked], fontsize=10)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Mean Success Rate (task-averaged, B1-B11)", fontsize=11)
    ax.set_title("Model Leaderboard", fontsize=14, fontweight="bold")
    for yi, (_, v) in zip(y, ranked):
        ax.text(v + 0.01, yi, f"{v:.0%}", va="center", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_model_leaderboard.png", dpi=150)
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
    plt.colorbar(im, ax=ax, label="Mean Total Duration (s)", shrink=0.8)

    ax.set_xticks(range(len(bids)))
    ax.set_xticklabels([f"B{bid}" for bid in bids], fontsize=9)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered, fontsize=10)

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
                    fontsize=8,
                    color="white" if val > 0.55 * vmax else "black",
                )

    ax.set_title("Model x Task Mean Duration", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_duration_by_model.png", dpi=150)
    plt.close(fig)
    print("  [4] duration by model")


# --------------------------------------------------------------------------- #
# Operation-level figures (pooled across models)
# --------------------------------------------------------------------------- #
def plot_op_latency(groups: dict[int, list[dict]]) -> None:
    op_bid_durations: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for bid, runs in groups.items():
        for run in runs:
            for step in run["steps"]:
                op_bid_durations[step["operation"]][bid].append(step["duration_ms"])

    # Ops appearing in 2+ benchmarks with a meaningful (>100ms) duration.
    interesting_ops = sorted(
        op
        for op, bid_data in op_bid_durations.items()
        if len(bid_data) >= 2 and max(max(v) for v in bid_data.values()) > 100
    )
    if not interesting_ops:
        return

    ncols = math.ceil(math.sqrt(len(interesting_ops)))
    nrows = math.ceil(len(interesting_ops) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.5 * ncols, 4 * nrows),
        sharey=False,
    )
    axes = axes.flatten() if nrows * ncols > 1 else [axes]

    for ax, op in zip(axes, interesting_ops):
        bid_data = op_bid_durations[op]
        bids_present = sorted(bid_data)
        bp = ax.boxplot(
            [bid_data[bid] for bid in bids_present],
            tick_labels=[f"B{bid}" for bid in bids_present],
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(op_color(op))
            patch.set_alpha(0.8)
        ax.set_title(op.replace("_", "\n"), fontsize=8, fontweight="bold")
        ax.set_ylabel("Duration (ms)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)

    for ax in axes[len(interesting_ops):]:
        ax.set_visible(False)

    fig.suptitle(
        "Per-Operation Step Duration by Benchmark (all models pooled)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
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

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(sorted_ops) + 2)))
    cmap = plt.get_cmap("RdYlGn")
    cmap.set_bad("0.9")
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Success Rate", shrink=0.8)

    ax.set_xticks(range(len(bids)))
    ax.set_xticklabels([f"B{bid}" for bid in bids], fontsize=10)
    ax.set_yticks(range(len(sorted_ops)))
    ax.set_yticklabels([op.replace("_", " ") for op in sorted_ops], fontsize=9)

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
        run = groups[bid][0]  # representative first run
        name = run["benchmark_name"]

        seen_groups: set[int] = set()
        group_steps: dict[int | None, list[dict]] = defaultdict(list)
        for step in run["steps"]:
            group_steps[step["parallel_group_id"]].append(step)

        cursor: dict[str, float] = defaultdict(float)
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
                robots_in_group = {
                    s["robot_id"] for s in group_steps[gid] if s.get("robot_id")
                }
                group_start = max(cursor[r] for r in (robots_in_group or ["Robot1"]))
                for s in group_steps[gid]:
                    cursor[s.get("robot_id") or "Robot1"] = group_start

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
            if dur > 0.5:
                ax.text(
                    start + dur / 2,
                    y,
                    op.replace("_", "\n"),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
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
    fig.savefig(PLOTS_DIR / "07_coordination_timeline.png", dpi=150)
    plt.close(fig)
    print("  [7] coordination timeline")


# --------------------------------------------------------------------------- #
# Ablation figure (b12-b16)
# --------------------------------------------------------------------------- #
def plot_ablation(groups: dict[int, list[dict]]) -> None:
    """Per-ablation success rate by condition.

    Conditions are taken from each run's ``ablation`` block plus its
    ``ablation_baseline`` when present, so the figure shows exactly the
    conditions that exist in the data. When both an enabled/variant and a
    disabled/baseline condition are present, the delta is annotated.
    """
    abl_bids = [bid for bid in sorted(groups) if bid in ABLATION_IDS]
    if not abl_bids:
        return

    # bid -> condition -> list of success rates
    per_bid: dict[int, dict[str, list[float]]] = {}
    missing_baseline: list[int] = []
    for bid in abl_bids:
        cond_sr: dict[str, list[float]] = defaultdict(list)
        has_baseline = False
        for r in groups[bid]:
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

    fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(abl_bids)), 5.5))
    x = np.arange(len(abl_bids))
    n = len(all_conds)
    width = 0.8 / max(n, 1)

    for i, cond in enumerate(all_conds):
        means = [
            np.mean(per_bid[bid][cond]) if cond in per_bid[bid] else np.nan
            for bid in abl_bids
        ]
        stds = [
            np.std(per_bid[bid][cond]) if cond in per_bid[bid] else 0.0
            for bid in abl_bids
        ]
        offset = (i - (n - 1) / 2) * width
        ax.bar(
            x + offset,
            np.nan_to_num(np.array(means, dtype=float)),
            width=width,
            yerr=stds,
            capsize=3,
            color=cond_color.get(cond, "#BBBBBB"),
            edgecolor="white",
            label=cond,
        )

    # Annotate delta for the treatment vs control pair (feature on vs off, or
    # ros vs unity for the movement-backend comparison).
    for j, bid in enumerate(abl_bids):
        cs = per_bid[bid]
        treat = next((c for c in ("enabled", "ros") if c in cs), None)
        control = (
            "disabled" if "disabled" in cs else ("unity" if "unity" in cs else None)
        )
        if treat and control and treat != control:
            delta = np.mean(cs[treat]) - np.mean(cs[control])
            top = float(max(np.mean(cs[treat]), np.mean(cs[control])))
            ax.text(
                j,
                top + 0.05,
                f"Δ {delta:+.0%}",
                ha="center",
                fontsize=9,
                fontweight="bold",
                color="#333333",
            )

    labels = [f"B{bid}\n{groups[bid][0]['benchmark_name']}" for bid in abl_bids]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Success Rate", fontsize=12)
    title = "Ablation: Success Rate by Condition"
    if missing_baseline:
        miss = ", ".join(f"B{b}" for b in missing_baseline)
        title += f"\n(no paired baseline in data for {miss} - re-run paired ablation)"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=9, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "08_ablation.png", dpi=150)
    plt.close(fig)
    print("  [8] ablation")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    groups = load_results()
    print(
        f"Loaded {sum(len(v) for v in groups.values())} runs "
        f"across {len(groups)} benchmarks"
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

    plot_success_rate_by_model(groups)
    plot_model_task_heatmap(groups)
    plot_model_leaderboard(groups)
    plot_duration_by_model(groups)
    plot_op_latency(groups)
    plot_reliability_heatmap(groups)
    plot_coordination_timeline(groups)
    plot_ablation(groups)

    print(f"Saved plots to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
