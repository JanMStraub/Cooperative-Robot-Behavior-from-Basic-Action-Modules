#!/usr/bin/env python3
"""Plot the operation registry as a complexity x category map.

Every registered operation occupies one cell of a 2D grid:
  * y-axis -- OperationComplexity (atomic -> complex)
  * x-axis -- OperationCategory   (perception, navigation, ...)

Populated cells are tinted by complexity tier (empty cells stay gray) and list
their numbered operation IDs. Saved to Misc/images/.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from operations.Base import OperationCategory, OperationComplexity
from operations.Registry import OperationRegistry, get_global_registry

IMAGES_DIR = Path(__file__).parent.parent.parent / "Misc" / "images"

# Display order along each axis (rows top->bottom, columns left->right).
COMPLEXITY_ORDER = [
    OperationComplexity.ATOMIC,
    OperationComplexity.BASIC,
    OperationComplexity.INTERMEDIATE,
    OperationComplexity.COMPLEX,
]
CATEGORY_ORDER = [
    OperationCategory.PERCEPTION,
    OperationCategory.NAVIGATION,
    OperationCategory.MANIPULATION,
    OperationCategory.STATE_CHECK,
    OperationCategory.COORDINATION,
    OperationCategory.SYNC,
]


def build_operation_grid(
    registry: OperationRegistry,
) -> Dict[Tuple[OperationComplexity, OperationCategory], List[str]]:
    """Map every operation to its (complexity, category) cell.

    Returns a dict keyed by (complexity, category) whose values are the sorted
    operation names in that bucket. Empty buckets are omitted.
    """
    grid: Dict[Tuple[OperationComplexity, OperationCategory], List[str]] = defaultdict(
        list
    )
    for op in registry.get_all_operations():
        grid[(op.complexity, op.category)].append(op.name)
    return {key: sorted(names) for key, names in grid.items()}


def _wrap_name(name: str, max_chars: int = 22) -> str:
    """Wrap an over-long operation id onto two lines at an underscore so it
    stays within its cell instead of spilling into the neighbouring column."""
    if len(name) <= max_chars:
        return name
    parts = name.split("_")
    first: list[str] = []
    length = 0
    for k, part in enumerate(parts):
        if k > 0 and length + 1 + len(part) > max_chars and first:
            break
        first.append(part)
        length += (1 if k > 0 else 0) + len(part)
    return "_".join(first) + "_\n" + "_".join(parts[len(first) :])


def plot_operation_map(
    registry: OperationRegistry | None = None,
    output_path: Path | None = None,
) -> Path:
    """Render the operation registry as a complexity x category heatmap."""
    registry = registry or get_global_registry()
    output_path = output_path or (IMAGES_DIR / "09_operation_map.png")
    grid = build_operation_grid(registry)

    n_rows, n_cols = len(COMPLEXITY_ORDER), len(CATEGORY_ORDER)
    # Populated cells are tinted by their complexity tier; empty cells stay gray.
    row_color = {
        OperationComplexity.ATOMIC: "#cfe8cf",
        OperationComplexity.BASIC: "#cfe2f3",
        OperationComplexity.INTERMEDIATE: "#ffe1b3",
        OperationComplexity.COMPLEX: "#e3cdec",
    }
    empty_rgb = to_rgb("0.96")
    rgb = np.empty((n_rows, n_cols, 3))
    for i, complexity in enumerate(COMPLEXITY_ORDER):
        for j, category in enumerate(CATEGORY_ORDER):
            names = grid.get((complexity, category))
            rgb[i, j] = to_rgb(row_color[complexity]) if names else empty_rgb

    fig, ax = plt.subplots(figsize=(3.5 * n_cols + 1, 2.2 * n_rows + 1.5))
    ax.imshow(rgb, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [c.value for c in CATEGORY_ORDER], fontsize=17, rotation=20, ha="right"
    )
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([c.value for c in COMPLEXITY_ORDER], fontsize=17)
    ax.set_xlabel("Operation category", fontsize=19, fontweight="bold")
    ax.set_ylabel("Operation complexity", fontsize=19, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3.0)
    ax.tick_params(which="minor", length=0)
    ax.invert_yaxis()  # atomic at the bottom, complexity increasing upward

    total = 0
    for i, complexity in enumerate(COMPLEXITY_ORDER):
        for j, category in enumerate(CATEGORY_ORDER):
            names = grid.get((complexity, category))
            if not names:
                continue
            total += len(names)
            label = "\n".join(
                f"{k}. {_wrap_name(name)}" for k, name in enumerate(names, start=1)
            )
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=17,
                color="black",
                linespacing=1.7,
            )

    ax.set_title(
        f"Operation Registry Map  ({total} operations)",
        fontsize=21,
        fontweight="bold",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main() -> None:
    path = plot_operation_map()
    print(f"Saved operation map to {path}")


if __name__ == "__main__":
    main()
