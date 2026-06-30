#!/usr/bin/env python3
"""Arrange a directory of images into a grid figure and save to Thesis/images/.

Usage:
    python -m tools.PlotImageGrid --input documents/images --output 12_handoff_sequence.png
    python -m tools.PlotImageGrid --input path/to/frames --glob "frame_*.png" --cols 4 --output 18_my_sequence.png --title "My Sequence"
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

REPO_ROOT = Path(__file__).parents[2]
PLOTS_DIR = REPO_ROOT / "Thesis" / "images"

FS_TITLE = 16
FS_SUPTITLE = 18
FS_TICK = 11


def _apply_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.titlesize": FS_TICK,
            "figure.titlesize": FS_SUPTITLE,
            "figure.titleweight": "bold",
            "figure.dpi": 100,
            "savefig.dpi": 150,
        }
    )


def plot_image_grid(
    input_dir: Path,
    glob: str,
    output: str,
    cols: int | None,
    title: str | None,
) -> None:
    images = sorted(input_dir.glob(glob))
    if not images:
        raise SystemExit(f"No files matching '{glob}' in {input_dir}")

    n = len(images)
    ncols = cols if cols is not None else math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    axes_flat = (
        [axes] if n == 1 else list(axes.flat if hasattr(axes, "flat") else [axes])
    )

    for ax, img_path in zip(axes_flat, images):
        ax.imshow(mpimg.imread(img_path))
        ax.set_title(img_path.stem.replace("_", " "), fontsize=FS_TICK, pad=3)
        ax.axis("off")

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.subplots_adjust(wspace=0.03, hspace=0.12)
    if title:
        fig.suptitle(title, fontsize=FS_SUPTITLE, fontweight="bold")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / output
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Directory containing images (absolute or relative to repo root)",
    )
    parser.add_argument(
        "--glob",
        default="*.png",
        help="Glob pattern to select images (default: *.png)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output filename inside Thesis/images/ (e.g. 12_handoff_sequence.png)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Number of columns (default: sqrt(n))",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure suptitle",
    )
    args = parser.parse_args()

    input_dir = args.input if args.input.is_absolute() else REPO_ROOT / args.input
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    _apply_rc()
    plot_image_grid(input_dir, args.glob, args.output, args.cols, args.title)


if __name__ == "__main__":
    main()
