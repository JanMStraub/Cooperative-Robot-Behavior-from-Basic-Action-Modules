#!/usr/bin/env python3
"""
Knowledge Graph Configuration
==============================

Controls the NetworkX-based knowledge graph for spatial reasoning
and multi-hop relationship queries.

Environment variable overrides:
    KNOWLEDGE_GRAPH_ENABLED=true      — activate the knowledge graph
    KG_NEAR_THRESHOLD=0.15            — override the NEAR edge distance (meters)
    KG_VIZ_AUTO_SAVE=true             — save a PNG after every graph update
    KG_VIZ_OUTPUT_DIR=./kg_snapshots  — directory for auto-saved PNGs
    KG_VIZ_DPI=150                    — PNG resolution
    KG_VIZ_FIGSIZE=14x10              — figure size in inches (WxH)
"""

import logging
import os

# Master switch: enable/disable knowledge graph (off by default)
KNOWLEDGE_GRAPH_ENABLED = os.environ.get("KNOWLEDGE_GRAPH_ENABLED", "true").lower() in (
    "true",
    "1",
    "yes",
)

# Distance threshold for NEAR edges (meters)
# Objects/robots closer than this threshold will be connected by a NEAR edge.
try:
    KG_NEAR_THRESHOLD = float(os.environ.get("KG_NEAR_THRESHOLD", "0.02"))
except ValueError:
    logging.warning("Invalid KG_NEAR_THRESHOLD env var; using default 0.02m")
    KG_NEAR_THRESHOLD = 0.02

# ---------------------------------------------------------------------------
# Visualisation settings (used by KnowledgeGraph.save_png and kg_inspect CLI)
# ---------------------------------------------------------------------------

# Automatically save a PNG snapshot after every graph update.
KG_VIZ_AUTO_SAVE = os.environ.get("KG_VIZ_AUTO_SAVE", "false").lower() in ("true", "1", "yes")

# Directory where auto-saved PNGs are written.
KG_VIZ_OUTPUT_DIR = os.environ.get("KG_VIZ_OUTPUT_DIR", "./kg_snapshots")

# PNG resolution (dots per inch).
try:
    KG_VIZ_DPI = int(os.environ.get("KG_VIZ_DPI", "150"))
except ValueError:
    logging.warning("Invalid KG_VIZ_DPI env var; using default 150")
    KG_VIZ_DPI = 150

# Figure size as (width_inches, height_inches).
try:
    _figsize_raw = os.environ.get("KG_VIZ_FIGSIZE", "14x10").lower().split("x")
    KG_VIZ_FIGSIZE: tuple = (float(_figsize_raw[0]), float(_figsize_raw[1]))
except (ValueError, IndexError):
    logging.warning("Invalid KG_VIZ_FIGSIZE env var; using default 14x10")
    KG_VIZ_FIGSIZE = (14.0, 10.0)
