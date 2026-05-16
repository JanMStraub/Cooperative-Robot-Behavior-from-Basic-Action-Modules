#!/usr/bin/env python3
"""
CLI tool to inspect and visualise the ACRL Knowledge Graph.

Modes:
  stats   — print node/edge counts
  dump    — print all nodes and edges as JSON
  png     — render graph to PNG
  graphml — export GraphML (opens in Gephi / Cytoscape)
  b12     — populate the B12 synthetic KG, render it, then clear

Usage:
  python -m tools.KGInspect stats
  python -m tools.KGInspect dump
  python -m tools.KGInspect png [--out kg.png] [--title "My KG"]
  python -m tools.KGInspect graphml [--out kg.graphml]
  python -m tools.KGInspect b12 [--out b12_kg.png]
"""

from __future__ import annotations

import argparse
import json
import sys


def _get_kg():
    from knowledge_graph._singleton import get_knowledge_graph

    return get_knowledge_graph()


def cmd_stats(args) -> None:
    kg = _get_kg()
    stats = kg.get_stats()
    print(json.dumps(stats, indent=2))


def cmd_dump(args) -> None:
    kg = _get_kg()
    from knowledge_graph.Core import KnowledgeGraph

    with kg._lock:
        nodes = {nid: dict(attrs) for nid, attrs in kg._graph.nodes(data=True)}
        edges = [
            {
                "from": u,
                "to": v,
                **{
                    k: str(v2) if isinstance(v2, tuple) else v2
                    for k, v2 in data.items()
                },
            }
            for u, v, data in kg._graph.edges(data=True)
        ]
    print(json.dumps({"nodes": nodes, "edges": edges}, indent=2, default=str))


def cmd_png(args) -> None:
    kg = _get_kg()
    out = getattr(args, "out", None) or "kg.png"
    title = getattr(args, "title", None) or "ACRL Knowledge Graph"
    kg.save_png(out, title=title)
    print(f"Saved: {out}")


def cmd_graphml(args) -> None:
    kg = _get_kg()
    out = getattr(args, "out", None) or "kg.graphml"
    kg.save_graphml(out)
    print(f"Saved: {out}")


def cmd_snapshot(args) -> None:
    """
    Load the latest auto-saved graphml from the running server and render it.

    The running server writes graphml snapshots to KG_VIZ_OUTPUT_DIR when
    KG_VIZ_AUTO_SAVE is enabled. This command finds the newest file, loads it
    into a fresh KG, and renders a PNG — bridging the process boundary.
    """
    import os
    import glob
    import config.KnowledgeGraph as kg_cfg

    search_dir = getattr(args, "dir", None) or kg_cfg.KG_VIZ_OUTPUT_DIR
    out = getattr(args, "out", None) or "kg_live.png"

    # Prefer graphml (lossless) over png snapshots
    graphml_files = sorted(glob.glob(os.path.join(search_dir, "*.graphml")))
    png_files = sorted(glob.glob(os.path.join(search_dir, "*.png")))

    if graphml_files:
        latest = graphml_files[-1]
        print(f"Loading: {latest}")
        kg = _get_kg()
        kg.load_graphml(latest)
        stats = kg.get_stats()
        print(f"Loaded: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
        kg.save_png(out, title=f"KG from {os.path.basename(latest)}")
        print(f"Saved: {out}")
    elif png_files:
        latest = png_files[-1]
        print(f"Latest snapshot PNG: {latest}")
        print("(No graphml found — copy the PNG path above to view it directly)")
    else:
        print(f"No snapshots found in {search_dir}")
        print(
            "Ensure KG_VIZ_AUTO_SAVE=true and the server has processed at least one WorldState update."
        )
        print("Or export manually from the server with: kg.save_graphml('kg.graphml')")


def cmd_b12(args) -> None:
    """Populate the B12 synthetic KG, render it, then clear."""
    from benchmarks.cases.B12KgAblation import (
        populate_synthetic_kg,
        clear_synthetic_kg,
        KG_OBJECTS,
        KG_ROBOT_NEARBY,
    )
    import config.KnowledgeGraph as kg_cfg

    prev = kg_cfg.KNOWLEDGE_GRAPH_ENABLED
    kg_cfg.KNOWLEDGE_GRAPH_ENABLED = True
    try:
        populate_synthetic_kg("Robot1")
        kg = _get_kg()
        stats = kg.get_stats()
        print(
            f"B12 synthetic KG: {stats['total_nodes']} nodes, {stats['total_edges']} edges"
        )
        print(f"Objects: {KG_OBJECTS}")
        print(f"Nearby robot: {KG_ROBOT_NEARBY}")
        out = getattr(args, "out", None) or "b12_kg.png"
        kg.save_png(out, title="B12 Synthetic Knowledge Graph")
        print(f"Saved: {out}")
    finally:
        clear_synthetic_kg()
        kg_cfg.KNOWLEDGE_GRAPH_ENABLED = prev


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tools.KGInspect",
        description="Inspect and visualise the ACRL Knowledge Graph",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stats", help="Print node/edge statistics")
    sub.add_parser("dump", help="Dump all nodes and edges as JSON")

    p_png = sub.add_parser("png", help="Render graph to PNG")
    p_png.add_argument("--out", default="kg.png", help="Output file (default: kg.png)")
    p_png.add_argument("--title", default="ACRL Knowledge Graph", help="Figure title")

    p_gml = sub.add_parser("graphml", help="Export GraphML")
    p_gml.add_argument(
        "--out", default="kg.graphml", help="Output file (default: kg.graphml)"
    )

    p_b12 = sub.add_parser("b12", help="Render the B12 synthetic KG")
    p_b12.add_argument(
        "--out", default="b12_kg.png", help="Output file (default: b12_kg.png)"
    )

    p_snap = sub.add_parser(
        "snapshot", help="Load latest server snapshot and render PNG"
    )
    p_snap.add_argument(
        "--dir", default=None, help="Snapshot directory (default: KG_VIZ_OUTPUT_DIR)"
    )
    p_snap.add_argument(
        "--out", default="kg_live.png", help="Output PNG (default: kg_live.png)"
    )

    args = parser.parse_args()
    {
        "stats": cmd_stats,
        "dump": cmd_dump,
        "png": cmd_png,
        "graphml": cmd_graphml,
        "snapshot": cmd_snapshot,
        "b12": cmd_b12,
    }[args.command](args)


if __name__ == "__main__":
    main()
