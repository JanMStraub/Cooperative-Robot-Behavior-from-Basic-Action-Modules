#!/usr/bin/env python3
"""Thread-safe wrapper over NetworkX MultiDiGraph with graph CRUD and persistence."""

import threading
from collections import defaultdict
from typing import Optional, List, Dict, Any, Set

try:
    import networkx as nx
except ImportError:
    raise ImportError(
        "NetworkX is required for KnowledgeGraph. Install with: pip install networkx"
    )

from core.LoggingSetup import get_logger

logger = get_logger(__name__)


class KnowledgeGraph:
    """Thread-safe knowledge graph using NetworkX MultiDiGraph with RLock protection."""

    def __init__(self):
        self._graph = nx.MultiDiGraph()
        self._lock = threading.RLock()
        # Secondary index: node_type -> set of node IDs for O(1) typed lookups.
        self._nodes_by_type: defaultdict = defaultdict(set)
        logger.info("KnowledgeGraph initialized (empty)")

    def add_node(self, node_id: str, **attrs):
        with self._lock:
            existing_type = None
            if node_id in self._graph:
                existing_type = self._graph.nodes[node_id].get("node_type")
                new_type = attrs.get("node_type")
                if existing_type and new_type and existing_type != new_type:
                    self._nodes_by_type[existing_type].discard(node_id)
            # If update doesn't supply node_type, preserve the existing one
            if "node_type" not in attrs and existing_type:
                attrs = {**attrs, "node_type": existing_type}
            self._graph.add_node(node_id, **attrs)
            node_type = attrs.get("node_type")
            if node_type:
                self._nodes_by_type[node_type].add(node_id)
            logger.debug(f"Added node: {node_id} ({attrs.get('node_type', 'unknown')})")

    def remove_node(self, node_id: str):
        with self._lock:
            if node_id in self._graph:
                node_type = self._graph.nodes[node_id].get("node_type")
                if node_type:
                    self._nodes_by_type[node_type].discard(node_id)
                self._graph.remove_node(node_id)
                logger.debug(f"Removed node: {node_id}")
                return True
            return False

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if node_id in self._graph:
                return dict(self._graph.nodes[node_id])
            return None

    def has_node(self, node_id: str) -> bool:
        with self._lock:
            return node_id in self._graph

    def add_edge(self, source: str, target: str, edge_type: str, **attrs):
        with self._lock:
            self._graph.add_edge(source, target, edge_type=edge_type, **attrs)
            logger.debug(f"Added edge: {source} --[{edge_type}]--> {target}")

    def remove_edge(self, source: str, target: str, edge_type: Optional[str] = None):
        with self._lock:
            if not self._graph.has_edge(source, target):
                return 0

            removed_count = 0
            # Get all edges between source and target
            edges_to_remove = []
            for key, data in self._graph[source][target].items():
                if edge_type is None or data.get("edge_type") == edge_type:
                    edges_to_remove.append(key)

            for key in edges_to_remove:
                self._graph.remove_edge(source, target, key)
                removed_count += 1

            if removed_count > 0:
                logger.debug(
                    f"Removed {removed_count} edge(s): {source} --> {target}"
                    + (f" (type: {edge_type})" if edge_type else "")
                )

            return removed_count

    def remove_edges_by_type(self, nodes: List[str], edge_types: Set[str]) -> int:
        """
        Batch-remove all edges of the given types from the given nodes.

        More efficient than per-pair remove_edge: collects existing edges first,
        removes in one locked pass.
        """
        with self._lock:
            edges_to_remove = [
                (u, v, key)
                for u in nodes
                if u in self._graph
                for v, edge_dict in self._graph[u].items()
                for key, data in edge_dict.items()
                if data.get("edge_type") in edge_types
            ]
            for u, v, key in edges_to_remove:
                self._graph.remove_edge(u, v, key)
            if edges_to_remove:
                logger.debug(f"Batch removed {len(edges_to_remove)} spatial edge(s)")
            return len(edges_to_remove)

    def get_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[str]:
        with self._lock:
            if node_id not in self._graph:
                return []

            if edge_type is None:
                return list(self._graph.successors(node_id))

            neighbors = []
            for target in self._graph.successors(node_id):
                # Check all edges to this target
                for _, edge_data in self._graph[node_id][target].items():
                    if edge_data.get("edge_type") == edge_type:
                        neighbors.append(target)
                        break  # Only add target once

            return neighbors

    def get_predecessors(
        self, node_id: str, edge_type: Optional[str] = None
    ) -> List[str]:
        with self._lock:
            if node_id not in self._graph:
                return []

            if edge_type is None:
                return list(self._graph.predecessors(node_id))

            predecessors = []
            for source in self._graph.predecessors(node_id):
                for _, edge_data in self._graph[source][node_id].items():
                    if edge_data.get("edge_type") == edge_type:
                        predecessors.append(source)
                        break

            return predecessors

    def get_all_nodes(self, node_type: Optional[str] = None) -> List[str]:
        with self._lock:
            if node_type is None:
                return list(self._graph.nodes())

            # O(1) index lookup instead of O(N) list comprehension
            return list(self._nodes_by_type.get(node_type, set()))

    def node_count(self) -> int:
        """Get total number of nodes."""
        with self._lock:
            return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        """Get total number of edges."""
        with self._lock:
            return self._graph.number_of_edges()

    def clear(self):
        """Remove all nodes and edges from the graph."""
        with self._lock:
            self._graph.clear()
            self._nodes_by_type.clear()
            logger.info("KnowledgeGraph cleared")

    def save_graphml(self, path: str):
        """Save graph to GraphML. Converts non-primitive attrs for compatibility."""
        # Copy inside lock, then write outside lock to avoid holding the lock
        # during slow file I/O.
        with self._lock:
            graph_copy = self._graph.copy()
            node_count = self._graph.number_of_nodes()
            edge_count = self._graph.number_of_edges()

        # GraphML only supports str/int/float/bool — convert everything else
        import json as _json

        def _graphml_safe(v):
            if isinstance(v, (str, int, float, bool)):
                return v
            return _json.dumps(v, default=str)

        for node_id, attrs in graph_copy.nodes(data=True):
            for key, value in list(attrs.items()):
                attrs[key] = _graphml_safe(value)

        for u, v, key, attrs in graph_copy.edges(data=True, keys=True):
            for attr_key, value in list(attrs.items()):
                attrs[attr_key] = _graphml_safe(value)

        try:
            nx.write_graphml(graph_copy, path)
        except OSError as e:
            logger.error(f"Failed to save graph to {path}: {e}")
            raise
        logger.info(f"Saved graph to {path} ({node_count} nodes, {edge_count} edges)")

    def load_graphml(self, path: str):
        with self._lock:
            try:
                self._graph = nx.read_graphml(path, node_type=str)
            except OSError as e:
                logger.error(f"Failed to load graph from {path}: {e}")
                raise
            # Rebuild _nodes_by_type index from the newly loaded graph
            self._nodes_by_type.clear()
            for node_id, attrs in self._graph.nodes(data=True):
                node_type = attrs.get("node_type")
                if node_type:
                    self._nodes_by_type[node_type].add(node_id)
            logger.info(
                f"Loaded graph from {path} ({self._graph.number_of_nodes()} nodes, {self._graph.number_of_edges()} edges)"
            )

    def save_png(
        self,
        path: str,
        title: str = "Knowledge Graph",
        dpi: Optional[int] = None,
        figsize: Optional[tuple] = None,
    ) -> None:
        """
        Render graph to PNG via matplotlib.

        Node color: robot=steelblue, object=coral, region=mediumseagreen.
        DPI/figsize fall back to KG_VIZ_DPI/KG_VIZ_FIGSIZE config values.
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            raise ImportError(
                "matplotlib is required for save_png. pip install matplotlib"
            )

        import config.KnowledgeGraph as _viz_cfg

        _dpi = dpi if dpi is not None else _viz_cfg.KG_VIZ_DPI
        _figsize = figsize if figsize is not None else _viz_cfg.KG_VIZ_FIGSIZE

        _NODE_COLORS = {
            "robot": "steelblue",
            "object": "coral",
            "region": "mediumseagreen",
        }
        _EDGE_COLORS = {
            "CAN_REACH": "#2ecc71",
            "NEAR": "#f39c12",
            "GRASPING": "#e74c3c",
            "IN_REGION": "#9b59b6",
            "ALLOCATED": "#1abc9c",
            "ADJACENT_TO": "#95a5a6",
            "EXECUTED": "#3498db",
            "REQUIRES": "#e67e22",
            "CONFLICTS_WITH": "#c0392b",
        }

        with self._lock:
            graph_copy = self._graph.copy()

        if graph_copy.number_of_nodes() == 0:
            logger.warning("Graph is empty — nothing to render")
            return

        fig, ax = plt.subplots(figsize=_figsize)
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.axis("off")

        try:
            pos = nx.spring_layout(graph_copy, seed=42, k=2.5)
        except Exception:
            pos = nx.shell_layout(graph_copy)

        node_colors = [
            _NODE_COLORS.get(graph_copy.nodes[n].get("node_type", ""), "lightgrey")
            for n in graph_copy.nodes()
        ]

        nx.draw_networkx_nodes(
            graph_copy, pos, node_color=node_colors, node_size=1200, ax=ax, alpha=0.9
        )
        nx.draw_networkx_labels(
            graph_copy, pos, font_size=14, font_weight="bold", ax=ax
        )

        seen_types: set = set()
        for u, v, data in graph_copy.edges(data=True):
            etype = data.get("edge_type", "")
            color = _EDGE_COLORS.get(etype, "#bdc3c7")
            nx.draw_networkx_edges(
                graph_copy,
                pos,
                edgelist=[(u, v)],
                edge_color=color,
                arrows=True,
                arrowsize=15,
                width=1.5,
                connectionstyle="arc3,rad=0.1",
                ax=ax,
            )
            seen_types.add(etype)

        handles = [
            mpatches.Patch(color=_EDGE_COLORS.get(t, "#bdc3c7"), label=t)
            for t in sorted(seen_types)
        ]
        for ntype, color in _NODE_COLORS.items():
            handles.append(mpatches.Patch(color=color, label=f"[{ntype}]"))
        ax.legend(
            handles=handles, loc="lower left", fontsize=12, framealpha=0.8, ncol=2
        )

        stats = (
            f"{graph_copy.number_of_nodes()} nodes  "
            f"{graph_copy.number_of_edges()} edges"
        )
        fig.text(0.5, 0.01, stats, ha="center", fontsize=12, color="grey")

        plt.tight_layout()
        plt.savefig(path, dpi=_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved KG visualisation to {path}")

    def auto_save_png_if_enabled(self, label: str = "") -> None:
        """
        Save timestamped PNG + GraphML if KG_VIZ_AUTO_SAVE enabled.

        Both formats written so kg_inspect can reload graphml for accurate re-rendering.
        """
        import config.KnowledgeGraph as _viz_cfg

        if not _viz_cfg.KG_VIZ_AUTO_SAVE:
            return
        import os
        import time

        out_dir = _viz_cfg.KG_VIZ_OUTPUT_DIR
        os.makedirs(out_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        suffix = f"_{label}" if label else ""
        base = os.path.join(out_dir, f"kg_{ts}{suffix}")
        try:
            self.save_png(
                f"{base}.png", title=f"Knowledge Graph snapshot {label or ts}"
            )
        except Exception as exc:
            logger.warning(f"auto_save_png failed: {exc}")
        try:
            self.save_graphml(f"{base}.graphml")
        except Exception as exc:
            logger.warning(f"auto_save_graphml failed: {exc}")

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            # Build type counts from the O(1) index
            node_type_counts = {
                node_type: len(node_ids)
                for node_type, node_ids in self._nodes_by_type.items()
                if node_ids  # skip empty sets from discard operations
            }

            return {
                "total_nodes": self._graph.number_of_nodes(),
                "total_edges": self._graph.number_of_edges(),
                "node_types": node_type_counts,
            }
