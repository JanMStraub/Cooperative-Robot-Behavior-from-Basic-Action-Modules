#!/usr/bin/env python3
"""
Knowledge Graph Core Implementation
====================================

Thin wrapper over NetworkX MultiDiGraph with thread-safe operations.
Provides graph CRUD operations and persistence.

Edge Types:
- CAN_REACH: Robot -> Object (with distance, approach_direction)
- NEAR: Object <-> Object, Robot <-> Object (with distance)
- IN_REGION: Robot/Object -> Region
- GRASPING: Robot -> Object (with grasp_time)
- ADJACENT_TO: Region <-> Region (static)
- ALLOCATED: Region -> Robot (with allocated_at, timeout)
- EXECUTED: Robot -> Operation (with timestamp)
- REQUIRES: Operation -> Operation (from OperationRelationship)
- CONFLICTS_WITH: Operation <-> Operation (from OperationRelationship)
"""

import threading
from collections import defaultdict
from typing import Optional, List, Dict, Any, Set

try:
    import networkx as nx
except ImportError:
    raise ImportError(
        "NetworkX is required for KnowledgeGraph. Install with: pip install networkx"
    )

# Configure logging
from core.LoggingSetup import get_logger

logger = get_logger(__name__)


class KnowledgeGraph:
    """
    Thread-safe knowledge graph for spatial and temporal reasoning.

    Uses NetworkX MultiDiGraph to support multiple edges between nodes.
    All operations are protected by RLock for thread safety.
    """

    def __init__(self):
        """
        Initialize empty knowledge graph.

        Graph structure:
        - Nodes: RobotNode, ObjectNode, RegionNode (stored as attributes dict)
        - Edges: Typed edges with optional attributes (weight, timestamp, etc.)
        """
        self._graph = nx.MultiDiGraph()
        self._lock = threading.RLock()
        # Secondary index: node_type -> set of node IDs for O(1) typed lookups.
        self._nodes_by_type: defaultdict = defaultdict(set)
        logger.info("KnowledgeGraph initialized (empty)")

    def add_node(self, node_id: str, **attrs):
        """
        Add or update a node in the graph.

        Args:
            node_id: Unique node identifier
            **attrs: Node attributes (e.g., node_type, position, etc.)

        Example:
            >>> kg = KnowledgeGraph()
            >>> kg.add_node("Robot1", node_type="robot", position=(-0.3, 0.2, 0.1))
        """
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
        """
        Remove a node and all its edges from the graph.

        Args:
            node_id: Node identifier to remove

        Returns:
            True if node was removed, False if it didn't exist
        """
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
        """
        Get node attributes.

        Args:
            node_id: Node identifier

        Returns:
            Dictionary of node attributes, or None if node doesn't exist
        """
        with self._lock:
            if node_id in self._graph:
                return dict(self._graph.nodes[node_id])
            return None

    def has_node(self, node_id: str) -> bool:
        """
        Check if node exists in graph.

        Args:
            node_id: Node identifier

        Returns:
            True if node exists
        """
        with self._lock:
            return node_id in self._graph

    def add_edge(self, source: str, target: str, edge_type: str, **attrs):
        """
        Add a typed edge between two nodes.

        Supports multiple edges between same nodes (MultiDiGraph).

        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Edge type (e.g., "CAN_REACH", "NEAR", "IN_REGION")
            **attrs: Edge attributes (e.g., distance, weight, timestamp)

        Example:
            >>> kg.add_edge("Robot1", "RedCube", "CAN_REACH", distance=0.5)
        """
        with self._lock:
            self._graph.add_edge(source, target, edge_type=edge_type, **attrs)
            logger.debug(f"Added edge: {source} --[{edge_type}]--> {target}")

    def remove_edge(self, source: str, target: str, edge_type: Optional[str] = None):
        """
        Remove edge(s) between two nodes.

        If edge_type is specified, removes only edges of that type.
        Otherwise, removes all edges between the nodes.

        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Optional edge type filter

        Returns:
            Number of edges removed
        """
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
        Batch-remove all edges of the given types incident to any of the given nodes.

        More efficient than calling remove_edge for every possible (src, dst) pair
        because it collects existing edges first and removes them in one locked pass.

        Args:
            nodes: List of node IDs whose outgoing edges should be scanned.
            edge_types: Set of edge type strings to remove (e.g. {"CAN_REACH", "NEAR"}).

        Returns:
            Number of edges removed.
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
        """
        Get neighbors of a node, optionally filtered by edge type.

        Returns successor nodes (outgoing edges).

        Args:
            node_id: Node identifier
            edge_type: Optional edge type filter (e.g., "CAN_REACH")

        Returns:
            List of neighbor node IDs

        Example:
            >>> kg.get_neighbors("Robot1", edge_type="CAN_REACH")
            ['RedCube', 'BlueCube']
        """
        with self._lock:
            if node_id not in self._graph:
                return []

            if edge_type is None:
                # Return all successors
                return list(self._graph.successors(node_id))

            # Filter by edge type
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
        """
        Get predecessors of a node (incoming edges), optionally filtered by edge type.

        Args:
            node_id: Node identifier
            edge_type: Optional edge type filter

        Returns:
            List of predecessor node IDs
        """
        with self._lock:
            if node_id not in self._graph:
                return []

            if edge_type is None:
                return list(self._graph.predecessors(node_id))

            # Filter by edge type
            predecessors = []
            for source in self._graph.predecessors(node_id):
                for _, edge_data in self._graph[source][node_id].items():
                    if edge_data.get("edge_type") == edge_type:
                        predecessors.append(source)
                        break

            return predecessors

    def get_all_nodes(self, node_type: Optional[str] = None) -> List[str]:
        """
        Get all node IDs, optionally filtered by node_type.

        Args:
            node_type: Optional node type filter (e.g., "robot", "object", "region")

        Returns:
            List of node IDs
        """
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
        """
        Save graph to GraphML format for offline analysis.

        Converts tuple attributes to strings for GraphML compatibility.

        Args:
            path: File path to save (e.g., "knowledge_graph.graphml")

        Example:
            >>> kg.save_graphml("session_graph.graphml")
            # Can be opened in Gephi, Cytoscape, or other graph tools
        """
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
        """
        Load graph from GraphML format.

        Args:
            path: File path to load
        """
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

    def save_png(self, path: str, title: str = "Knowledge Graph",
                 dpi: Optional[int] = None,
                 figsize: Optional[tuple] = None) -> None:
        """
        Render the graph to a PNG using matplotlib.

        DPI and figsize default to config.KnowledgeGraph.KG_VIZ_DPI /
        KG_VIZ_FIGSIZE when not supplied.

        Node colour encodes type: robot=steelblue, object=coral, region=mediumseagreen,
        other=lightgrey. Edge labels show edge_type. Nodes are labelled with their ID.

        Args:
            path: Output file path (e.g. "kg.png").
            title: Figure title.
            dpi: PNG resolution; falls back to KG_VIZ_DPI config value.
            figsize: (width, height) in inches; falls back to KG_VIZ_FIGSIZE config value.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            raise ImportError("matplotlib is required for save_png. pip install matplotlib")

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

        nx.draw_networkx_nodes(graph_copy, pos, node_color=node_colors,
                               node_size=1200, ax=ax, alpha=0.9)
        nx.draw_networkx_labels(graph_copy, pos, font_size=14,
                                font_weight="bold", ax=ax)

        seen_types: set = set()
        for u, v, data in graph_copy.edges(data=True):
            etype = data.get("edge_type", "")
            color = _EDGE_COLORS.get(etype, "#bdc3c7")
            nx.draw_networkx_edges(
                graph_copy, pos, edgelist=[(u, v)],
                edge_color=color, arrows=True,
                arrowsize=15, width=1.5,
                connectionstyle="arc3,rad=0.1", ax=ax,
            )
            seen_types.add(etype)

        handles = [
            mpatches.Patch(color=_EDGE_COLORS.get(t, "#bdc3c7"), label=t)
            for t in sorted(seen_types)
        ]
        for ntype, color in _NODE_COLORS.items():
            handles.append(mpatches.Patch(color=color, label=f"[{ntype}]"))
        ax.legend(handles=handles, loc="lower left", fontsize=12,
                  framealpha=0.8, ncol=2)

        stats = (f"{graph_copy.number_of_nodes()} nodes  "
                 f"{graph_copy.number_of_edges()} edges")
        fig.text(0.5, 0.01, stats, ha="center", fontsize=12, color="grey")

        plt.tight_layout()
        plt.savefig(path, dpi=_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved KG visualisation to {path}")

    def auto_save_png_if_enabled(self, label: str = "") -> None:
        """
        Save timestamped PNG and GraphML snapshots if KG_VIZ_AUTO_SAVE is enabled.

        Writes both formats so kg_inspect can reload the graphml for accurate
        re-rendering and stats, and the PNG can be viewed immediately.

        Args:
            label: Optional suffix appended to the filename (e.g. "world_state").
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
            self.save_png(f"{base}.png", title=f"Knowledge Graph snapshot {label or ts}")
        except Exception as exc:
            logger.warning(f"auto_save_png failed: {exc}")
        try:
            self.save_graphml(f"{base}.graphml")
        except Exception as exc:
            logger.warning(f"auto_save_graphml failed: {exc}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get graph statistics.

        Returns:
            Dictionary with node counts, edge counts, and node type breakdown
        """
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
