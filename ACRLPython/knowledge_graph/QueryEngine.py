#!/usr/bin/env python3
"""
Knowledge Graph Query Engine
=============================

High-level queries over the knowledge graph for spatial reasoning,
multi-hop relationship queries, and operation history tracking.

Provides semantic queries that leverage graph structure:
- Reachability queries (which robots can reach object X?)
- Proximity queries (what's near robot Y?)
- Handoff planning (where can Robot1 and Robot2 meet?)
- Path checking (is path blocked by obstacles?)
- Operation history (what did Robot1 do recently?)
"""

import math
from typing import List, Dict, Any, Tuple
from .Core import KnowledgeGraph

# Configure logging
from core.LoggingSetup import get_logger

logger = get_logger(__name__)


class GraphQueryEngine:
    """
    High-level query interface for knowledge graph.

    Provides semantic queries that combine multiple graph traversals
    and spatial reasoning.
    """

    def __init__(self, graph: KnowledgeGraph):
        """
        Initialize query engine.

        Args:
            graph: KnowledgeGraph instance to query
        """
        self._graph = graph
        logger.info("GraphQueryEngine initialized")

    def find_reachable_robots(self, object_id: str) -> List[str]:
        """
        Find which robots can reach a given object.

        Uses reverse CAN_REACH lookup (finds predecessors with CAN_REACH edge).

        Args:
            object_id: Object identifier

        Returns:
            List of robot IDs that can reach the object

        Example:
            >>> query_engine.find_reachable_robots("RedCube")
            ['Robot1', 'Robot2']
        """
        if not self._graph.has_node(object_id):
            logger.warning(f"Object {object_id} not found in graph")
            return []

        # Get all predecessors with CAN_REACH edge
        robots = self._graph.get_predecessors(object_id, edge_type="CAN_REACH")

        logger.debug(f"Found {len(robots)} robots that can reach {object_id}")
        return robots

    def find_robots_near(
        self, robot_id: str, max_distance: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        Find robots within a distance threshold of a given robot.

        Uses NEAR edges or computes from positions.

        Args:
            robot_id: Robot identifier
            max_distance: Maximum distance threshold in meters

        Returns:
            List of dicts with robot_id and distance

        Example:
            >>> query_engine.find_robots_near("Robot1", max_distance=0.3)
            [{'robot_id': 'Robot2', 'distance': 0.25}]
        """
        if not self._graph.has_node(robot_id):
            logger.warning(f"Robot {robot_id} not found in graph")
            return []

        robot_node = self._graph.get_node(robot_id)
        if not robot_node:
            return []
        robot_pos = robot_node.get("position")
        if not robot_pos:
            return []

        # Find all robots
        all_robots = self._graph.get_all_nodes(node_type="robot")
        nearby_robots = []

        for other_robot_id in all_robots:
            if other_robot_id == robot_id:
                continue

            other_node = self._graph.get_node(other_robot_id)
            if not other_node:
                continue
            other_pos = other_node.get("position")
            if not other_pos:
                continue

            distance = math.dist(robot_pos, other_pos)
            if distance <= max_distance:
                nearby_robots.append({"robot_id": other_robot_id, "distance": distance})

        # Sort by distance
        nearby_robots.sort(key=lambda x: x["distance"])

        logger.debug(f"Found {len(nearby_robots)} robots near {robot_id}")
        return nearby_robots

    def get_handoff_candidates(
        self, robot1: str, robot2: str, object_id: str
    ) -> List[Dict[str, Any]]:
        """
        Find positions where two robots can both reach an object for handoff.

        Checks for positions in shared regions where both robots can reach.

        Args:
            robot1: First robot ID
            robot2: Second robot ID
            object_id: Object to be handed off

        Returns:
            List of candidate handoff positions with metadata

        Example:
            >>> query_engine.get_handoff_candidates("Robot1", "Robot2", "RedCube")
            [{'position': (0.0, 0.3, 0.0), 'region': 'shared_zone', 'r1_dist': 0.4, 'r2_dist': 0.4}]
        """
        candidates = []

        # Check if both robots can reach the object
        robot1_can_reach = self._graph.get_neighbors(robot1, edge_type="CAN_REACH")
        robot2_can_reach = self._graph.get_neighbors(robot2, edge_type="CAN_REACH")

        if object_id not in robot1_can_reach or object_id not in robot2_can_reach:
            logger.debug(f"Not both robots can reach {object_id}")
            return []

        # Get shared zones (regions accessible by both)
        robot1_regions = self._graph.get_neighbors(robot1, edge_type="IN_REGION")
        robot2_regions = self._graph.get_neighbors(robot2, edge_type="IN_REGION")

        # Find adjacent regions
        all_regions = set(robot1_regions)
        for region in robot1_regions:
            adjacent = self._graph.get_neighbors(region, edge_type="ADJACENT_TO")
            all_regions.update(adjacent)

        for region in robot2_regions:
            adjacent = self._graph.get_neighbors(region, edge_type="ADJACENT_TO")
            all_regions.update(adjacent)

        # Check shared_zone specifically (common handoff region)
        if "shared_zone" in all_regions:
            # Get object position as potential handoff point
            obj_node = self._graph.get_node(object_id)
            obj_pos = obj_node.get("position") if obj_node else None

            if obj_pos:
                robot1_node = self._graph.get_node(robot1)
                robot2_node = self._graph.get_node(robot2)
                if not robot1_node or not robot2_node:
                    return candidates
                robot1_pos = robot1_node.get("position")
                robot2_pos = robot2_node.get("position")

                if robot1_pos and robot2_pos:
                    candidates.append(
                        {
                            "position": obj_pos,
                            "region": "shared_zone",
                            "r1_distance": math.dist(robot1_pos, obj_pos),
                            "r2_distance": math.dist(robot2_pos, obj_pos),
                        }
                    )

        logger.debug(f"Found {len(candidates)} handoff candidates for {object_id}")
        return candidates

    def is_path_blocked(
        self, robot_id: str, target: Tuple[float, float, float]
    ) -> bool:
        """
        Check if a straight-line path from robot to target is blocked by objects.

        Uses NEAR edges of the robot (and object nodes near the target) as
        candidates, then applies the point-to-line-segment formula to determine
        whether any candidate lies within 5cm of the path.

        The segment formula is used instead of a midpoint check to avoid false
        negatives for objects near the start or end of long paths.

        Args:
            robot_id: Robot identifier
            target: Target position (x, y, z)

        Returns:
            True if path appears blocked
        """
        if not self._graph.has_node(robot_id):
            return False

        robot_node = self._graph.get_node(robot_id)
        if not robot_node:
            return False
        robot_pos = robot_node.get("position")
        if not robot_pos:
            return False

        # Collect candidate obstacle nodes: NEAR neighbors of the robot plus all
        # object nodes (to catch obstacles near the far end of a long path that
        # may not be NEAR the robot start position).
        near_objects = set(self._graph.get_neighbors(robot_id, edge_type="NEAR"))
        all_objects = set(self._graph.get_all_nodes(node_type="object"))
        candidates = near_objects | all_objects

        # Exclude objects the robot is currently grasping — they travel with it.
        grasped = set(self._graph.get_neighbors(robot_id, edge_type="GRASPING"))

        # Precompute segment vector for point-to-segment projection
        ax, ay, az = robot_pos
        bx, by, bz = target
        dx, dy, dz = bx - ax, by - ay, bz - az
        seg_len_sq = dx * dx + dy * dy + dz * dz

        blocking_threshold = 0.05  # 5cm

        for obj_id in candidates:
            if obj_id == robot_id:
                continue

            # Skip objects being carried by this robot.
            if obj_id in grasped:
                continue

            # Skip objects that are at (or very near) the target itself —
            # the robot is moving *toward* them, not being blocked by them.
            obj_node = self._graph.get_node(obj_id)
            if not obj_node:
                continue
            obj_pos = obj_node.get("position")
            if not obj_pos:
                continue

            # Skip objects at the target destination — the robot is moving toward
            # them, not being blocked by them.
            if math.dist(obj_pos, target) < blocking_threshold:
                continue

            # Point-to-line-segment distance:
            # Project obj_pos onto the segment [robot_pos, target], clamp t to [0,1],
            # compute the closest point on the segment, measure distance.
            if seg_len_sq == 0.0:
                # Degenerate segment: robot is at target
                dist = math.dist(obj_pos, robot_pos)
            else:
                px, py, pz = obj_pos
                t = ((px - ax) * dx + (py - ay) * dy + (pz - az) * dz) / seg_len_sq
                t = max(0.0, min(1.0, t))
                closest = (ax + t * dx, ay + t * dy, az + t * dz)
                dist = math.dist(obj_pos, closest)

            if dist < blocking_threshold:
                logger.debug(f"Path blocked by {obj_id} (dist={dist:.4f}m)")
                return True

        return False

    def get_operation_history(
        self, robot_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent operations executed by a robot.

        Uses EXECUTED edges (Robot -> Operation) with timestamps.

        Args:
            robot_id: Robot identifier
            limit: Maximum number of operations to return

        Returns:
            List of operation dicts sorted by timestamp (recent first)

        Note:
            Requires Phase 6 (temporal history) to be implemented.
            Returns empty list if operation tracking is not enabled.
        """
        if not self._graph.has_node(robot_id):
            logger.warning(f"Robot {robot_id} not found in graph")
            return []

        # Get operations executed by robot
        operation_nodes = self._graph.get_neighbors(robot_id, edge_type="EXECUTED")

        # This feature requires operation nodes to be tracked
        # For now, return empty (Phase 6 implementation)
        logger.debug("Operation history tracking requires Phase 6 implementation")
        return []

    def get_objects_in_reach(self, robot_id: str) -> List[Dict[str, Any]]:
        """
        Get all objects reachable by a robot with metadata.

        Args:
            robot_id: Robot identifier

        Returns:
            List of dicts with object_id, distance, color, stale status

        Example:
            >>> query_engine.get_objects_in_reach("Robot1")
            [
                {'object_id': 'RedCube', 'distance': 0.5, 'color': 'red', 'stale': False},
                {'object_id': 'BlueCube', 'distance': 0.6, 'color': 'blue', 'stale': False}
            ]
        """
        if not self._graph.has_node(robot_id):
            return []

        reachable_obj_ids = self._graph.get_neighbors(robot_id, edge_type="CAN_REACH")
        robot_node = self._graph.get_node(robot_id)
        robot_pos = robot_node.get("position") if robot_node else None

        objects = []
        for obj_id in reachable_obj_ids:
            obj_node = self._graph.get_node(obj_id)
            if not obj_node:
                continue

            obj_pos = obj_node.get("position")
            distance = (
                math.dist(robot_pos, obj_pos) if (robot_pos and obj_pos) else None
            )

            objects.append(
                {
                    "object_id": obj_id,
                    "distance": distance,
                    "color": obj_node.get("color", "unknown"),
                    "stale": obj_node.get("stale", False),
                    "grasped_by": obj_node.get("grasped_by"),
                }
            )

        # Sort by distance
        objects.sort(key=lambda x: x["distance"] if x["distance"] else float("inf"))

        return objects

    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive graph statistics.

        Returns:
            Dict with node/edge counts, connectivity metrics, etc.
        """
        stats = self._graph.get_stats()

        # Add edge type breakdown
        robots = self._graph.get_all_nodes(node_type="robot")
        objects = self._graph.get_all_nodes(node_type="object")
        regions = self._graph.get_all_nodes(node_type="region")

        edge_counts = {
            "CAN_REACH": 0,
            "NEAR": 0,
            "IN_REGION": 0,
            "GRASPING": 0,
            "ALLOCATED": 0,
            "ADJACENT_TO": 0,
        }

        for robot_id in robots:
            edge_counts["CAN_REACH"] += len(
                self._graph.get_neighbors(robot_id, edge_type="CAN_REACH")
            )
            edge_counts["GRASPING"] += len(
                self._graph.get_neighbors(robot_id, edge_type="GRASPING")
            )
            edge_counts["IN_REGION"] += len(
                self._graph.get_neighbors(robot_id, edge_type="IN_REGION")
            )

        for obj_id in objects:
            edge_counts["NEAR"] += len(
                self._graph.get_neighbors(obj_id, edge_type="NEAR")
            )
            edge_counts["IN_REGION"] += len(
                self._graph.get_neighbors(obj_id, edge_type="IN_REGION")
            )

        for region_id in regions:
            edge_counts["ALLOCATED"] += len(
                self._graph.get_neighbors(region_id, edge_type="ALLOCATED")
            )
            edge_counts["ADJACENT_TO"] += len(
                self._graph.get_neighbors(region_id, edge_type="ADJACENT_TO")
            )

        stats["edge_types"] = edge_counts

        return stats
