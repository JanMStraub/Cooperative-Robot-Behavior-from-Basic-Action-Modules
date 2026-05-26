#!/usr/bin/env python3
"""High-level spatial reasoning queries over the knowledge graph."""

import math
from typing import List, Dict, Any, Tuple
from .Core import KnowledgeGraph

from core.LoggingSetup import get_logger

logger = get_logger(__name__)


class GraphQueryEngine:
    """High-level query interface combining graph traversals and spatial reasoning."""

    def __init__(self, graph: KnowledgeGraph):
        self._graph = graph
        logger.info("GraphQueryEngine initialized")

    def find_reachable_robots(self, object_id: str) -> List[str]:
        """Find robots that can reach object via reverse CAN_REACH lookup."""
        if not self._graph.has_node(object_id):
            logger.warning(f"Object {object_id} not found in graph")
            return []

        robots = self._graph.get_predecessors(object_id, edge_type="CAN_REACH")

        logger.debug(f"Found {len(robots)} robots that can reach {object_id}")
        return robots

    def find_robots_near(
        self, robot_id: str, max_distance: float = 0.2
    ) -> List[Dict[str, Any]]:
        if not self._graph.has_node(robot_id):
            logger.warning(f"Robot {robot_id} not found in graph")
            return []

        robot_node = self._graph.get_node(robot_id)
        if not robot_node:
            return []
        robot_pos = robot_node.get("position")
        if not robot_pos:
            return []

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

        nearby_robots.sort(key=lambda x: x["distance"])

        logger.debug(f"Found {len(nearby_robots)} robots near {robot_id}")
        return nearby_robots

    def get_handoff_candidates(
        self, robot1: str, robot2: str, object_id: str
    ) -> List[Dict[str, Any]]:
        candidates = []

        robot1_can_reach = self._graph.get_neighbors(robot1, edge_type="CAN_REACH")
        robot2_can_reach = self._graph.get_neighbors(robot2, edge_type="CAN_REACH")

        if object_id not in robot1_can_reach or object_id not in robot2_can_reach:
            logger.debug(f"Not both robots can reach {object_id}")
            return []

        robot1_regions = self._graph.get_neighbors(robot1, edge_type="IN_REGION")
        robot2_regions = self._graph.get_neighbors(robot2, edge_type="IN_REGION")

        all_regions = set(robot1_regions)
        for region in robot1_regions:
            adjacent = self._graph.get_neighbors(region, edge_type="ADJACENT_TO")
            all_regions.update(adjacent)

        for region in robot2_regions:
            adjacent = self._graph.get_neighbors(region, edge_type="ADJACENT_TO")
            all_regions.update(adjacent)

        if "shared_zone" in all_regions:
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
        robot_pos = self._get_robot_position(robot_id)
        if not robot_pos:
            return False

        candidates = self._collect_obstacle_candidates(robot_id)

        ax, ay, az = robot_pos
        bx, by, bz = target
        dx, dy, dz = bx - ax, by - ay, bz - az
        seg_len_sq = dx * dx + dy * dy + dz * dz
        blocking_threshold = 0.05  # 5cm

        for obj_id in candidates:
            obj_node = self._graph.get_node(obj_id)
            if not obj_node:
                continue
            obj_pos = obj_node.get("position")
            if not obj_pos:
                continue

            if self._obstacle_blocks_path(
                obj_pos, robot_pos, target, seg_len_sq, dx, dy, dz, blocking_threshold
            ):
                logger.warning(
                    f"Path blocked: obj={obj_id} pos={obj_pos} "
                    f"robot_pos={robot_pos} target={target}"
                )
                return True

        return False

    def _get_robot_position(self, robot_id: str):
        """Return robot EE position from live WorldState, falling back to KG node."""
        if not self._graph.has_node(robot_id):
            return None
        robot_node = self._graph.get_node(robot_id)
        if not robot_node:
            return None
        try:
            from core.Imports import get_world_state

            ws = get_world_state()
            pos = ws.get_robot_ee_position(robot_id) if ws else None
            return pos or robot_node.get("position")
        except Exception:
            return robot_node.get("position")

    def _collect_obstacle_candidates(self, robot_id: str) -> set:
        """Return object IDs that are potential path obstacles, excluding grasped objects.

        Only objects with a NEAR edge to the robot are checked. Previously, all
        graph objects were included which caused false positives: objects on the
        table that MoveIt's collision avoidance would route around were flagged as
        blocking valid paths, aborting sequences prematurely.
        """
        near_objects = set(self._graph.get_neighbors(robot_id, edge_type="NEAR"))
        grasped = set(self._graph.get_neighbors(robot_id, edge_type="GRASPING"))
        return near_objects - grasped - {robot_id}

    def _obstacle_blocks_path(
        self,
        obj_pos,
        robot_pos,
        target,
        seg_len_sq: float,
        dx: float,
        dy: float,
        dz: float,
        threshold: float,
    ) -> bool:
        """Return True if obj_pos lies within threshold of the robot→target segment.

        Skips objects co-located with robot or target (they are not blocking obstacles).
        Uses point-to-line-segment projection; degenerate (zero-length) segment falls
        back to point distance.
        """
        if math.dist(obj_pos, robot_pos) < threshold:
            return False
        if math.dist(obj_pos, target) < threshold:
            return False

        ax, ay, az = robot_pos
        if seg_len_sq == 0.0:
            dist = math.dist(obj_pos, robot_pos)
        else:
            px, py, pz = obj_pos
            t = ((px - ax) * dx + (py - ay) * dy + (pz - az) * dz) / seg_len_sq
            t = max(0.0, min(1.0, t))
            closest = (ax + t * dx, ay + t * dy, az + t * dz)
            dist = math.dist(obj_pos, closest)

        return dist < threshold

    def get_operation_history(
        self, robot_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Operation history via EXECUTED edges. Not yet tracked; returns empty."""
        if not self._graph.has_node(robot_id):
            logger.warning(f"Robot {robot_id} not found in graph")
            return []

        return []

    def get_objects_in_reach(self, robot_id: str) -> List[Dict[str, Any]]:
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

        objects.sort(key=lambda x: x["distance"] if x["distance"] else float("inf"))

        return objects

    def can_reach_position(
        self,
        robot_id: str,
        position: Tuple[float, float, float],
    ) -> Dict[str, Any]:
        """
        Check whether robot can physically reach an arbitrary world position.

        Combines reach-radius check (MAX_ROBOT_REACH) and path-obstruction check
        (5cm clearance along straight-line path). More general than
        find_reachable_robots() which only works for graph-registered objects.
        """
        try:
            from operations.SpatialPredicates import target_within_reach
        except ImportError:
            logger.debug("SpatialPredicates unavailable; skipping reach radius check")
            target_within_reach = None  # type: ignore[assignment]

        x, y, z = position
        within_reach = True
        reason = ""

        if target_within_reach is not None:
            within_reach, reason = target_within_reach(robot_id, x, y, z)

        path_blocked = self.is_path_blocked(robot_id, position)

        if not within_reach:
            return {
                "reachable": False,
                "reason": reason,
                "within_reach": False,
                "path_blocked": path_blocked,
            }

        if path_blocked:
            return {
                "reachable": False,
                "reason": f"Path to ({x:.3f}, {y:.3f}, {z:.3f}) is blocked by an obstacle",
                "within_reach": True,
                "path_blocked": True,
            }

        return {
            "reachable": True,
            "reason": "",
            "within_reach": True,
            "path_blocked": False,
        }

    def get_graph_stats(self) -> Dict[str, Any]:
        stats = self._graph.get_stats()

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
