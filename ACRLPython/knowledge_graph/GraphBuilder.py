#!/usr/bin/env python3
"""Builds and maintains the knowledge graph from WorldState updates."""

import math
import time
from typing import Dict, Any
from operations.WorldState import WorldState
from operations.SpatialPredicates import object_accessible_by_robot
from config.Robot import WORKSPACE_REGIONS, ROBOT_BASE_POSITIONS
from .Core import KnowledgeGraph
from .Schema import RobotNode, ObjectNode, RegionNode

from core.LoggingSetup import get_logger

logger = get_logger(__name__)

# Configuration constants
from config.KnowledgeGraph import KG_NEAR_THRESHOLD as NEAR_THRESHOLD


class GraphBuilder:
    """Builds and maintains knowledge graph from WorldState updates via callback."""

    def __init__(self, graph: KnowledgeGraph, world_state: WorldState):
        self._graph = graph
        self._world_state = world_state
        self._init_static_regions()
        self._seed_robot_nodes()
        logger.info("GraphBuilder initialized")

    def _init_static_regions(self):
        """Create static region nodes and ADJACENT_TO edges once at init."""
        for region_name, bounds in WORKSPACE_REGIONS.items():
            region_node = RegionNode(node_id=region_name, bounds=bounds)
            self._graph.add_node(region_node.node_id, **region_node.to_dict())
            logger.debug(f"Added static region node: {region_name}")

        adjacencies = [
            ("left_workspace", "shared_zone"),
            ("shared_zone", "right_workspace"),
            # Center is adjacent to shared_zone
            ("center", "shared_zone"),
        ]

        for region1, region2 in adjacencies:
            if region1 in WORKSPACE_REGIONS and region2 in WORKSPACE_REGIONS:
                self._graph.add_edge(region1, region2, "ADJACENT_TO")
                self._graph.add_edge(region2, region1, "ADJACENT_TO")

    def _seed_robot_nodes(self):
        """
        Pre-populate robot nodes from ROBOT_BASE_POSITIONS at init time.

        Eliminates the race where QueryEngine warns about missing robot nodes
        before the first WorldState packet arrives from Unity. Base positions
        serve as placeholders until on_state_update overwrites them with real
        positions.
        """
        for robot_id, base_pos in ROBOT_BASE_POSITIONS.items():
            robot_state = self._world_state.get_robot_state(robot_id)
            if robot_state and robot_state.position:
                position = robot_state.position
            else:
                position = (base_pos[0], base_pos[1], base_pos[2])
            robot_node = RobotNode(
                node_id=robot_id,
                position=position,
                workspace_region=None,
                gripper_state="unknown",
                is_moving=False,
                timestamp=time.time(),
            )
            self._graph.add_node(robot_node.node_id, **robot_node.to_dict())
            logger.debug(f"Seeded robot node at startup: {robot_id}")

    def on_object_updated(self, object_id: str, position: tuple) -> None:
        """
        Lightweight callback fired by WorldState after a Python detection write.

        Avoids full graph rebuild: refreshes single object node + spatial edges,
        keeping CAN_REACH/NEAR current without waiting for next Unity packet.
        """
        try:
            obj_data = self._world_state.get_object_state(object_id)
            if not obj_data:
                return

            raw_pos = obj_data.get("position")
            if isinstance(raw_pos, dict):
                raw_pos = (raw_pos["x"], raw_pos["y"], raw_pos["z"])
            object_node = ObjectNode(
                node_id=object_id,
                position=raw_pos,
                color=obj_data.get("color", "unknown"),
                object_type=obj_data.get("object_type", "unknown"),
                is_graspable=obj_data.get("is_graspable", False),
                grasped_by=obj_data.get("grasped_by"),
                confidence=obj_data.get("confidence", 1.0),
                stale=obj_data.get("stale", False),
                timestamp=obj_data.get("timestamp", time.time()),
            )
            self._graph.add_node(object_node.node_id, **object_node.to_dict())
            self._recompute_spatial_edges()
            logger.debug(f"KG updated from Python detection: {object_id} at {position}")
        except Exception as e:
            logger.error(
                f"Error in on_object_updated for {object_id}: {e}", exc_info=True
            )

    def on_state_update(self, state_data: Dict[str, Any]):
        try:
            self._update_robot_nodes(state_data)
            self._update_object_nodes(state_data)
            self._recompute_spatial_edges()
            self._update_grasp_edges()
            self._update_allocation_edges()
            logger.debug("Graph updated from world state")
            self._graph.auto_save_png_if_enabled("world_state")

        except Exception as e:
            logger.error(f"Error updating graph from state: {e}", exc_info=True)

    def _update_robot_nodes(self, state_data: Dict[str, Any]):
        robots = state_data.get("robots", [])

        for robot_data in robots:
            robot_id = robot_data.get("robot_id")
            if not robot_id:
                continue

            robot_state = self._world_state.get_robot_state(robot_id)
            if not robot_state:
                continue

            workspace_region = None
            if robot_state.position:
                workspace_region = self._world_state.get_region_for_position(
                    robot_state.position
                )

            robot_node = RobotNode(
                node_id=robot_id,
                position=robot_state.position,
                workspace_region=workspace_region,
                gripper_state=robot_state.gripper_state,
                is_moving=robot_state.is_moving,
                timestamp=robot_state.timestamp,
            )
            self._graph.add_node(robot_node.node_id, **robot_node.to_dict())

    def _update_object_nodes(self, state_data: Dict[str, Any]):
        objects = state_data.get("objects", [])
        seen_object_ids = set()
        # Build lookup once - O(N) total instead of O(N²)
        world_objs_by_id = {o.object_id: o for o in self._world_state.get_all_objects()}

        for obj_data in objects:
            object_id = obj_data.get("object_id")
            if not object_id:
                continue

            # Skip robot IDs - robots appear in the Unity objects array but
            # are tracked as robot nodes, not object nodes.
            if object_id in ROBOT_BASE_POSITIONS:
                continue

            seen_object_ids.add(object_id)

            obj_state = world_objs_by_id.get(object_id)

            if not obj_state:
                continue

            object_node = ObjectNode(
                node_id=object_id,
                position=obj_state.position,
                color=obj_state.color,
                object_type=obj_state.object_type,
                is_graspable=obj_state.is_graspable,
                grasped_by=obj_state.grasped_by,
                confidence=obj_state.confidence,
                stale=obj_state.stale,
                timestamp=obj_state.timestamp,
            )
            self._graph.add_node(object_node.node_id, **object_node.to_dict())

        # Remove objects no longer in WorldState (TTL expired)
        current_object_nodes = self._graph.get_all_nodes(node_type="object")
        for obj_id in current_object_nodes:
            if obj_id not in seen_object_ids:
                self._graph.remove_node(obj_id)
                logger.debug(f"Removed stale object node: {obj_id}")

    @staticmethod
    def _to_pos_tuple(pos):
        """Normalize position to (x, y, z) tuple regardless of dict/tuple/list input."""
        if isinstance(pos, dict):
            return (pos["x"], pos["y"], pos["z"])
        return pos

    def _recompute_spatial_edges(self):
        robots = self._graph.get_all_nodes(node_type="robot")
        objects = self._graph.get_all_nodes(node_type="object")

        # Single locked batch removal avoids O(N²) per-pair remove_edge calls.
        self._graph.remove_edges_by_type(
            nodes=robots + objects, edge_types={"CAN_REACH", "NEAR"}
        )

        self._compute_can_reach_edges(robots, objects)
        self._compute_near_edges(robots, objects)
        self._compute_in_region_edges(robots, objects)

    def _compute_can_reach_edges(self, robots: list, objects: list) -> None:
        """Add CAN_REACH edges from each robot to objects within accessible reach."""
        for robot_id in robots:
            robot_node = self._graph.get_node(robot_id)
            if not robot_node:
                continue
            robot_pos = self._to_pos_tuple(robot_node.get("position"))
            if not robot_pos:
                continue

            for obj_id in objects:
                obj_node = self._graph.get_node(obj_id)
                if not obj_node:
                    continue
                obj_pos = self._to_pos_tuple(obj_node.get("position"))
                if not obj_pos:
                    continue

                is_accessible, _ = object_accessible_by_robot(
                    robot_id, obj_pos, world_state=self._world_state
                )
                if is_accessible:
                    distance = math.dist(robot_pos, obj_pos)
                    self._graph.add_edge(
                        robot_id,
                        obj_id,
                        "CAN_REACH",
                        distance=distance,
                        approach_direction=None,
                    )

    def _compute_near_edges(self, robots: list, objects: list) -> None:
        """Add bidirectional NEAR edges between robots/objects within NEAR_THRESHOLD."""
        for robot_id in robots:
            robot_node = self._graph.get_node(robot_id)
            if not robot_node:
                continue
            robot_pos = self._to_pos_tuple(robot_node.get("position"))
            if not robot_pos:
                continue

            for obj_id in objects:
                obj_node = self._graph.get_node(obj_id)
                if not obj_node:
                    continue
                obj_pos = self._to_pos_tuple(obj_node.get("position"))
                if not obj_pos:
                    continue

                distance = math.dist(robot_pos, obj_pos)
                if distance < NEAR_THRESHOLD:
                    self._graph.add_edge(robot_id, obj_id, "NEAR", distance=distance)
                    self._graph.add_edge(obj_id, robot_id, "NEAR", distance=distance)

        for i, obj1_id in enumerate(objects):
            obj1_node = self._graph.get_node(obj1_id)
            if not obj1_node:
                continue
            obj1_pos = self._to_pos_tuple(obj1_node.get("position"))
            if not obj1_pos:
                continue

            for obj2_id in objects[i + 1 :]:
                obj2_node = self._graph.get_node(obj2_id)
                if not obj2_node:
                    continue
                obj2_pos = self._to_pos_tuple(obj2_node.get("position"))
                if not obj2_pos:
                    continue

                distance = math.dist(obj1_pos, obj2_pos)
                if distance < NEAR_THRESHOLD:
                    self._graph.add_edge(obj1_id, obj2_id, "NEAR", distance=distance)
                    self._graph.add_edge(obj2_id, obj1_id, "NEAR", distance=distance)

    def _compute_in_region_edges(self, robots: list, objects: list) -> None:
        """Update IN_REGION edges for all robots and objects based on current positions."""
        regions = list(WORKSPACE_REGIONS.keys())

        for node_id in robots + objects:
            node = self._graph.get_node(node_id)
            if not node:
                continue
            pos = self._to_pos_tuple(node.get("position"))
            if not pos:
                continue
            region = self._world_state.get_region_for_position(pos)
            if region:
                for old_region in regions:
                    self._graph.remove_edge(node_id, old_region, edge_type="IN_REGION")
                self._graph.add_edge(node_id, region, "IN_REGION")

    def _update_grasp_edges(self):
        robots = self._graph.get_all_nodes(node_type="robot")
        objects = self._graph.get_all_nodes(node_type="object")

        for robot_id in robots:
            for obj_id in objects:
                self._graph.remove_edge(robot_id, obj_id, edge_type="GRASPING")

        for obj_id in objects:
            obj_node = self._graph.get_node(obj_id)
            if not obj_node:
                continue
            grasped_by = obj_node.get("grasped_by")

            if grasped_by:
                self._graph.add_edge(
                    grasped_by, obj_id, "GRASPING", grasp_time=time.time()
                )

    def _update_allocation_edges(self):
        regions = list(WORKSPACE_REGIONS.keys())
        robots = self._graph.get_all_nodes(node_type="robot")

        for region in regions:
            for robot_id in robots:
                self._graph.remove_edge(region, robot_id, edge_type="ALLOCATED")

        for region in regions:
            owner = self._world_state.get_workspace_owner(region)
            if owner:
                self._graph.add_edge(
                    region, owner, "ALLOCATED", allocated_at=time.time()
                )
