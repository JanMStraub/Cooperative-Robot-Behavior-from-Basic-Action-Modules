"""
Knowledge Graph Builder
=======================

Builds and maintains the knowledge graph from WorldState updates.
Registers as a WorldStateServer callback to keep graph synchronized.

Key Responsibilities:
- Create/update robot, object, and region nodes from WorldState
- Compute and maintain spatial edges (CAN_REACH, NEAR, IN_REGION)
- Update temporal edges (GRASPING, ALLOCATED)
- Recompute edges when entities move
"""

import math
import time
from typing import Dict, Any
from operations.WorldState import WorldState
from operations.SpatialPredicates import object_accessible_by_robot
from config.Robot import WORKSPACE_REGIONS
from .Core import KnowledgeGraph
from .Schema import RobotNode, ObjectNode, RegionNode

# Configure logging
from core.LoggingSetup import get_logger
logger = get_logger(__name__)

# Configuration constants
NEAR_THRESHOLD = 0.1  # meters - objects closer than this are considered "NEAR"


class GraphBuilder:
    """
    Builds and maintains knowledge graph from WorldState updates.

    Automatically updates graph structure when WorldState changes via callback.
    """

    def __init__(self, graph: KnowledgeGraph, world_state: WorldState):
        """
        Initialize GraphBuilder.

        Args:
            graph: KnowledgeGraph instance to populate
            world_state: WorldState instance to read from
        """
        self._graph = graph
        self._world_state = world_state
        self._init_static_regions()
        logger.info("GraphBuilder initialized")

    def _init_static_regions(self):
        """
        Create static region nodes and ADJACENT_TO edges.

        Regions are static (don't move), so create them once at initialization.
        """
        # Add region nodes
        for region_name, bounds in WORKSPACE_REGIONS.items():
            region_node = RegionNode(node_id=region_name, bounds=bounds)
            self._graph.add_node(region_node.node_id, **region_node.to_dict())
            logger.debug(f"Added static region node: {region_name}")

        # Add ADJACENT_TO edges between neighboring regions
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
                logger.debug(f"Added ADJACENT_TO edge: {region1} <-> {region2}")

    def on_state_update(self, state_data: Dict[str, Any]):
        """
        Callback invoked by WorldStateServer on each state update.

        Rebuilds graph nodes and edges based on current world state.

        Args:
            state_data: World state update dictionary from Unity
        """
        try:
            self._update_robot_nodes(state_data)
            self._update_object_nodes(state_data)
            self._recompute_spatial_edges()
            self._update_grasp_edges()
            self._update_allocation_edges()
            logger.debug("Graph updated from world state")

        except Exception as e:
            logger.error(f"Error updating graph from state: {e}", exc_info=True)

    def _update_robot_nodes(self, state_data: Dict[str, Any]):
        """
        Update robot nodes from state data.

        Creates or updates nodes for each robot in the state.

        Args:
            state_data: World state dictionary
        """
        robots = state_data.get("robots", [])

        for robot_data in robots:
            robot_id = robot_data.get("robot_id")
            if not robot_id:
                continue

            # Get robot from WorldState for full data
            robot_state = self._world_state.get_robot_state(robot_id)
            if not robot_state:
                continue

            # Determine workspace region
            workspace_region = None
            if robot_state.position:
                workspace_region = self._world_state.get_region_for_position(robot_state.position)

            # Create node
            robot_node = RobotNode(
                node_id=robot_id,
                position=robot_state.position,
                workspace_region=workspace_region,
                gripper_state=robot_state.gripper_state,
                is_moving=robot_state.is_moving,
                timestamp=robot_state.timestamp,
            )

            # Add or update node
            self._graph.add_node(robot_node.node_id, **robot_node.to_dict())

    def _update_object_nodes(self, state_data: Dict[str, Any]):
        """
        Update object nodes from state data.

        Args:
            state_data: World state dictionary
        """
        objects = state_data.get("objects", [])

        # Track seen objects for cleanup
        seen_object_ids = set()

        for obj_data in objects:
            object_id = obj_data.get("object_id")
            if not object_id:
                continue

            seen_object_ids.add(object_id)

            # Get object from WorldState for full data
            world_objs = self._world_state.get_all_objects()
            obj_state = next((o for o in world_objs if o.object_id == object_id), None)

            if not obj_state:
                continue

            # Create node
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

            # Add or update node
            self._graph.add_node(object_node.node_id, **object_node.to_dict())

        # Remove objects that are no longer in WorldState (TTL expired)
        current_object_nodes = self._graph.get_all_nodes(node_type="object")
        for obj_id in current_object_nodes:
            if obj_id not in seen_object_ids:
                self._graph.remove_node(obj_id)
                logger.debug(f"Removed stale object node: {obj_id}")

    def _recompute_spatial_edges(self):
        """
        Recompute all spatial edges based on current positions.

        Edges computed:
        - CAN_REACH: Robot -> Object (if within reach and accessible)
        - NEAR: Object <-> Object, Robot <-> Object (if distance < threshold)
        - IN_REGION: Robot/Object -> Region (based on position)
        """
        # Clear old spatial edges
        robots = self._graph.get_all_nodes(node_type="robot")
        objects = self._graph.get_all_nodes(node_type="object")

        # Remove old spatial edges in a single locked batch operation to avoid
        # the O(N²) overhead of per-pair remove_edge calls.
        self._graph.remove_edges_by_type(
            nodes=robots + objects, edge_types={"CAN_REACH", "NEAR"}
        )

        # Recompute CAN_REACH edges (Robot -> Object)
        for robot_id in robots:
            robot_node = self._graph.get_node(robot_id)
            if not robot_node:
                continue
            robot_pos = robot_node.get("position")

            if not robot_pos:
                continue

            for obj_id in objects:
                obj_node = self._graph.get_node(obj_id)
                if not obj_node:
                    continue
                obj_pos = obj_node.get("position")

                if not obj_pos:
                    continue

                # Check if object is accessible
                is_accessible, _ = object_accessible_by_robot(
                    robot_id, obj_pos, world_state=self._world_state
                )

                if is_accessible:
                    distance = math.dist(robot_pos, obj_pos)
                    self._graph.add_edge(
                        robot_id, obj_id, "CAN_REACH",
                        distance=distance,
                        approach_direction=None  # Could compute from positions
                    )

        # Recompute NEAR edges (proximity)
        # Robot <-> Object
        for robot_id in robots:
            robot_node = self._graph.get_node(robot_id)
            if not robot_node:
                continue
            robot_pos = robot_node.get("position")
            if not robot_pos:
                continue

            for obj_id in objects:
                obj_node = self._graph.get_node(obj_id)
                if not obj_node:
                    continue
                obj_pos = obj_node.get("position")
                if not obj_pos:
                    continue

                distance = math.dist(robot_pos, obj_pos)
                if distance < NEAR_THRESHOLD:
                    self._graph.add_edge(robot_id, obj_id, "NEAR", distance=distance)
                    self._graph.add_edge(obj_id, robot_id, "NEAR", distance=distance)

        # Object <-> Object
        for i, obj1_id in enumerate(objects):
            obj1_node = self._graph.get_node(obj1_id)
            if not obj1_node:
                continue
            obj1_pos = obj1_node.get("position")
            if not obj1_pos:
                continue

            for obj2_id in objects[i+1:]:
                obj2_node = self._graph.get_node(obj2_id)
                if not obj2_node:
                    continue
                obj2_pos = obj2_node.get("position")
                if not obj2_pos:
                    continue

                distance = math.dist(obj1_pos, obj2_pos)
                if distance < NEAR_THRESHOLD:
                    self._graph.add_edge(obj1_id, obj2_id, "NEAR", distance=distance)
                    self._graph.add_edge(obj2_id, obj1_id, "NEAR", distance=distance)

        # Recompute IN_REGION edges
        regions = list(WORKSPACE_REGIONS.keys())

        for robot_id in robots:
            robot_node = self._graph.get_node(robot_id)
            if robot_node:
                robot_pos = robot_node.get("position")
                if robot_pos:
                    region = self._world_state.get_region_for_position(robot_pos)
                    if region:
                        # Remove old IN_REGION edges
                        for old_region in regions:
                            self._graph.remove_edge(robot_id, old_region, edge_type="IN_REGION")
                        # Add new edge
                        self._graph.add_edge(robot_id, region, "IN_REGION")

        for obj_id in objects:
            obj_node = self._graph.get_node(obj_id)
            if obj_node:
                obj_pos = obj_node.get("position")
                if obj_pos:
                    region = self._world_state.get_region_for_position(obj_pos)
                    if region:
                        # Remove old IN_REGION edges
                        for old_region in regions:
                            self._graph.remove_edge(obj_id, old_region, edge_type="IN_REGION")
                        # Add new edge
                        self._graph.add_edge(obj_id, region, "IN_REGION")

    def _update_grasp_edges(self):
        """
        Update GRASPING edges based on object grasp state.

        GRASPING edge: Robot -> Object (when robot has object grasped)
        """
        robots = self._graph.get_all_nodes(node_type="robot")
        objects = self._graph.get_all_nodes(node_type="object")

        # Clear old GRASPING edges
        for robot_id in robots:
            for obj_id in objects:
                self._graph.remove_edge(robot_id, obj_id, edge_type="GRASPING")

        # Add new GRASPING edges based on object state
        for obj_id in objects:
            obj_node = self._graph.get_node(obj_id)
            if not obj_node:
                continue
            grasped_by = obj_node.get("grasped_by")

            if grasped_by:
                self._graph.add_edge(
                    grasped_by, obj_id, "GRASPING",
                    grasp_time=time.time()
                )

    def _update_allocation_edges(self):
        """
        Update ALLOCATED edges based on workspace allocation.

        ALLOCATED edge: Region -> Robot (when region is allocated to robot)
        """
        regions = list(WORKSPACE_REGIONS.keys())
        robots = self._graph.get_all_nodes(node_type="robot")

        # Clear old ALLOCATED edges
        for region in regions:
            for robot_id in robots:
                self._graph.remove_edge(region, robot_id, edge_type="ALLOCATED")

        # Add new ALLOCATED edges
        for region in regions:
            owner = self._world_state.get_workspace_owner(region)
            if owner:
                self._graph.add_edge(
                    region, owner, "ALLOCATED",
                    allocated_at=time.time()
                )
