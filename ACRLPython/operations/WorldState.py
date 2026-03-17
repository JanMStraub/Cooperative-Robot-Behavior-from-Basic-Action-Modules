#!/usr/bin/env python3
"""
World State Tracking System
============================

This module provides a centralized world state manager for tracking robot
positions, object locations, workspace allocations, and in-flight commands.

Features:
- TTL-based caching for robot status queries
- Singleton pattern for global state access
- Thread-safe operations
- Workspace allocation management
- Object tracking from vision system
"""

import time
import threading
import math
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

try:
    from config.Robot import (
        WORKSPACE_REGIONS,
        ROBOT_STATUS_CACHE_TTL,
        WORKSPACE_ALLOCATION_TIMEOUT,
        CONFIDENCE_DECAY_PER_FRAME,
        STALE_CONFIDENCE_THRESHOLD,
        OBJECT_TTL_SECONDS,
    )
except ImportError:
    from ..config.Robot import (
        WORKSPACE_REGIONS,
        ROBOT_STATUS_CACHE_TTL,
        WORKSPACE_ALLOCATION_TIMEOUT,
        CONFIDENCE_DECAY_PER_FRAME,
        STALE_CONFIDENCE_THRESHOLD,
        OBJECT_TTL_SECONDS,
    )
from .StatusOperations import check_robot_status

# Configure logging
from core.LoggingSetup import get_logger
from core.SingletonBase import SingletonBase

logger = get_logger(__name__)


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class CachedValue:
    """
    TTL-based cached value for robot status queries.

    Attributes:
        value: Cached data
        timestamp: Time when value was cached (defaults to current time)
        ttl: Time-to-live in seconds
    """

    value: Any
    ttl: float
    timestamp: float = field(default_factory=time.time)

    def is_valid(self) -> bool:
        """Check if cached value is still valid."""
        age = time.time() - self.timestamp
        return age < self.ttl

    def get(self) -> Optional[Any]:
        """Get value if valid, None if expired."""
        return self.value if self.is_valid() else None


@dataclass
class RobotState:
    """
    Complete state of a robot.

    Attributes:
        robot_id: Robot identifier
        position: End effector position (x, y, z) in world coordinates
        rotation: End effector rotation (roll, pitch, yaw) in degrees
        target_position: Target position for movement (x, y, z)
        target_rotation: Target rotation for movement
        gripper_state: "open", "closed", or "unknown"
        is_moving: True if robot is currently moving
        is_initialized: True if robot is initialized and ready
        joint_angles: List of joint angles in radians
        timestamp: Time of last update
    """

    robot_id: str
    position: Optional[Tuple[float, float, float]] = None
    rotation: Optional[Tuple[float, float, float]] = None
    target_position: Optional[Tuple[float, float, float]] = None
    target_rotation: Optional[Tuple[float, float, float]] = None
    gripper_state: str = "unknown"
    is_moving: bool = False
    is_initialized: bool = False
    joint_angles: Optional[list[float]] = None
    start_joint_angles: Optional[list[float]] = None  # Saved at registration; radians, ROS convention
    timestamp: float = field(default_factory=time.time)


@dataclass
class ObjectState:
    """
    State of a detected object.

    Attributes:
        object_id: Unique object identifier
        position: Object position (x, y, z) in world coordinates
        color: Object color (e.g., "red", "blue", "green")
        object_type: Object type (e.g., "cube", "sphere")
        is_graspable: True if object can be grasped
        grasped_by: Robot ID if currently grasped, None otherwise
        confidence: Detection confidence (0.0 - 1.0)
        dimensions: Object dimensions (width, height, depth) in meters
        timestamp: Time of last detection
        last_seen: Time when object was last seen (for liveness tracking)
        stale: True if confidence < threshold (indicates potentially outdated state)
    """

    object_id: str
    position: Tuple[float, float, float]
    color: str = "unknown"
    object_type: str = "unknown"
    is_graspable: bool = True
    grasped_by: Optional[str] = None
    confidence: float = 1.0
    dimensions: Optional[Tuple[float, float, float]] = None
    timestamp: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    stale: bool = False


@dataclass
class WorkspaceAllocation:
    """
    Workspace region allocation with timeout tracking.

    Attributes:
        robot_id: Robot that owns this workspace region
        region: Workspace region name
        allocated_at: Time when allocation occurred
    """

    robot_id: str
    region: str
    allocated_at: float = field(default_factory=time.time)


# ============================================================================
# World State Manager (Singleton)
# ============================================================================


class WorldState(SingletonBase):
    """
    Singleton manager for global world state.

    This class tracks:
    - Robot states with TTL-based caching
    - Detected objects from vision system
    - Workspace allocations for multi-robot coordination
    - In-flight commands for request tracking

    Thread-safe for concurrent access.
    """

    @classmethod
    def get_instance(cls):
        """
        Get the singleton instance of WorldState.

        Returns:
            WorldState singleton instance
        """
        return cls()

    def _singleton_init(self):
        """Initialize world state manager (called once by SingletonBase)."""
        self._lock = threading.RLock()

        # Robot state cache
        self._robot_cache: Dict[str, CachedValue] = {}
        self._robot_states: Dict[str, RobotState] = {}

        # Object tracking
        self._objects: Dict[str, ObjectState] = {}
        # Cache of {normalized_key: original_key} for O(1) partial-match
        # lookups. Invalidated whenever _objects changes.
        self._normalized_object_keys: Optional[Dict[str, str]] = None

        # Workspace allocation with timeout tracking
        self._workspace_allocations: Dict[str, Optional[WorkspaceAllocation]] = {
            region: None for region in WORKSPACE_REGIONS.keys()
        }
        self._workspace_timeout = WORKSPACE_ALLOCATION_TIMEOUT

        # In-flight command tracking
        self._pending_commands: Dict[int, Dict[str, Any]] = {}

        logger.info("WorldState initialized")

    # ========================================================================
    # Robot Status Queries
    # ========================================================================

    def get_robot_status(
        self, robot_id: str, force_refresh: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get robot status with TTL-based caching.

        Args:
            robot_id: Robot identifier
            force_refresh: If True, bypass cache and query Unity

        Returns:
            Robot status dict or None if unavailable
        """
        with self._lock:
            # Check cache first
            if not force_refresh and robot_id in self._robot_cache:
                cached = self._robot_cache[robot_id]
                if cached.is_valid():
                    logger.debug(f"Using cached status for {robot_id}")
                    return cached.get()

            # Query Unity for fresh status
            logger.debug(f"Querying Unity for {robot_id} status")
            try:
                # Generate request ID for tracking
                request_id = int(time.time() * 1000) % (2**32)

                result = check_robot_status(
                    robot_id, detailed=True, request_id=request_id
                )

                if result.success:
                    # Note: This returns query_sent status, not actual robot state
                    # In a real system, we'd wait for the response from Unity
                    # For now, cache the acknowledgment
                    status = result.result
                    self._robot_cache[robot_id] = CachedValue(
                        value=status,
                        timestamp=time.time(),
                        ttl=ROBOT_STATUS_CACHE_TTL,
                    )
                    return status
                else:
                    logger.warning(f"Failed to query {robot_id}: {result.error}")
                    return None

            except Exception as e:
                logger.error(f"Error querying robot status: {e}")
                return None

    def get_robot_position(self, robot_id: str) -> Optional[Tuple[float, float, float]]:
        """
        Get robot end effector position (cached).

        Args:
            robot_id: Robot identifier

        Returns:
            Position tuple (x, y, z) or None if unavailable
        """
        # First check if we have a robot state with position
        with self._lock:
            if robot_id in self._robot_states:
                robot_state = self._robot_states[robot_id]
                if robot_state.position is not None:
                    return robot_state.position

        # Fall back to querying status
        status = self.get_robot_status(robot_id)
        if status is None:
            return None

        # Extract position from status (if available)
        # Note: Actual position extraction depends on Unity response format
        return status.get("position")

    def get_robot_position_fresh(
        self, robot_id: str, max_age: float = 1.0
    ) -> Optional[Tuple[float, float, float]]:
        """
        Get robot end effector position with freshness guarantee.
        Forces refresh if cached data is older than max_age.

        Args:
            robot_id: Robot identifier
            max_age: Maximum age in seconds for cached position (default 1s for collision checks)

        Returns:
            Position tuple (x, y, z) or None if unavailable
        """
        with self._lock:
            # Check if we have a recent robot state with position
            if robot_id in self._robot_states:
                robot_state = self._robot_states[robot_id]
                age = time.time() - robot_state.timestamp
                if robot_state.position is not None and age < max_age:
                    logger.debug(
                        f"Using fresh position for {robot_id} (age: {age:.3f}s)"
                    )
                    return robot_state.position

        # Force refresh if cached data is stale
        logger.debug(f"Forcing position refresh for {robot_id} (max_age: {max_age}s)")
        status = self.get_robot_status(robot_id, force_refresh=True)
        if status is None:
            return None

        # Extract position from status (if available)
        return status.get("position")

    def get_robot_target(self, robot_id: str) -> Optional[Tuple[float, float, float]]:
        """
        Get robot movement target position.

        Args:
            robot_id: Robot identifier

        Returns:
            Target position tuple (x, y, z) or None if unavailable
        """
        with self._lock:
            robot_state = self._robot_states.get(robot_id)
            if robot_state:
                return robot_state.target_position
            return None

    @staticmethod
    def _to_position_tuple(value) -> Optional[Tuple[float, float, float]]:
        """
        Normalize a position value to a (float, float, float) tuple.

        Unity serializes Vector3 as {"x": ..., "y": ..., "z": ...}.  This
        helper converts that dict form to the tuple form expected everywhere
        else in the codebase.  Plain tuples/lists pass through unchanged.

        Args:
            value: Position as dict {"x","y","z"}, list, tuple, or None.

        Returns:
            (float, float, float) tuple, or the original value if conversion
            is not possible (so existing None defaults are preserved).
        """
        if isinstance(value, dict):
            return (
                float(value.get("x", 0.0)),
                float(value.get("y", 0.0)),
                float(value.get("z", 0.0)),
            )
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return (float(value[0]), float(value[1]), float(value[2]))
        return None

    def update_robot_state(self, robot_id: str, state_data: Dict[str, Any]):
        """
        Update robot state from Unity response.

        Args:
            robot_id: Robot identifier
            state_data: State data dict from Unity
        """
        with self._lock:
            if robot_id not in self._robot_states:
                self._robot_states[robot_id] = RobotState(robot_id=robot_id)

            state = self._robot_states[robot_id]
            state.position = self._to_position_tuple(
                state_data.get("position", state.position)
            )
            state.rotation = state_data.get("rotation", state.rotation)
            state.target_position = self._to_position_tuple(
                state_data.get("target_position", state.target_position)
            )
            state.target_rotation = state_data.get(
                "target_rotation", state.target_rotation
            )
            state.gripper_state = state_data.get("gripper_state", state.gripper_state)
            state.is_moving = state_data.get("is_moving", state.is_moving)
            state.is_initialized = state_data.get(
                "is_initialized", state.is_initialized
            )
            state.joint_angles = state_data.get("joint_angles", state.joint_angles)
            state.start_joint_angles = state_data.get("start_joint_angles", state.start_joint_angles)
            state.timestamp = time.time()

            logger.debug(f"Updated robot state for {robot_id}")

    def update_robot(
        self,
        robot_id: str,
        position: Optional[Tuple[float, float, float]] = None,
        rotation: Optional[Tuple[float, float, float]] = None,
        joint_angles: Optional[list[float]] = None,
        is_moving: Optional[bool] = None,
        **kwargs,
    ):
        """
        Update robot state (simplified interface for tests).

        Args:
            robot_id: Robot identifier
            position: Robot position (x, y, z)
            rotation: Robot rotation (roll, pitch, yaw)
            joint_angles: List of joint angles
            is_moving: True if robot is moving
            **kwargs: Additional state data
        """
        state_data = {}
        if position is not None:
            state_data["position"] = position
        if rotation is not None:
            state_data["rotation"] = rotation
        if joint_angles is not None:
            state_data["joint_angles"] = joint_angles
        if is_moving is not None:
            state_data["is_moving"] = is_moving
        state_data.update(kwargs)

        self.update_robot_state(robot_id, state_data)

    def get_robot_state(self, robot_id: str) -> Optional[RobotState]:
        """
        Get robot state object.

        Args:
            robot_id: Robot identifier

        Returns:
            RobotState object or None if not found
        """
        with self._lock:
            return self._robot_states.get(robot_id)

    # ========================================================================
    # Object Tracking
    # ========================================================================

    def update_object_position(
        self,
        object_id: str,
        position: Tuple[float, float, float],
        color: str = "unknown",
        object_type: str = "unknown",
        confidence: float = 1.0,
        dimensions: Optional[Tuple[float, float, float]] = None,
    ):
        """
        Update object position from detection results.

        Args:
            object_id: Unique object identifier
            position: Object position (x, y, z)
            color: Object color
            object_type: Object type
            confidence: Detection confidence
            dimensions: Optional object dimensions (width, height, depth) in meters
        """
        with self._lock:
            if object_id not in self._objects:
                self._objects[object_id] = ObjectState(
                    object_id=object_id,
                    position=position,
                    color=color,
                    object_type=object_type,
                    confidence=confidence,
                    dimensions=dimensions,
                )
            else:
                obj = self._objects[object_id]
                obj.position = position
                obj.color = color
                obj.object_type = object_type
                obj.confidence = confidence
                obj.dimensions = dimensions
                obj.timestamp = time.time()

            logger.debug(f"Updated object {object_id} at {position}")
            # Invalidate normalized key cache on any structural change
            self._normalized_object_keys = None

    def _get_normalized_keys(self) -> Dict[str, str]:
        """
        Return (building if necessary) the normalized-key-to-original-key cache.

        Must be called under self._lock.

        Returns:
            Dict mapping normalized key strings to their original _objects keys.
        """
        if self._normalized_object_keys is None:
            self._normalized_object_keys = {
                k.lower().replace(" ", "_").replace("-", "_"): k for k in self._objects
            }
        return self._normalized_object_keys

    def get_object_state(self, object_id: str) -> Optional[Dict[str, Any]]:
        """
        Get object state as a dictionary (compatibility method).

        Uses the same partial-match fallback as get_object_position so that
        compound names like "red_cube" resolve to an object stored as "red".

        Args:
            object_id: Object identifier

        Returns:
            Dict with object state data, or None if not found
        """
        with self._lock:
            obj = self._objects.get(object_id)
            if obj is None:
                normalised = object_id.lower().replace(" ", "_").replace("-", "_")
                norm_cache = self._get_normalized_keys()
                # Exact normalized match first, then substring fallback
                original_key = norm_cache.get(normalised)
                if original_key is None:
                    for key_norm, orig in norm_cache.items():
                        if key_norm in normalised or normalised in key_norm:
                            original_key = orig
                            break
                if original_key is not None:
                    obj = self._objects.get(original_key)
            if obj is None:
                return None
            return {
                "position": (
                    {"x": obj.position[0], "y": obj.position[1], "z": obj.position[2]}
                    if obj.position
                    else None
                ),
                "color": obj.color,
                "object_type": obj.object_type,
                "is_graspable": obj.is_graspable,
                "grasped_by": obj.grasped_by,
                "confidence": obj.confidence,
            }

    def get_object_position(
        self, object_id: str
    ) -> Optional[Tuple[float, float, float]]:
        """
        Get object position.

        Performs an exact key lookup first. If that fails, falls back to
        partial matching so that compound names like "red_cube" or "red cube"
        still resolve to an object stored under the key "red" (as written by
        VisionOperations which uses just the color as the key).

        Args:
            object_id: Object identifier (exact key, color, or compound like "red_cube")

        Returns:
            Position tuple (x, y, z) or None if not found
        """
        with self._lock:
            # 1. Exact match
            obj = self._objects.get(object_id)
            if obj:
                return obj.position

            # 2. Normalise: replace spaces/hyphens with underscores, lowercase
            normalised = object_id.lower().replace(" ", "_").replace("-", "_")

            # 3. Use cached normalized keys for O(1) exact then substring match
            norm_cache = self._get_normalized_keys()
            original_key = norm_cache.get(normalised)
            if original_key is None:
                for key_norm, orig in norm_cache.items():
                    if key_norm in normalised or normalised in key_norm:
                        original_key = orig
                        break

            if original_key is not None:
                logger.debug(
                    f"get_object_position: resolved '{object_id}' → '{original_key}' via partial match"
                )
                return self._objects[original_key].position

            return None

    def get_object_dimensions(
        self, object_id: str
    ) -> Optional[Tuple[float, float, float]]:
        """
        Get object dimensions.

        Uses the same partial-match fallback as get_object_position so that
        compound names like "red_cube" resolve to an object stored as "red".

        Args:
            object_id: Object identifier

        Returns:
            Dimensions tuple (width, height, depth) in meters or None if not found
        """
        with self._lock:
            obj = self._objects.get(object_id)
            if obj:
                return obj.dimensions

            normalised = object_id.lower().replace(" ", "_").replace("-", "_")
            norm_cache = self._get_normalized_keys()
            original_key = norm_cache.get(normalised)
            if original_key is None:
                for key_norm, orig in norm_cache.items():
                    if key_norm in normalised or normalised in key_norm:
                        original_key = orig
                        break

            if original_key is not None:
                return self._objects[original_key].dimensions

            return None

    def get_objects_by_color(self, color: str) -> list[ObjectState]:
        """
        Get all objects of a specific color.

        Args:
            color: Color to filter by

        Returns:
            List of ObjectState instances
        """
        with self._lock:
            return [obj for obj in self._objects.values() if obj.color == color]

    def mark_object_grasped(self, object_id: str, robot_id: str):
        """
        Mark an object as grasped by a robot.

        Args:
            object_id: Object identifier
            robot_id: Robot that grasped the object
        """
        with self._lock:
            if object_id in self._objects:
                self._objects[object_id].grasped_by = robot_id
                logger.info(f"Object {object_id} grasped by {robot_id}")

    def mark_object_released(self, object_id: str):
        """
        Mark an object as released (no longer grasped).

        Args:
            object_id: Object identifier
        """
        with self._lock:
            if object_id in self._objects:
                self._objects[object_id].grasped_by = None
                logger.info(f"Object {object_id} released")

    def register_object(
        self,
        object_id: str,
        object_type: str = "unknown",
        position: Tuple[float, float, float] = (0, 0, 0),
        graspable: bool = True,
        **kwargs,
    ):
        """
        Register a new object (simplified interface for tests).

        Args:
            object_id: Unique object identifier
            object_type: Type of object (e.g., "cube")
            position: Object position (x, y, z)
            graspable: True if object can be grasped
            **kwargs: Additional object properties
        """
        color = kwargs.get("color", "unknown")
        confidence = kwargs.get("confidence", 1.0)
        self.update_object_position(object_id, position, color, object_type, confidence)

    def get_all_objects(self) -> list[ObjectState]:
        """
        Get all registered objects.

        Returns:
            List of all ObjectState instances
        """
        with self._lock:
            return list(self._objects.values())

    def decay_object_confidence(self, seen_object_ids: set[str]):
        """
        Update object confidence based on recent detections.

        Called after each detection frame. Decays confidence for objects
        not seen in this frame and refreshes those that were seen. Objects
        with very low confidence or not seen for a long time are removed.

        Args:
            seen_object_ids: Set of object IDs detected in current frame
        """
        with self._lock:
            now = time.time()
            to_delete = []

            for obj_id, obj in self._objects.items():
                if obj_id in seen_object_ids:
                    # Object was seen - refresh confidence
                    obj.confidence = 1.0
                    obj.last_seen = now
                    obj.stale = False
                    obj.timestamp = now
                else:
                    # Object not seen - decay confidence.
                    # Round to 10 decimal places to prevent floating-point
                    # accumulation errors from repeated subtraction.
                    obj.confidence = round(
                        max(0.0, obj.confidence - CONFIDENCE_DECAY_PER_FRAME), 10
                    )
                    obj.stale = obj.confidence < STALE_CONFIDENCE_THRESHOLD

                    # Mark for deletion if TTL exceeded
                    if now - obj.last_seen > OBJECT_TTL_SECONDS:
                        to_delete.append(obj_id)

            # Remove stale objects
            for obj_id in to_delete:
                logger.debug(
                    f"Removing stale object {obj_id} (not seen for {OBJECT_TTL_SECONDS}s)"
                )
                del self._objects[obj_id]

            if to_delete:
                self._normalized_object_keys = None
                logger.info(f"Removed {len(to_delete)} stale objects from world state")

    def find_objects_near(
        self,
        position: Tuple[float, float, float],
        radius: float = 0.1,
        exclude_stale: bool = True,
    ) -> list[ObjectState]:
        """
        Find all objects within radius of a position.

        Uses simple Euclidean distance calculation (sufficient for <50 objects).

        Args:
            position: Center position (x, y, z) to search from
            radius: Search radius in meters (default 0.1m)
            exclude_stale: If True, exclude objects marked as stale (default True)

        Returns:
            List of ObjectState instances within radius
        """
        with self._lock:
            nearby = []
            for obj in self._objects.values():
                if exclude_stale and obj.stale:
                    continue

                # Calculate Euclidean distance
                distance = math.dist(position, obj.position)
                if distance <= radius:
                    nearby.append(obj)

            return nearby

    def find_robots_near(
        self, position: Tuple[float, float, float], radius: float = 0.2
    ) -> list[RobotState]:
        """
        Find all robots within radius of a position.

        Args:
            position: Center position (x, y, z) to search from
            radius: Search radius in meters (default 0.2m)

        Returns:
            List of RobotState instances within radius
        """
        with self._lock:
            nearby = []
            for robot in self._robot_states.values():
                if robot.position is None:
                    continue

                # Calculate Euclidean distance
                distance = math.dist(position, robot.position)
                if distance <= radius:
                    nearby.append(robot)

            return nearby

    def get_reachable_objects(
        self, robot_id: str, exclude_stale: bool = True
    ) -> list[ObjectState]:
        """
        Get all objects reachable by a robot.

        Uses spatial predicates to determine reachability:
        - target_within_reach: Distance from robot base within MAX_ROBOT_REACH
        - object_accessible_by_robot: Within workspace and collision-free

        Args:
            robot_id: Robot identifier
            exclude_stale: If True, exclude objects marked as stale (default True)

        Returns:
            List of reachable ObjectState instances
        """
        with self._lock:
            # Import here to avoid circular dependencies
            try:
                from .SpatialPredicates import (
                    target_within_reach,
                    object_accessible_by_robot,
                )
            except ImportError:
                from operations.SpatialPredicates import (
                    target_within_reach,
                    object_accessible_by_robot,
                )

            reachable = []
            for obj in self._objects.values():
                if exclude_stale and obj.stale:
                    continue

                # Check if target is within reach
                x, y, z = obj.position
                is_reachable, _ = target_within_reach(
                    robot_id, x, y, z, world_state=self
                )

                if is_reachable:
                    # Additional check: object must be accessible (workspace, no collision)
                    is_accessible, _ = object_accessible_by_robot(
                        robot_id, obj.position, world_state=self
                    )
                    if is_accessible:
                        reachable.append(obj)

            return reachable

    def get_objects_in_region(
        self, region: str, exclude_stale: bool = True
    ) -> list[ObjectState]:
        """
        Get all objects in a workspace region.

        Args:
            region: Region name (e.g., "left_workspace", "shared_zone")
            exclude_stale: If True, exclude objects marked as stale (default True)

        Returns:
            List of ObjectState instances in the region
        """
        with self._lock:
            if region not in WORKSPACE_REGIONS:
                logger.warning(f"Unknown workspace region: {region}")
                return []

            bounds = WORKSPACE_REGIONS[region]
            objects_in_region = []

            for obj in self._objects.values():
                if exclude_stale and obj.stale:
                    continue

                # Check if object position is within region bounds
                x, y, z = obj.position
                if (
                    bounds["x_min"] <= x <= bounds["x_max"]
                    and bounds["y_min"] <= y <= bounds["y_max"]
                    and bounds["z_min"] <= z <= bounds["z_max"]
                ):
                    objects_in_region.append(obj)

            return objects_in_region

    def get_region_for_position(
        self, position: Tuple[float, float, float]
    ) -> Optional[str]:
        """
        Get which workspace region contains a position.

        Args:
            position: Position (x, y, z) to check

        Returns:
            Region name or None if position is outside all regions
        """
        x, y, z = position

        for region, bounds in WORKSPACE_REGIONS.items():
            if (
                bounds["x_min"] <= x <= bounds["x_max"]
                and bounds["y_min"] <= y <= bounds["y_max"]
                and bounds["z_min"] <= z <= bounds["z_max"]
            ):
                return region

        return None

    def get_world_context_string(self, robot_id: str) -> str:
        """
        Generate a natural language context string for LLM consumption.

        Provides robot state and annotated object list with spatial relationships.

        Args:
            robot_id: Robot identifier for context perspective

        Returns:
            Formatted context string with robot state and object annotations

        Example:
            "Robot1 at (-0.3, 0.2, 0.1), gripper open. Objects: RedCube at (0.1, 0.3, 0.0)
            [reachable, in shared_zone], BlueCube at (0.4, 0.2, 0.1) [not reachable,
            in right_workspace]."
        """
        with self._lock:
            # Get robot state
            robot = self._robot_states.get(robot_id)
            if robot is None or robot.position is None:
                return f"{robot_id} state unknown."

            # Format robot state
            pos = robot.position
            gripper = robot.gripper_state
            context = f"{robot_id} at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), gripper {gripper}."

            # Get reachable objects for this robot
            reachable_ids = {
                obj.object_id for obj in self.get_reachable_objects(robot_id)
            }

            # Format object list
            if not self._objects:
                context += " No objects detected."
                return context

            context += " Objects: "
            obj_descriptions = []

            for obj in self._objects.values():
                obj_pos = obj.position
                desc = f"{obj.object_id} at ({obj_pos[0]:.2f}, {obj_pos[1]:.2f}, {obj_pos[2]:.2f})"

                # Add annotations
                annotations = []
                if obj.object_id in reachable_ids:
                    annotations.append("reachable")
                else:
                    annotations.append("not reachable")

                region = self.get_region_for_position(obj_pos)
                if region:
                    annotations.append(f"in {region}")

                if obj.stale:
                    annotations.append("stale")

                if obj.grasped_by:
                    annotations.append(f"grasped by {obj.grasped_by}")

                desc += f" [{', '.join(annotations)}]"
                obj_descriptions.append(desc)

            context += ", ".join(obj_descriptions) + "."
            return context

    # ========================================================================
    # Workspace Allocation
    # ========================================================================

    def allocate_workspace(self, region: str, robot_id: str) -> bool:
        """
        Allocate a workspace region to a robot with timeout tracking.

        Args:
            region: Region name (e.g., "left_workspace")
            robot_id: Robot identifier

        Returns:
            True if allocation successful, False if already allocated
        """
        with self._lock:
            if region not in self._workspace_allocations:
                logger.warning(f"Unknown workspace region: {region}")
                return False

            # Cleanup stale allocations first
            self._cleanup_stale_allocations()

            current_allocation = self._workspace_allocations[region]
            if (
                current_allocation is not None
                and current_allocation.robot_id != robot_id
            ):
                logger.warning(
                    f"Region {region} already allocated to {current_allocation.robot_id}"
                )
                return False

            self._workspace_allocations[region] = WorkspaceAllocation(
                robot_id=robot_id, region=region
            )
            logger.info(f"Allocated {region} to {robot_id}")
            return True

    def release_workspace(self, region: str, robot_id: str) -> bool:
        """
        Release a workspace region allocation.

        Args:
            region: Region name
            robot_id: Robot identifier (must match current owner)

        Returns:
            True if release successful, False otherwise
        """
        with self._lock:
            if region not in self._workspace_allocations:
                logger.warning(f"Unknown workspace region: {region}")
                return False

            current_allocation = self._workspace_allocations[region]
            if current_allocation is None:
                logger.warning(f"Region {region} is not allocated")
                return False

            if current_allocation.robot_id != robot_id:
                logger.warning(f"Region {region} not allocated to {robot_id}")
                return False

            self._workspace_allocations[region] = None
            logger.info(f"Released {region} from {robot_id}")
            return True

    def get_workspace_owner(self, region: str) -> Optional[str]:
        """
        Get the robot that owns a workspace region.

        Args:
            region: Region name

        Returns:
            Robot ID or None if not allocated
        """
        with self._lock:
            self._cleanup_stale_allocations()
            allocation = self._workspace_allocations.get(region)
            return allocation.robot_id if allocation else None

    def _cleanup_stale_allocations(self):
        """
        Cleanup stale workspace allocations that have exceeded timeout.
        Called automatically by allocate_workspace.
        """
        now = time.time()
        stale_regions = []

        for region, allocation in self._workspace_allocations.items():
            if allocation is not None:
                age = now - allocation.allocated_at
                if age > self._workspace_timeout:
                    stale_regions.append(region)
                    logger.warning(
                        f"Auto-releasing stale allocation: {region} from {allocation.robot_id} (age: {age:.1f}s)"
                    )

        # Release stale allocations
        for region in stale_regions:
            self._workspace_allocations[region] = None

    def set_workspace_timeout(self, timeout: float):
        """
        Set workspace allocation timeout.

        Args:
            timeout: Timeout in seconds (default 60s)
        """
        with self._lock:
            self._workspace_timeout = max(1.0, timeout)
            logger.info(f"Set workspace timeout to {self._workspace_timeout}s")

    # ========================================================================
    # Command Tracking
    # ========================================================================

    def register_command(self, request_id: int, command: Dict[str, Any]):
        """
        Register an in-flight command for tracking.

        Args:
            request_id: Unique request identifier
            command: Command data dict
        """
        with self._lock:
            self._pending_commands[request_id] = {
                "command": command,
                "timestamp": time.time(),
                "status": "pending",
            }
            logger.debug(f"Registered command {request_id}")

    def update_command_status(
        self, request_id: int, status: str, result: Optional[Any] = None
    ):
        """
        Update status of a tracked command.

        Args:
            request_id: Request identifier
            status: New status (e.g., "completed", "failed")
            result: Optional result data
        """
        with self._lock:
            if request_id in self._pending_commands:
                self._pending_commands[request_id]["status"] = status
                self._pending_commands[request_id]["result"] = result
                self._pending_commands[request_id]["completion_time"] = time.time()
                logger.debug(f"Command {request_id} status: {status}")

    def get_command_status(self, request_id: int) -> Optional[Dict[str, Any]]:
        """
        Get status of a tracked command.

        Args:
            request_id: Request identifier

        Returns:
            Command status dict or None if not found
        """
        with self._lock:
            return self._pending_commands.get(request_id)

    def cleanup_old_commands(self, max_age_seconds: float = 300.0):
        """
        Remove old completed commands from tracking.

        Args:
            max_age_seconds: Maximum age for completed commands
        """
        with self._lock:
            now = time.time()
            to_remove = []

            for request_id, cmd_data in self._pending_commands.items():
                if cmd_data["status"] in ["completed", "failed"]:
                    age = now - cmd_data.get("completion_time", cmd_data["timestamp"])
                    if age > max_age_seconds:
                        to_remove.append(request_id)

            for request_id in to_remove:
                del self._pending_commands[request_id]

            if to_remove:
                logger.debug(f"Cleaned up {len(to_remove)} old commands")

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def clear_cache(self):
        """Clear all cached robot statuses."""
        with self._lock:
            self._robot_cache.clear()
            logger.info("Cleared robot status cache")

    def reset(self):
        """Reset all world state (for testing)."""
        with self._lock:
            self._robot_cache.clear()
            self._robot_states.clear()
            self._objects.clear()
            self._normalized_object_keys = None

            # Reinitialize workspace allocations
            self._workspace_allocations = {
                region: None for region in WORKSPACE_REGIONS.keys()
            }
            self._workspace_timeout = WORKSPACE_ALLOCATION_TIMEOUT  # Reset to default
            self._pending_commands.clear()
            logger.info("Reset world state")


# ============================================================================
# Global Instance
# ============================================================================


def get_world_state() -> WorldState:
    """
    Get the global WorldState singleton instance.

    Returns:
        WorldState instance
    """
    return WorldState()
