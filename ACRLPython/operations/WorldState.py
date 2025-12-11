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
import logging
import threading
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import LLMConfig
from operations.StatusOperations import check_robot_status

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        timestamp: Time of last detection
    """
    object_id: str
    position: Tuple[float, float, float]
    color: str = "unknown"
    object_type: str = "unknown"
    is_graspable: bool = True
    grasped_by: Optional[str] = None
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# World State Manager (Singleton)
# ============================================================================


class WorldState:
    """
    Singleton manager for global world state.

    This class tracks:
    - Robot states with TTL-based caching
    - Detected objects from vision system
    - Workspace allocations for multi-robot coordination
    - In-flight commands for request tracking

    Thread-safe for concurrent access.
    """

    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        """Singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize world state manager."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            # Robot state cache
            self._robot_cache: Dict[str, CachedValue] = {}
            self._robot_states: Dict[str, RobotState] = {}

            # Object tracking
            self._objects: Dict[str, ObjectState] = {}

            # Workspace allocation
            self._workspace_allocations: Dict[str, Optional[str]] = {
                region: None for region in LLMConfig.WORKSPACE_REGIONS.keys()
            }

            # In-flight command tracking
            self._pending_commands: Dict[int, Dict[str, Any]] = {}

            self._initialized = True
            logger.info("WorldState initialized")

    # ========================================================================
    # Robot Status Queries
    # ========================================================================

    def get_robot_status(self, robot_id: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
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

                result = check_robot_status(robot_id, detailed=True, request_id=request_id)

                if result.success:
                    # Note: This returns query_sent status, not actual robot state
                    # In a real system, we'd wait for the response from Unity
                    # For now, cache the acknowledgment
                    status = result.result
                    self._robot_cache[robot_id] = CachedValue(
                        value=status,
                        timestamp=time.time(),
                        ttl=LLMConfig.ROBOT_STATUS_CACHE_TTL
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
            state.position = state_data.get("position", state.position)
            state.rotation = state_data.get("rotation", state.rotation)
            state.target_position = state_data.get("target_position", state.target_position)
            state.target_rotation = state_data.get("target_rotation", state.target_rotation)
            state.gripper_state = state_data.get("gripper_state", state.gripper_state)
            state.is_moving = state_data.get("is_moving", state.is_moving)
            state.is_initialized = state_data.get("is_initialized", state.is_initialized)
            state.joint_angles = state_data.get("joint_angles", state.joint_angles)
            state.timestamp = time.time()

            logger.debug(f"Updated robot state for {robot_id}")

    # ========================================================================
    # Object Tracking
    # ========================================================================

    def update_object_position(self, object_id: str, position: Tuple[float, float, float],
                               color: str = "unknown", object_type: str = "unknown",
                               confidence: float = 1.0):
        """
        Update object position from detection results.

        Args:
            object_id: Unique object identifier
            position: Object position (x, y, z)
            color: Object color
            object_type: Object type
            confidence: Detection confidence
        """
        with self._lock:
            if object_id not in self._objects:
                self._objects[object_id] = ObjectState(
                    object_id=object_id,
                    position=position,
                    color=color,
                    object_type=object_type,
                    confidence=confidence
                )
            else:
                obj = self._objects[object_id]
                obj.position = position
                obj.color = color
                obj.object_type = object_type
                obj.confidence = confidence
                obj.timestamp = time.time()

            logger.debug(f"Updated object {object_id} at {position}")

    def get_object_position(self, object_id: str) -> Optional[Tuple[float, float, float]]:
        """
        Get object position.

        Args:
            object_id: Object identifier

        Returns:
            Position tuple (x, y, z) or None if not found
        """
        with self._lock:
            obj = self._objects.get(object_id)
            return obj.position if obj else None

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

    # ========================================================================
    # Workspace Allocation
    # ========================================================================

    def allocate_workspace(self, region: str, robot_id: str) -> bool:
        """
        Allocate a workspace region to a robot.

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

            current_owner = self._workspace_allocations[region]
            if current_owner is not None and current_owner != robot_id:
                logger.warning(f"Region {region} already allocated to {current_owner}")
                return False

            self._workspace_allocations[region] = robot_id
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

            current_owner = self._workspace_allocations[region]
            if current_owner != robot_id:
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
            return self._workspace_allocations.get(region)

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
                "status": "pending"
            }
            logger.debug(f"Registered command {request_id}")

    def update_command_status(self, request_id: int, status: str, result: Optional[Any] = None):
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
            self._workspace_allocations = {
                region: None for region in LLMConfig.WORKSPACE_REGIONS.keys()
            }
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
