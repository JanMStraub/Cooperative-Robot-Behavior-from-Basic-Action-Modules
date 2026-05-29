#!/usr/bin/env python3
"""Thread-safe singleton for tracking robot positions, object locations, workspace allocations, and in-flight commands."""

import time
import threading
import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    from config.Robot import (
        WORKSPACE_REGIONS,
        ROBOT_STATUS_CACHE_TTL,
        CONFIDENCE_DECAY_PER_FRAME,
        STALE_CONFIDENCE_THRESHOLD,
        OBJECT_TTL_SECONDS,
        JOINT_MOVEMENT_THRESHOLD,
    )
    from config.Robot import ROBOT_BASE_POSITIONS
except ImportError:
    from ..config.Robot import (
        WORKSPACE_REGIONS,
        ROBOT_STATUS_CACHE_TTL,
        CONFIDENCE_DECAY_PER_FRAME,
        STALE_CONFIDENCE_THRESHOLD,
        OBJECT_TTL_SECONDS,
        JOINT_MOVEMENT_THRESHOLD,
    )
    from ..config.Robot import ROBOT_BASE_POSITIONS
from .StatusOperations import check_robot_status

from core.LoggingSetup import get_logger
from core.SingletonBase import SingletonBase

logger = get_logger(__name__)


@dataclass
class CachedValue:
    value: Any
    ttl: float
    timestamp: float = field(default_factory=time.time)

    def is_valid(self) -> bool:
        return (time.time() - self.timestamp) < self.ttl

    def get(self) -> Optional[Any]:
        return self.value if self.is_valid() else None


@dataclass
class RobotState:
    robot_id: str
    position: Optional[Tuple[float, float, float]] = None
    rotation: Optional[Tuple[float, float, float]] = None
    target_position: Optional[Tuple[float, float, float]] = None
    target_rotation: Optional[Tuple[float, float, float]] = None
    gripper_state: str = "unknown"
    is_moving: bool = False
    is_initialized: bool = False
    joint_angles: Optional[list[float]] = None
    start_joint_angles: Optional[list[float]] = (
        None  # Saved at registration; radians, ROS convention
    )
    proximity_frozen: bool = (
        False  # True when Unity ProximityGuard has halted this robot
    )
    moving_toward_object: Optional[str] = (
        None  # object_id this robot is currently targeting
    )
    workspace_intent: Optional[str] = (
        None  # workspace region this robot intends to enter
    )
    timestamp: float = field(default_factory=time.time)


@dataclass
class ObjectState:
    object_id: str
    position: Tuple[float, float, float]
    color: str = "unknown"
    object_type: str = "unknown"
    is_graspable: bool = True
    grasped_by: Optional[str] = None
    confidence: float = 1.0
    dimensions: Optional[Tuple[float, float, float]] = None
    rotation: Optional[Tuple[float, float, float]] = None
    timestamp: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    stale: bool = False
    source: str = "unity"  # "vision" or "unity"; tracks which system last set position


try:
    from .WorkspaceAllocator import WorkspaceAllocation, WorkspaceAllocator  # type: ignore[import]
except ImportError:
    from operations.WorkspaceAllocator import WorkspaceAllocation, WorkspaceAllocator  # type: ignore[no-redef]


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
        """Get the singleton instance of WorldState."""
        return cls()

    def _singleton_init(self):
        """Initialize world state manager (called once by SingletonBase)."""
        self._lock = threading.RLock()
        self._robot_cache: Dict[str, CachedValue] = {}
        self._robot_states: Dict[str, RobotState] = {}

        self._objects: Dict[str, ObjectState] = {}
        # Cache of {normalized_key: original_key} for O(1) partial-match
        # lookups. Invalidated whenever _objects changes.
        self._normalized_object_keys: Optional[Dict[str, str]] = None

        self._workspace_allocator = WorkspaceAllocator()

        self._task_outcomes: List[Dict[str, Any]] = []
        self._pending_commands: Dict[int, Dict[str, Any]] = {}
        # Observers notified on every update_object_position write.
        # Each callable receives (object_id: str, position: tuple).
        self._object_observers: List = []

        logger.info("WorldState initialized")

    def get_robot_status(
        self, robot_id: str, force_refresh: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Get robot status with TTL-based caching."""
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

        Priority:
        1. Stored position in robot state (set by Unity stream or FK update).
        2. On-the-fly FK computation from stored joint angles (no side effects).
        3. Fall back to querying Unity status.
        """
        # First check if we have a robot state with position
        with self._lock:
            if robot_id in self._robot_states:
                robot_state = self._robot_states[robot_id]
                if robot_state.position is not None:
                    return robot_state.position

                # FK fallback: compute position on-the-fly from joint angles
                if robot_state.joint_angles and len(robot_state.joint_angles) == 6:
                    try:
                        import math as _math
                        from operations.AR4Kinematics import (
                            compute_end_effector_position,
                        )

                        base_pos = ROBOT_BASE_POSITIONS.get(robot_id)
                        if base_pos is not None:
                            base_yaw = _math.pi if robot_id == "Robot2" else 0.0
                            return compute_end_effector_position(
                                robot_state.joint_angles, base_pos, base_yaw
                            )
                    except Exception as exc:
                        logger.warning(f"FK fallback failed for {robot_id}: {exc}")

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
        """Get robot end effector position; forces refresh if cached data older than max_age."""
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
        """Get robot movement target position."""
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

    @staticmethod
    def _to_rotation_tuple(value) -> Optional[Tuple[float, float, float]]:
        """
        Normalize a rotation value to a (roll, pitch, yaw) tuple in degrees.

        Unity serializes Quaternion as {"x":..., "y":..., "z":..., "w":...}.
        This helper converts that to Euler angles using pure-math intrinsic
        ZXY decomposition (roll=X, pitch=Y, yaw=Z), which matches Unity's
        convention.  Plain tuples/lists of length ≥ 3 are passed through as-is
        (assumed to already be in Euler-degree form).

        Args:
            value: Rotation as dict {"x","y","z","w"}, list, tuple, or None.

        Returns:
            (roll, pitch, yaw) tuple in degrees, or None if not convertible.
        """
        if isinstance(value, dict) and "w" in value:
            x = float(value.get("x", 0.0))
            y = float(value.get("y", 0.0))
            z = float(value.get("z", 0.0))
            w = float(value.get("w", 1.0))
            roll = math.degrees(
                math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
            )
            pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))
            yaw = math.degrees(
                math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            )
            return (roll, pitch, yaw)
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return (float(value[0]), float(value[1]), float(value[2]))
        return None

    def update_robot_state(self, robot_id: str, state_data: Dict[str, Any]):
        """
        Update robot state from Unity response or command-tracked updates.

        When joint_angles are provided and no explicit position is in state_data,
        derives end-effector position via AR4 FK so WorldState stays accurate
        without a Unity WorldStateServer connection.
        """
        with self._lock:
            if robot_id not in self._robot_states:
                self._robot_states[robot_id] = RobotState(robot_id=robot_id)

            state = self._robot_states[robot_id]

            # Store previous joint angles before updating (used for is_moving detection)
            prev_joint_angles = state.joint_angles

            state.position = self._to_position_tuple(
                state_data.get("position", state.position)
            )
            state.rotation = self._to_rotation_tuple(
                state_data.get("rotation", state.rotation)
            )
            state.target_position = self._to_position_tuple(
                state_data.get("target_position", state.target_position)
            )
            state.target_rotation = self._to_rotation_tuple(
                state_data.get("target_rotation", state.target_rotation)
            )
            state.gripper_state = state_data.get("gripper_state", state.gripper_state)
            state.is_moving = state_data.get("is_moving", state.is_moving)
            state.is_initialized = state_data.get(
                "is_initialized", state.is_initialized
            )
            new_joint_angles = state_data.get("joint_angles", None)
            if new_joint_angles is not None:
                state.joint_angles = new_joint_angles
            state.start_joint_angles = state_data.get(
                "start_joint_angles", state.start_joint_angles
            )
            state.proximity_frozen = state_data.get("proximity_frozen", False)
            if "moving_toward_object" in state_data:
                state.moving_toward_object = state_data["moving_toward_object"]
            if "workspace_intent" in state_data:
                state.workspace_intent = state_data["workspace_intent"]
            state.timestamp = time.time()

            # Derive end-effector pose from FK when joint_angles were just updated
            # and no explicit ground-truth position was provided in this update.
            # This makes WorldState self-sufficient when Unity WorldStateServer is absent.
            if new_joint_angles is not None and "position" not in state_data:
                self._update_position_from_fk(
                    robot_id, state, new_joint_angles, prev_joint_angles
                )

            logger.debug(f"Updated robot state for {robot_id}")

    def _update_position_from_fk(
        self,
        robot_id: str,
        state: "RobotState",
        new_joint_angles: list,
        prev_joint_angles,
    ):
        """Compute and store FK-derived end-effector position and is_moving flag.

        Called from update_robot_state when joint_angles are present but no
        explicit position was provided by Unity. Assumes self._lock is held.
        """
        try:
            import math as _math
            from operations.AR4Kinematics import compute_end_effector_pose

            base_pos = ROBOT_BASE_POSITIONS.get(robot_id)
            if base_pos is None:
                logger.debug(
                    f"FK skipped for {robot_id}: no base position in ROBOT_BASE_POSITIONS"
                )
                return

            if len(new_joint_angles) != 6:
                logger.debug(
                    f"FK skipped for {robot_id}: expected 6 joint angles, got {len(new_joint_angles)}"
                )
                return

            # Robot2 is mounted mirrored (180° yaw from Robot1)
            base_yaw = _math.pi if robot_id == "Robot2" else 0.0

            pos, quat = compute_end_effector_pose(new_joint_angles, base_pos, base_yaw)
            state.position = pos
            # Store quaternion as rotation tuple (treated as (x,y,z,w) 4-tuple)
            # existing consumers of state.rotation expect (roll,pitch,yaw) degrees,
            # so we convert here for compatibility.
            qx, qy, qz, qw = quat
            roll = _math.degrees(
                _math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
            )
            pitch = _math.degrees(
                _math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx))))
            )
            yaw = _math.degrees(
                _math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            )
            state.rotation = (roll, pitch, yaw)

            # Derive is_moving from joint angle delta
            if prev_joint_angles and len(prev_joint_angles) == 6:
                delta = sum(
                    abs(a - b) for a, b in zip(new_joint_angles, prev_joint_angles)
                )
                state.is_moving = delta > JOINT_MOVEMENT_THRESHOLD

            logger.debug(
                f"FK pose for {robot_id}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
            )
        except Exception as exc:
            logger.warning(f"FK computation failed for {robot_id}: {exc}")

    def update_robot(
        self,
        robot_id: str,
        position: Optional[Tuple[float, float, float]] = None,
        rotation: Optional[Tuple[float, float, float]] = None,
        joint_angles: Optional[list[float]] = None,
        is_moving: Optional[bool] = None,
        **kwargs,
    ):
        """Update robot state (simplified interface for tests)."""
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
        """Get robot state object."""
        with self._lock:
            return self._robot_states.get(robot_id)

    def get_robot_ee_position(
        self, robot_id: str
    ) -> Optional[Tuple[float, float, float]]:
        """
        Return the live end-effector position for a robot from WorldState.

        Used by QueryEngine.is_path_blocked to prefer the freshest position
        over the potentially stale value stored in the knowledge graph node.
        """
        with self._lock:
            state = self._robot_states.get(robot_id)
            return state.position if state else None

    def update_object_position(
        self,
        object_id: str,
        position: Tuple[float, float, float],
        color: str = "unknown",
        object_type: str = "unknown",
        confidence: float = 1.0,
        dimensions: Optional[Tuple[float, float, float]] = None,
        rotation: Optional[Tuple[float, float, float]] = None,
        source: str = "vision",
    ):
        """Update object position from detection results."""
        with self._lock:
            if object_id not in self._objects:
                self._objects[object_id] = ObjectState(
                    object_id=object_id,
                    position=position,
                    color=color,
                    object_type=object_type,
                    confidence=confidence,
                    dimensions=dimensions,
                    rotation=rotation,
                    source=source,
                )
            else:
                obj = self._objects[object_id]
                obj.position = position
                obj.color = color
                obj.object_type = object_type
                obj.confidence = confidence
                obj.source = source
                # Preserve existing dimensions when the new update has none.
                # Unity streams accurate collider-based dimensions; Python detection
                # (especially cached SharedVisionState) often sends None.
                # Overwriting with None would erase good dimension data.
                if dimensions is not None:
                    obj.dimensions = dimensions
                # Preserve existing rotation if the new update doesn't carry one
                # (vision-detected objects don't have rotation; physics-scene objects do).
                if rotation is not None:
                    obj.rotation = rotation
                obj.timestamp = time.time()

            logger.debug(f"Updated object {object_id} at {position}")
            # Invalidate normalized key cache on any structural change
            self._normalized_object_keys = None
            observers = list(self._object_observers)

        for cb in observers:
            try:
                cb(object_id, position)
            except Exception:
                pass

    def supplement_object_from_unity(
        self,
        object_id: str,
        position: Tuple[float, float, float],
        color: str = "unknown",
        object_type: str = "unknown",
        confidence: float = 1.0,
        dimensions: Optional[Tuple[float, float, float]] = None,
        rotation: Optional[Tuple[float, float, float]] = None,
    ):
        """
        Update object state from Unity, but only as a supplement to vision.

        Creates the object if not yet in WorldState (source="unity").
        If already present (populated by vision), only fills in missing
        dimensions/rotation/color/type — never overwrites position.
        Does not fire object observers since no position change occurs for
        already-tracked objects.
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
                    rotation=rotation,
                    source="unity",
                )
                self._normalized_object_keys = None
                logger.debug(
                    f"Supplemented (created) object {object_id} from Unity at {position}"
                )
            else:
                obj = self._objects[object_id]
                # Fill missing metadata only — never overwrite position set by vision
                if dimensions is not None and obj.dimensions is None:
                    obj.dimensions = dimensions
                if rotation is not None and obj.rotation is None:
                    obj.rotation = rotation
                if obj.color == "unknown" and color != "unknown":
                    obj.color = color
                if obj.object_type == "unknown" and object_type != "unknown":
                    obj.object_type = object_type
                obj.timestamp = time.time()
                logger.debug(
                    f"Supplemented (metadata only) object {object_id} from Unity"
                )

    def register_object_observer(self, callback) -> None:
        """Register a callback invoked after every update_object_position write."""
        with self._lock:
            if callback not in self._object_observers:
                self._object_observers.append(callback)

    @staticmethod
    def _strip_to_alnum(s: str) -> str:
        """Return lowercase string with all non-alphanumeric chars removed.

        'Red Cube' → 'redcube', 'red_cube' → 'redcube', 'redCube' → 'redcube'.
        Used as a last-resort normalization so camelCase, snake_case, and
        space-separated names all resolve to the same canonical form.
        """
        return "".join(c for c in s.lower() if c.isalnum())

    def _get_normalized_keys(self) -> Dict[str, str]:
        """Return (building if necessary) the normalized-key-to-original-key cache. Must be called under self._lock."""
        if self._normalized_object_keys is None:
            self._normalized_object_keys = {
                k.lower().replace(" ", "_").replace("-", "_"): k for k in self._objects
            }
        return self._normalized_object_keys

    def resolve_canonical_id(self, object_id: str) -> Optional[str]:
        """Return the canonical WorldState key for object_id, or None if not found.

        Uses the same resolution order as get_object_position (exact → normalised →
        substring → alnum-strip) but returns the stored key rather than the position.
        Call this before sending object_id to Unity so the command uses the actual
        scene object name instead of the raw LLM-generated identifier.
        """
        with self._lock:
            if object_id in self._objects:
                return object_id

            normalised = object_id.lower().replace(" ", "_").replace("-", "_")
            norm_cache = self._get_normalized_keys()

            original_key = norm_cache.get(normalised)
            if original_key is None:
                for key_norm, orig in norm_cache.items():
                    if key_norm in normalised or normalised in key_norm:
                        original_key = orig
                        break

            if original_key is not None:
                return original_key

            # Last resort: strip all non-alphanumeric chars and compare.
            # Handles camelCase ↔ snake_case ↔ space-separated mismatches,
            # e.g. "redCube" → "redcube" matches stored "Red Cube" → "redcube".
            query_alnum = self._strip_to_alnum(object_id)
            for orig_key in self._objects:
                if self._strip_to_alnum(orig_key) == query_alnum:
                    logger.debug(
                        f"resolve_canonical_id: alnum-strip resolved '{object_id}' → '{orig_key}'"
                    )
                    return orig_key

            return None

    def get_object_state(self, object_id: str) -> Optional[Dict[str, Any]]:
        """
        Get object state as a dictionary (compatibility method).

        Uses the same partial-match fallback as get_object_position so that
        compound names like "red_cube" resolve to an object stored as "red".
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
                "timestamp": obj.timestamp,
            }

    def get_object_rotation(
        self, object_id: str
    ) -> Optional[Tuple[float, float, float]]:
        """Get object rotation (roll, pitch, yaw) in degrees with partial-match fallback.

        Uses the same normalised-key lookup as ``get_object_state`` and
        ``get_object_position`` so compound names like "red_cube" resolve to an
        object stored as "red".  Returns ``None`` when the object is not found or
        has no rotation recorded.

        Args:
            object_id: Object identifier (exact key, color, or compound name).

        Returns:
            Rotation tuple ``(roll_deg, pitch_deg, yaw_deg)`` or ``None``.
        """
        with self._lock:
            obj = self._objects.get(object_id)
            if obj is None:
                normalised = object_id.lower().replace(" ", "_").replace("-", "_")
                norm_cache = self._get_normalized_keys()
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
            return obj.rotation

    def get_object_position(
        self, object_id: str
    ) -> Optional[Tuple[float, float, float]]:
        """
        Get object position.

        Exact key lookup first. Falls back to partial matching so compound names
        like "red_cube" or "red cube" resolve to an object stored as "red" (as
        written by VisionOperations which uses just the color as the key).
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

            # Last resort: strip all non-alphanumeric chars and compare.
            # Handles camelCase ↔ snake_case ↔ space-separated mismatches,
            # e.g. "redCube" → "redcube" matches stored "Red Cube" → "redcube".
            query_alnum = self._strip_to_alnum(object_id)
            for orig_key, obj in self._objects.items():
                if self._strip_to_alnum(orig_key) == query_alnum:
                    logger.debug(
                        f"get_object_position: alnum-strip resolved '{object_id}' → '{orig_key}'"
                    )
                    return obj.position

            return None

    def get_object_dimensions(
        self, object_id: str
    ) -> Optional[Tuple[float, float, float]]:
        """Get object dimensions with same partial-match fallback as get_object_position."""
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

            # Last resort: alnum-strip match (same as get_object_position).
            query_alnum = self._strip_to_alnum(object_id)
            for orig_key, obj in self._objects.items():
                if self._strip_to_alnum(orig_key) == query_alnum:
                    return obj.dimensions

            return None

    def get_objects_by_color(self, color: str) -> list[ObjectState]:
        """Get all objects of a specific color."""
        with self._lock:
            return [obj for obj in self._objects.values() if obj.color == color]

    def mark_object_grasped(self, object_id: str, robot_id: str):
        """Mark an object as grasped by a robot."""
        with self._lock:
            if object_id in self._objects:
                self._objects[object_id].grasped_by = robot_id
                logger.info(f"Object {object_id} grasped by {robot_id}")

    def mark_object_released(self, object_id: str):
        """Mark an object as released (no longer grasped)."""
        with self._lock:
            if object_id in self._objects:
                self._objects[object_id].grasped_by = None
                logger.info(f"Object {object_id} released")

    def register_object(
        self,
        object_id: str,
        object_type: str = "unknown",
        position: Tuple[float, float, float] = (0, 0, 0),
        _graspable: bool = True,
        **kwargs,
    ):
        """Register a new object (simplified interface for tests)."""
        color = kwargs.get("color", "unknown")
        confidence = kwargs.get("confidence", 1.0)
        self.update_object_position(object_id, position, color, object_type, confidence)

    def get_all_objects(self) -> list[ObjectState]:
        """Get all registered objects."""
        with self._lock:
            return list(self._objects.values())

    def decay_object_confidence(self, seen_object_ids: set[str]):
        """
        Update object confidence based on recent detections.

        Decays confidence for objects not seen this frame; refreshes seen objects.
        Removes objects not seen for OBJECT_TTL_SECONDS.
        """
        with self._lock:
            now = time.time()
            to_delete = []

            for obj_id, obj in self._objects.items():
                # Fields are static landmarks not tracked by Unity — skip decay.
                if getattr(obj, "object_type", None) == "field":
                    continue
                if obj_id in seen_object_ids:
                    obj.confidence = 1.0
                    obj.last_seen = now
                    obj.stale = False
                    obj.timestamp = now
                else:
                    # Round to 10 decimal places to prevent floating-point
                    # accumulation errors from repeated subtraction.
                    obj.confidence = round(
                        max(0.0, obj.confidence - CONFIDENCE_DECAY_PER_FRAME), 10
                    )
                    obj.stale = obj.confidence < STALE_CONFIDENCE_THRESHOLD
                    if now - obj.last_seen > OBJECT_TTL_SECONDS:
                        to_delete.append(obj_id)

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
        """Find all objects within radius of a position (Euclidean, sufficient for <50 objects)."""
        with self._lock:
            nearby = []
            for obj in self._objects.values():
                if exclude_stale and obj.stale:
                    continue
                distance = math.dist(position, obj.position)
                if distance <= radius:
                    nearby.append(obj)

            return nearby

    def find_robots_near(
        self, position: Tuple[float, float, float], radius: float = 0.2
    ) -> list[RobotState]:
        """Find all robots within radius of a position."""
        with self._lock:
            nearby = []
            for robot in self._robot_states.values():
                if robot.position is None:
                    continue
                distance = math.dist(position, robot.position)
                if distance <= radius:
                    nearby.append(robot)

            return nearby

    def get_reachable_objects(
        self, robot_id: str, exclude_stale: bool = True
    ) -> list[ObjectState]:
        """Get all objects reachable by a robot via spatial predicates."""
        with self._lock:
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
                    is_accessible, _ = object_accessible_by_robot(
                        robot_id, obj.position, world_state=self
                    )
                    if is_accessible:
                        reachable.append(obj)

            return reachable

    def get_objects_in_region(
        self, region: str, exclude_stale: bool = True
    ) -> list[ObjectState]:
        """Get all objects in a workspace region."""
        with self._lock:
            if region not in WORKSPACE_REGIONS:
                logger.warning(f"Unknown workspace region: {region}")
                return []

            bounds = WORKSPACE_REGIONS[region]
            objects_in_region = []

            for obj in self._objects.values():
                if exclude_stale and obj.stale:
                    continue
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
        """Get which workspace region contains a position."""
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

        Returns robot state and annotated object list with spatial relationships.
        Example: "Robot1 at (-0.3, 0.2, 0.1), gripper open. Objects: RedCube at (0.1, 0.3, 0.0)
        [reachable, in shared_zone], BlueCube at (0.4, 0.2, 0.1) [not reachable, in right_workspace]."
        """
        with self._lock:
            robot = self._robot_states.get(robot_id)
            if robot is None or robot.position is None:
                return f"{robot_id} state unknown."
            pos = robot.position
            gripper = robot.gripper_state
            context = f"{robot_id} at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), gripper {gripper}."
            reachable_ids = {
                obj.object_id for obj in self.get_reachable_objects(robot_id)
            }

            if not self._objects:
                context += " No objects detected."
                return context

            context += " Objects: "
            obj_descriptions = []
            for obj in self._objects.values():
                obj_pos = obj.position
                desc = f"{obj.object_id} at ({obj_pos[0]:.2f}, {obj_pos[1]:.2f}, {obj_pos[2]:.2f})"
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

    def allocate_workspace(
        self,
        region: str,
        robot_id: str,
        urgency: int = 1,
        estimated_duration: float = 30.0,
    ) -> bool:
        """Allocate a workspace region to a robot. Delegates to WorkspaceAllocator."""
        return self._workspace_allocator.allocate(
            region, robot_id, urgency, estimated_duration
        )

    def release_workspace(self, region: str, robot_id: str) -> bool:
        """Release a workspace region allocation."""
        return self._workspace_allocator.release(region, robot_id)

    def get_workspace_owner(self, region: str) -> Optional[str]:
        """Get the robot that owns a workspace region."""
        return self._workspace_allocator.get_owner(region)

    def get_free_workspace_regions(self) -> list:
        """Return list of region names currently unallocated."""
        return self._workspace_allocator.get_free_regions()

    def get_robot_intents(self) -> Dict[str, str]:
        """Return {robot_id: object_id} for all robots with active movement intent."""
        with self._lock:
            return {
                rid: state.moving_toward_object
                for rid, state in self._robot_states.items()
                if state.moving_toward_object is not None
            }

    def get_all_robots(self) -> list:
        """Return list of all RobotState objects currently tracked."""
        with self._lock:
            return list(self._robot_states.values())

    def broadcast_task_outcome(
        self,
        robot_id: str,
        task_id: str,
        success: bool,
        duration_ms: float,
        final_object_states: Dict[str, Any],
    ) -> None:
        """Publish a completed sequence outcome so peer robots can reason about world state."""
        outcome = {
            "robot_id": robot_id,
            "task_id": task_id,
            "success": success,
            "duration_ms": duration_ms,
            "final_object_states": final_object_states,
            "timestamp": time.time(),
        }
        with self._lock:
            self._task_outcomes.append(outcome)
            if len(self._task_outcomes) > 50:
                self._task_outcomes = self._task_outcomes[-50:]

    def get_task_outcomes(
        self,
        last_n: int = 20,
        robot_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return recent task outcomes, optionally filtered by robot_id."""
        with self._lock:
            outcomes = list(self._task_outcomes)
        if robot_id is not None:
            outcomes = [o for o in outcomes if o["robot_id"] == robot_id]
        return outcomes[-last_n:]

    # ------------------------------------------------------------------
    # Backward-compatibility shims: tests and external code may read/write
    # _workspace_allocations and _workspace_timeout directly on WorldState.
    # These properties proxy through to the WorkspaceAllocator.
    # ------------------------------------------------------------------

    @property
    def _workspace_allocations(self) -> Dict[str, Optional["WorkspaceAllocation"]]:
        return self._workspace_allocator._allocations

    @_workspace_allocations.setter
    def _workspace_allocations(self, value):
        self._workspace_allocator._allocations = value

    @property
    def _workspace_timeout(self) -> float:
        return self._workspace_allocator._timeout

    @_workspace_timeout.setter
    def _workspace_timeout(self, value: float):
        # Route through set_timeout() to enforce max(1.0, ...) floor and RLock.
        self._workspace_allocator.set_timeout(value)

    def _cleanup_stale_allocations(self):
        """No longer used directly; kept for safety. Delegates to WorkspaceAllocator."""
        with self._workspace_allocator._lock:
            self._workspace_allocator._cleanup_stale()

    def set_workspace_timeout(self, timeout: float):
        """Set workspace allocation timeout in seconds."""
        self._workspace_allocator.set_timeout(timeout)

    def register_command(self, request_id: int, command: Dict[str, Any]):
        """Register an in-flight command for tracking."""
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
        """Update status of a tracked command."""
        with self._lock:
            if request_id in self._pending_commands:
                self._pending_commands[request_id]["status"] = status
                self._pending_commands[request_id]["result"] = result
                self._pending_commands[request_id]["completion_time"] = time.time()
                logger.debug(f"Command {request_id} status: {status}")

    def get_command_status(self, request_id: int) -> Optional[Dict[str, Any]]:
        """Get status of a tracked command."""
        with self._lock:
            return self._pending_commands.get(request_id)

    def cleanup_old_commands(self, max_age_seconds: float = 300.0):
        """Remove old completed commands from tracking."""
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

            self._workspace_allocator.reset()
            self._pending_commands.clear()
            logger.info("Reset world state")


def get_world_state() -> WorldState:
    """Get the global WorldState singleton instance."""
    return WorldState()
