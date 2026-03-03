#!/usr/bin/env python3
"""
SharedVisionState.py - Thread-safe shared vision state for multi-robot coordination

Provides a centralized state for vision detections with claim/release mechanism
to prevent conflicts when multiple robots target the same object.

Features:
- Thread-safe detection storage with timestamps
- Object claim/release mechanism (prevents conflicts)
- Auto-timeout for stale claims (10s default)
- Query available objects (exclude claimed)
- Conflict resolution strategies (closest robot, first claim)
- Integration with VisionProcessor for continuous updates

Usage:
    from operations.SharedVisionState import get_shared_vision_state

    # Get singleton instance
    state = get_shared_vision_state()

    # Update with new detections
    state.update_detections(detections)

    # Query available objects
    available = state.get_available_objects(color="blue")

    # Claim object for robot
    if state.claim_object("blue_cube_1", "Robot1"):
        print("Object claimed successfully")

    # Release when done
    state.release_object("blue_cube_1", "Robot1")
"""

import threading
import time
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import math

# Import detection data models
try:
    from ..vision.DetectionDataModels import DetectionObject
except ImportError:
    try:
        from vision.DetectionDataModels import DetectionObject
    except ImportError:
        DetectionObject = None

# Import config
try:
    from config.Vision import (
        CONFLICT_RESOLUTION_STRATEGY,
        CONFLICT_MIN_DISTANCE_DIFF,
        OBJECT_CLAIM_TIMEOUT,
    )
except ImportError:
    from ..config.Vision import (
        CONFLICT_RESOLUTION_STRATEGY,
        CONFLICT_MIN_DISTANCE_DIFF,
        OBJECT_CLAIM_TIMEOUT,
    )

from core.LoggingSetup import get_logger
logger = get_logger(__name__)


@dataclass
class ClaimedObject:
    """
    Object with claim information for multi-robot coordination.

    Attributes:
        object_id: Unique object identifier (e.g., "blue_cube_1")
        color: Object class/color name
        world_position: 3D world position (x, y, z) in meters
        claimed_by: Robot ID that claimed this object (None = unclaimed)
        claim_timestamp: Time when claim was made
        track_id: Persistent track ID across frames (optional)
        confidence: Detection confidence (0.0-1.0)
        depth_m: Depth from camera in meters
        last_seen: Timestamp of last detection update
    """

    object_id: str
    color: str
    world_position: Tuple[float, float, float]
    claimed_by: Optional[str] = None
    claim_timestamp: float = 0.0
    track_id: Optional[int] = None
    confidence: float = 1.0
    depth_m: Optional[float] = None
    last_seen: float = 0.0


class SharedVisionState:
    """
    Thread-safe shared vision state for multi-robot coordination.

    Maintains a centralized registry of detected objects with claim/release
    mechanism to prevent conflicts when multiple robots target the same object.

    Attributes:
        detections: Dictionary of object_id -> ClaimedObject
        lock: Thread lock for synchronized access
        claim_timeout: Auto-release timeout for stale claims (seconds)
        conflict_strategy: Strategy for resolving conflicts

    Example:
        state = SharedVisionState(claim_timeout=10.0)

        # Update with new detections
        state.update_detections(vision_detections)

        # Robot 1 queries available blue cubes
        available = state.get_available_objects(color="blue")

        # Robot 1 claims an object
        state.claim_object(available[0].object_id, "Robot1")

        # Robot 2 tries to claim same object (conflict!)
        # Returns False - object already claimed
        state.claim_object(available[0].object_id, "Robot2")
    """

    def __init__(self, claim_timeout: float = 10.0):
        """
        Initialize shared vision state.

        Args:
            claim_timeout: Auto-release timeout for stale claims in seconds
        """
        self.detections: Dict[str, ClaimedObject] = {}
        self.lock = threading.Lock()
        self.claim_timeout = claim_timeout
        self.conflict_strategy = CONFLICT_RESOLUTION_STRATEGY

        logger.info(
            f"SharedVisionState initialized: timeout={claim_timeout}s, "
            f"strategy={self.conflict_strategy}"
        )

    def update_detections(self, detections: List):
        """
        Update shared state with new detections from vision system.

        - Adds new objects
        - Updates existing object positions
        - Preserves claim information
        - Removes stale claims

        Args:
            detections: List of DetectionObject from vision system
        """
        with self.lock:
            current_time = time.time()

            # Clean up stale claims first
            self._cleanup_stale_claims()

            # Update or add detections
            for det in detections:
                # Generate object ID (use track_id if available, else use color + position)
                if det.track_id is not None:
                    object_id = f"{det.color}_track_{det.track_id}"
                else:
                    # Fallback: use color + approximate position
                    if det.world_position:
                        x, y, z = det.world_position
                        object_id = f"{det.color}_{x:.2f}_{y:.2f}_{z:.2f}"
                    else:
                        # No world position, use pixel center
                        object_id = f"{det.color}_{det.center_x}_{det.center_y}"

                # Check if object already exists
                if object_id in self.detections:
                    # Update existing object (preserve claim)
                    obj = self.detections[object_id]
                    if det.world_position:
                        obj.world_position = det.world_position
                    obj.confidence = det.confidence
                    obj.depth_m = det.depth_m
                    obj.last_seen = current_time
                else:
                    # Add new object
                    if det.world_position:
                        claimed_obj = ClaimedObject(
                            object_id=object_id,
                            color=det.color,
                            world_position=det.world_position,
                            track_id=det.track_id,
                            confidence=det.confidence,
                            depth_m=det.depth_m,
                            last_seen=current_time,
                        )
                        self.detections[object_id] = claimed_obj

            logger.debug(f"Updated vision state: {len(self.detections)} objects")

    def claim_object(self, object_id: str, robot_id: str) -> bool:
        """
        Claim an object for a robot.

        Args:
            object_id: Object identifier
            robot_id: Robot identifier (e.g., "Robot1")

        Returns:
            True if claim successful, False if already claimed by another robot
        """
        with self.lock:
            if object_id not in self.detections:
                logger.warning(f"Cannot claim unknown object: {object_id}")
                return False

            obj = self.detections[object_id]

            # Check if already claimed
            if obj.claimed_by is not None:
                if obj.claimed_by == robot_id:
                    # Already claimed by this robot, refresh timestamp
                    obj.claim_timestamp = time.time()
                    return True
                else:
                    logger.debug(
                        f"Object {object_id} already claimed by {obj.claimed_by}"
                    )
                    return False

            # Claim object
            obj.claimed_by = robot_id
            obj.claim_timestamp = time.time()
            logger.info(f"Robot {robot_id} claimed object {object_id}")
            return True

    def release_object(self, object_id: str, robot_id: str) -> bool:
        """
        Release a claimed object.

        Args:
            object_id: Object identifier
            robot_id: Robot identifier

        Returns:
            True if release successful, False otherwise
        """
        with self.lock:
            if object_id not in self.detections:
                logger.warning(f"Cannot release unknown object: {object_id}")
                return False

            obj = self.detections[object_id]

            # Verify robot owns the claim
            if obj.claimed_by != robot_id:
                logger.warning(
                    f"Robot {robot_id} cannot release object {object_id} "
                    f"(claimed by {obj.claimed_by})"
                )
                return False

            # Release claim
            obj.claimed_by = None
            obj.claim_timestamp = 0.0
            logger.info(f"Robot {robot_id} released object {object_id}")
            return True

    def get_available_objects(self, color: Optional[str] = None) -> List[ClaimedObject]:
        """
        Get list of objects not currently claimed.

        Args:
            color: Optional filter by color (e.g., "blue", "red_cube")

        Returns:
            List of unclaimed ClaimedObject instances
        """
        with self.lock:
            # Clean up stale claims
            self._cleanup_stale_claims()

            available = []
            for obj in self.detections.values():
                # Skip claimed objects
                if obj.claimed_by is not None:
                    continue

                # Apply color filter if specified
                if color is not None:
                    # Flexible matching: "blue" matches "blue_cube"
                    if color not in obj.color:
                        continue

                available.append(obj)

            return available.copy()

    def get_claimed_objects(self, robot_id: str) -> List[ClaimedObject]:
        """
        Get objects claimed by a specific robot.

        Args:
            robot_id: Robot identifier

        Returns:
            List of ClaimedObject instances claimed by robot
        """
        with self.lock:
            claimed = [
                obj for obj in self.detections.values() if obj.claimed_by == robot_id
            ]
            return claimed.copy()

    def resolve_conflict(
        self,
        object_id: str,
        robot1_id: str,
        robot2_id: str,
        robot1_pos: Tuple[float, float, float],
        robot2_pos: Tuple[float, float, float],
    ) -> str:
        """
        Resolve conflict when both robots want the same object.

        Strategies:
        - "closest_robot": Assign to closer robot (if distance diff > 5cm)
        - "first_claim": Keep existing claim holder

        Args:
            object_id: Object identifier
            robot1_id: First robot identifier
            robot2_id: Second robot identifier
            robot1_pos: First robot position (x, y, z)
            robot2_pos: Second robot position (x, y, z)

        Returns:
            Robot ID that should get the object
        """
        with self.lock:
            if object_id not in self.detections:
                logger.warning(
                    f"Cannot resolve conflict for unknown object: {object_id}"
                )
                return robot1_id  # Default to first robot

            obj = self.detections[object_id]

            # Check existing claim
            if obj.claimed_by is not None:
                logger.debug(f"Conflict resolved by existing claim: {obj.claimed_by}")
                return obj.claimed_by

            # Apply conflict resolution strategy
            if self.conflict_strategy == "closest_robot":
                # Calculate distances
                dist1 = self._calculate_distance(robot1_pos, obj.world_position)
                dist2 = self._calculate_distance(robot2_pos, obj.world_position)

                min_diff = CONFLICT_MIN_DISTANCE_DIFF

                if abs(dist1 - dist2) > min_diff:
                    # Clear winner: assign to closer robot
                    winner = robot1_id if dist1 < dist2 else robot2_id
                    logger.info(
                        f"Conflict resolved by distance: {winner} "
                        f"(d1={dist1:.3f}m, d2={dist2:.3f}m)"
                    )
                    return winner
                else:
                    # Too close to call, use tie-breaker (alphabetical)
                    winner = robot1_id if robot1_id < robot2_id else robot2_id
                    logger.info(f"Conflict tie-breaker (distances equal): {winner}")
                    return winner
            else:
                # "first_claim" strategy: first robot wins
                logger.info(f"Conflict resolved by first claim: {robot1_id}")
                return robot1_id

    def _calculate_distance(
        self, pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]
    ) -> float:
        """
        Calculate Euclidean distance between two 3D points.

        Args:
            pos1: First position (x, y, z)
            pos2: Second position (x, y, z)

        Returns:
            Distance in meters
        """
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        dz = pos1[2] - pos2[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _cleanup_stale_claims(self):
        """
        Remove claims older than timeout.

        Must be called with lock held.
        """
        current_time = time.time()

        for obj in self.detections.values():
            if obj.claimed_by is not None:
                age = current_time - obj.claim_timestamp
                if age > self.claim_timeout:
                    logger.info(
                        f"Auto-releasing stale claim: {obj.object_id} "
                        f"(claimed by {obj.claimed_by}, age={age:.1f}s)"
                    )
                    obj.claimed_by = None
                    obj.claim_timestamp = 0.0

    def get_stats(self) -> dict:
        """
        Get state statistics.

        Returns:
            Dictionary with state stats
        """
        with self.lock:
            total = len(self.detections)
            claimed = sum(
                1 for obj in self.detections.values() if obj.claimed_by is not None
            )
            available = total - claimed

            return {
                "total_objects": total,
                "claimed_objects": claimed,
                "available_objects": available,
                "claim_timeout": self.claim_timeout,
                "conflict_strategy": self.conflict_strategy,
            }

    def clear(self):
        """
        Clear all detections and claims.
        """
        with self.lock:
            self.detections.clear()
            logger.info("SharedVisionState cleared")


# ===========================
# Singleton Instance
# ===========================

_shared_vision_state: Optional[SharedVisionState] = None
_shared_vision_state_lock = threading.Lock()


def get_shared_vision_state() -> SharedVisionState:
    """
    Get singleton SharedVisionState instance (thread-safe).

    Uses double-checked locking to prevent duplicate initialization when
    multiple threads call this function simultaneously during startup.

    Returns:
        SharedVisionState singleton
    """
    global _shared_vision_state
    if _shared_vision_state is None:
        with _shared_vision_state_lock:
            if _shared_vision_state is None:
                _shared_vision_state = SharedVisionState(
                    claim_timeout=OBJECT_CLAIM_TIMEOUT
                )
    return _shared_vision_state


# ===========================
# Main (for testing)
# ===========================


def main():
    """Test SharedVisionState with mock detections"""
    print("=== SharedVisionState Test ===\n")

    state = get_shared_vision_state()

    # Mock detection objects (would normally come from vision system)
    if DetectionObject is not None:
        detections = [
            DetectionObject(
                object_id=1,
                color="blue_cube",
                bbox=(100, 100, 50, 50),
                confidence=0.9,
                world_position=(0.3, 0.1, 0.0),
                track_id=1,
            ),
            DetectionObject(
                object_id=2,
                color="red_cube",
                bbox=(200, 100, 50, 50),
                confidence=0.85,
                world_position=(0.5, 0.1, 0.2),
                track_id=2,
            ),
        ]

        # Update state
        state.update_detections(detections)
        print(f"Updated state: {state.get_stats()}")

        # Query available objects
        available = state.get_available_objects()
        print(f"\nAvailable objects: {len(available)}")
        for obj in available:
            print(f"  {obj.object_id}: {obj.color} at {obj.world_position}")

        # Robot 1 claims blue cube
        print("\nRobot1 claims blue cube:")
        success = state.claim_object(available[0].object_id, "Robot1")
        print(f"  Claim success: {success}")

        # Robot 2 tries to claim same object
        print("\nRobot2 tries to claim same object:")
        success = state.claim_object(available[0].object_id, "Robot2")
        print(f"  Claim success: {success} (should be False)")

        # Check stats
        print(f"\nState after claims: {state.get_stats()}")

        # Robot 1 releases
        print("\nRobot1 releases blue cube:")
        state.release_object(available[0].object_id, "Robot1")

        # Final stats
        print(f"\nFinal state: {state.get_stats()}")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
