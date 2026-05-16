#!/usr/bin/env python3
"""Thread-safe shared vision state with claim/release mechanism to prevent multi-robot conflicts."""

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
    """Detected object with claim state for multi-robot coordination."""

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
    """Thread-safe centralized registry of detected objects with claim/release to prevent multi-robot conflicts."""

    def __init__(self, claim_timeout: float = 10.0):
        self.detections: Dict[str, ClaimedObject] = {}
        self.lock = threading.Lock()
        self.claim_timeout = claim_timeout
        self.conflict_strategy = CONFLICT_RESOLUTION_STRATEGY

        logger.debug(
            f"SharedVisionState initialized: timeout={claim_timeout}s, "
            f"strategy={self.conflict_strategy}"
        )

    def update_detections(self, detections: List):
        """Update shared state with new detections; preserves claims, removes stale ones."""
        with self.lock:
            current_time = time.time()

            # Clean up stale claims first
            self._cleanup_stale_claims()

            for det in detections:
                if det.track_id is not None:
                    object_id = f"{det.color}_track_{det.track_id}"
                else:
                    # Fallback: use color + approximate position
                    if det.world_position:
                        x, y, z = det.world_position
                        object_id = f"{det.color}_{x:.2f}_{y:.2f}_{z:.2f}"
                    else:
                        object_id = f"{det.color}_{det.center_x}_{det.center_y}"

                if object_id in self.detections:
                    obj = self.detections[object_id]
                    if det.world_position:
                        obj.world_position = det.world_position
                    obj.confidence = det.confidence
                    obj.depth_m = det.depth_m
                    obj.last_seen = current_time
                else:
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
        """Claim an object for a robot. Returns False if already claimed by another robot."""
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
        """Release a claimed object."""
        with self.lock:
            if object_id not in self.detections:
                logger.warning(f"Cannot release unknown object: {object_id}")
                return False

            obj = self.detections[object_id]
            if obj.claimed_by != robot_id:
                logger.warning(
                    f"Robot {robot_id} cannot release object {object_id} "
                    f"(claimed by {obj.claimed_by})"
                )
                return False

            obj.claimed_by = None
            obj.claim_timestamp = 0.0
            logger.info(f"Robot {robot_id} released object {object_id}")
            return True

    def get_available_objects(self, color: Optional[str] = None) -> List[ClaimedObject]:
        """Get unclaimed objects, optionally filtered by color."""
        with self.lock:
            self._cleanup_stale_claims()
            available = []
            for obj in self.detections.values():
                if obj.claimed_by is not None:
                    continue
                if color is not None:
                    # Flexible matching: "blue" matches "blue_cube"
                    if color not in obj.color:
                        continue

                available.append(obj)

            return available.copy()

    def get_claimed_objects(self, robot_id: str) -> List[ClaimedObject]:
        """Get objects claimed by a specific robot."""
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

        Strategies: "closest_robot" (assign to closer robot if diff > 5cm), "first_claim".
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
        """Euclidean distance between two 3D points."""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        dz = pos1[2] - pos2[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _cleanup_stale_claims(self):
        """Remove claims older than timeout. Must be called with lock held."""
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
        """Get state statistics."""
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
        """Clear all detections and claims."""
        with self.lock:
            self.detections.clear()
            logger.info("SharedVisionState cleared")


_shared_vision_state: Optional[SharedVisionState] = None
_shared_vision_state_lock = threading.Lock()


def get_shared_vision_state() -> SharedVisionState:
    """Get singleton SharedVisionState instance (thread-safe, double-checked locking)."""
    global _shared_vision_state
    if _shared_vision_state is None:
        with _shared_vision_state_lock:
            if _shared_vision_state is None:
                _shared_vision_state = SharedVisionState(
                    claim_timeout=OBJECT_CLAIM_TIMEOUT
                )
    return _shared_vision_state


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
