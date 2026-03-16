#!/usr/bin/env python3
"""
VisionConflictResolver - Multi-robot object claim conflict resolution.

When multiple robots detect the same object and want to interact with it,
this module decides which robot gets priority based on the configured strategy.

Config (config/Vision.py):
    OBJECT_CLAIM_TIMEOUT:       Seconds before a claim expires (default 10.0)
    CONFLICT_RESOLUTION_STRATEGY: "closest_robot" or "first_come" (default "closest_robot")
    CONFLICT_MIN_DISTANCE_DIFF: Min distance difference to break ties (default 0.05m)
"""

import time
from typing import Dict, Optional, Tuple

from core.LoggingSetup import get_logger

logger = get_logger(__name__)

try:
    from config.Vision import (
        OBJECT_CLAIM_TIMEOUT,
        CONFLICT_RESOLUTION_STRATEGY,
        CONFLICT_MIN_DISTANCE_DIFF,
    )
except ImportError:
    OBJECT_CLAIM_TIMEOUT = 10.0
    CONFLICT_RESOLUTION_STRATEGY = "closest_robot"
    CONFLICT_MIN_DISTANCE_DIFF = 0.05


class VisionConflictResolver:
    """
    Resolves ownership conflicts when multiple robots claim the same detected object.

    Strategies:
        "closest_robot": The robot physically closest to the object wins.
            If distance difference is below CONFLICT_MIN_DISTANCE_DIFF, the first
            claimant wins to break the tie.
        "first_come": The robot that claimed first (by timestamp) wins.

    Claims expire after OBJECT_CLAIM_TIMEOUT seconds.
    """

    def __init__(self):
        """Initialize conflict resolver with empty claim registry."""
        # Maps object_id -> {robot_id: claim_timestamp}
        self._claims: Dict[str, Dict[str, float]] = {}

    def claim_object(self, robot_id: str, object_id: str) -> None:
        """
        Register a claim for an object by a robot.

        Args:
            robot_id: Robot identifier (e.g., "Robot1")
            object_id: Object identifier (e.g., "red_cube_0")
        """
        now = time.time()
        if object_id not in self._claims:
            self._claims[object_id] = {}
        self._claims[object_id][robot_id] = now
        logger.debug(f"Claim registered: {robot_id} -> {object_id}")

    def resolve_conflict(
        self,
        robot_id: str,
        object_id: str,
        robot_position: Optional[Tuple[float, float, float]] = None,
        object_position: Optional[Tuple[float, float, float]] = None,
    ) -> bool:
        """
        Determine if the given robot gets access to the claimed object.

        Expired claims are pruned before resolution. If no conflict exists
        (only one valid claimant), the robot always gets the object.

        Args:
            robot_id: Robot requesting access
            object_id: Object being contested
            robot_position: Robot's current position (x, y, z) in world space
            object_position: Object's position (x, y, z) in world space

        Returns:
            True if this robot should proceed; False if another robot has priority.
        """
        self._prune_expired_claims(object_id)

        claims = self._claims.get(object_id, {})

        # If no competing claims, robot gets access
        if not claims or robot_id not in claims:
            return True

        competitors = [r for r in claims if r != robot_id]
        if not competitors:
            return True

        if CONFLICT_RESOLUTION_STRATEGY == "first_come":
            # Winner is the robot with the oldest (lowest) timestamp
            earliest_ts = min(claims[r] for r in claims)
            result = claims[robot_id] == earliest_ts
            if not result:
                logger.debug(
                    f"Conflict: {robot_id} loses to earlier claim on {object_id}"
                )
            return result

        # "closest_robot" strategy (default)
        if robot_position is None or object_position is None:
            # No position info: fall back to first_come
            logger.debug(
                f"No position data for closest_robot resolution; falling back to first_come"
            )
            earliest_ts = min(claims[r] for r in claims)
            return claims[robot_id] == earliest_ts

        def _dist(pos_a: Tuple, pos_b: Tuple) -> float:
            return sum((a - b) ** 2 for a, b in zip(pos_a, pos_b)) ** 0.5

        my_dist = _dist(robot_position, object_position)

        for competitor in competitors:
            # We can only compare distance when the caller provides its own position;
            # competitors' positions are not stored in claims. Use claim timestamp as
            # proxy for competitor proximity when only our position is known.
            # A full implementation would pass all robot positions from WorldState.
            pass

        # When only this robot's position is available, grant access if
        # no competitor timestamp is significantly earlier (tie-break heuristic)
        earliest_ts = min(claims[r] for r in claims)
        my_ts = claims[robot_id]

        if my_ts == earliest_ts:
            # We claimed first; grant access
            return True

        # We claimed later; check if we're meaningfully closer (requires position of all robots)
        # Without competitor positions, defer to first_come for safety
        logger.debug(
            f"Conflict: {robot_id} yielding to earlier claimant on {object_id}"
        )
        return False

    def resolve_conflict_with_positions(
        self,
        robot_id: str,
        object_id: str,
        all_robot_positions: Dict[str, Tuple[float, float, float]],
        object_position: Tuple[float, float, float],
    ) -> bool:
        """
        Resolve conflict using positions of ALL active robots.

        This is the preferred method when WorldState is available, as it provides
        accurate closest-robot resolution.

        Args:
            robot_id: Robot requesting access
            object_id: Object being contested
            all_robot_positions: Dict mapping robot_id -> (x, y, z) world position
            object_position: Object's position in world space

        Returns:
            True if this robot should proceed.
        """
        self._prune_expired_claims(object_id)

        claims = self._claims.get(object_id, {})
        if not claims or robot_id not in claims:
            return True

        competitors = [r for r in claims if r != robot_id]
        if not competitors:
            return True

        if CONFLICT_RESOLUTION_STRATEGY == "first_come":
            earliest_ts = min(claims[r] for r in claims)
            return claims[robot_id] == earliest_ts

        # Closest robot strategy
        def _dist(pos_a: Tuple, pos_b: Tuple) -> float:
            return sum((a - b) ** 2 for a, b in zip(pos_a, pos_b)) ** 0.5

        my_pos = all_robot_positions.get(robot_id)
        if my_pos is None:
            return True  # Can't determine, grant access

        my_dist = _dist(my_pos, object_position)

        for competitor in competitors:
            comp_pos = all_robot_positions.get(competitor)
            if comp_pos is None:
                continue
            comp_dist = _dist(comp_pos, object_position)

            if comp_dist < my_dist - CONFLICT_MIN_DISTANCE_DIFF:
                # Competitor is meaningfully closer
                logger.debug(
                    f"Conflict: {competitor} ({comp_dist:.3f}m) closer than "
                    f"{robot_id} ({my_dist:.3f}m) to {object_id}"
                )
                return False

        return True

    def release_claim(self, robot_id: str, object_id: str) -> None:
        """
        Release a robot's claim on an object (e.g., after successful grasp or failure).

        Args:
            robot_id: Robot releasing the claim
            object_id: Object being released
        """
        if object_id in self._claims and robot_id in self._claims[object_id]:
            del self._claims[object_id][robot_id]
            if not self._claims[object_id]:
                del self._claims[object_id]
            logger.debug(f"Claim released: {robot_id} -> {object_id}")

    def _prune_expired_claims(self, object_id: str) -> None:
        """
        Remove claims older than OBJECT_CLAIM_TIMEOUT seconds.

        Args:
            object_id: Object whose claims to prune
        """
        if object_id not in self._claims:
            return

        now = time.time()
        expired = [
            robot_id
            for robot_id, ts in self._claims[object_id].items()
            if now - ts > OBJECT_CLAIM_TIMEOUT
        ]
        for robot_id in expired:
            del self._claims[object_id][robot_id]
            logger.debug(f"Expired claim pruned: {robot_id} -> {object_id}")

        if not self._claims[object_id]:
            del self._claims[object_id]

    def get_active_claims(self, object_id: str) -> Dict[str, float]:
        """
        Get all active (non-expired) claims for an object.

        Args:
            object_id: Object to query

        Returns:
            Dict mapping robot_id -> claim timestamp
        """
        self._prune_expired_claims(object_id)
        return dict(self._claims.get(object_id, {}))
