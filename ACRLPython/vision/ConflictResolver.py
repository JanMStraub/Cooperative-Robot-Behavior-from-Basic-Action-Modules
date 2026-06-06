#!/usr/bin/env python3
"""Multi-robot object claim conflict resolution."""

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
    """Resolves ownership conflicts when multiple robots claim the same object."""

    def __init__(self):
        self._claims: Dict[str, Dict[str, float]] = {}

    def claim_object(self, robot_id: str, object_id: str) -> None:
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
        self._prune_expired_claims(object_id)

        claims = self._claims.get(object_id, {})

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

        earliest_ts = min(claims[r] for r in claims)
        my_ts = claims[robot_id]

        if my_ts == earliest_ts:
            logger.debug(
                f"Conflict: {robot_id} wins (first claim, dist={my_dist:.3f}m) on {object_id}"
            )
            return True

        logger.debug(
            f"Conflict: {robot_id} yielding to earlier claimant on {object_id} "
            f"(dist={my_dist:.3f}m; use resolve_conflict_with_positions for full logic)"
        )
        return False

    def resolve_conflict_with_positions(
        self,
        robot_id: str,
        object_id: str,
        all_robot_positions: Dict[str, Tuple[float, float, float]],
        object_position: Tuple[float, float, float],
    ) -> bool:
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

        def _dist(pos_a: Tuple, pos_b: Tuple) -> float:
            return sum((a - b) ** 2 for a, b in zip(pos_a, pos_b)) ** 0.5

        my_pos = all_robot_positions.get(robot_id)
        if my_pos is None:
            return True

        my_dist = _dist(my_pos, object_position)

        for competitor in competitors:
            comp_pos = all_robot_positions.get(competitor)
            if comp_pos is None:
                continue
            comp_dist = _dist(comp_pos, object_position)

            if comp_dist < my_dist - CONFLICT_MIN_DISTANCE_DIFF:
                logger.debug(
                    f"Conflict: {competitor} ({comp_dist:.3f}m) closer than "
                    f"{robot_id} ({my_dist:.3f}m) to {object_id}"
                )
                return False

        return True

    def release_claim(self, robot_id: str, object_id: str) -> None:
        if object_id in self._claims and robot_id in self._claims[object_id]:
            del self._claims[object_id][robot_id]
            if not self._claims[object_id]:
                del self._claims[object_id]
            logger.debug(f"Claim released: {robot_id} -> {object_id}")

    def _prune_expired_claims(self, object_id: str) -> None:
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
        self._prune_expired_claims(object_id)
        return dict(self._claims.get(object_id, {}))
