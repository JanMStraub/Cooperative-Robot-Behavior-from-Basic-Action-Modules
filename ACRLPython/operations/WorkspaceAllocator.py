#!/usr/bin/env python3
"""Self-contained workspace allocation logic extracted from WorldState."""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

try:
    from config.Robot import WORKSPACE_REGIONS, WORKSPACE_ALLOCATION_TIMEOUT
except ImportError:
    from ..config.Robot import WORKSPACE_REGIONS, WORKSPACE_ALLOCATION_TIMEOUT

from core.LoggingSetup import get_logger

logger = get_logger(__name__)


@dataclass
class WorkspaceAllocation:
    robot_id: str
    region: str
    allocated_at: float = field(default_factory=time.time)
    urgency: int = 1  # 1 (low) to 5 (high); higher urgency can preempt lower
    estimated_duration: float = 30.0  # seconds this robot expects to need the region


class WorkspaceAllocator:
    """Manages workspace region allocations with timeout and preemption support."""

    def __init__(self):
        self._lock = threading.RLock()
        self._allocations: Dict[str, Optional[WorkspaceAllocation]] = {
            region: None for region in WORKSPACE_REGIONS.keys()
        }
        self._timeout: float = WORKSPACE_ALLOCATION_TIMEOUT

    def reset(self):
        """Reset all allocations and timeout to defaults (for testing)."""
        with self._lock:
            for region in self._allocations:
                self._allocations[region] = None
            self._timeout = WORKSPACE_ALLOCATION_TIMEOUT

    def allocate(
        self,
        region: str,
        robot_id: str,
        urgency: int = 1,
        estimated_duration: float = 30.0,
    ) -> bool:
        """
        Allocate a workspace region to a robot with timeout tracking.

        High-urgency requests (urgency > current holder's urgency) can preempt
        low-urgency holders with >10s remaining.
        """
        with self._lock:
            if region not in self._allocations:
                logger.warning(f"Unknown workspace region: {region}")
                return False

            self._cleanup_stale()

            current = self._allocations[region]
            if current is None:
                self._allocations[region] = WorkspaceAllocation(
                    robot_id=robot_id,
                    region=region,
                    urgency=urgency,
                    estimated_duration=estimated_duration,
                )
                logger.info(f"Allocated {region} to {robot_id}")
                return True

            if current.robot_id == robot_id:
                current.urgency = urgency
                current.estimated_duration = estimated_duration
                current.allocated_at = time.time()
                return True

            # Preemption: strictly higher urgency AND holder has >10s remaining
            elapsed = time.time() - current.allocated_at
            remaining = max(0.0, current.estimated_duration - elapsed)
            if urgency > current.urgency and remaining > 10.0:
                logger.info(
                    f"Preempting {region} from {current.robot_id} (urgency {current.urgency}, "
                    f"{remaining:.1f}s remaining) for {robot_id} (urgency {urgency})"
                )
                self._allocations[region] = WorkspaceAllocation(
                    robot_id=robot_id,
                    region=region,
                    urgency=urgency,
                    estimated_duration=estimated_duration,
                )
                return True

            logger.warning(
                f"Region {region} allocated to {current.robot_id}, preemption denied"
            )
            return False

    def release(self, region: str, robot_id: str) -> bool:
        """Release a workspace region allocation."""
        with self._lock:
            if region not in self._allocations:
                logger.warning(f"Unknown workspace region: {region}")
                return False

            current_allocation = self._allocations[region]
            if current_allocation is None:
                logger.warning(f"Region {region} is not allocated")
                return False

            if current_allocation.robot_id != robot_id:
                logger.warning(f"Region {region} not allocated to {robot_id}")
                return False

            self._allocations[region] = None
            logger.info(f"Released {region} from {robot_id}")
            return True

    def get_owner(self, region: str) -> Optional[str]:
        """Get the robot that owns a workspace region."""
        with self._lock:
            self._cleanup_stale()
            allocation = self._allocations.get(region)
            return allocation.robot_id if allocation else None

    def get_free_regions(self) -> list:
        """Return list of region names currently unallocated."""
        with self._lock:
            self._cleanup_stale()
            return [r for r, alloc in self._allocations.items() if alloc is None]

    def set_timeout(self, timeout: float):
        """Set workspace allocation timeout in seconds."""
        with self._lock:
            self._timeout = max(1.0, timeout)
            logger.info(f"Set workspace timeout to {self._timeout}s")

    def _cleanup_stale(self):
        """Release workspace allocations that have exceeded timeout.

        Must be called while self._lock is already held.
        """
        now = time.time()
        stale_regions = []
        for region, allocation in self._allocations.items():
            if allocation is not None:
                age = now - allocation.allocated_at
                if age > self._timeout:
                    stale_regions.append(region)
                    logger.warning(
                        f"Auto-releasing stale allocation: {region} from {allocation.robot_id} (age: {age:.1f}s)"
                    )
        for region in stale_regions:
            self._allocations[region] = None
