"""
Multi-Robot Coordination Safety Verifier
=========================================

This module provides safety verification for multi-robot operations,
ensuring robots don't collide, conflict over resources, or deadlock.

Checks performed:
- Path collision detection
- Workspace conflict detection
- Object access conflicts
- Deadlock prevention
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import LLMConfig
from operations.WorldState import get_world_state
from operations.SpatialPredicates import robots_will_collide, is_in_shared_zone
from operations.Base import OperationCategory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class CoordinationIssue:
    """
    A coordination safety issue detected.

    Attributes:
        issue_type: Type of issue ("collision", "workspace_conflict", "object_conflict", "deadlock")
        severity: "blocking" (prevents execution) or "warning" (risky but allowed)
        description: Human-readable description
        affected_robots: List of robot IDs involved
        resolution_suggestions: How to resolve the issue
    """

    issue_type: str
    severity: str
    description: str
    affected_robots: List[str] = field(default_factory=list)
    resolution_suggestions: List[str] = field(default_factory=list)


@dataclass
class CoordinationCheckResult:
    """
    Result of multi-robot safety check.

    Attributes:
        safe: True if no blocking issues found
        issues: List of coordination issues detected
        warnings: List of warnings (non-blocking)
        checked_robots: List of robots that were checked
        recommendations: General recommendations for safer execution
    """

    safe: bool = True
    issues: List[CoordinationIssue] = field(default_factory=list)
    warnings: List[CoordinationIssue] = field(default_factory=list)
    checked_robots: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def add_issue(self, issue: CoordinationIssue):
        """Add an issue to the result."""
        if issue.severity == "blocking":
            self.issues.append(issue)
            self.safe = False
        else:
            self.warnings.append(issue)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "safe": self.safe,
            "issues": [
                {
                    "type": i.issue_type,
                    "severity": i.severity,
                    "description": i.description,
                    "affected_robots": i.affected_robots,
                    "resolution_suggestions": i.resolution_suggestions,
                }
                for i in self.issues
            ],
            "warnings": [
                {
                    "type": w.issue_type,
                    "severity": w.severity,
                    "description": w.description,
                    "affected_robots": w.affected_robots,
                    "resolution_suggestions": w.resolution_suggestions,
                }
                for w in self.warnings
            ],
            "checked_robots": self.checked_robots,
            "recommendations": self.recommendations,
        }


# ============================================================================
# Coordination Verifier
# ============================================================================


class CoordinationVerifier:
    """
    Verifier for multi-robot coordination safety.

    Checks for potential conflicts when multiple robots operate simultaneously:
    - Collision detection (path intersection)
    - Workspace conflicts (both in same exclusive workspace)
    - Object access conflicts (both trying to grasp same object)
    - Deadlock detection (circular dependencies)
    """

    def __init__(self):
        """Initialize the coordination verifier."""
        self.world_state = get_world_state()

    def verify_multi_robot_safety(
        self,
        robot_id: str,
        operation_category: OperationCategory,
        params: Dict[str, Any],
        world_state=None,
    ) -> CoordinationCheckResult:
        """
        Verify safety of operation in multi-robot context.

        Checks if the planned operation will conflict with other robots
        currently operating or with pending operations.

        Args:
            robot_id: ID of robot planning to execute operation
            operation_category: Category of operation (NAVIGATION, MANIPULATION, etc.)
            params: Operation parameters (including target positions)
            world_state: Optional WorldState instance

        Returns:
            CoordinationCheckResult with safety status and detected issues

        Example:
            >>> verifier = CoordinationVerifier()
            >>> result = verifier.verify_multi_robot_safety(
            ...     "Robot1",
            ...     OperationCategory.NAVIGATION,
            ...     {"x": 0.3, "y": 0.2, "z": 0.1}
            ... )
            >>> if not result.safe:
            ...     print("Coordination issues:", result.issues)
        """
        if world_state is None:
            world_state = self.world_state

        result = CoordinationCheckResult()
        result.checked_robots.append(robot_id)

        # Get all other robots in system
        other_robots = [
            rid for rid in LLMConfig.ROBOT_BASE_POSITIONS.keys() if rid != robot_id
        ]

        # Check 1: Path collision with other robots
        if operation_category == OperationCategory.NAVIGATION:
            target_pos = self._extract_target_position(params)
            if target_pos is not None:
                for other_robot in other_robots:
                    result.checked_robots.append(other_robot)
                    collision_issue = self._check_collision(
                        robot_id, target_pos, other_robot, world_state
                    )
                    if collision_issue:
                        result.add_issue(collision_issue)

        # Check 2: Workspace conflicts
        workspace_issue = self._check_workspace_conflict(
            robot_id, params, other_robots, world_state
        )
        if workspace_issue:
            result.add_issue(workspace_issue)

        # Check 3: Object access conflicts
        if operation_category == OperationCategory.MANIPULATION:
            object_issue = self._check_object_conflict(
                robot_id, params, other_robots, world_state
            )
            if object_issue:
                result.add_issue(object_issue)

        # Check 4: Deadlock potential
        deadlock_issue = self._check_deadlock_potential(
            robot_id, params, other_robots, world_state
        )
        if deadlock_issue:
            result.add_issue(deadlock_issue)

        # Add general recommendations
        if not result.safe:
            result.recommendations.extend(
                [
                    "Consider serializing operations instead of parallel execution",
                    "Use coordinate_simultaneous_move for safe multi-robot movement",
                    "Allocate workspace regions before operations",
                ]
            )

        return result

    def _extract_target_position(
        self, params: Dict[str, Any]
    ) -> Optional[Tuple[float, float, float]]:
        """
        Extract target position from operation parameters.

        Args:
            params: Operation parameters

        Returns:
            (x, y, z) tuple or None if no target position
        """
        if "x" in params and "y" in params and "z" in params:
            return (params["x"], params["y"], params["z"])
        elif "target_position" in params:
            return params["target_position"]
        elif "position" in params:
            return params["position"]
        return None

    def _check_collision(
        self,
        robot_id: str,
        target_pos: Tuple[float, float, float],
        other_robot_id: str,
        world_state,
    ) -> Optional[CoordinationIssue]:
        """
        Check if robot will collide with another robot.

        Args:
            robot_id: ID of moving robot
            target_pos: Target position
            other_robot_id: ID of other robot
            world_state: WorldState instance

        Returns:
            CoordinationIssue if collision detected, None otherwise
        """
        # Get other robot's current target (if moving)
        other_robot_state = world_state._robot_states.get(other_robot_id)
        if other_robot_state is None:
            return None  # Other robot not tracked

        # If other robot is not moving, check static position
        if not other_robot_state.is_moving:
            # Check distance to other robot's current position
            if other_robot_state.position is not None:
                dx = target_pos[0] - other_robot_state.position[0]
                dy = target_pos[1] - other_robot_state.position[1]
                dz = target_pos[2] - other_robot_state.position[2]
                distance = (dx * dx + dy * dy + dz * dz) ** 0.5

                min_sep = LLMConfig.MIN_ROBOT_SEPARATION
                if distance < min_sep:
                    return CoordinationIssue(
                        issue_type="collision",
                        severity="blocking",
                        description=f"Target position too close to {other_robot_id} (distance: {distance:.3f}m)",
                        affected_robots=[robot_id, other_robot_id],
                        resolution_suggestions=[
                            f"Maintain minimum separation of {min_sep}m",
                            f"Wait for {other_robot_id} to move",
                            "Choose different target position",
                        ],
                    )
            return None

        # Other robot is moving - check path collision
        other_target = other_robot_state.target_position
        if other_target is None:
            return None

        # Use robots_will_collide predicate
        will_collide, reason = robots_will_collide(
            robot_id, target_pos, other_robot_id, other_target, world_state
        )

        if will_collide:
            return CoordinationIssue(
                issue_type="collision",
                severity="blocking",
                description=f"Path collision with {other_robot_id}: {reason}",
                affected_robots=[robot_id, other_robot_id],
                resolution_suggestions=[
                    "Serialize movements (one robot at a time)",
                    "Use coordinate_simultaneous_move for safe parallel movement",
                    f"Wait for {other_robot_id} to complete movement",
                ],
            )

        return None

    def _check_workspace_conflict(
        self,
        robot_id: str,
        params: Dict[str, Any],
        other_robots: List[str],
        world_state,
    ) -> Optional[CoordinationIssue]:
        """
        Check for workspace allocation conflicts.

        Args:
            robot_id: Robot planning operation
            params: Operation parameters
            other_robots: Other robots in system
            world_state: WorldState instance

        Returns:
            CoordinationIssue if conflict detected
        """
        target_pos = self._extract_target_position(params)
        if target_pos is None:
            return None

        # Determine which workspace this operation targets
        target_workspace = self._get_workspace_for_position(target_pos)
        if target_workspace is None:
            return None  # Not in any defined workspace

        # Shared zones are allowed for all robots
        is_shared, _ = is_in_shared_zone(*target_pos)
        if is_shared:
            return None

        # Check if another robot has allocated this workspace
        current_owner = world_state.get_workspace_owner(target_workspace)
        if current_owner is not None and current_owner != robot_id:
            return CoordinationIssue(
                issue_type="workspace_conflict",
                severity="blocking",
                description=f"Workspace '{target_workspace}' allocated to {current_owner}",
                affected_robots=[robot_id, current_owner],
                resolution_suggestions=[
                    f"Wait for {current_owner} to release workspace",
                    f"Use allocate_workspace_region to claim workspace first",
                    "Move to shared_zone instead",
                    "Use different workspace",
                ],
            )

        return None

    def _check_object_conflict(
        self,
        robot_id: str,
        params: Dict[str, Any],
        other_robots: List[str],
        world_state,
    ) -> Optional[CoordinationIssue]:
        """
        Check if multiple robots are trying to grasp same object.

        Args:
            robot_id: Robot planning manipulation
            params: Operation parameters
            other_robots: Other robots in system
            world_state: WorldState instance

        Returns:
            CoordinationIssue if conflict detected
        """
        # Extract target object ID from params
        target_object_id = params.get("object_id") or params.get("object_ref")
        if target_object_id is None or not isinstance(target_object_id, str):
            return None  # No specific object targeted

        # Check if object is already grasped
        obj_state = world_state._objects.get(target_object_id)
        if obj_state is None:
            return None  # Object not tracked

        if obj_state.grasped_by is not None and obj_state.grasped_by != robot_id:
            return CoordinationIssue(
                issue_type="object_conflict",
                severity="blocking",
                description=f"Object '{target_object_id}' already grasped by {obj_state.grasped_by}",
                affected_robots=[robot_id, obj_state.grasped_by],
                resolution_suggestions=[
                    f"Wait for {obj_state.grasped_by} to release object",
                    "Use coordinate_handoff for transfer",
                    "Target different object",
                ],
            )

        return None

    def _check_deadlock_potential(
        self,
        robot_id: str,
        params: Dict[str, Any],
        other_robots: List[str],
        world_state,
    ) -> Optional[CoordinationIssue]:
        """
        Check for potential deadlock situations.

        Simplified deadlock detection:
        - Two robots in each other's target workspaces
        - Circular wait for workspace release

        Args:
            robot_id: Robot planning operation
            params: Operation parameters
            other_robots: Other robots in system
            world_state: WorldState instance

        Returns:
            CoordinationIssue if deadlock potential detected
        """
        target_pos = self._extract_target_position(params)
        if target_pos is None:
            return None

        target_workspace = self._get_workspace_for_position(target_pos)
        if target_workspace is None:
            return None

        # Check if any other robot is trying to move to this robot's current workspace
        robot_current_workspace = LLMConfig.ROBOT_WORKSPACE_ASSIGNMENTS.get(robot_id)
        if robot_current_workspace is None:
            return None

        for other_robot in other_robots:
            other_state = world_state._robot_states.get(other_robot)
            if other_state is None or other_state.target_position is None:
                continue

            # Get other robot's target workspace
            other_target_workspace = self._get_workspace_for_position(
                other_state.target_position
            )

            # Potential deadlock: this robot going to other's workspace,
            # other robot going to this robot's workspace
            if (
                target_workspace == other_state.target_position
                and other_target_workspace == robot_current_workspace
            ):
                return CoordinationIssue(
                    issue_type="deadlock",
                    severity="warning",  # Warning, not blocking (may resolve naturally)
                    description=f"Potential deadlock with {other_robot}: circular workspace dependency",
                    affected_robots=[robot_id, other_robot],
                    resolution_suggestions=[
                        "Serialize movements to avoid circular wait",
                        "One robot should complete movement first",
                        "Use intermediate position via shared_zone",
                    ],
                )

        return None

    def _get_workspace_for_position(
        self, position: Tuple[float, float, float]
    ) -> Optional[str]:
        """
        Determine which workspace region contains a position.

        Args:
            position: (x, y, z) coordinates

        Returns:
            Workspace region name or None
        """
        x, y, z = position
        for region_name, bounds in LLMConfig.WORKSPACE_REGIONS.items():
            if (
                bounds["x_min"] <= x <= bounds["x_max"]
                and bounds["y_min"] <= y <= bounds["y_max"]
                and bounds["z_min"] <= z <= bounds["z_max"]
            ):
                return region_name
        return None


# ============================================================================
# Utility Functions
# ============================================================================


def quick_check_multi_robot_safety(
    robot_id: str, operation_category: OperationCategory, params: Dict[str, Any]
) -> Tuple[bool, CoordinationCheckResult]:
    """
    Quick helper to check if operation is safe in multi-robot context.

    Args:
        robot_id: Robot ID
        operation_category: Operation category
        params: Operation parameters

    Returns:
        (is_safe, check_result)

    Example:
        >>> is_safe, result = quick_check_multi_robot_safety("Robot1", OperationCategory.NAVIGATION, {...})
        >>> if not is_safe:
        ...     print("Coordination issues:", result.issues)
    """
    verifier = CoordinationVerifier()
    result = verifier.verify_multi_robot_safety(robot_id, operation_category, params)
    return result.safe, result
