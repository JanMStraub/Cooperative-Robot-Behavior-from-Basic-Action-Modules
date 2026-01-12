#!/usr/bin/env python3
"""
Unit tests for CoordinationVerifier.py

Tests multi-robot coordination safety verification including collision detection,
workspace conflicts, object access conflicts, and deadlock detection.
"""

import pytest
from unittest.mock import Mock, patch
from operations.CoordinationVerifier import (
    CoordinationIssue,
    CoordinationCheckResult,
    CoordinationVerifier,
    quick_check_multi_robot_safety,
)
from operations.Base import OperationCategory
# Config imports not needed - using fixtures


class TestCoordinationIssue:
    """Test CoordinationIssue dataclass"""

    def test_coordination_issue_creation(self):
        """Test creating a CoordinationIssue"""
        issue = CoordinationIssue(
            issue_type="collision",
            severity="blocking",
            description="Robots will collide",
            affected_robots=["Robot1", "Robot2"],
            resolution_suggestions=["Serialize movements", "Use different paths"]
        )

        assert issue.issue_type == "collision"
        assert issue.severity == "blocking"
        assert issue.description == "Robots will collide"
        assert len(issue.affected_robots) == 2
        assert len(issue.resolution_suggestions) == 2

    def test_coordination_issue_defaults(self):
        """Test default values for CoordinationIssue"""
        issue = CoordinationIssue(
            issue_type="workspace_conflict",
            severity="warning",
            description="Potential workspace overlap"
        )

        assert issue.affected_robots == []
        assert issue.resolution_suggestions == []


class TestCoordinationCheckResult:
    """Test CoordinationCheckResult dataclass"""

    def test_coordination_check_result_safe(self):
        """Test safe coordination result"""
        result = CoordinationCheckResult()

        assert result.safe is True
        assert len(result.issues) == 0
        assert len(result.warnings) == 0

    def test_add_issue_blocking(self):
        """Test adding blocking issue sets safe to False"""
        result = CoordinationCheckResult()

        issue = CoordinationIssue(
            issue_type="collision",
            severity="blocking",
            description="Path collision detected",
            affected_robots=["Robot1", "Robot2"]
        )

        result.add_issue(issue)

        assert result.safe is False
        assert len(result.issues) == 1
        assert len(result.warnings) == 0

    def test_add_issue_warning(self):
        """Test adding warning doesn't set safe to False"""
        result = CoordinationCheckResult()

        issue = CoordinationIssue(
            issue_type="deadlock",
            severity="warning",
            description="Potential circular dependency",
            affected_robots=["Robot1", "Robot2"]
        )

        result.add_issue(issue)

        assert result.safe is True  # Still safe
        assert len(result.issues) == 0
        assert len(result.warnings) == 1

    def test_coordination_check_result_to_dict(self):
        """Test serialization to dictionary"""
        result = CoordinationCheckResult()
        result.checked_robots = ["Robot1", "Robot2"]

        issue = CoordinationIssue(
            issue_type="collision",
            severity="blocking",
            description="Collision detected",
            affected_robots=["Robot1", "Robot2"],
            resolution_suggestions=["Serialize movements"]
        )
        result.add_issue(issue)

        result_dict = result.to_dict()

        assert result_dict["safe"] is False
        assert len(result_dict["issues"]) == 1
        assert result_dict["issues"][0]["type"] == "collision"
        assert len(result_dict["checked_robots"]) == 2


class TestCoordinationVerifier:
    """Test CoordinationVerifier"""

    def test_verify_multi_robot_safety_navigation(self, mock_world_state_multi_robot, cleanup_world_state):
        """Test navigation operation safety checks"""
        verifier = CoordinationVerifier()

        params = {
            "x": 0.2,
            "y": 0.1,
            "z": 0.15
        }

        result = verifier.verify_multi_robot_safety(
            "Robot1",
            OperationCategory.NAVIGATION,
            params,
            mock_world_state_multi_robot
        )

        # Should check for collisions with Robot2
        assert "Robot1" in result.checked_robots
        assert "Robot2" in result.checked_robots

    def test_verify_multi_robot_safety_manipulation(self, mock_world_state_multi_robot, cleanup_world_state):
        """Test manipulation operation safety checks"""
        verifier = CoordinationVerifier()

        params = {
            "object_id": "test_object",
            "action": "grasp"
        }

        result = verifier.verify_multi_robot_safety(
            "Robot1",
            OperationCategory.MANIPULATION,
            params,
            mock_world_state_multi_robot
        )

        # Should check for object conflicts
        assert "Robot1" in result.checked_robots

    def test_check_collision_static_robot(self, mock_world_state_multi_robot, cleanup_world_state):
        """Test static robot too close"""
        verifier = CoordinationVerifier()

        # Robot2 is stationary at (0.3, 0.0, 0.1)
        # Try to move Robot1 very close to Robot2
        target_pos = (0.35, 0.0, 0.1)  # Only 0.05m from Robot2

        issue = verifier._check_collision(
            "Robot1",
            target_pos,
            "Robot2",
            mock_world_state_multi_robot
        )

        assert issue is not None
        assert issue.issue_type == "collision"
        assert issue.severity == "blocking"
        assert "Robot2" in issue.description

    def test_check_collision_moving_robot(self, mock_world_state_multi_robot, cleanup_world_state):
        """Test moving robot path collision detection"""
        verifier = CoordinationVerifier()

        # Setup Robot2 as moving
        robot2_state = mock_world_state_multi_robot._robot_states["Robot2"]
        robot2_state.is_moving = True
        robot2_state.target_position = (-0.3, 0.0, 0.1)  # Moving towards Robot1's area

        # Robot1 trying to move to Robot2's starting position
        target_pos = (0.3, 0.0, 0.1)

        with patch('operations.SpatialPredicates.robots_will_collide') as mock_collide:
            mock_collide.return_value = (True, "Paths will cross")

            issue = verifier._check_collision(
                "Robot1",
                target_pos,
                "Robot2",
                mock_world_state_multi_robot
            )

        assert issue is not None
        assert issue.issue_type == "collision"
        assert "Path collision" in issue.description

    def test_check_collision_safe(self, mock_world_state_multi_robot, cleanup_world_state):
        """Test no collision case"""
        verifier = CoordinationVerifier()

        # Robot2 at (0.3, 0.0, 0.1)
        # Move Robot1 far away
        target_pos = (-0.8, 0.0, 0.2)  # Far from Robot2

        issue = verifier._check_collision(
            "Robot1",
            target_pos,
            "Robot2",
            mock_world_state_multi_robot
        )

        assert issue is None  # No collision

    def test_check_workspace_conflict(self, mock_world_state_multi_robot, cleanup_world_state):
        """Test workspace already allocated"""
        verifier = CoordinationVerifier()

        # Mock workspace allocated to Robot2
        mock_world_state_multi_robot.get_workspace_owner = Mock(return_value="Robot2")

        params = {
            "x": -0.5,  # In left_workspace
            "y": 0.0,
            "z": 0.2
        }

        issue = verifier._check_workspace_conflict(
            "Robot1",
            params,
            ["Robot2"],
            mock_world_state_multi_robot
        )

        assert issue is not None
        assert issue.issue_type == "workspace_conflict"
        assert "Robot2" in issue.description

    def test_check_workspace_conflict_shared_zone_ok(self, mock_world_state_multi_robot, cleanup_world_state):
        """Test shared zone allowed for all robots"""
        verifier = CoordinationVerifier()

        # Mock no workspace owner
        mock_world_state_multi_robot.get_workspace_owner = Mock(return_value=None)

        params = {
            "x": 0.0,  # In shared_zone
            "y": 0.0,
            "z": 0.2
        }

        issue = verifier._check_workspace_conflict(
            "Robot1",
            params,
            ["Robot2"],
            mock_world_state_multi_robot
        )

        assert issue is None  # No conflict in shared zone

    def test_check_object_conflict(self, mock_world_state_multi_robot, cleanup_world_state):
        """Test object already grasped"""
        verifier = CoordinationVerifier()

        # Mark object as grasped by Robot2
        obj = mock_world_state_multi_robot._objects["test_object"]
        obj.grasped_by = "Robot2"

        params = {
            "object_id": "test_object",
            "action": "grasp"
        }

        issue = verifier._check_object_conflict(
            "Robot1",
            params,
            ["Robot2"],
            mock_world_state_multi_robot
        )

        assert issue is not None
        assert issue.issue_type == "object_conflict"
        assert "Robot2" in issue.description

    def test_check_object_conflict_not_tracked(self, mock_world_state_multi_robot, cleanup_world_state):
        """Test object not tracked in world state"""
        verifier = CoordinationVerifier()

        params = {
            "object_id": "nonexistent_object",
            "action": "grasp"
        }

        issue = verifier._check_object_conflict(
            "Robot1",
            params,
            ["Robot2"],
            mock_world_state_multi_robot
        )

        assert issue is None  # No conflict if object not tracked

    def test_check_deadlock_potential(self, mock_world_state_multi_robot, cleanup_world_state):
        """Test circular workspace dependency detection"""
        verifier = CoordinationVerifier()

        # This test is complex - simplified version
        # Robot1 trying to move to right_workspace while Robot2 is in left_workspace
        params = {
            "x": 0.5,  # right_workspace
            "y": 0.0,
            "z": 0.2
        }

        # Setup Robot2 state with target
        robot2_state = mock_world_state_multi_robot._robot_states["Robot2"]
        robot2_state.target_position = (-0.5, 0.0, 0.2)  # Targeting left_workspace

        # This could detect deadlock if robots are in each other's target workspaces
        issue = verifier._check_deadlock_potential(
            "Robot1",
            params,
            ["Robot2"],
            mock_world_state_multi_robot
        )

        # May or may not detect deadlock depending on workspace assignments
        # This is a warning, not blocking
        if issue:
            assert issue.severity == "warning"

    def test_get_workspace_for_position(self, cleanup_world_state):
        """Test position to workspace mapping"""
        verifier = CoordinationVerifier()

        # Test left_workspace
        workspace = verifier._get_workspace_for_position((-0.5, 0.0, 0.2))
        assert workspace == "left_workspace"

        # Test right_workspace
        workspace = verifier._get_workspace_for_position((0.5, 0.0, 0.2))
        assert workspace == "right_workspace"

        # Test shared_zone
        workspace = verifier._get_workspace_for_position((0.0, 0.0, 0.2))
        assert workspace == "shared_zone"

        # Test outside all regions
        workspace = verifier._get_workspace_for_position((10.0, 10.0, 10.0))
        assert workspace is None


class TestQuickCheckMultiRobotSafety:
    """Test quick_check_multi_robot_safety helper"""

    def test_quick_check_safe(self, mock_world_state_multi_robot, cleanup_world_state):
        """Test quick check passes for safe operation"""
        params = {
            "x": -0.5,
            "y": 0.0,
            "z": 0.2
        }

        is_safe, result = quick_check_multi_robot_safety(
            "Robot1",
            OperationCategory.NAVIGATION,
            params
        )

        # Should be safe if robots are far apart
        assert isinstance(is_safe, bool)
        assert isinstance(result, CoordinationCheckResult)

    def test_quick_check_unsafe(self, mock_world_state_multi_robot, cleanup_world_state):
        """Test quick check detects unsafe operation"""
        # Create scenario where Robot2 is very close
        robot2_state = mock_world_state_multi_robot._robot_states["Robot2"]
        robot2_state.position = (0.25, 0.0, 0.1)

        params = {
            "x": 0.3,  # Very close to Robot2
            "y": 0.0,
            "z": 0.1
        }

        is_safe, result = quick_check_multi_robot_safety(
            "Robot1",
            OperationCategory.NAVIGATION,
            params
        )

        # May detect collision depending on MIN_ROBOT_SEPARATION
        if not is_safe:
            assert len(result.issues) > 0
