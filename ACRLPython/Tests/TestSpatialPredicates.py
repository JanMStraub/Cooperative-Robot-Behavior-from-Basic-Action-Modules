#!/usr/bin/env python3
"""
Unit tests for SpatialPredicates.py

Tests all spatial predicate functions including basic predicates
and multi-robot collision detection.
"""

import pytest
from unittest.mock import Mock, patch
import math

from operations.SpatialPredicates import (
    target_within_reach,
    is_in_robot_workspace,
    is_in_shared_zone,
    robot_is_initialized,
    robot_is_stationary,
    gripper_is_open,
    gripper_is_closed,
    object_accessible_by_robot,
    robots_will_collide,
    evaluate_predicate,
    list_predicates,
    PREDICATE_REGISTRY,
)
import LLMConfig as cfg


class TestTargetWithinReach:
    """Test target_within_reach predicate"""

    def test_target_within_reach_valid(self):
        """Test that robot can reach target within MAX_ROBOT_REACH"""
        # Target at 0.5m from Robot1's base at (-0.3, 0.0, 0.0)
        is_valid, reason = target_within_reach("Robot1", -0.1, 0.0, 0.4)

        assert is_valid is True
        assert reason == ""

    def test_target_within_reach_too_far(self):
        """Test that target beyond MAX_ROBOT_REACH fails"""
        # Target at 1.5m from Robot1's base (exceeds MAX_ROBOT_REACH of 0.8m)
        is_valid, reason = target_within_reach("Robot1", 0.8, 0.0, 0.7)

        assert is_valid is False
        assert "exceeds max reach" in reason
        assert "0.8" in reason  # MAX_ROBOT_REACH value

    def test_target_within_reach_at_boundary(self):
        """Test target exactly at MAX_ROBOT_REACH"""
        # Calculate position exactly at MAX_ROBOT_REACH distance
        robot_base = cfg.ROBOT_BASE_POSITIONS["Robot1"]  # (-0.3, 0.0, 0.0)
        max_reach = cfg.MAX_ROBOT_REACH  # 0.8

        # Target at exactly 0.8m away
        target_x = robot_base[0] + max_reach
        is_valid, reason = target_within_reach("Robot1", target_x, 0.0, 0.0)

        assert is_valid is True

    def test_target_within_reach_unknown_robot(self):
        """Test that unknown robot ID fails"""
        is_valid, reason = target_within_reach("UnknownRobot", 0.0, 0.0, 0.0)

        assert is_valid is False
        assert "Unknown robot" in reason
        assert "UnknownRobot" in reason


class TestWorkspacePredicates:
    """Test workspace-related predicates"""

    def test_is_in_robot_workspace_valid(self):
        """Test position within assigned workspace"""
        # Position within Robot1's left_workspace
        is_valid, reason = is_in_robot_workspace("Robot1", -0.5, 0.0, 0.2)

        assert is_valid is True
        assert reason == ""

    def test_is_in_robot_workspace_outside_x(self):
        """Test position outside workspace X bounds"""
        # Position too far right for left_workspace
        is_valid, reason = is_in_robot_workspace("Robot1", 0.5, 0.0, 0.2)

        assert is_valid is False
        assert "X coordinate" in reason
        assert "outside workspace" in reason

    def test_is_in_robot_workspace_outside_y(self):
        """Test position outside workspace Y bounds"""
        # Position with Y out of bounds
        is_valid, reason = is_in_robot_workspace("Robot1", -0.5, 2.0, 0.2)

        assert is_valid is False
        assert "Y coordinate" in reason

    def test_is_in_robot_workspace_outside_z(self):
        """Test position outside workspace Z bounds"""
        # Position with Z out of bounds
        is_valid, reason = is_in_robot_workspace("Robot1", -0.5, 0.0, 1.0)

        assert is_valid is False
        assert "Z coordinate" in reason

    def test_is_in_robot_workspace_unknown_robot(self):
        """Test unknown robot fails"""
        is_valid, reason = is_in_robot_workspace("UnknownRobot", 0.0, 0.0, 0.0)

        assert is_valid is False
        assert "no assigned workspace" in reason

    def test_is_in_shared_zone_valid(self):
        """Test position in shared zone"""
        # Position within shared_zone (-0.1 to 0.1 in X)
        is_valid, reason = is_in_shared_zone(0.05, 0.0, 0.2)

        assert is_valid is True
        assert reason == ""

    def test_is_in_shared_zone_outside(self):
        """Test position outside shared zone"""
        # Position outside shared zone
        is_valid, reason = is_in_shared_zone(0.5, 0.0, 0.2)

        assert is_valid is False
        assert "outside shared zone" in reason


class TestRobotStatePredicates:
    """Test robot state checking predicates"""

    def test_robot_is_initialized_exists(self, mock_world_state):
        """Test robot exists and is initialized"""
        mock_world_state.get_robot_status = Mock(return_value={"is_initialized": True})

        is_valid, reason = robot_is_initialized("Robot1", mock_world_state)

        assert is_valid is True
        assert reason == ""

    def test_robot_is_initialized_not_initialized(self, mock_world_state):
        """Test robot not initialized"""
        mock_world_state.get_robot_status = Mock(return_value={"is_initialized": False})

        is_valid, reason = robot_is_initialized("Robot1", mock_world_state)

        assert is_valid is False
        assert "not initialized" in reason

    def test_robot_is_initialized_no_world_state(self):
        """Test basic check without world state"""
        # Should pass basic check if robot exists in config
        is_valid, reason = robot_is_initialized("Robot1", None)

        assert is_valid is True  # Basic check passes

    def test_robot_is_initialized_unknown_robot(self):
        """Test unknown robot fails"""
        is_valid, reason = robot_is_initialized("UnknownRobot", None)

        assert is_valid is False
        assert "not found" in reason

    def test_robot_is_stationary_true(self, mock_world_state):
        """Test robot is stationary"""
        mock_world_state.get_robot_status = Mock(return_value={"is_moving": False})

        is_valid, reason = robot_is_stationary("Robot1", mock_world_state)

        assert is_valid is True

    def test_robot_is_stationary_false(self, mock_world_state):
        """Test robot is moving"""
        mock_world_state.get_robot_status = Mock(return_value={"is_moving": True})

        is_valid, reason = robot_is_stationary("Robot1", mock_world_state)

        assert is_valid is False
        assert "currently moving" in reason

    def test_robot_is_stationary_no_world_state(self):
        """Test requires world state"""
        is_valid, reason = robot_is_stationary("Robot1", None)

        assert is_valid is False
        assert "WorldState required" in reason


class TestGripperPredicates:
    """Test gripper state predicates"""

    def test_gripper_is_open_true(self, mock_world_state):
        """Test gripper is open"""
        mock_world_state.get_robot_status = Mock(return_value={"gripper_state": "open"})

        is_valid, reason = gripper_is_open("Robot1", mock_world_state)

        assert is_valid is True
        assert reason == ""

    def test_gripper_is_open_false(self, mock_world_state):
        """Test gripper is closed"""
        mock_world_state.get_robot_status = Mock(return_value={"gripper_state": "closed"})

        is_valid, reason = gripper_is_open("Robot1", mock_world_state)

        assert is_valid is False
        assert "gripper is closed" in reason

    def test_gripper_is_closed_true(self, mock_world_state):
        """Test gripper is closed"""
        mock_world_state.get_robot_status = Mock(return_value={"gripper_state": "closed"})

        is_valid, reason = gripper_is_closed("Robot1", mock_world_state)

        assert is_valid is True
        assert reason == ""

    def test_gripper_is_closed_false(self, mock_world_state):
        """Test gripper is open"""
        mock_world_state.get_robot_status = Mock(return_value={"gripper_state": "open"})

        is_valid, reason = gripper_is_closed("Robot1", mock_world_state)

        assert is_valid is False
        assert "gripper is open" in reason

    def test_gripper_state_unknown(self, mock_world_state):
        """Test unknown gripper state"""
        mock_world_state.get_robot_status = Mock(return_value={"gripper_state": "unknown"})

        is_valid, reason = gripper_is_open("Robot1", mock_world_state)

        assert is_valid is False
        assert "unknown" in reason


class TestObjectAccessibility:
    """Test object_accessible_by_robot predicate"""

    def test_object_accessible_within_workspace(self, mock_world_state):
        """Test object within robot workspace and reach"""
        # Object within Robot1's left_workspace and reach
        object_pos = (-0.4, 0.0, 0.2)

        is_valid, reason = object_accessible_by_robot("Robot1", object_pos, mock_world_state)

        assert is_valid is True

    def test_object_accessible_in_shared_zone(self, mock_world_state):
        """Test object in shared zone is accessible"""
        # Object in shared_zone
        object_pos = (0.0, 0.0, 0.2)

        is_valid, reason = object_accessible_by_robot("Robot1", object_pos, mock_world_state)

        assert is_valid is True  # Shared zone accessible by all

    def test_object_not_reachable(self, mock_world_state):
        """Test object beyond reach"""
        # Object too far from Robot1
        object_pos = (2.0, 0.0, 0.2)

        is_valid, reason = object_accessible_by_robot("Robot1", object_pos, mock_world_state)

        assert is_valid is False
        assert "not reachable" in reason

    def test_object_not_in_workspace(self, mock_world_state):
        """Test object in other robot's workspace"""
        # Object in Robot2's right_workspace
        object_pos = (0.5, 0.0, 0.2)

        is_valid, reason = object_accessible_by_robot("Robot1", object_pos, mock_world_state)

        assert is_valid is False
        # The predicate checks reachability first, so we expect "not reachable"
        assert "not reachable" in reason


class TestMultiRobotCollision:
    """Test robots_will_collide predicate"""

    def test_robots_will_collide_target_too_close(self, mock_world_state):
        """Test MIN_ROBOT_SEPARATION violation"""
        target1 = (0.0, 0.0, 0.2)
        target2 = (0.1, 0.0, 0.2)  # Only 0.1m apart

        will_collide, reason = robots_will_collide(
            "Robot1", target1, "Robot2", target2, mock_world_state
        )

        assert will_collide is True
        assert "too close" in reason
        assert str(cfg.MIN_ROBOT_SEPARATION) in reason

    def test_robots_will_collide_safe_parallel(self, mock_world_state_multi_robot):
        """Test no collision case"""
        target1 = (-0.3, 0.0, 0.2)
        target2 = (0.3, 0.0, 0.2)  # 0.6m apart - safe

        will_collide, reason = robots_will_collide(
            "Robot1", target1, "Robot2", target2, mock_world_state_multi_robot
        )

        assert will_collide is False
        assert reason == ""

    def test_robots_will_collide_path_intersection(self, mock_world_state_multi_robot):
        """Test linear path collision detection"""
        # Setup robots with positions that will cross paths
        mock_world_state_multi_robot._robot_states["Robot1"].position = (-0.3, -0.5, 0.1)
        mock_world_state_multi_robot._robot_states["Robot2"].position = (0.3, 0.5, 0.1)

        # Targets that cross paths
        target1 = (0.3, 0.5, 0.2)  # Robot1 moves to Robot2's position
        target2 = (-0.3, -0.5, 0.2)  # Robot2 moves to Robot1's position

        will_collide, reason = robots_will_collide(
            "Robot1", target1, "Robot2", target2, mock_world_state_multi_robot
        )

        # Paths cross - should detect collision
        assert will_collide is True

    def test_robots_will_collide_same_workspace(self):
        """Test exclusive workspace conflict"""
        # Both targets in left_workspace (exclusive to Robot1)
        # Targets must be >= MIN_ROBOT_SEPARATION (0.2m) apart to pass distance check
        target1 = (-0.5, 0.0, 0.2)
        target2 = (-0.25, 0.0, 0.2)  # 0.25m apart in X - passes distance check

        will_collide, reason = robots_will_collide(
            "Robot1", target1, "Robot2", target2, None
        )

        assert will_collide is True
        assert "same exclusive workspace" in reason


class TestPredicateRegistry:
    """Test predicate registration and lookup"""

    def test_predicate_registry_populated(self):
        """Test all predicates are registered"""
        predicates = list_predicates()

        # Check key predicates are registered
        assert "target_within_reach" in predicates
        assert "is_in_robot_workspace" in predicates
        assert "is_in_shared_zone" in predicates
        assert "robot_is_initialized" in predicates
        assert "robot_is_stationary" in predicates
        assert "gripper_is_open" in predicates
        assert "gripper_is_closed" in predicates
        assert "object_accessible_by_robot" in predicates
        assert "robots_will_collide" in predicates

        # Should have at least 9 predicates
        assert len(predicates) >= 9

    def test_evaluate_predicate_valid(self, mock_world_state):
        """Test dynamic predicate evaluation"""
        is_valid, reason = evaluate_predicate(
            "target_within_reach",
            robot_id="Robot1",
            x=0.0,
            y=0.0,
            z=0.2)

        assert is_valid is True

    def test_evaluate_predicate_unknown(self, mock_world_state):
        """Test unknown predicate evaluation"""
        is_valid, reason = evaluate_predicate(
            "unknown_predicate",
            robot_id="Robot1")

        assert is_valid is False
        assert "Unknown predicate" in reason
