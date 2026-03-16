#!/usr/bin/env python3
"""
Unit tests for CoordinationOperations.py

Tests the multi-robot coordination operations including:
- detect_other_robot: Robot detection in workspace
- mirror_movement_of_other_robot: Movement mirroring
- Multi-robot workspace safety
- Coordination with missing/offline robots
"""

import pytest
from unittest.mock import Mock

from operations.CoordinationOperations import (
    detect_other_robot,
    mirror_movement_of_other_robot,
    DETECT_OTHER_ROBOT_OPERATION,
    MIRROR_MOVEMENT_OPERATION,
)


# ============================================================================
# Test Class: detect_other_robot - Robot Detection
# ============================================================================


class TestDetectOtherRobot:
    """Test robot-to-robot detection operation."""

    def test_detect_other_robot_success(
        self, mock_world_state_multi_robot, patch_world_state
    ):
        """Test successful detection of another robot."""
        # Mock robot states with end effector positions
        mock_world_state_multi_robot.get_robot_state = Mock(
            side_effect=lambda rid: {
                "Robot1": {
                    "robot_id": "Robot1",
                    "end_effector_position": {"x": -0.3, "y": 0.0, "z": 0.1},
                },
                "Robot2": {
                    "robot_id": "Robot2",
                    "end_effector_position": {"x": 0.3, "y": 0.0, "z": 0.1},
                },
            }.get(rid)
        )

        with patch_world_state(mock_world_state_multi_robot):
            result = detect_other_robot("Robot1", "Robot2")

        assert result.success is True
        assert result.result is not None
        assert result.result["robot_id"] == "Robot1"
        assert result.result["target_robot_id"] == "Robot2"
        assert result.result["detected"] is True
        assert "position" in result.result
        assert "distance" in result.result
        # Distance should be approximately 0.6m (0.3 - (-0.3))
        assert 0.55 <= result.result["distance"] <= 0.65

    def test_detect_other_robot_invalid_robot_id(self):
        """Test detection with invalid robot ID."""
        result = detect_other_robot("", "Robot2")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_ROBOT_ID"

    def test_detect_other_robot_invalid_target_robot_id(self):
        """Test detection with invalid target robot ID."""
        result = detect_other_robot("Robot1", "")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_TARGET_ROBOT_ID"

    def test_detect_other_robot_target_not_found(
        self, mock_world_state_multi_robot, patch_world_state
    ):
        """Test detection when target robot not in world state."""
        # Mock get_robot_state to return None for target
        mock_world_state_multi_robot.get_robot_state = Mock(
            side_effect=lambda rid: {
                "Robot1": {
                    "robot_id": "Robot1",
                    "end_effector_position": {"x": -0.3, "y": 0.0, "z": 0.1},
                }
            }.get(rid)
        )

        with patch_world_state(mock_world_state_multi_robot):
            result = detect_other_robot("Robot1", "Robot3")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "TARGET_ROBOT_NOT_FOUND"

    def test_detect_other_robot_detector_not_found(
        self, mock_world_state_multi_robot, patch_world_state
    ):
        """Test detection when detecting robot not in world state."""
        # Mock get_robot_state to return None for detector
        mock_world_state_multi_robot.get_robot_state = Mock(
            side_effect=lambda rid: {
                "Robot2": {
                    "robot_id": "Robot2",
                    "end_effector_position": {"x": 0.3, "y": 0.0, "z": 0.1},
                }
            }.get(rid)
        )

        with patch_world_state(mock_world_state_multi_robot):
            result = detect_other_robot("Robot3", "Robot2")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "DETECTOR_ROBOT_NOT_FOUND"

    def test_detect_other_robot_missing_position_data(
        self, mock_world_state_multi_robot, patch_world_state
    ):
        """Test detection when position data is missing."""
        # Mock robot states without position data
        mock_world_state_multi_robot.get_robot_state = Mock(
            side_effect=lambda rid: {
                "Robot1": {"robot_id": "Robot1"},
                "Robot2": {"robot_id": "Robot2"},
            }.get(rid)
        )

        with patch_world_state(mock_world_state_multi_robot):
            result = detect_other_robot("Robot1", "Robot2")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "POSITION_DATA_MISSING"

    def test_detect_other_robot_uses_base_position_fallback(
        self, mock_world_state_multi_robot, patch_world_state
    ):
        """Test detection falls back to base position if end effector position unavailable."""
        # Mock robot states with only base position
        mock_world_state_multi_robot.get_robot_state = Mock(
            side_effect=lambda rid: {
                "Robot1": {
                    "robot_id": "Robot1",
                    "position": {"x": -0.3, "y": 0.0, "z": 0.0},
                },
                "Robot2": {
                    "robot_id": "Robot2",
                    "position": {"x": 0.3, "y": 0.0, "z": 0.0},
                },
            }.get(rid)
        )

        with patch_world_state(mock_world_state_multi_robot):
            result = detect_other_robot("Robot1", "Robot2")

        assert result.success is True
        assert result.result is not None
        assert result.result["distance"] == pytest.approx(0.6, abs=0.01)

    def test_detect_other_robot_with_camera_id(
        self, mock_world_state_multi_robot, patch_world_state
    ):
        """Test detection with custom camera ID."""
        mock_world_state_multi_robot.get_robot_state = Mock(
            side_effect=lambda rid: {
                "Robot1": {
                    "robot_id": "Robot1",
                    "end_effector_position": {"x": -0.3, "y": 0.0, "z": 0.1},
                },
                "Robot2": {
                    "robot_id": "Robot2",
                    "end_effector_position": {"x": 0.3, "y": 0.0, "z": 0.1},
                },
            }.get(rid)
        )

        with patch_world_state(mock_world_state_multi_robot):
            result = detect_other_robot("Robot1", "Robot2", camera_id="stereo")

        assert result.success is True
        assert result.result is not None
        assert result.result["camera_id"] == "stereo"


# ============================================================================
# Test Class: mirror_movement_of_other_robot - Movement Mirroring
# ============================================================================


class TestMirrorMovement:
    """Test robot movement mirroring operation."""

    def test_mirror_movement_success(self, patch_command_broadcaster):
        """Test successful movement mirroring activation."""
        result = mirror_movement_of_other_robot("Robot2", "Robot1", "x")

        assert result.success is True
        assert result.result is not None
        assert result.result["robot_id"] == "Robot2"
        assert result.result["target_robot_id"] == "Robot1"
        assert result.result["mirror_axis"] == "x"
        assert result.result["scale_factor"] == 1.0
        assert result.result["status"] == "mirroring_active"
        patch_command_broadcaster.send_command.assert_called_once()

    def test_mirror_movement_with_scale_factor(self, patch_command_broadcaster):
        """Test mirroring with custom scale factor."""
        result = mirror_movement_of_other_robot(
            "Robot2", "Robot1", "x", scale_factor=0.5
        )

        assert result.success is True
        assert result.result is not None
        assert result.result["scale_factor"] == 0.5

    def test_mirror_movement_all_axes(self, patch_command_broadcaster):
        """Test mirroring on different axes."""
        for axis in ["x", "y", "z", "none"]:
            result = mirror_movement_of_other_robot("Robot2", "Robot1", axis)
            assert result.success is True
            assert result.result is not None
            assert result.result["mirror_axis"] == axis

    def test_mirror_movement_invalid_robot_id(self):
        """Test mirroring with invalid robot ID."""
        result = mirror_movement_of_other_robot("", "Robot1", "x")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_ROBOT_ID"

    def test_mirror_movement_invalid_target_robot_id(self):
        """Test mirroring with invalid target robot ID."""
        result = mirror_movement_of_other_robot("Robot2", "", "x")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_TARGET_ROBOT_ID"

    def test_mirror_movement_invalid_mirror_axis(self):
        """Test mirroring with invalid mirror axis."""
        result = mirror_movement_of_other_robot("Robot2", "Robot1", "invalid")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_MIRROR_AXIS"

    def test_mirror_movement_invalid_scale_factor_too_low(self):
        """Test mirroring with scale factor below minimum."""
        result = mirror_movement_of_other_robot(
            "Robot2", "Robot1", "x", scale_factor=0.05
        )

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_SCALE_FACTOR"

    def test_mirror_movement_invalid_scale_factor_too_high(self):
        """Test mirroring with scale factor above maximum."""
        result = mirror_movement_of_other_robot(
            "Robot2", "Robot1", "x", scale_factor=3.0
        )

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_SCALE_FACTOR"

    def test_mirror_movement_command_structure(self, patch_command_broadcaster):
        """Test that mirror command has correct structure."""
        result = mirror_movement_of_other_robot(
            "Robot2", "Robot1", "y", scale_factor=1.5, request_id=456
        )

        patch_command_broadcaster.send_command.assert_called_once()
        call_args = patch_command_broadcaster.send_command.call_args

        command = call_args[0][0]
        assert command["command_type"] == "mirror_movement"
        assert command["robot_id"] == "Robot2"
        assert command["parameters"]["target_robot_id"] == "Robot1"
        assert command["parameters"]["mirror_axis"] == "y"
        assert command["parameters"]["scale_factor"] == 1.5
        assert "timestamp" in command

        request_id = call_args[0][1]
        assert request_id == 456

    def test_mirror_movement_communication_failed(self, patch_command_broadcaster):
        """Test mirroring when communication fails."""
        patch_command_broadcaster.send_command = Mock(return_value=False)

        result = mirror_movement_of_other_robot("Robot2", "Robot1", "x")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "COMMUNICATION_FAILED"

    def test_mirror_movement_network_error(self, patch_command_broadcaster):
        """Test mirroring when broadcaster raises exception."""
        patch_command_broadcaster.send_command = Mock(
            side_effect=Exception("Network error")
        )

        result = mirror_movement_of_other_robot("Robot2", "Robot1", "x")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "UNEXPECTED_ERROR"


# ============================================================================
# Test Class: Operation Definitions
# ============================================================================


class TestCoordinationOperationDefinitions:
    """Test BasicOperation definitions for coordination operations."""

    def test_detect_other_robot_operation_definition(self):
        """Test DETECT_OTHER_ROBOT_OPERATION is properly defined."""
        assert DETECT_OTHER_ROBOT_OPERATION is not None
        assert DETECT_OTHER_ROBOT_OPERATION.name == "detect_other_robot"
        assert (
            DETECT_OTHER_ROBOT_OPERATION.operation_id == "coordination_detect_robot_001"
        )

    def test_detect_other_robot_has_metadata(self):
        """Test detect operation has required metadata."""
        op = DETECT_OTHER_ROBOT_OPERATION

        assert op.description is not None
        assert len(op.parameters) >= 2  # robot_id, target_robot_id minimum
        assert op.preconditions is not None
        assert op.postconditions is not None
        assert op.implementation is not None

    def test_mirror_movement_operation_definition(self):
        """Test MIRROR_MOVEMENT_OPERATION is properly defined."""
        assert MIRROR_MOVEMENT_OPERATION is not None
        assert MIRROR_MOVEMENT_OPERATION.name == "mirror_movement_of_other_robot"
        assert (
            MIRROR_MOVEMENT_OPERATION.operation_id == "coordination_mirror_movement_002"
        )

    def test_mirror_movement_has_metadata(self):
        """Test mirror operation has required metadata."""
        op = MIRROR_MOVEMENT_OPERATION

        assert op.description is not None
        assert len(op.parameters) >= 2  # robot_id, target_robot_id minimum
        assert op.preconditions is not None
        assert op.postconditions is not None
        assert op.implementation is not None

    def test_detect_operation_execution_through_definition(
        self, mock_world_state_multi_robot, patch_world_state
    ):
        """Test executing detect operation through BasicOperation.execute()."""
        mock_world_state_multi_robot.get_robot_state = Mock(
            side_effect=lambda rid: {
                "Robot1": {
                    "robot_id": "Robot1",
                    "end_effector_position": {"x": -0.3, "y": 0.0, "z": 0.1},
                },
                "Robot2": {
                    "robot_id": "Robot2",
                    "end_effector_position": {"x": 0.3, "y": 0.0, "z": 0.1},
                },
            }.get(rid)
        )

        with patch_world_state(mock_world_state_multi_robot):
            result = DETECT_OTHER_ROBOT_OPERATION.execute(
                robot_id="Robot1", target_robot_id="Robot2"
            )

        assert result.success is True

    def test_mirror_operation_execution_through_definition(
        self, patch_command_broadcaster
    ):
        """Test executing mirror operation through BasicOperation.execute()."""
        result = MIRROR_MOVEMENT_OPERATION.execute(
            robot_id="Robot2", target_robot_id="Robot1", mirror_axis="x"
        )

        assert result.success is True


# ============================================================================
# Test Class: Concurrent Execution
# ============================================================================


class TestCoordinationConcurrency:
    """Test thread safety for coordination operations."""

    def test_concurrent_detections(
        self, mock_world_state_multi_robot, patch_world_state
    ):
        """Test concurrent robot detection from multiple threads."""
        import threading

        mock_world_state_multi_robot.get_robot_state = Mock(
            side_effect=lambda rid: {
                "Robot1": {
                    "robot_id": "Robot1",
                    "end_effector_position": {"x": -0.3, "y": 0.0, "z": 0.1},
                },
                "Robot2": {
                    "robot_id": "Robot2",
                    "end_effector_position": {"x": 0.3, "y": 0.0, "z": 0.1},
                },
            }.get(rid)
        )

        results = []
        lock = threading.Lock()

        # Patch once at test level, not inside threads (patch is not thread-safe)
        with patch_world_state(mock_world_state_multi_robot):

            def detect_worker():
                result = detect_other_robot("Robot1", "Robot2")
                with lock:
                    results.append(result)

            threads = [threading.Thread(target=detect_worker) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(results) == 5
        assert all(r.success for r in results)

    def test_concurrent_mirror_commands(self, patch_command_broadcaster):
        """Test concurrent mirroring commands don't interfere."""
        import threading

        results = []

        def mirror_worker(robot_id, target_id):
            result = mirror_movement_of_other_robot(robot_id, target_id, "x")
            results.append(result)

        threads = [
            threading.Thread(target=mirror_worker, args=(f"Robot{i}", "Robot1"))
            for i in range(2, 6)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 4
        assert all(r.success for r in results)


# ============================================================================
# Test Class: Edge Cases
# ============================================================================


class TestCoordinationEdgeCases:
    """Test edge cases for coordination operations."""

    def test_detect_robot_same_id(
        self, mock_world_state_multi_robot, patch_world_state
    ):
        """Test detecting a robot with the same ID (should still work)."""
        mock_world_state_multi_robot.get_robot_state = Mock(
            return_value={
                "robot_id": "Robot1",
                "end_effector_position": {"x": -0.3, "y": 0.0, "z": 0.1},
            }
        )

        # Detecting self should work (distance = 0)
        with patch_world_state(mock_world_state_multi_robot):
            result = detect_other_robot("Robot1", "Robot1")

        assert result.success is True
        assert result.result is not None
        assert result.result["distance"] == pytest.approx(0.0)

    def test_mirror_with_boundary_scale_factors(self, patch_command_broadcaster):
        """Test mirroring with boundary scale factor values."""
        # Minimum valid scale factor
        result = mirror_movement_of_other_robot(
            "Robot2", "Robot1", "x", scale_factor=0.1
        )
        assert result.success is True

        # Maximum valid scale factor
        result = mirror_movement_of_other_robot(
            "Robot2", "Robot1", "x", scale_factor=2.0
        )
        assert result.success is True

    def test_detect_with_large_distance(
        self, mock_world_state_multi_robot, patch_world_state
    ):
        """Test detection with robots far apart."""
        mock_world_state_multi_robot.get_robot_state = Mock(
            side_effect=lambda rid: {
                "Robot1": {
                    "robot_id": "Robot1",
                    "end_effector_position": {"x": -1.0, "y": -1.0, "z": 0.0},
                },
                "Robot2": {
                    "robot_id": "Robot2",
                    "end_effector_position": {"x": 1.0, "y": 1.0, "z": 0.5},
                },
            }.get(rid)
        )

        with patch_world_state(mock_world_state_multi_robot):
            result = detect_other_robot("Robot1", "Robot2")

        assert result.success is True
        assert result.result is not None
        # Distance should be sqrt((2)^2 + (2)^2 + (0.5)^2) ≈ 2.915
        assert result.result["distance"] > 2.8
        assert result.result["distance"] < 3.0
