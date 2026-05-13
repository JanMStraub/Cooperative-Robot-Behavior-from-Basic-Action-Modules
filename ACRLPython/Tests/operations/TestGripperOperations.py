#!/usr/bin/env python3
"""
Unit tests for GripperOperations.py

Tests the gripper control operations including:
- Open/close gripper commands
- Timeout handling
- Command broadcasting to Unity
- Failure recovery
- Invalid robot ID handling
- State validation
- Parameter validation
- Error handling
"""

import pytest
from unittest.mock import Mock, patch

from operations.GripperOperations import (
    control_gripper,
    CONTROL_GRIPPER_OPERATION,
    place_object,
    PLACE_OBJECT_OPERATION,
    place_between_objects,
    PLACE_BETWEEN_OBJECTS_OPERATION,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_broadcaster():
    """
    Create a mock CommandBroadcaster for testing.

    Returns:
        Mock CommandBroadcaster with send_command method
    """
    broadcaster = Mock()
    broadcaster.send_command = Mock(return_value=True)
    return broadcaster


# ============================================================================
# Test Class: Basic Gripper Operations
# ============================================================================


class TestGripperOperations:
    """Test basic gripper control operations."""

    def test_open_gripper_success(self, patch_command_broadcaster):
        """Test opening gripper successfully."""

        result = control_gripper("Robot1", open_gripper=True)

        assert result.success is True
        assert result.result is not None
        assert result.result["robot_id"] == "Robot1"
        assert result.result["open_gripper"] is True
        assert result.result["status"] == "command_sent"
        patch_command_broadcaster.send_command.assert_called_once()

    def test_close_gripper_success(self, patch_command_broadcaster):
        """Test closing gripper successfully."""

        result = control_gripper("Robot1", open_gripper=False)

        assert result.success is True
        assert result.result is not None
        assert result.result["robot_id"] == "Robot1"
        assert result.result["open_gripper"] is False
        patch_command_broadcaster.send_command.assert_called_once()

    def test_gripper_command_structure(self, patch_command_broadcaster):
        """Test that gripper command has correct structure."""

        result = control_gripper("Robot1", open_gripper=True, request_id=123)

        # Verify command was sent
        patch_command_broadcaster.send_command.assert_called_once()
        call_args = patch_command_broadcaster.send_command.call_args

        # Check command structure
        command = call_args[0][0]
        assert command["command_type"] == "control_gripper"
        assert command["robot_id"] == "Robot1"
        assert command["parameters"]["open_gripper"] is True
        assert "timestamp" in command

        # Check request_id parameter
        request_id = call_args[0][1]
        assert request_id == 123


# ============================================================================
# Test Class: Error Handling
# ============================================================================


class TestGripperErrors:
    """Test error handling for gripper operations."""

    def test_gripper_invalid_robot_id_empty(self, patch_command_broadcaster):
        """Test gripper control with empty robot ID."""

        result = control_gripper("", open_gripper=True)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_ROBOT_ID"

    def test_gripper_invalid_robot_id_none(self, patch_command_broadcaster):
        """Test gripper control with None robot ID."""

        result = control_gripper(None, open_gripper=True)  # type: ignore[arg-type]

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_ROBOT_ID"

    def test_gripper_invalid_parameter_type(self, patch_command_broadcaster):
        """Test gripper control with invalid open_gripper parameter type."""

        result = control_gripper("Robot1", open_gripper="yes")  # type: ignore[arg-type]

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_OPEN_GRIPPER_PARAMETER"

    def test_gripper_communication_failed(self, patch_command_broadcaster):
        """Test gripper control when communication fails."""
        patch_command_broadcaster.send_command = Mock(return_value=False)

        result = control_gripper("Robot1", open_gripper=True)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "COMMUNICATION_FAILED"

    def test_gripper_network_failure(self, patch_command_broadcaster):
        """Test gripper control when broadcaster raises exception."""
        patch_command_broadcaster.send_command = Mock(
            side_effect=Exception("Network error")
        )

        result = control_gripper("Robot1", open_gripper=True)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "UNEXPECTED_ERROR"


# ============================================================================
# Test Class: Command Broadcasting
# ============================================================================


class TestGripperBroadcasting:
    """Test command broadcasting to Unity."""

    def test_gripper_command_broadcast_open(self, patch_command_broadcaster):
        """Test opening gripper broadcasts correct command to Unity."""

        control_gripper("CustomRobot", open_gripper=True, request_id=999)

        call_args = patch_command_broadcaster.send_command.call_args
        command = call_args[0][0]
        request_id = call_args[0][1]

        assert command["robot_id"] == "CustomRobot"
        assert command["parameters"]["open_gripper"] is True
        assert request_id == 999

    def test_gripper_command_broadcast_close(self, patch_command_broadcaster):
        """Test closing gripper broadcasts correct command to Unity."""

        control_gripper("AR4_Robot", open_gripper=False, request_id=555)

        call_args = patch_command_broadcaster.send_command.call_args
        command = call_args[0][0]
        request_id = call_args[0][1]

        assert command["robot_id"] == "AR4_Robot"
        assert command["parameters"]["open_gripper"] is False
        assert request_id == 555


# ============================================================================
# Test Class: Operation Definition
# ============================================================================


class TestGripperOperationDefinition:
    """Test the BasicOperation definition for gripper control."""

    def test_operation_definition_exists(self):
        """Test that CONTROL_GRIPPER_OPERATION is properly defined."""
        assert CONTROL_GRIPPER_OPERATION is not None
        assert CONTROL_GRIPPER_OPERATION.name == "control_gripper"
        assert (
            CONTROL_GRIPPER_OPERATION.operation_id == "manipulation_control_gripper_001"
        )

    def test_operation_has_metadata(self):
        """Test that operation has required metadata."""
        op = CONTROL_GRIPPER_OPERATION

        assert op.description is not None
        assert len(op.parameters) >= 2  # robot_id, open_gripper
        assert op.preconditions is not None
        assert op.postconditions is not None
        assert op.implementation is not None

    def test_operation_execution_through_definition(self, patch_command_broadcaster):
        """Test executing operation through BasicOperation.execute()."""

        result = CONTROL_GRIPPER_OPERATION.execute(robot_id="Robot1", open_gripper=True)

        assert result.success is True


# ============================================================================
# Test Class: place_between_objects
# ============================================================================


def _make_ws_two_objects():
    """WorldState mock with blue and red cube at known positions."""
    ws = Mock()

    def _resolve(obj_id):
        return obj_id  # pass-through — IDs already canonical

    def _get_pos(obj_id):
        return {"blue": (0.1, 0.0, 0.2), "red": (0.5, 0.0, 0.6)}.get(obj_id)

    ws.resolve_canonical_id = Mock(side_effect=_resolve)
    ws.get_object_position = Mock(side_effect=_get_pos)
    ws.get_object_dimensions = Mock(return_value=None)
    return ws


class TestPlaceBetweenObjects:
    """Tests for place_between_objects operation."""

    def _patch_ws(self, monkeypatch, ws):
        try:
            import operations._imports as imp
            monkeypatch.setattr(imp, "get_world_state", lambda: ws)
        except (ImportError, AttributeError):
            pass
        try:
            import core.Imports as ci
            monkeypatch.setattr(ci, "get_world_state", lambda: ws)
        except (ImportError, AttributeError):
            pass

    def test_midpoint_computed_correctly(self, monkeypatch, patch_command_broadcaster):
        """Midpoint x/z correct for two known object positions."""
        ws = _make_ws_two_objects()
        self._patch_ws(monkeypatch, ws)

        result = place_between_objects("Robot1", "blue", "red")

        assert result.success is True
        # x: (0.1 + 0.5) / 2 = 0.3,  z: (0.2 + 0.6) / 2 = 0.4
        assert abs(result.result["placed_at"]["x"] - 0.3) < 1e-6
        assert abs(result.result["placed_at"]["z"] - 0.4) < 1e-6

    def test_midpoint_in_result_metadata(self, monkeypatch, patch_command_broadcaster):
        """Result includes midpoint and reference_objects metadata."""
        ws = _make_ws_two_objects()
        self._patch_ws(monkeypatch, ws)

        result = place_between_objects("Robot1", "blue", "red")

        assert result.success is True
        assert "midpoint" in result.result
        assert result.result["reference_objects"] == ["blue", "red"]

    def test_first_object_not_found(self, monkeypatch, patch_command_broadcaster):
        """Returns OBJECT_NOT_FOUND error when first object missing."""
        ws = Mock()
        ws.resolve_canonical_id = Mock(return_value=None)
        self._patch_ws(monkeypatch, ws)

        result = place_between_objects("Robot1", "missing", "red")

        assert result.success is False
        assert result.error["code"] == "OBJECT_NOT_FOUND"
        assert "missing" in result.error["message"]

    def test_second_object_not_found(self, monkeypatch, patch_command_broadcaster):
        """Returns OBJECT_NOT_FOUND error when second object missing."""
        ws = Mock()

        def _resolve(obj_id):
            return obj_id if obj_id == "blue" else None

        def _get_pos(obj_id):
            return (0.1, 0.0, 0.2) if obj_id == "blue" else None

        ws.resolve_canonical_id = Mock(side_effect=_resolve)
        ws.get_object_position = Mock(side_effect=_get_pos)
        self._patch_ws(monkeypatch, ws)

        result = place_between_objects("Robot1", "blue", "missing")

        assert result.success is False
        assert result.error["code"] == "OBJECT_NOT_FOUND"
        assert "missing" in result.error["message"]

    def test_invalid_robot_id(self, monkeypatch):
        """Returns INVALID_ROBOT_ID for empty robot_id."""
        result = place_between_objects("", "blue", "red")
        assert result.success is False
        assert result.error["code"] == "INVALID_ROBOT_ID"

    def test_operation_registered(self):
        """PLACE_BETWEEN_OBJECTS_OPERATION has required metadata."""
        op = PLACE_BETWEEN_OBJECTS_OPERATION
        param_names = {p.name for p in op.parameters}
        assert "object_id_1" in param_names
        assert "object_id_2" in param_names
        assert op.implementation is place_between_objects

    def test_default_y_used_when_no_on_top_of(self, monkeypatch, patch_command_broadcaster):
        """Explicit y passes through unchanged when on_top_of not given."""
        ws = _make_ws_two_objects()
        self._patch_ws(monkeypatch, ws)

        result = place_between_objects("Robot1", "blue", "red", y=0.08)

        assert result.success is True
        assert abs(result.result["placed_at"]["y"] - 0.08) < 1e-6


# ============================================================================
# Test Class: place_object — on_top_of stacking
# ============================================================================


def _make_world_state(
    obj_id: str = "target_cube",
    position=(0.0, 0.05, 0.0),
    dimensions=(0.05, 0.10, 0.05),
):
    """Build a minimal WorldState mock for place_object stacking tests."""
    ws = Mock()
    ws.resolve_canonical_id = Mock(return_value=obj_id)
    ws.get_object_position = Mock(return_value=position)
    ws.get_object_dimensions = Mock(return_value=dimensions)
    return ws


class TestPlaceObjectOnTopOf:
    """Tests for place_object's on_top_of stacking behaviour."""

    def _patch_ws(self, monkeypatch, ws):
        """Patch get_world_state in GripperOperations module."""
        import operations.GripperOperations as mod
        monkeypatch.setattr(mod, "_resolve_placement_y.__globals__", {}, raising=False)
        # Patch at the _imports level used by _resolve_placement_y
        try:
            import operations._imports as imp
            monkeypatch.setattr(imp, "get_world_state", lambda: ws)
        except (ImportError, AttributeError):
            pass
        try:
            import core.Imports as ci
            monkeypatch.setattr(ci, "get_world_state", lambda: ws)
        except (ImportError, AttributeError):
            pass

    def test_resolves_y_from_worldstate(self, monkeypatch, patch_command_broadcaster):
        """Placement Y computed from object center + half-height when on_top_of set."""
        ws = _make_world_state(position=(0.0, 0.05, 0.0), dimensions=(0.05, 0.10, 0.05))
        self._patch_ws(monkeypatch, ws)

        result = place_object("Robot1", x=0.0, y=0.0, z=0.0, on_top_of="target_cube")

        assert result.success is True
        # expected: 0.05 + 0.10/2 + 0.0/2 = 0.10
        placed_y = result.result["placed_at"]["y"]
        assert abs(placed_y - 0.10) < 1e-6, f"expected 0.10, got {placed_y}"
        assert result.result["resolution"] == "stacked_on:target_cube"

    def test_with_placed_object_height(self, monkeypatch, patch_command_broadcaster):
        """placed_object_height shifts the TCP up by half the held object's height."""
        ws = _make_world_state(position=(0.0, 0.05, 0.0), dimensions=(0.05, 0.10, 0.05))
        self._patch_ws(monkeypatch, ws)

        result = place_object(
            "Robot1", x=0.0, y=0.0, z=0.0,
            on_top_of="target_cube", placed_object_height=0.04,
        )

        assert result.success is True
        # expected: 0.05 + 0.05 + 0.02 = 0.12
        placed_y = result.result["placed_at"]["y"]
        assert abs(placed_y - 0.12) < 1e-6, f"expected 0.12, got {placed_y}"

    def test_object_not_found_fallback(self, monkeypatch, patch_command_broadcaster):
        """Falls back to explicit y when object not in WorldState."""
        ws = Mock()
        ws.resolve_canonical_id = Mock(return_value=None)
        self._patch_ws(monkeypatch, ws)

        result = place_object("Robot1", x=0.0, y=0.5, z=0.0, on_top_of="missing_obj")

        assert result.success is True
        assert abs(result.result["placed_at"]["y"] - 0.5) < 1e-6
        assert result.result["resolution"] == "fallback_object_not_found"

    def test_no_dimensions_fallback(self, monkeypatch, patch_command_broadcaster):
        """Falls back to explicit y when object has no dimensions (vision-only)."""
        ws = Mock()
        ws.resolve_canonical_id = Mock(return_value="target_cube")
        ws.get_object_position = Mock(return_value=(0.0, 0.05, 0.0))
        ws.get_object_dimensions = Mock(return_value=None)
        self._patch_ws(monkeypatch, ws)

        result = place_object("Robot1", x=0.0, y=0.5, z=0.0, on_top_of="target_cube")

        assert result.success is True
        assert abs(result.result["placed_at"]["y"] - 0.5) < 1e-6
        assert result.result["resolution"] == "fallback_no_dimensions"

    def test_no_position_fallback(self, monkeypatch, patch_command_broadcaster):
        """Falls back to explicit y when object position is None."""
        ws = Mock()
        ws.resolve_canonical_id = Mock(return_value="target_cube")
        ws.get_object_position = Mock(return_value=None)
        ws.get_object_dimensions = Mock(return_value=(0.05, 0.10, 0.05))
        self._patch_ws(monkeypatch, ws)

        result = place_object("Robot1", x=0.0, y=0.5, z=0.0, on_top_of="target_cube")

        assert result.success is True
        assert abs(result.result["placed_at"]["y"] - 0.5) < 1e-6
        assert result.result["resolution"] == "fallback_no_position"

    def test_none_on_top_of_uses_explicit_coords(self, patch_command_broadcaster):
        """Without on_top_of, explicit y passes through unchanged."""
        result = place_object("Robot1", x=0.0, y=0.42, z=0.0)

        assert result.success is True
        assert abs(result.result["placed_at"]["y"] - 0.42) < 1e-6
        assert result.result["resolution"] == "explicit_coords"

    def test_tcp_command_y_matches_computed(self, monkeypatch, patch_command_broadcaster):
        """The y value in the Unity command dict equals the computed effective_y."""
        ws = _make_world_state(position=(0.0, 0.10, 0.0), dimensions=(0.05, 0.20, 0.05))
        self._patch_ws(monkeypatch, ws)

        place_object("Robot1", x=0.0, y=0.0, z=0.0, on_top_of="target_cube")

        call_args = patch_command_broadcaster.send_command.call_args
        command = call_args[0][0]
        sent_y = command["parameters"]["target_position"]["y"]
        # expected: 0.10 + 0.10 = 0.20
        assert abs(sent_y - 0.20) < 1e-6, f"expected 0.20, got {sent_y}"

    def test_place_object_operation_has_on_top_of_param(self):
        """PLACE_OBJECT_OPERATION metadata exposes on_top_of and placed_object_height."""
        param_names = {p.name for p in PLACE_OBJECT_OPERATION.parameters}
        assert "on_top_of" in param_names
        assert "placed_object_height" in param_names
