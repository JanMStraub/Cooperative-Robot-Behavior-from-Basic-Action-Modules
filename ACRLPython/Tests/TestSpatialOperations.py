#!/usr/bin/env python3
"""
Unit tests for SpatialOperations.py

Tests spatial relationship operations including relative positioning,
interpolation between objects, and region-based movement.
"""

import pytest
from unittest.mock import Mock, patch, call
from operations.SpatialOperations import (
    move_relative_to_object,
    move_between_objects,
    move_to_region,
)
from operations.Base import OperationResult
import LLMConfig as cfg


class TestMoveRelativeToObject:
    """Test move_relative_to_object operation"""

    @patch('operations.SpatialOperations.get_world_state')
    @patch('operations.SpatialOperations.move_to_coordinate')
    @patch('operations.SpatialOperations.get_command_broadcaster')
    def test_move_relative_left_of(self, mock_broadcast, mock_move, mock_get_ws, mock_world_state_with_objects):
        """Test left_of relation with offset"""
        mock_get_ws.return_value = mock_world_state_with_objects
        mock_move.return_value = OperationResult.success_result({"final_position": (0.15, 0.2, 0.1)})
        result = move_relative_to_object(
            robot_id="Robot1",
            object_ref="cube_01",
            relation="left_of",
            offset=0.15)

        assert result.success is True
        # cube_01 is at (0.3, 0.2, 0.1), left_of means x - offset
        mock_move.assert_called_once()
        args = mock_move.call_args[1]
        assert args["robot_id"] == "Robot1"
        assert args["x"] == pytest.approx(0.3 - 0.15)  # 0.15
        assert args["y"] == pytest.approx(0.2)
        assert args["z"] == pytest.approx(0.1)

    @patch('operations.SpatialOperations.get_world_state')
    @patch('operations.SpatialOperations.move_to_coordinate')
    @patch('operations.SpatialOperations.get_command_broadcaster')
    def test_move_relative_right_of(self, mock_broadcast, mock_move, mock_get_ws, mock_world_state_with_objects):
        mock_get_ws.return_value = mock_world_state_with_objects
        mock_move.return_value = OperationResult.success_result({"final_position": (0.0, 0.0, 0.0)})
        """Test right_of relation"""
        result = move_relative_to_object(
            robot_id="Robot1",
            object_ref="cube_01",
            relation="right_of",
            offset=0.15)

        assert result.success is True
        # right_of means x + offset
        args = mock_move.call_args[1]
        assert args["x"] == pytest.approx(0.3 + 0.15)  # 0.45

    @patch('operations.SpatialOperations.get_world_state')
    @patch('operations.SpatialOperations.move_to_coordinate')
    @patch('operations.SpatialOperations.get_command_broadcaster')
    def test_move_relative_above(self, mock_broadcast, mock_move, mock_get_ws, mock_world_state_with_objects):
        mock_get_ws.return_value = mock_world_state_with_objects
        mock_move.return_value = OperationResult.success_result({"final_position": (0.0, 0.0, 0.0)})
        """Test above relation"""
        result = move_relative_to_object(
            robot_id="Robot1",
            object_ref="cube_01",
            relation="above",
            offset=0.15)

        assert result.success is True
        # above means z + offset
        args = mock_move.call_args[1]
        assert args["z"] == pytest.approx(0.1 + 0.15)  # 0.25

    @patch('operations.SpatialOperations.get_world_state')
    @patch('operations.SpatialOperations.move_to_coordinate')
    @patch('operations.SpatialOperations.get_command_broadcaster')
    def test_move_relative_below(self, mock_broadcast, mock_move, mock_get_ws, mock_world_state_with_objects):
        mock_get_ws.return_value = mock_world_state_with_objects
        mock_move.return_value = OperationResult.success_result({"final_position": (0.0, 0.0, 0.0)})
        """Test below relation"""
        result = move_relative_to_object(
            robot_id="Robot1",
            object_ref="cube_01",
            relation="below",
            offset=0.1)

        assert result.success is True
        # below means z - offset
        args = mock_move.call_args[1]
        assert args["z"] == pytest.approx(0.1 - 0.1)  # 0.0

    @patch('operations.SpatialOperations.get_world_state')
    @patch('operations.SpatialOperations.move_to_coordinate')
    @patch('operations.SpatialOperations.get_command_broadcaster')
    def test_move_relative_in_front_of(self, mock_broadcast, mock_move, mock_get_ws, mock_world_state_with_objects):
        mock_get_ws.return_value = mock_world_state_with_objects
        mock_move.return_value = OperationResult.success_result({"final_position": (0.0, 0.0, 0.0)})
        """Test in_front_of relation"""
        result = move_relative_to_object(
            robot_id="Robot1",
            object_ref="cube_01",
            relation="in_front_of",
            offset=0.15)

        assert result.success is True
        # in_front_of means y + offset
        args = mock_move.call_args[1]
        assert args["y"] == pytest.approx(0.2 + 0.15)  # 0.35

    @patch('operations.SpatialOperations.get_world_state')
    @patch('operations.SpatialOperations.move_to_coordinate')
    @patch('operations.SpatialOperations.get_command_broadcaster')
    def test_move_relative_behind(self, mock_broadcast, mock_move, mock_get_ws, mock_world_state_with_objects):
        mock_get_ws.return_value = mock_world_state_with_objects
        mock_move.return_value = OperationResult.success_result({"final_position": (0.0, 0.0, 0.0)})
        """Test behind relation"""
        result = move_relative_to_object(
            robot_id="Robot1",
            object_ref="cube_01",
            relation="behind",
            offset=0.15)

        assert result.success is True
        # behind means y - offset
        args = mock_move.call_args[1]
        assert args["y"] == pytest.approx(0.2 - 0.15)  # 0.05

    @patch('operations.SpatialOperations.move_to_coordinate')
    @patch('operations.SpatialOperations.get_command_broadcaster')
    def test_move_relative_with_position(self, mock_broadcast, mock_move):
        """Test using direct position instead of object_id"""
        mock_broadcast.return_value = Mock()
        mock_broadcast.return_value.send_command = Mock(return_value=True)
        mock_move.return_value = OperationResult.success_result({"final_position": (0.5, 0.3, 0.3)})
        result = move_relative_to_object(
            robot_id="Robot1",
            object_ref=(0.5, 0.3, 0.2),  # Direct position tuple
            relation="above",
            offset=0.1
        )

        assert result.success is True
        args = mock_move.call_args[1]
        assert args["x"] == pytest.approx(0.5)
        assert args["y"] == pytest.approx(0.3)
        assert args["z"] == pytest.approx(0.2 + 0.1)  # 0.3

    @patch('operations.SpatialOperations.get_world_state')
    @patch('operations.SpatialOperations.move_to_coordinate')
    @patch('operations.SpatialOperations.get_command_broadcaster')
    def test_move_relative_z_override(self, mock_broadcast, mock_move, mock_get_ws, mock_world_state_with_objects):
        """Test Z coordinate override"""
        mock_get_ws.return_value = mock_world_state_with_objects
        mock_move.return_value = OperationResult.success_result({"final_position": (0.15, 0.2, 0.25)})
        result = move_relative_to_object(
            robot_id="Robot1",
            object_ref="cube_01",
            relation="left_of",
            offset=0.15,
            z_override=0.25)

        assert result.success is True
        args = mock_move.call_args[1]
        assert args["z"] == pytest.approx(0.25)  # Overridden value

    def test_move_relative_invalid_relation(self):
        """Test invalid relation error"""
        # Use position tuple to bypass world state lookup
        result = move_relative_to_object(
            robot_id="Robot1",
            object_ref=(0.3, 0.2, 0.1),  # Use position directly
            relation="invalid_relation",
            offset=0.1)

        assert result.success is False
        assert result.error is not None
        assert "Invalid relation" in result.error["message"]

    @patch('operations.SpatialOperations.get_world_state')
    def test_move_relative_object_not_found(self, mock_get_ws):
        """Test object not found error"""
        # Create world state with no objects
        world_state = Mock()
        world_state.get_object_position = Mock(return_value=None)
        mock_get_ws.return_value = world_state

        result = move_relative_to_object(
            robot_id="Robot1",
            object_ref="nonexistent",
            relation="above",
            offset=0.1)

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error["message"].lower()


class TestMoveBetweenObjects:
    """Test move_between_objects operation"""

    @patch('operations.SpatialOperations.get_world_state')
    @patch('operations.SpatialOperations.get_command_broadcaster')
    @patch('operations.SpatialOperations.move_to_coordinate')
    def test_move_between_midpoint(self, mock_move, mock_broadcast, mock_get_ws, mock_world_state_with_objects):
        mock_get_ws.return_value = mock_world_state_with_objects
        mock_move.return_value = OperationResult.success_result({"final_position": (0.35, 0.25, 0.1)})
        """Test bias=0.5 calculates center"""
        result = move_between_objects(
            robot_id="Robot1",
            object1="cube_01",
            object2="cube_02",
            bias=0.5)

        assert result.success is True
        # cube_01 at (0.3, 0.2, 0.1), cube_02 at (0.4, 0.3, 0.1)
        # Midpoint: (0.35, 0.25, 0.1)
        args = mock_move.call_args[1]
        assert args["x"] == pytest.approx(0.35)
        assert args["y"] == pytest.approx(0.25)
        assert args["z"] == pytest.approx(0.1)

    @patch('operations.SpatialOperations.get_world_state')
    @patch('operations.SpatialOperations.get_command_broadcaster')
    @patch('operations.SpatialOperations.move_to_coordinate')
    def test_move_between_bias_towards_first(self, mock_move, mock_broadcast, mock_get_ws, mock_world_state_with_objects):
        mock_get_ws.return_value = mock_world_state_with_objects
        mock_move.return_value = OperationResult.success_result({"final_position": (0.33, 0.23, 0.1)})
        """Test bias=0.3 closer to object1"""
        result = move_between_objects(
            robot_id="Robot1",
            object1="cube_01",
            object2="cube_02",
            bias=0.3)

        assert result.success is True
        # bias=0.3 means 70% towards object1, 30% towards object2
        # x: 0.3 + 0.3 * (0.4 - 0.3) = 0.3 + 0.03 = 0.33
        # y: 0.2 + 0.3 * (0.3 - 0.2) = 0.2 + 0.03 = 0.23
        args = mock_move.call_args[1]
        assert args["x"] == pytest.approx(0.33)
        assert args["y"] == pytest.approx(0.23)

    @patch('operations.SpatialOperations.get_world_state')
    @patch('operations.SpatialOperations.get_command_broadcaster')
    @patch('operations.SpatialOperations.move_to_coordinate')
    def test_move_between_bias_towards_second(self, mock_move, mock_broadcast, mock_get_ws, mock_world_state_with_objects):
        mock_get_ws.return_value = mock_world_state_with_objects
        mock_move.return_value = OperationResult.success_result({"final_position": (0.37, 0.27, 0.1)})
        """Test bias=0.7 closer to object2"""
        result = move_between_objects(
            robot_id="Robot1",
            object1="cube_01",
            object2="cube_02",
            bias=0.7)

        assert result.success is True
        # bias=0.7 means 30% towards object1, 70% towards object2
        # x: 0.3 + 0.7 * (0.4 - 0.3) = 0.3 + 0.07 = 0.37
        args = mock_move.call_args[1]
        assert args["x"] == pytest.approx(0.37)

    @patch('operations.SpatialOperations.get_world_state')
    @patch('operations.SpatialOperations.get_command_broadcaster')
    @patch('operations.SpatialOperations.move_to_coordinate')
    def test_move_between_z_offset(self, mock_move, mock_broadcast, mock_get_ws, mock_world_state_with_objects):
        mock_get_ws.return_value = mock_world_state_with_objects
        mock_move.return_value = OperationResult.success_result({"final_position": (0.35, 0.25, 0.25)})
        """Test vertical offset application"""
        result = move_between_objects(
            robot_id="Robot1",
            object1="cube_01",
            object2="cube_02",
            bias=0.5,
            z_offset=0.15)

        assert result.success is True
        # Midpoint z is 0.1, plus offset 0.15 = 0.25
        args = mock_move.call_args[1]
        assert args["z"] == pytest.approx(0.25)

    @patch('operations.SpatialOperations.get_world_state')
    @patch('operations.SpatialOperations.get_command_broadcaster')
    @patch('operations.SpatialOperations.move_to_coordinate')
    def test_move_between_with_positions(self, mock_move, mock_broadcast, mock_get_ws):
        """Test using direct position tuples"""
        mock_move.return_value = OperationResult.success_result({"final_position": (0.4, 0.3, 0.1)})
        result = move_between_objects(
            robot_id="Robot1",
            object1=(0.2, 0.1, 0.05),
            object2=(0.6, 0.5, 0.15),
            bias=0.5
        )

        assert result.success is True
        # Midpoint: (0.4, 0.3, 0.1)
        args = mock_move.call_args[1]
        assert args["x"] == pytest.approx(0.4)
        assert args["y"] == pytest.approx(0.3)
        assert args["z"] == pytest.approx(0.1)

    @patch('operations.SpatialOperations.get_world_state')
    def test_move_between_object_not_found(self, mock_get_ws):
        """Test object not found error"""
        world_state = Mock()
        world_state.get_object_position = Mock(return_value=None)
        mock_get_ws.return_value = world_state

        result = move_between_objects(
            robot_id="Robot1",
            object1="nonexistent",
            object2="cube_02",
            bias=0.5)

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error["message"].lower()


class TestMoveToRegion:
    """Test move_to_region operation"""

    @patch('operations.SpatialOperations.get_world_state')
    @patch('operations.SpatialOperations.get_command_broadcaster')
    @patch('operations.SpatialOperations.move_to_coordinate')
    def test_move_to_region_center(self, mock_move, mock_broadcast, mock_get_ws):
        """Test center position calculation"""
        mock_move.return_value = OperationResult.success_result({"final_position": (-0.55, 0.0, 0.25)})
        result = move_to_region(
            robot_id="Robot1",
            region_name="left_workspace",
            position_in_region="center"
        )

        assert result.success is True
        # left_workspace: x_min=-1.0, x_max=-0.1, y_min=-1.0, y_max=1.0, z_min=0.0, z_max=0.5
        # Center: x=-0.55, y=0.0, z=0.25
        args = mock_move.call_args[1]
        assert args["x"] == pytest.approx(-0.55)
        assert args["y"] == pytest.approx(0.0)
        assert args["z"] == pytest.approx(0.25)

    @patch('operations.SpatialOperations.get_world_state')
    @patch('operations.SpatialOperations.get_command_broadcaster')
    @patch('operations.SpatialOperations.move_to_coordinate')
    def test_move_to_region_near(self, mock_move, mock_broadcast, mock_get_ws):
        """Test near position (near the workspace edge closest to robot)"""
        mock_move.return_value = OperationResult.success_result({"final_position": (-0.4, 0.0, 0.15)})
        result = move_to_region(
            robot_id="Robot1",
            region_name="left_workspace",
            position_in_region="near"
        )

        assert result.success is True
        # Robot1 base at (-0.3, 0.0, 0.0), left_workspace: x_min=-1.0, x_max=-0.1
        # For left-side robot, "near" means near the right edge: x_max - 0.1 = -0.2
        args = mock_move.call_args[1]
        assert args["x"] == pytest.approx(-0.2)
        assert args["y"] == pytest.approx(0.0)
        assert args["z"] == pytest.approx(0.25)

    @patch('operations.SpatialOperations.get_world_state')
    @patch('operations.SpatialOperations.get_command_broadcaster')
    @patch('operations.SpatialOperations.move_to_coordinate')
    def test_move_to_region_far(self, mock_move, mock_broadcast, mock_get_ws):
        """Test far position (farther from robot base)"""
        mock_move.return_value = OperationResult.success_result({"final_position": (-0.7, 0.0, 0.35)})
        result = move_to_region(
            robot_id="Robot1",
            region_name="left_workspace",
            position_in_region="far"
        )

        assert result.success is True
        # Far position should be farther from robot base
        args = mock_move.call_args[1]
        # Far position should have x farther than center from -0.3
        assert args["x"] < -0.55

    @patch('operations.SpatialOperations.get_world_state')
    @patch('operations.SpatialOperations.get_command_broadcaster')
    @patch('operations.SpatialOperations.move_to_coordinate')
    def test_move_to_all_regions(self, mock_move, mock_broadcast, mock_get_ws):
        """Test all workspace regions"""
        mock_move.return_value = OperationResult.success_result({"final_position": (0.0, 0.0, 0.2)})
        regions = ["left_workspace", "right_workspace", "shared_zone", "center"]

        for region in regions:
            result = move_to_region(
                robot_id="Robot1",
                region_name=region,
                position_in_region="center"
            )

            assert result.success is True, f"Failed for region: {region}"
            mock_move.assert_called()
            mock_move.reset_mock()
            # Reset mock return value after reset_mock
            mock_move.return_value = OperationResult.success_result({"final_position": (0.0, 0.0, 0.2)})

    @patch('operations.SpatialOperations.get_world_state')
    @patch('operations.SpatialOperations.get_command_broadcaster')
    @patch('operations.SpatialOperations.move_to_coordinate')
    def test_move_to_region_z_override(self, mock_move, mock_broadcast, mock_get_ws):
        """Test custom Z height"""
        mock_move.return_value = OperationResult.success_result({"final_position": (0.0, 0.0, 0.35)})
        result = move_to_region(
            robot_id="Robot1",
            region_name="shared_zone",
            position_in_region="center",
            z_height=0.35
        )

        assert result.success is True
        args = mock_move.call_args[1]
        assert args["z"] == pytest.approx(0.35)  # Custom height

    def test_move_to_region_invalid_region(self):
        """Test invalid region error"""
        result = move_to_region(
            robot_id="Robot1",
            region_name="invalid_region",
            position_in_region="center"
        )

        assert result.success is False
        assert result.error is not None
        assert "Unknown region" in result.error["message"]

    def test_move_to_region_invalid_position(self):
        """Test invalid position type error"""
        result = move_to_region(
            robot_id="Robot1",
            region_name="left_workspace",
            position_in_region="invalid_position"
        )

        assert result.success is False
        assert result.error is not None
        assert "Invalid position" in result.error["message"]


class TestSpatialOperationsIntegration:
    """Integration tests for spatial operations"""

    @patch('operations.SpatialOperations.get_world_state')
    @patch('operations.SpatialOperations.get_command_broadcaster')
    @patch('operations.SpatialOperations.move_to_coordinate')
    def test_chained_spatial_operations(self, mock_move, mock_broadcast, mock_get_ws, mock_world_state_with_objects):
        mock_get_ws.return_value = mock_world_state_with_objects
        mock_move.return_value = OperationResult.success_result({"final_position": (0.3, 0.2, 0.25)})
        """Test multiple spatial operations in sequence"""
        # Move relative to object
        result1 = move_relative_to_object(
            robot_id="Robot1",
            object_ref="cube_01",
            relation="above",
            offset=0.15)
        assert result1.success is True

        # Move between two objects
        result2 = move_between_objects(
            robot_id="Robot1",
            object1="cube_01",
            object2="cube_02",
            bias=0.5)
        assert result2.success is True

        # Move to region
        result3 = move_to_region(
            robot_id="Robot1",
            region_name="shared_zone",
            position_in_region="center"
        )
        assert result3.success is True

        # All operations should have called move_to_coordinate
        assert mock_move.call_count == 3
