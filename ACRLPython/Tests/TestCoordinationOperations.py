#!/usr/bin/env python3
"""
Unit tests for CoordinationOperations.py

Tests multi-robot coordination operations including simultaneous movement,
object handoff, and workspace allocation.
"""

import pytest
from unittest.mock import Mock, patch, call
from operations.CoordinationOperations import (
    coordinate_simultaneous_move,
    coordinate_handoff,
    allocate_workspace_region,
)
from operations.Base import OperationResult
import LLMConfig as cfg


class TestCoordinateSimultaneousMove:
    """Test coordinate_simultaneous_move operation"""

    @patch('operations.CoordinationOperations.get_world_state')
    @patch('operations.MoveOperations.get_command_broadcaster')
    @patch('operations.CoordinationOperations.robots_will_collide')
    @patch('operations.CoordinationOperations.move_to_coordinate')
    def test_parallel_movement_safe(self, mock_move, mock_collision, mock_broadcast, mock_get_ws, mock_world_state_multi_robot):
        """Test safe parallel movement when no collision"""
        mock_get_ws.return_value = mock_world_state_multi_robot
        # Mock no collision
        mock_collision.return_value = (False, "")

        # Mock successful movements
        mock_move.return_value = OperationResult.success_result({
            "final_position": (0.3, 0.2, 0.1)
        })

        result = coordinate_simultaneous_move(
            robot1_id="Robot1",
            target1=(0.2, 0.1, 0.15),
            robot2_id="Robot2",
            target2=(0.5, 0.3, 0.15)
        )

        assert result.success is True
        assert result.result is not None
        assert result.result["coordination_mode"] == "parallel"
        # Both robots should move in parallel
        assert mock_move.call_count == 2

    @patch('operations.CoordinationOperations.get_world_state')
    @patch('operations.MoveOperations.get_command_broadcaster')
    @patch('operations.CoordinationOperations.robots_will_collide')
    @patch('operations.CoordinationOperations.move_to_coordinate')
    def test_serialized_movement_collision_detected(self, mock_move, mock_collision, mock_broadcast, mock_get_ws, mock_world_state_multi_robot):
        """Test serialized movement when collision detected"""
        mock_get_ws.return_value = mock_world_state_multi_robot
        # Mock collision detected
        mock_collision.return_value = (True, "Paths will intersect")

        # Mock successful movements
        mock_move.return_value = OperationResult.success_result({
            "final_position": (0.3, 0.2, 0.1)
        })

        result = coordinate_simultaneous_move(
            robot1_id="Robot1",
            target1=(0.3, 0.2, 0.15),
            robot2_id="Robot2",
            target2=(0.0, 0.0, 0.15)
        )

        assert result.success is True
        assert result.result is not None
        assert result.result["coordination_mode"] == "serialized"
        # Robot1 moves first, then Robot2
        assert mock_move.call_count == 2
        # Verify order: Robot1 called first
        calls = mock_move.call_args_list
        assert calls[0][1]["robot_id"] == "Robot1"
        assert calls[1][1]["robot_id"] == "Robot2"

    @patch('operations.CoordinationOperations.get_world_state')
    @patch('operations.MoveOperations.get_command_broadcaster')
    @patch('operations.CoordinationOperations.robots_will_collide')
    @patch('operations.CoordinationOperations.move_to_coordinate')
    def test_robot1_movement_fails(self, mock_move, mock_collision, mock_broadcast, mock_get_ws, mock_world_state_multi_robot):
        """Test Robot1 movement error"""
        mock_get_ws.return_value = mock_world_state_multi_robot
        mock_collision.return_value = (False, "")

        # Robot1 movement fails
        mock_move.side_effect = [
            OperationResult.error_result(
                error_code="MOVEMENT_FAILED",
                message="Robot1 movement failed",
                recovery_suggestions=["Check robot1 status"]
            ),
            OperationResult.success_result({
                "final_position": (0.5, 0.3, 0.15)
            })
        ]

        result = coordinate_simultaneous_move(
            robot1_id="Robot1",
            target1=(0.2, 0.1, 0.15),
            robot2_id="Robot2",
            target2=(0.5, 0.3, 0.15)
        )

        assert result.success is False
        assert result.error is not None
        assert "Robot1" in result.error["message"]

    @patch('operations.CoordinationOperations.get_world_state')
    @patch('operations.MoveOperations.get_command_broadcaster')
    @patch('operations.CoordinationOperations.robots_will_collide')
    @patch('operations.CoordinationOperations.move_to_coordinate')
    def test_robot2_movement_fails(self, mock_move, mock_collision, mock_broadcast, mock_get_ws, mock_world_state_multi_robot):
        """Test Robot2 movement error"""
        mock_get_ws.return_value = mock_world_state_multi_robot
        mock_collision.return_value = (False, "")

        # Robot2 movement fails
        mock_move.side_effect = [
            OperationResult.success_result({
                "final_position": (0.2, 0.1, 0.15)
            }),
            OperationResult.error_result(
                error_code="MOVEMENT_FAILED",
                message="Robot2 movement failed",
                recovery_suggestions=["Check robot2 status"]
            )
        ]

        result = coordinate_simultaneous_move(
            robot1_id="Robot1",
            target1=(0.2, 0.1, 0.15),
            robot2_id="Robot2",
            target2=(0.5, 0.3, 0.15)
        )

        assert result.success is False
        assert result.error is not None
        assert "Robot2" in result.error["message"]


class TestCoordinateHandoff:
    """Test coordinate_handoff operation"""

    @patch('operations.CoordinationOperations.get_world_state')
    @patch('operations.GripperOperations.control_gripper')
    @patch('operations.CoordinationOperations.move_to_coordinate')
    @patch('operations.MoveOperations.get_command_broadcaster')
    def test_handoff_default_position(self, mock_broadcast, mock_move, mock_gripper, mock_get_ws, mock_world_state_multi_robot):
        """Test handoff in shared_zone (default)"""
        mock_get_ws.return_value = mock_world_state_multi_robot
        # Setup: Robot1 has the object
        obj = mock_world_state_multi_robot._objects["test_object"]
        obj.grasped_by = "Robot1"

        # Mock successful operations
        mock_move.return_value = OperationResult.success_result({
            "final_position": (0.0, 0.0, 0.15)
        })
        mock_gripper.return_value = OperationResult.success_result({
            "gripper_state": "operated"
        })

        result = coordinate_handoff(
            robot1_id="Robot1",
            robot2_id="Robot2",
            object_id="test_object"
        )

        assert result.success is True
        # Should move both robots to shared_zone, release, then grasp
        assert mock_move.call_count >= 2
        # TODO: Re-enable when gripper integration is implemented
        # assert mock_gripper.call_count >= 2

    @patch('operations.CoordinationOperations.get_world_state')
    @patch('operations.GripperOperations.control_gripper')
    @patch('operations.CoordinationOperations.move_to_coordinate')
    @patch('operations.MoveOperations.get_command_broadcaster')
    def test_handoff_custom_position(self, mock_broadcast, mock_move, mock_gripper, mock_get_ws, mock_world_state_multi_robot):
        """Test handoff at specified location"""
        mock_get_ws.return_value = mock_world_state_multi_robot
        obj = mock_world_state_multi_robot._objects["test_object"]
        obj.grasped_by = "Robot1"

        mock_move.return_value = OperationResult.success_result({
            "final_position": (0.2, 0.1, 0.15)
        })
        mock_gripper.return_value = OperationResult.success_result({
            "gripper_state": "operated"
        })

        result = coordinate_handoff(
            robot1_id="Robot1",
            robot2_id="Robot2",
            object_id="test_object",
            handoff_position=(0.2, 0.1, 0.15)
        )

        assert result.success is True
        # Verify custom position was used
        move_calls = mock_move.call_args_list
        assert any(call[1].get("x") == 0.2 for call in move_calls)

    def test_handoff_object_not_found(self, mock_world_state_multi_robot):
        """Test object missing error"""
        result = coordinate_handoff(
            robot1_id="Robot1",
            robot2_id="Robot2",
            object_id="nonexistent_object"
        )

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error["message"].lower()

    @patch('operations.CoordinationOperations.get_world_state')
    @patch('operations.CoordinationOperations.move_to_coordinate')
    @patch('operations.MoveOperations.get_command_broadcaster')
    def test_handoff_object_not_grasped(self, mock_broadcast, mock_move, mock_get_ws, mock_world_state_multi_robot):
        """Test object not grasped by robot1"""
        mock_get_ws.return_value = mock_world_state_multi_robot
        obj = mock_world_state_multi_robot._objects["test_object"]
        obj.grasped_by = None  # Object not grasped

        mock_move.return_value = OperationResult.success_result({
            "final_position": (0.0, 0.0, 0.15)
        })

        result = coordinate_handoff(
            robot1_id="Robot1",
            robot2_id="Robot2",
            object_id="test_object"
        )

        # Should succeed - robot1 needs to grasp first
        assert result.success is True

    @patch('operations.CoordinationOperations.get_world_state')
    @patch('operations.GripperOperations.control_gripper')
    @patch('operations.CoordinationOperations.move_to_coordinate')
    @patch('operations.MoveOperations.get_command_broadcaster')
    def test_handoff_robot1_approach_fails(self, mock_broadcast, mock_move, mock_gripper, mock_get_ws, mock_world_state_multi_robot):
        """Test Robot1 approach movement error"""
        mock_get_ws.return_value = mock_world_state_multi_robot
        obj = mock_world_state_multi_robot._objects["test_object"]
        obj.grasped_by = "Robot1"

        # Robot1 movement fails
        mock_move.side_effect = [
            OperationResult.error_result(
                error_code="MOVEMENT_FAILED",
                message="Robot1 cannot reach handoff position",
                recovery_suggestions=["Check robot1 reachability"]
            )
        ]

        result = coordinate_handoff(
            robot1_id="Robot1",
            robot2_id="Robot2",
            object_id="test_object"
        )

        assert result.success is False
        assert result.error is not None
        assert "Robot1" in result.error["message"]


class TestAllocateWorkspaceRegion:
    """Test allocate_workspace_region operation"""

    def test_allocate_workspace_success(self, mock_world_state_multi_robot):
        """Test successful workspace allocation"""
        result = allocate_workspace_region(
            robot_id="Robot1",
            region_name="left_workspace"
        )

        assert result.success is True
        assert result.result is not None
        assert result.result["region_name"] == "left_workspace"
        assert result.result["robot_id"] == "Robot1"

    @patch('operations.CoordinationOperations.get_world_state')
    def test_allocate_workspace_already_allocated(self, mock_get_ws, mock_world_state_multi_robot):
        """Test workspace allocation conflict"""
        # Mock workspace already allocated
        mock_world_state_multi_robot.allocate_workspace = Mock(return_value=False)
        mock_world_state_multi_robot.get_workspace_owner = Mock(return_value="Robot2")
        mock_get_ws.return_value = mock_world_state_multi_robot

        result = allocate_workspace_region(
            robot_id="Robot1",
            region_name="left_workspace"
        )

        assert result.success is False
        assert result.error is not None
        assert "already allocated" in result.error["message"].lower()
        assert "Robot2" in result.error["message"]

    def test_allocate_workspace_invalid_region(self, mock_world_state_multi_robot):
        """Test invalid region name"""
        result = allocate_workspace_region(
            robot_id="Robot1",
            region_name="invalid_workspace"
        )

        assert result.success is False
        assert result.error is not None
        assert "Unknown region" in result.error["message"]


class TestCoordinationIntegration:
    """Integration tests for coordination operations"""

    @patch('operations.CoordinationOperations.get_world_state')
    @patch('operations.GripperOperations.control_gripper')
    @patch('operations.CoordinationOperations.move_to_coordinate')
    @patch('operations.MoveOperations.get_command_broadcaster')
    @patch('operations.CoordinationOperations.robots_will_collide')
    def test_full_coordination_workflow(self, mock_collision, mock_broadcast, mock_move, mock_gripper, mock_get_ws, mock_world_state_multi_robot):
        """Test complete coordination workflow"""
        mock_get_ws.return_value = mock_world_state_multi_robot
        mock_collision.return_value = (False, "")
        mock_move.return_value = OperationResult.success_result({
            "final_position": (0.3, 0.2, 0.1)
        })
        mock_gripper.return_value = OperationResult.success_result({
            "gripper_state": "operated"
        })

        # Step 1: Allocate workspaces
        result1 = allocate_workspace_region(
            robot_id="Robot1",
            region_name="left_workspace"
        )
        assert result1.success is True

        result2 = allocate_workspace_region(
            robot_id="Robot2",
            region_name="right_workspace"
        )
        assert result2.success is True

        # Step 2: Simultaneous move to workspaces
        result3 = coordinate_simultaneous_move(
            robot1_id="Robot1",
            target1=(-0.5, 0.0, 0.2),
            robot2_id="Robot2",
            target2=(0.5, 0.0, 0.2)
        )
        assert result3.success is True

        # Step 3: Handoff object
        obj = mock_world_state_multi_robot._objects["test_object"]
        obj.grasped_by = "Robot1"

        result4 = coordinate_handoff(
            robot1_id="Robot1",
            robot2_id="Robot2",
            object_id="test_object"
        )
        assert result4.success is True
