"""
Unit Tests for Basic Operations (Level 1-2)
============================================

Tests for:
- move_from_a_to_b
- adjust_end_effector_orientation
- release_object
- estimate_distance_to_object
- estimate_distance_between_objects
"""

import unittest
from unittest.mock import patch, MagicMock
from operations.MoveOperations import move_from_a_to_b, adjust_end_effector_orientation
from operations.GripperOperations import release_object
from operations.DetectionOperations import (
    estimate_distance_to_object,
    estimate_distance_between_objects,
)


class TestMoveFromAToB(unittest.TestCase):
    """Test cases for move_from_a_to_b operation"""

    @patch("operations.MoveOperations._get_command_broadcaster")
    def test_move_from_a_to_b_success(self, mock_broadcaster):
        """Test successful move from point A to point B"""
        mock_broadcaster().send_command = MagicMock(return_value=True)

        point_a = {"x": 0.0, "y": 0.0, "z": 0.3}
        point_b = {"x": 0.3, "y": 0.15, "z": 0.1}

        result = move_from_a_to_b("Robot1", point_a, point_b)

        self.assertTrue(result.success)
        self.assertEqual(result.result["robot_id"], "Robot1")
        self.assertEqual(result.result["point_a"], point_a)
        self.assertEqual(result.result["point_b"], point_b)
        mock_broadcaster().send_command.assert_called_once()

    def test_move_from_a_to_b_invalid_robot_id(self):
        """Test with invalid robot ID"""
        point_a = {"x": 0.0, "y": 0.0, "z": 0.3}
        point_b = {"x": 0.3, "y": 0.15, "z": 0.1}

        result = move_from_a_to_b("", point_a, point_b)

        self.assertFalse(result.success)
        self.assertEqual(result.error["code"], "INVALID_ROBOT_ID")

    def test_move_from_a_to_b_invalid_point_a(self):
        """Test with invalid point A"""
        point_b = {"x": 0.3, "y": 0.15, "z": 0.1}

        result = move_from_a_to_b("Robot1", {"x": 0.0}, point_b)

        self.assertFalse(result.success)
        self.assertEqual(result.error["code"], "INVALID_POINT_A")

    def test_move_from_a_to_b_invalid_point_b(self):
        """Test with invalid point B"""
        point_a = {"x": 0.0, "y": 0.0, "z": 0.3}

        result = move_from_a_to_b("Robot1", point_a, {"incomplete": True})

        self.assertFalse(result.success)
        self.assertEqual(result.error["code"], "INVALID_POINT_B")

    @patch("operations.MoveOperations._get_command_broadcaster")
    def test_move_from_a_to_b_communication_failed(self, mock_broadcaster):
        """Test communication failure"""
        mock_broadcaster().send_command = MagicMock(return_value=False)

        point_a = {"x": 0.0, "y": 0.0, "z": 0.3}
        point_b = {"x": 0.3, "y": 0.15, "z": 0.1}

        result = move_from_a_to_b("Robot1", point_a, point_b)

        self.assertFalse(result.success)
        self.assertEqual(result.error["code"], "COMMUNICATION_FAILED")


class TestAdjustEndEffectorOrientation(unittest.TestCase):
    """Test cases for adjust_end_effector_orientation operation"""

    @patch("operations.MoveOperations._get_command_broadcaster")
    def test_adjust_orientation_success(self, mock_broadcaster):
        """Test successful orientation adjustment"""
        mock_broadcaster().send_command = MagicMock(return_value=True)

        result = adjust_end_effector_orientation("Robot1", roll=90.0, pitch=0.0, yaw=45.0)

        self.assertTrue(result.success)
        self.assertEqual(result.result["robot_id"], "Robot1")
        self.assertEqual(result.result["orientation"]["roll"], 90.0)
        self.assertEqual(result.result["orientation"]["pitch"], 0.0)
        self.assertEqual(result.result["orientation"]["yaw"], 45.0)

    def test_adjust_orientation_invalid_angle(self):
        """Test with out-of-range angle"""
        result = adjust_end_effector_orientation("Robot1", roll=200.0)

        self.assertFalse(result.success)
        self.assertEqual(result.error["code"], "ANGLE_OUT_OF_RANGE")

    def test_adjust_orientation_invalid_type(self):
        """Test with invalid angle type"""
        result = adjust_end_effector_orientation("Robot1", roll="not_a_number")

        self.assertFalse(result.success)
        self.assertEqual(result.error["code"], "INVALID_ANGLE")


class TestReleaseObject(unittest.TestCase):
    """Test cases for release_object operation (ATOMIC - gripper only, no positioning)"""

    @patch("operations.GripperOperations._get_command_broadcaster")
    def test_release_object_atomic(self, mock_broadcaster):
        """Test atomic release (ONLY opens gripper, no movement)"""
        mock_broadcaster().send_command = MagicMock(return_value=True)

        result = release_object("Robot1")

        self.assertTrue(result.success)
        self.assertEqual(result.result["robot_id"], "Robot1")
        # Atomic operation - no place_position field in result
        self.assertNotIn("place_position", result.result)

    @patch("operations.GripperOperations._get_command_broadcaster")
    def test_release_object_invalid_robot(self, mock_broadcaster):
        """Test with invalid robot ID"""
        result = release_object("")

        self.assertFalse(result.success)
        self.assertEqual(result.error["code"], "INVALID_ROBOT_ID")

    @patch("operations.MoveOperations._get_command_broadcaster")
    @patch("operations.GripperOperations._get_command_broadcaster")
    def test_release_object_chaining_with_move(self, mock_gripper_broadcaster, mock_move_broadcaster):
        """Test chaining: move_to_coordinate → release_object (LLM pattern)"""
        from operations.MoveOperations import move_to_coordinate

        mock_move_broadcaster().send_command = MagicMock(return_value=True)
        mock_gripper_broadcaster().send_command = MagicMock(return_value=True)

        # Step 1: Move to drop position
        move_result = move_to_coordinate("Robot1", x=0.3, y=0.0, z=0.1)
        self.assertTrue(move_result.success)

        # Step 2: Release object (atomic - gripper only)
        release_result = release_object("Robot1")
        self.assertTrue(release_result.success)

        # Verify commands were sent to both broadcasters
        mock_move_broadcaster().send_command.assert_called_once()
        mock_gripper_broadcaster().send_command.assert_called_once()


class TestDistanceEstimation(unittest.TestCase):
    """Test cases for distance estimation operations"""

    @patch("operations.WorldState.WorldState")
    def test_estimate_distance_to_object_success(self, mock_world_state):
        """Test successful distance estimation to object"""
        # Mock WorldState
        mock_instance = MagicMock()
        mock_world_state.return_value = mock_instance

        # Mock robot state
        mock_instance.get_robot_state.return_value = {
            "end_effector_position": {"x": 0.0, "y": 0.0, "z": 0.3}
        }

        # Mock object state
        mock_instance.get_object_state.return_value = {
            "position": {"x": 0.3, "y": 0.0, "z": 0.1}
        }

        result = estimate_distance_to_object("Robot1", "RedCube")

        self.assertTrue(result.success)
        self.assertEqual(result.result["robot_id"], "Robot1")
        self.assertEqual(result.result["object_id"], "RedCube")
        # Distance should be sqrt(0.3^2 + 0.0^2 + 0.2^2) = sqrt(0.13) ≈ 0.36
        self.assertAlmostEqual(result.result["distance"], 0.36, places=1)

    @patch("operations.WorldState.WorldState")
    def test_estimate_distance_robot_not_found(self, mock_world_state):
        """Test with robot not found"""
        mock_instance = MagicMock()
        mock_world_state.return_value = mock_instance
        mock_instance.get_robot_state.return_value = None

        result = estimate_distance_to_object("Robot1", "RedCube")

        self.assertFalse(result.success)
        self.assertEqual(result.error["code"], "ROBOT_NOT_FOUND")

    @patch("operations.WorldState.WorldState")
    def test_estimate_distance_between_objects_success(self, mock_world_state):
        """Test successful distance estimation between objects"""
        mock_instance = MagicMock()
        mock_world_state.return_value = mock_instance

        # Mock object states
        mock_instance.get_object_state.side_effect = [
            {"position": {"x": 0.0, "y": 0.0, "z": 0.1}},  # RedCube
            {"position": {"x": 0.3, "y": 0.0, "z": 0.1}},  # BlueCube
        ]

        result = estimate_distance_between_objects("RedCube", "BlueCube")

        self.assertTrue(result.success)
        self.assertEqual(result.result["object_id1"], "RedCube")
        self.assertEqual(result.result["object_id2"], "BlueCube")
        self.assertAlmostEqual(result.result["distance"], 0.3, places=2)


if __name__ == "__main__":
    unittest.main()
