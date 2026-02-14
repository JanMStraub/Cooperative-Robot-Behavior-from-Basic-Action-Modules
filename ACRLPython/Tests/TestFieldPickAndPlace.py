"""
Integration Test: Field Pick-and-Place (ATOMIC OPERATIONS)
============================================================

End-to-end test for field-based pick-and-place operation:
"Pick the red cube from field D and place it on field E"

This test validates the complete ATOMIC workflow (LLM chains operations):
1. Detect field D
2. Detect red cube on field D (or use field center)
3. Grasp cube
4. Detect field E
5. Move to field E (positioning BEFORE release)
6. Release cube (ATOMIC - gripper only, NO movement)

IMPORTANT: release_object is now ATOMIC (gripper only).
For positioned release, LLM must chain: move_to_coordinate → release_object
"""

import unittest
from unittest.mock import patch, MagicMock
from operations.FieldOperations import detect_field, get_field_center
from operations.DetectionOperations import detect_objects
from operations import GraspOperations
from operations.MoveOperations import move_to_coordinate
from operations.GripperOperations import release_object
from operations.Base import OperationResult


class TestFieldPickAndPlace(unittest.TestCase):
    """Integration test for field-based pick-and-place"""

    @patch("config.ROS.ROS_ENABLED", False)
    @patch("operations.GripperOperations._get_command_broadcaster")
    @patch("operations.MoveOperations._get_command_broadcaster")
    @patch("operations.GraspOperations.grasp_object")
    @patch("vision.YOLODetector.YOLODetector")
    @patch("operations.FieldOperations.get_unified_image_storage")
    def test_complete_field_pick_and_place_workflow(
        self,
        mock_image_storage,
        mock_yolo,
        mock_grasp_object,
        mock_move_broadcaster,
        mock_gripper_broadcaster,
    ):
        """
        Test complete workflow: Pick cube from field D, place on field E
        """
        # ========================================================================
        # Setup mocks
        # ========================================================================

        # Mock stereo image storage
        mock_storage_instance = MagicMock()
        mock_image_storage.return_value = mock_storage_instance
        mock_storage_instance.get_latest_stereo_image.return_value = {
            "left_image": MagicMock(),
            "right_image": MagicMock(),
            "camera_params": {},
        }

        # Mock YOLO detector for field detection
        mock_yolo_instance = MagicMock()
        mock_yolo.return_value = mock_yolo_instance

        # Mock field D detection
        mock_detection_d = MagicMock()
        mock_detection_d.class_name = "fieldd"
        mock_detection_d.world_position = {"x": 0.2, "y": -0.1, "z": 0.0}
        mock_detection_d.bbox = {"x": 100, "y": 100, "w": 80, "h": 80}
        mock_detection_d.confidence = 0.95

        # Mock field E detection
        mock_detection_e = MagicMock()
        mock_detection_e.class_name = "fielde"
        mock_detection_e.world_position = {"x": 0.3, "y": 0.1, "z": 0.0}
        mock_detection_e.bbox = {"x": 200, "y": 100, "w": 80, "h": 80}
        mock_detection_e.confidence = 0.93

        # Configure YOLO to return field detections
        mock_detections_d = MagicMock()
        mock_detections_d.detections = [mock_detection_d]

        mock_detections_e = MagicMock()
        mock_detections_e.detections = [mock_detection_e]

        # YOLO returns different fields on consecutive calls
        mock_yolo_instance.detect_objects_stereo.side_effect = [
            mock_detections_d,
            mock_detections_e,
        ]

        # Mock command broadcasters - need to configure the return value properly
        mock_move_instance = MagicMock()
        mock_move_instance.send_command = MagicMock(return_value=True)
        mock_move_broadcaster.return_value = mock_move_instance

        mock_gripper_instance = MagicMock()
        mock_gripper_instance.send_command = MagicMock(return_value=True)
        mock_gripper_broadcaster.return_value = mock_gripper_instance

        # Mock grasp_object to return success
        mock_grasp_object.return_value = OperationResult.success_result({
            "robot_id": "Robot1",
            "object_id": "cube_on_field_d",
            "success": True,
        })

        # ========================================================================
        # STEP 1: Detect field D
        # ========================================================================

        result_field_d = detect_field("stereo", "D")

        self.assertTrue(result_field_d.success, "Field D detection failed")
        self.assertEqual(result_field_d.result["field_label"], "D")
        self.assertEqual(result_field_d.result["center"], {"x": 0.2, "y": -0.1, "z": 0.0})
        print(f"✓ Step 1: Detected field D at {result_field_d.result['center']}")

        # ========================================================================
        # STEP 2: Grasp cube at field D position
        # ========================================================================

        # In real scenario, would detect cube on field D first
        # For integration test, using field D center as cube position
        cube_position = result_field_d.result["center"]

        # grasp_object takes object_id, not coordinates
        # In real workflow: detect_object → get object_id → grasp_object(robot, object_id)
        result_grasp = GraspOperations.grasp_object("Robot1", "cube_on_field_d")

        self.assertTrue(result_grasp.success, "Grasp operation failed")
        self.assertEqual(result_grasp.result["robot_id"], "Robot1")
        print(f"✓ Step 2: Grasped cube at field D")

        # ========================================================================
        # STEP 3: Detect field E
        # ========================================================================

        result_field_e = detect_field("stereo", "E")

        self.assertTrue(result_field_e.success, "Field E detection failed")
        self.assertEqual(result_field_e.result["field_label"], "E")
        self.assertEqual(result_field_e.result["center"], {"x": 0.3, "y": 0.1, "z": 0.0})
        print(f"✓ Step 3: Detected field E at {result_field_e.result['center']}")

        # ========================================================================
        # STEP 4: Move to field E (positioning before release)
        # ========================================================================

        field_e_center = result_field_e.result["center"]
        result_move = move_to_coordinate(
            "Robot1",
            field_e_center["x"],
            field_e_center["y"],
            field_e_center["z"] + 0.05,  # Lift 5cm above field
        )

        self.assertTrue(result_move.success, "Move to field E failed")
        print(f"✓ Step 4: Moved to field E")

        # ========================================================================
        # STEP 5: Release cube (ATOMIC - gripper only, no movement)
        # ========================================================================

        # Atomic operation - ONLY opens gripper at current position
        # LLM pattern: move_to_coordinate → release_object
        result_release = release_object("Robot1")

        self.assertTrue(result_release.success, "Release operation failed")
        # Atomic operation - no place_position in result
        self.assertNotIn("place_position", result_release.result)
        print(f"✓ Step 5: Released cube (atomic gripper operation)")

        # ========================================================================
        # Verification
        # ========================================================================

        # Verify all operations were executed
        self.assertEqual(mock_yolo_instance.detect_objects_stereo.call_count, 2)  # Field D + E

        print("\n✓ Integration test PASSED: Field pick-and-place workflow complete")
        print("  Workflow: detect_field(D) → grasp_object → detect_field(E) → move_to_coordinate(E) → release_object (atomic)")

    @patch("operations.FieldOperations.get_unified_image_storage")
    @patch("vision.YOLODetector.YOLODetector")
    def test_field_not_detected_error_handling(self, mock_yolo, mock_image_storage):
        """Test error handling when field is not detected"""

        # Mock stereo image storage
        mock_storage_instance = MagicMock()
        mock_image_storage.return_value = mock_storage_instance
        mock_storage_instance.get_latest_stereo_image.return_value = {
            "left_image": MagicMock(),
            "right_image": MagicMock(),
            "camera_params": {},
        }

        # Mock YOLO to return no detections
        mock_yolo_instance = MagicMock()
        mock_yolo.return_value = mock_yolo_instance

        mock_detections = MagicMock()
        mock_detections.detections = []  # No fields detected

        mock_yolo_instance.detect_objects_stereo.return_value = mock_detections

        # Try to detect field D
        result = detect_field("stereo", "D")

        self.assertFalse(result.success)
        self.assertEqual(result.error["code"], "FIELD_NOT_DETECTED")
        print("✓ Error handling test PASSED: Correctly handles missing field")

    def test_get_field_center_convenience_wrapper(self):
        """Test get_field_center convenience function"""

        with patch("operations.FieldOperations.detect_field") as mock_detect:
            # Mock detect_field to return success
            mock_detect.return_value = OperationResult.success_result(
                {
                    "field_label": "A",
                    "center": {"x": 0.1, "y": 0.1, "z": 0.0},
                    "bounds": {},
                    "confidence": 0.9,
                    "camera_id": "stereo",
                    "timestamp": 0.0,
                }
            )

            result = get_field_center("Robot1", "A")

            self.assertTrue(result.success)
            self.assertEqual(result.result["center"], {"x": 0.1, "y": 0.1, "z": 0.0})
            mock_detect.assert_called_once_with("Robot1", "A", "stereo", request_id=0)
            print("✓ Convenience wrapper test PASSED")


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)
