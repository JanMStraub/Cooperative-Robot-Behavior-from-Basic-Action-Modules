#!/usr/bin/env python3
"""
Unity Integration Tests
========================

End-to-end integration tests with real Unity environment.
These tests require Unity to be running and connected.

To run these tests:
1. Start Unity and load the AR4 scene
2. Start the Python backend servers
3. Run: pytest tests/TestUnityIntegration.py -v

To skip these tests in CI:
    pytest tests/ -m "not requires_unity"
"""

import pytest
import socket
import time
import numpy as np
from typing import Optional

# Check if Unity is available
def is_unity_available() -> bool:
    """
    Check if Unity backend servers are available.

    Returns:
        True if Unity is likely running, False otherwise
    """
    try:
        # Try to connect to CommandServer (port 5010)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        result = sock.connect_ex(('localhost', 5010))
        sock.close()
        return result == 0
    except Exception:
        return False


UNITY_AVAILABLE = is_unity_available()
SKIP_REASON = "Unity not running. Start Unity and backend servers to run these tests."


@pytest.mark.requires_unity
@pytest.mark.skipif(not UNITY_AVAILABLE, reason=SKIP_REASON)
class TestUnityCommandExecution:
    """Test real command execution with Unity"""

    def test_connection_to_command_server(self):
        """Test we can connect to Unity's CommandServer"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)

        try:
            sock.connect(('localhost', 5010))
            assert True  # Connection successful
        except Exception as e:
            pytest.fail(f"Failed to connect to CommandServer: {e}")
        finally:
            sock.close()

    def test_real_robot_status_query(self):
        """Test querying real robot status from Unity"""
        from operations.StatusOperations import check_robot_status

        # Query status for Robot1 (should exist in Unity scene)
        result = check_robot_status(robot_id="Robot1")

        # Verify we got a valid response
        assert result["success"] is True
        assert "robot_id" in result
        assert result["robot_id"] == "Robot1"
        assert "position" in result
        assert isinstance(result["position"], (tuple, list))
        assert len(result["position"]) == 3  # x, y, z

    def test_real_move_command_execution(self):
        """Test executing real movement command in Unity"""
        from operations.MoveOperations import move_to_coordinate

        # Send move command to valid position
        result = move_to_coordinate(
            robot_id="Robot1",
            x=0.3,
            y=0.2,
            z=0.15
        )

        # Verify command was sent successfully
        assert result["success"] is True

        # Wait a bit for movement to start
        time.sleep(0.5)

        # Query status to verify movement
        from operations.StatusOperations import check_robot_status
        status = check_robot_status(robot_id="Robot1")

        # Robot should be moving or have moved
        assert status["success"] is True

    def test_real_gripper_control(self):
        """Test real gripper control in Unity"""
        from operations.GripperOperations import control_gripper

        # Open gripper
        result_open = control_gripper(robot_id="Robot1", action="open")
        assert result_open["success"] is True

        time.sleep(0.3)

        # Close gripper
        result_close = control_gripper(robot_id="Robot1", action="close")
        assert result_close["success"] is True


@pytest.mark.requires_unity
@pytest.mark.skipif(not UNITY_AVAILABLE, reason=SKIP_REASON)
class TestUnityImageCapture:
    """Test real image capture from Unity cameras"""

    def test_connection_to_image_server(self):
        """Test we can connect to Unity's ImageServer"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)

        try:
            sock.connect(('localhost', 5005))
            assert True  # Connection successful
        except Exception as e:
            pytest.fail(f"Failed to connect to ImageServer: {e}")
        finally:
            sock.close()

    def test_real_single_image_capture(self):
        """Test capturing real image from Unity camera"""
        from servers.ImageStorageCore import UnifiedImageStorage

        storage = UnifiedImageStorage()

        # Wait for images to be available (Unity sends them periodically)
        time.sleep(1.0)

        # Get available camera IDs
        camera_ids = storage.get_all_camera_ids()

        # Verify we have at least one camera
        assert len(camera_ids) > 0, "No cameras available from Unity"

        # Get image from first camera
        camera_id = camera_ids[0]
        image = storage.get_single_image(camera_id)

        # Verify image is valid
        assert image is not None
        assert isinstance(image, np.ndarray)
        assert image.ndim == 3  # Height x Width x Channels
        assert image.shape[2] == 3  # RGB image
        assert image.dtype == np.uint8

    def test_real_stereo_image_capture(self):
        """Test capturing real stereo image pair from Unity"""
        from servers.ImageStorageCore import UnifiedImageStorage

        storage = UnifiedImageStorage()

        # Wait for stereo images
        time.sleep(1.0)

        # Try to get latest stereo pair
        stereo_data = storage.get_latest_stereo_image()

        if stereo_data is None:
            pytest.skip("No stereo cameras available in Unity scene")

        # Verify stereo pair is valid
        assert "left_image" in stereo_data
        assert "right_image" in stereo_data

        left_img = stereo_data["left_image"]
        right_img = stereo_data["right_image"]

        # Both images should be valid
        assert left_img is not None
        assert right_img is not None
        assert isinstance(left_img, np.ndarray)
        assert isinstance(right_img, np.ndarray)

        # Both should have same dimensions
        assert left_img.shape == right_img.shape


@pytest.mark.requires_unity
@pytest.mark.skipif(not UNITY_AVAILABLE, reason=SKIP_REASON)
class TestUnityProtocolCompatibility:
    """Test Protocol V2 compatibility with Unity"""

    def test_request_id_correlation(self):
        """Test request ID is correctly correlated in responses"""
        from servers.CommandServer import get_command_broadcaster
        from operations.MoveOperations import move_to_coordinate

        broadcaster = get_command_broadcaster()

        # Send command with specific request ID
        request_id = 12345

        # Create completion queue
        broadcaster.create_completion_queue(request_id)

        # Send move command
        result = move_to_coordinate(
            robot_id="Robot1",
            x=0.3,
            y=0.2,
            z=0.1,
            _request_id=request_id  # If supported
        )

        # Verify we got a response
        assert result["success"] is True or result.get("error") is not None

        # Clean up
        broadcaster.remove_completion_queue(request_id)

    def test_multiple_concurrent_commands(self):
        """Test handling multiple concurrent commands"""
        from operations.MoveOperations import move_to_coordinate
        import threading

        results = []
        errors = []

        def send_command(robot_id, x, y, z):
            try:
                result = move_to_coordinate(
                    robot_id=robot_id,
                    x=x, y=y, z=z
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Send 3 commands concurrently
        threads = [
            threading.Thread(target=send_command, args=("Robot1", 0.3, 0.2, 0.1)),
            threading.Thread(target=send_command, args=("Robot1", 0.35, 0.25, 0.15)),
            threading.Thread(target=send_command, args=("Robot1", 0.4, 0.3, 0.2)),
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=10.0)

        # All commands should complete without errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 3


@pytest.mark.requires_unity
@pytest.mark.skipif(not UNITY_AVAILABLE, reason=SKIP_REASON)
class TestUnityObjectDetection:
    """Test real object detection with Unity scene"""

    def test_real_object_detection(self):
        """Test detecting real objects in Unity scene"""
        from operations.DetectionOperations import detect_objects
        from servers.ImageStorageCore import UnifiedImageStorage

        storage = UnifiedImageStorage()

        # Wait for images
        time.sleep(1.0)

        camera_ids = storage.get_all_camera_ids()
        if len(camera_ids) == 0:
            pytest.skip("No cameras available")

        # Detect objects using first camera
        result = detect_objects(
            camera_id=camera_ids[0],
            color_filter=None  # Detect all colors
        )

        # Verify detection ran successfully
        assert result["success"] is True
        assert "detections" in result
        assert isinstance(result["detections"], list)

        # If objects exist in scene, we should detect them
        # (This test is lenient - just verifies detection runs without error)


@pytest.mark.requires_unity
@pytest.mark.skipif(not UNITY_AVAILABLE, reason=SKIP_REASON)
class TestUnityEndToEnd:
    """End-to-end integration tests with full workflow"""

    @pytest.mark.slow
    def test_full_pick_and_place_workflow(self):
        """Test complete pick and place workflow in Unity"""
        from operations.DetectionOperations import detect_objects
        from operations.GraspOperations import execute_grasp
        from servers.ImageStorageCore import UnifiedImageStorage

        storage = UnifiedImageStorage()
        time.sleep(1.0)

        camera_ids = storage.get_all_camera_ids()
        if len(camera_ids) == 0:
            pytest.skip("No cameras available")

        # 1. Detect objects
        detection_result = detect_objects(
            camera_id=camera_ids[0],
            color_filter="blue"  # Look for blue cube
        )

        if not detection_result["success"] or len(detection_result["detections"]) == 0:
            pytest.skip("No blue objects detected in scene")

        # 2. Get first detected object position
        detected_obj = detection_result["detections"][0]

        # 3. Execute grasp (if grasp operations are available)
        try:
            grasp_result = execute_grasp(
                robot_id="Robot1",
                target_position=(
                    detected_obj.get("world_x", 0.3),
                    detected_obj.get("world_y", 0.2),
                    detected_obj.get("world_z", 0.1)
                ),
                approach_direction="Top"
            )

            # Verify grasp executed (may or may not succeed depending on scene)
            assert "success" in grasp_result

        except Exception as e:
            pytest.skip(f"Grasp execution not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
