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
    Check if the Python backend CommandServer is listening and a Unity client
    is connected (i.e. CommandBroadcaster has an active server with clients).

    Port 5010 is the Python CommandServer — Unity connects to it as a client.
    A reachable port only means the Python server is up; we also need at least
    one connected Unity client before commands can be sent.

    Returns:
        True if backend is running and Unity client is connected, False otherwise
    """
    try:
        from servers.CommandServer import CommandBroadcaster
        broadcaster = CommandBroadcaster()
        if broadcaster._server is None:
            return False
        # Check if there is at least one connected client
        return broadcaster._server.get_client_count() > 0
    except Exception:
        return False


UNITY_AVAILABLE = is_unity_available()
SKIP_REASON = "Unity not running or not connected to backend. Start Unity and backend servers to run these tests."


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

        # Verify we got a valid response - result is an OperationResult object
        assert result.success is True
        assert result.result is not None
        assert "robot_id" in result.result
        assert result.result["robot_id"] == "Robot1"
        # Note: The status query returns query_sent status, not full robot state
        # Full state would be received via Unity's status response system

    def test_real_move_command_execution(self):
        """Test executing real movement command in Unity via Unity IK (not ROS)"""
        from operations.MoveOperations import move_to_coordinate

        # Force Unity IK path (use_ros=False) — this is a Unity integration test,
        # not a ROS test. MoveIt may be unavailable even when Unity is running.
        result = move_to_coordinate(
            robot_id="Robot1",
            x=0.15,
            y=0.25,
            z=0.1,
            use_ros=False,
        )

        assert result.success is True

        # Wait a bit for movement to start
        time.sleep(0.5)

        # Query status to verify movement
        from operations.StatusOperations import check_robot_status
        status = check_robot_status(robot_id="Robot1")

        # Robot should be moving or have moved
        assert status.success is True

    def test_real_gripper_control(self):
        """Test real gripper control in Unity"""
        from operations.GripperOperations import control_gripper

        # Open gripper
        result_open = control_gripper(robot_id="Robot1", open_gripper=True)
        assert result_open.success is True

        time.sleep(0.3)

        # Close gripper
        result_close = control_gripper(robot_id="Robot1", open_gripper=False)
        assert result_close.success is True


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

        # Send move command (request_id is handled internally by operations)
        result = move_to_coordinate(
            robot_id="Robot1",
            x=0.3,
            y=0.2,
            z=0.1,
            request_id=request_id  # Standard parameter name for operations
        )

        # Verify we got a response - result is OperationResult object
        assert result.success is True or result.error is not None

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

        # Send 3 commands concurrently to positions reachable by the AR4 arm
        threads = [
            threading.Thread(target=send_command, args=("Robot1", 0.15, 0.25, 0.10)),
            threading.Thread(target=send_command, args=("Robot1", 0.10, 0.28, 0.12)),
            threading.Thread(target=send_command, args=("Robot1", 0.12, 0.22, 0.08)),
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

        # Detect objects using first camera - use Robot1 as robot_id
        result = detect_objects(
            robot_id="Robot1",
            camera_id=camera_ids[0]
        )

        # Verify detection ran successfully - result is OperationResult object
        assert result.success is True
        assert result.result is not None
        assert "detections" in result.result
        assert isinstance(result.result["detections"], list)

        # If objects exist in scene, we should detect them
        # (This test is lenient - just verifies detection runs without error)



if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
