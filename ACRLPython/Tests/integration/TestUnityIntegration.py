#!/usr/bin/env python3
"""
Unity Integration Tests
========================

End-to-end integration tests with real Unity environment.
These tests require Unity to be running and connected to the Python backend.

To run these tests:
1. Start Unity and load the AR4 scene
2. Start the Python backend servers:
       python -m orchestrators.RunRobotController
3. Run: pytest tests/integration/TestUnityIntegration.py -v

To skip these tests in CI:
    pytest tests/ -m "not requires_unity"
"""

import pytest
import time

from backend_client import BackendClient, backend_available, port_open  # type: ignore[import]


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


def _port_open(port: int, timeout: float = 2.0) -> bool:
    """Return True if a TCP server is accepting connections on *port*."""
    return port_open(port, timeout)


def is_unity_available() -> bool:
    """
    Check whether both the Python backend SequenceServer (port 5013) and the
    CommandServer (port 5010, where Unity connects) are reachable.

    We probe via raw sockets so the check works from any process, regardless of
    whether the backend singleton is initialised in this process.

    Returns:
        True if both backend ports are reachable, False otherwise.
    """
    return backend_available()


UNITY_AVAILABLE = is_unity_available()
SKIP_REASON = (
    "Unity not running or not connected to backend. "
    "Start Unity and backend servers to run these tests."
)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


@pytest.mark.requires_unity
@pytest.mark.skipif(not UNITY_AVAILABLE, reason=SKIP_REASON)
class TestUnityCommandExecution:
    """Test real command execution with Unity via SequenceServer."""

    def test_connection_to_command_server(self):
        """Verify direct TCP connection to the CommandServer (port 5010)."""
        assert _port_open(5010), "Could not connect to CommandServer on port 5010"

    def test_connection_to_sequence_server(self):
        """Verify direct TCP connection to the SequenceServer (port 5013)."""
        assert _port_open(5013), "Could not connect to SequenceServer on port 5013"

    def test_real_robot_status_query(self):
        """Query real robot status through the backend SequenceServer."""
        with BackendClient(timeout=15.0) as client:
            result = client.send_command(
                command="check robot status for Robot1",
                robot_id="Robot1",
                request_id=1,
            )

        assert (
            result.get("success") is True
        ), f"Status query failed: {result.get('error')}"

    def test_real_move_command_execution(self):
        """Execute a real movement command in Unity for Robot1 (left workspace).

        Robot1 is assigned to left_workspace: x in [-0.5, -0.15], y in [0.0, 0.6].
        Using x=-0.25 keeps the target well within reach.
        """
        with BackendClient(timeout=30.0) as client:
            result = client.send_command(
                command="move Robot1 to coordinate -0.25 0.3 0.1",
                robot_id="Robot1",
                request_id=2,
            )

        assert (
            result.get("success") is True
        ), f"Move command failed: {result.get('error')}"

    def test_real_gripper_control(self):
        """Test real gripper open/close through the backend."""
        with BackendClient(timeout=15.0) as client:
            result_open = client.send_command(
                command="open gripper for Robot1",
                robot_id="Robot1",
                request_id=3,
            )

        assert (
            result_open.get("success") is True
        ), f"Gripper open failed: {result_open.get('error')}"

        time.sleep(0.3)

        with BackendClient(timeout=15.0) as client:
            result_close = client.send_command(
                command="close gripper for Robot1",
                robot_id="Robot1",
                request_id=4,
            )

        assert (
            result_close.get("success") is True
        ), f"Gripper close failed: {result_close.get('error')}"


@pytest.mark.requires_unity
@pytest.mark.skipif(not UNITY_AVAILABLE, reason=SKIP_REASON)
class TestUnityImageCapture:
    """Test real image capture from Unity cameras."""

    def test_connection_to_image_server(self):
        """Verify direct TCP connection to the ImageServer (port 5005)."""
        assert _port_open(5005), "Could not connect to ImageServer on port 5005"

    def test_real_single_image_capture(self):
        """
        Test on-demand stereo image capture from Unity.

        Uses detect_object_stereo which sends a 'capture_stereo_images' command
        to Unity (via CommandBroadcaster) and waits for Unity to send back the
        images.  This does not rely on Unity already streaming images passively.
        """
        with BackendClient(timeout=30.0) as client:
            result = client.send_command(
                command="detect object stereo for Robot1",
                robot_id="Robot1",
                camera_id="TableStereoCamera",
                request_id=5,
            )

        assert (
            result.get("success") is True
        ), f"Stereo image capture failed: {result.get('error')}"


@pytest.mark.requires_unity
@pytest.mark.skipif(not UNITY_AVAILABLE, reason=SKIP_REASON)
class TestUnityProtocolCompatibility:
    """Test Protocol V2 request/response correlation with the backend."""

    def test_request_id_correlation(self):
        """Verify that request_id is echoed back correctly in the response."""
        request_id = 12345
        with BackendClient(timeout=15.0) as client:
            result = client.send_command(
                command="check robot status for Robot1",
                robot_id="Robot1",
                request_id=request_id,
            )

        assert (
            result.get("request_id") == request_id
        ), f"request_id mismatch: sent {request_id}, got {result.get('request_id')}"

    def test_multiple_concurrent_commands(self):
        """Test that the backend handles multiple concurrent connections correctly.

        Uses check_robot_status (a pure-Python perception op) so each command
        returns immediately without waiting for Unity completion.  This tests
        the Protocol V2 connection-per-request model without conflicting robot
        movements that would cause Unity to drop or ignore completion signals.
        """
        import threading

        results = []
        errors = []
        lock = threading.Lock()

        def send_command(command: str, rid: int):
            try:
                with BackendClient(timeout=30.0) as client:
                    result = client.send_command(
                        command=command,
                        robot_id="Robot1",
                        request_id=rid,
                    )
                with lock:
                    results.append(result)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        # Use status queries rather than move commands: status ops return
        # immediately (no Unity completion wait) so all three connections
        # can resolve concurrently without robot conflict.
        commands = [
            ("check robot status for Robot1", 10),
            ("check robot status for Robot1", 11),
            ("check robot status for Robot1", 12),
        ]

        threads = [
            threading.Thread(target=send_command, args=cmd, daemon=True)
            for cmd in commands
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)

        assert not errors, f"Errors in concurrent commands: {errors}"
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"


@pytest.mark.requires_unity
@pytest.mark.skipif(not UNITY_AVAILABLE, reason=SKIP_REASON)
class TestUnityObjectDetection:
    """Test real object detection with the Unity scene."""

    def test_real_object_detection(self):
        """Detect objects in the Unity scene using on-demand stereo capture.

        Uses detect_object_stereo which triggers Unity to capture images first,
        rather than reading from pre-existing image storage.
        """
        with BackendClient(timeout=30.0) as client:
            result = client.send_command(
                command="detect object stereo for Robot1",
                robot_id="Robot1",
                camera_id="TableStereoCamera",
                request_id=20,
            )

        assert (
            result.get("success") is True
        ), f"Object detection failed: {result.get('error')}"
        # The detection result is nested inside the sequence execution result.
        # We verify the overall call succeeded; detailed shape is tested in unit tests.


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
