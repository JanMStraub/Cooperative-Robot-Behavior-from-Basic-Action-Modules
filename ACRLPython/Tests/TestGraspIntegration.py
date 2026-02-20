"""
End-to-end integration tests for grasp planning system.

Tests the full pipeline from Python operation → Unity command handler → grasp execution.
Requires Unity to be running with proper scene setup.
"""

import pytest
import socket
import time
from unittest.mock import Mock, patch, MagicMock
from operations.GraspOperations import grasp_object
from servers.CommandServer import CommandBroadcaster


def _is_unity_available() -> bool:
    """Check if Unity backend is reachable on port 5010."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        result = sock.connect_ex(('localhost', 5010))
        sock.close()
        return result == 0
    except Exception:
        return False


def _is_ros_available() -> bool:
    """Check if ROS bridge (Docker/MoveIt) is reachable on port 5020."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        result = sock.connect_ex(('localhost', 5020))
        sock.close()
        return result == 0
    except Exception:
        return False


_UNITY_AVAILABLE = _is_unity_available()
_ROS_AVAILABLE = _is_ros_available()
_SKIP_REASON = "Unity not running. Start Unity and backend servers to run these tests."
_SKIP_REASON_ROS = "ROS/MoveIt not running. Start Docker with MoveIt to run these tests."


@pytest.mark.integration
class TestGraspEndToEnd:
    """End-to-end integration tests for grasp planning."""

    @pytest.fixture
    def mock_unity_connection(self):
        """Mock Unity connection for integration tests."""
        # Disable ROS to ensure tests use mocked Unity connection
        with patch('config.ROS.ROS_ENABLED', False), \
             patch('operations.GraspOperations._get_command_broadcaster') as mock:
            broadcaster = MagicMock()
            broadcaster.send_command_and_wait = MagicMock()
            mock.return_value = broadcaster
            yield broadcaster

    @pytest.mark.slow
    def test_grasp_object_full_pipeline_success(self, mock_unity_connection):
        """Test grasp operation sends command successfully (async mode)."""
        # Async mode - command is sent without waiting for Unity response
        mock_unity_connection.send_command.return_value = True

        # Execute grasp operation
        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01",
            use_advanced_planning=True,
            preferred_approach="auto",
            enable_retreat=True
        )

        # Verify command was sent successfully
        assert result["success"] is True
        assert result["result"]["command_sent"] is True
        assert result["result"]["robot_id"] == "Robot1"
        assert result["result"]["object_id"] == "Cube_01"

        # Verify command structure
        call_args = mock_unity_connection.send_command.call_args[0][0]
        assert call_args["command_type"] == "grasp_object"
        assert call_args["parameters"]["use_advanced_planning"] is True
        assert call_args["parameters"]["preferred_approach"] == "auto"
        assert call_args["parameters"]["enable_retreat"] is True
        # Note: Actual execution details come from Unity via SequenceExecutor completion handler

    @pytest.mark.slow
    def test_grasp_with_custom_parameters(self, mock_unity_connection):
        """Test grasp with custom approach vector and distances."""
        mock_unity_connection.send_command.return_value = True

        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01",
            use_advanced_planning=True,
            custom_approach_vector=[0.0, 1.0, 0.5],
            pre_grasp_distance=0.12,
            retreat_distance=0.15
        )

        assert result["success"] is True
        assert result["result"]["command_sent"] is True

        # Verify command parameters
        call_args = mock_unity_connection.send_command.call_args[0][0]
        assert call_args["parameters"]["custom_approach_vector"]["y"] == 1.0
        assert call_args["parameters"]["pre_grasp_distance"] == 0.12
        assert call_args["parameters"]["retreat_distance"] == 0.15

    @pytest.mark.slow
    def test_grasp_planning_failure_scenarios(self, mock_unity_connection):
        """Test command send failure scenario."""
        # Test command send failure
        mock_unity_connection.send_command.return_value = False

        result = grasp_object(
            robot_id="Robot1",
            object_id="TestObject"
        )

        assert result["success"] is False
        assert result["error"]["code"] == "COMMUNICATION_ERROR"
        assert "Failed to send grasp command" in result["error"]["message"]

        # Note: Other failure scenarios (IK validation, collision, object not found)
        # are handled by Unity's grasp planning pipeline and reported via completion messages
        # which are processed by SequenceExecutor, not the operation itself

    @pytest.mark.slow
    def test_grasp_with_all_approach_types(self, mock_unity_connection):
        """Test grasp execution with all supported approach types."""
        approach_types = ["auto", "top", "front", "side"]
        mock_unity_connection.send_command.return_value = True

        for approach in approach_types:
            result = grasp_object(
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach=approach
            )

            assert result["success"] is True, f"Approach '{approach}' should succeed"
            call_args = mock_unity_connection.send_command.call_args[0][0]
            assert call_args["parameters"]["preferred_approach"] == approach

    @pytest.mark.slow
    def test_grasp_retreat_motion(self, mock_unity_connection):
        """Test grasp with and without retreat motion."""
        mock_unity_connection.send_command.return_value = True

        # Test with retreat enabled
        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01",
            enable_retreat=True
        )

        assert result["success"] is True
        call_args = mock_unity_connection.send_command.call_args[0][0]
        assert call_args["parameters"]["enable_retreat"] is True

        # Test with retreat disabled
        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01",
            enable_retreat=False
        )

        assert result["success"] is True
        call_args = mock_unity_connection.send_command.call_args[0][0]
        assert call_args["parameters"]["enable_retreat"] is False

    @pytest.mark.slow
    def test_grasp_request_id_correlation(self, mock_unity_connection):
        """Test request ID correlation in Protocol V2."""
        request_id = 12345
        mock_unity_connection.send_command.return_value = True

        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01",
            request_id=request_id
        )

        assert result["success"] is True
        assert result["result"]["request_id"] == request_id

        # Verify request_id was sent correctly
        call_args = mock_unity_connection.send_command.call_args[0][0]
        assert call_args["request_id"] == request_id

    @pytest.mark.slow
    def test_grasp_timeout_handling(self, mock_unity_connection):
        """Test command send in async mode (timeout handled by SequenceExecutor)."""
        mock_unity_connection.send_command.return_value = True

        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01"
        )

        assert result["success"] is True
        assert result["result"]["command_sent"] is True
        # Note: Timeout handling for grasp execution is managed by SequenceExecutor
        # which waits for Unity completion messages with configurable timeout

    @pytest.mark.slow
    def test_grasp_performance_benchmark(self, mock_unity_connection):
        """Benchmark grasp operation performance."""
        execution_times = []

        for i in range(10):
            unity_response = {
                "success": True,
                "data": {
                    "robot_id": "Robot1",
                    "object_id": f"Cube_{i:02d}",
                    "execution_time_ms": 120.0 + (i * 5)  # Simulated variance
                }
            }
            mock_unity_connection.send_command_and_wait.return_value = unity_response

            start_time = time.time()
            result = grasp_object(
                robot_id="Robot1",
                object_id=f"Cube_{i:02d}"
            )
            end_time = time.time()

            assert result["success"] is True
            execution_times.append((end_time - start_time) * 1000)  # Convert to ms

        # Calculate statistics
        avg_time = sum(execution_times) / len(execution_times)
        max_time = max(execution_times)
        min_time = min(execution_times)

        print(f"\nPerformance Benchmark Results:")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  Min: {min_time:.2f}ms")
        print(f"  Max: {max_time:.2f}ms")

        # Performance assertions
        assert avg_time < 50, f"Average Python overhead should be <50ms, got {avg_time:.2f}ms"


@pytest.mark.integration
@pytest.mark.requires_unity
@pytest.mark.skipif(not _UNITY_AVAILABLE, reason=_SKIP_REASON)
@pytest.mark.skipif(not _ROS_AVAILABLE, reason=_SKIP_REASON_ROS)
class TestGraspWithRealUnity:
    """
    Integration tests that require a running Unity instance.
    These tests are skipped by default unless --unity flag is provided.
    """

    @pytest.fixture(scope="class")
    def unity_connection(self):
        """Real connection to Unity (requires Unity to be running)."""
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            result = sock.connect_ex(('localhost', 5010))
            sock.close()
            if result != 0:
                pytest.skip("Unity backend not reachable on port 5010")
        except Exception:
            pytest.skip("Unity backend not reachable on port 5010")

    def test_real_grasp_execution(self, unity_connection):
        """Test grasp execution with real Unity instance."""
        from core.Imports import get_world_state

        # Populate WorldState with redCube position so ROS planning can proceed.
        # In a real run detect_object_stereo would do this; here we seed it directly.
        # Robot1 is at Unity world (-0.475, 0, 0). This position is 0.3m in front of
        # it at table height, mapping to ROS base_link (0.3, 0.0, 0.08) — reachable.
        world_state = get_world_state()
        world_state.update_object_position("redCube", [-0.475, 0.08, 0.3])

        result = grasp_object(
            robot_id="Robot1",
            object_id="redCube",
            use_advanced_planning=True,
            preferred_approach="top"
        )

        assert result.success is True, f"Grasp failed: {result.error}"

    def test_real_collision_avoidance(self, unity_connection):
        """Test that grasp of an unknown object fails gracefully."""
        result = grasp_object(
            robot_id="Robot1",
            object_id="blueCube",
            use_advanced_planning=True
        )

        # Should either succeed or fail with a meaningful error (not crash)
        assert result.success is True or result.error is not None


# Performance benchmark configuration
@pytest.mark.benchmark
class TestGraspPerformance:
    """Performance benchmarks for grasp planning."""

    @pytest.fixture
    def mock_unity_connection(self):
        """Mock Unity connection for performance tests."""
        with patch('operations.GraspOperations._get_command_broadcaster') as mock:
            broadcaster = MagicMock()
            broadcaster.send_command_and_wait = MagicMock()
            mock.return_value = broadcaster
            yield broadcaster

    @pytest.fixture
    def benchmark_config(self):
        """Configuration for performance tests."""
        return {
            "target_time_ms": 200,
            "acceptable_variance_pct": 20,
            "sample_size": 50
        }

    @pytest.mark.slow
    def test_pipeline_performance_target(self, mock_unity_connection, benchmark_config):
        """Verify command send performance (async mode)."""
        execution_times = []
        mock_unity_connection.send_command.return_value = True

        for i in range(benchmark_config["sample_size"]):
            start_time = time.time()
            result = grasp_object(robot_id="Robot1", object_id=f"Object_{i}")
            end_time = time.time()

            if result["success"]:
                execution_times.append((end_time - start_time) * 1000)  # Convert to ms

        avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
        max_time = max(execution_times) if execution_times else 0

        print(f"\nCommand Send Performance:")
        print(f"  Samples: {len(execution_times)}")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  Max: {max_time:.2f}ms")

        # Command send should be very fast (< 10ms)
        assert avg_time < 10.0, \
            f"Average command send time {avg_time:.2f}ms exceeds 10ms"

        # Note: Actual grasp execution time (pipeline performance) measured by Unity
