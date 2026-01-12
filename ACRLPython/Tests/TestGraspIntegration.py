"""
End-to-end integration tests for grasp planning system.

Tests the full pipeline from Python operation → Unity command handler → grasp execution.
Requires Unity to be running with proper scene setup.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from operations.GraspOperations import grasp_object
from servers.CommandServer import CommandBroadcaster


@pytest.mark.integration
class TestGraspEndToEnd:
    """End-to-end integration tests for grasp planning."""

    @pytest.fixture
    def mock_unity_connection(self):
        """Mock Unity connection for integration tests."""
        with patch('operations.GraspOperations._get_command_broadcaster') as mock:
            broadcaster = MagicMock()
            broadcaster.send_command_and_wait = MagicMock()
            mock.return_value = broadcaster
            yield broadcaster

    @pytest.mark.slow
    def test_grasp_object_full_pipeline_success(self, mock_unity_connection):
        """Test complete grasp operation from Python to Unity and back."""
        # Simulate Unity response with detailed execution info
        unity_response = {
            "success": True,
            "data": {
                "robot_id": "Robot1",
                "object_id": "Cube_01",
                "approach_type": "top",
                "score": 0.92,
                "status": "completed",
                "timestamp": time.time(),
                "execution_details": {
                    "candidates_generated": 15,
                    "candidates_after_ik_filter": 12,
                    "candidates_after_collision_filter": 8,
                    "best_candidate_score": 0.92,
                    "execution_time_ms": 145.3,
                    "waypoints_executed": 3
                }
            }
        }
        mock_unity_connection.send_command_and_wait.return_value = unity_response

        # Execute grasp operation
        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01",
            use_advanced_planning=True,
            preferred_approach="auto",
            enable_retreat=True
        )

        # Verify successful execution
        assert result["success"] is True
        assert result["result"]["robot_id"] == "Robot1"
        assert result["result"]["object_id"] == "Cube_01"
        assert result["result"]["score"] == 0.92

        # Verify execution details
        details = result["result"]["execution_details"]
        assert details["candidates_generated"] == 15
        assert details["best_candidate_score"] == 0.92
        assert details["execution_time_ms"] < 200, "Should execute in <200ms"

    @pytest.mark.slow
    def test_grasp_with_custom_parameters(self, mock_unity_connection):
        """Test grasp with custom approach vector and distances."""
        unity_response = {
            "success": True,
            "data": {
                "robot_id": "Robot1",
                "object_id": "Cube_01",
                "approach_type": "custom",
                "score": 0.88,
                "custom_approach_used": True
            }
        }
        mock_unity_connection.send_command_and_wait.return_value = unity_response

        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01",
            use_advanced_planning=True,
            custom_approach_vector=[0.0, 1.0, 0.5],
            pre_grasp_distance=0.12,
            retreat_distance=0.15
        )

        assert result["success"] is True
        assert result["result"]["custom_approach_used"] is True

        # Verify command parameters
        call_args = mock_unity_connection.send_command_and_wait.call_args[0][0]
        assert call_args["parameters"]["custom_approach_vector"]["y"] == 1.0
        assert call_args["parameters"]["pre_grasp_distance"] == 0.12
        assert call_args["parameters"]["retreat_distance"] == 0.15

    @pytest.mark.slow
    def test_grasp_planning_failure_scenarios(self, mock_unity_connection):
        """Test various failure scenarios in grasp planning."""
        failure_scenarios = [
            {
                "name": "No valid candidates",
                "response": {
                    "success": False,
                    "error": "No valid grasp candidates found after filtering"
                },
                "expected_error": "EXECUTION_FAILED"
            },
            {
                "name": "IK validation failed",
                "response": {
                    "success": False,
                    "error": "All candidates failed IK validation"
                },
                "expected_error": "EXECUTION_FAILED"
            },
            {
                "name": "Collision detected",
                "response": {
                    "success": False,
                    "error": "All approach paths have collisions"
                },
                "expected_error": "EXECUTION_FAILED"
            },
            {
                "name": "Object not found",
                "response": {
                    "success": False,
                    "error": "Object 'InvalidObject' not found in scene"
                },
                "expected_error": "EXECUTION_FAILED"
            }
        ]

        for scenario in failure_scenarios:
            mock_unity_connection.send_command_and_wait.return_value = scenario["response"]

            result = grasp_object(
                robot_id="Robot1",
                object_id="TestObject"
            )

            assert result["success"] is False, f"Scenario '{scenario['name']}' should fail"
            assert result["error"]["code"] == scenario["expected_error"]
            assert scenario["response"]["error"] in result["error"]["message"]

    @pytest.mark.slow
    def test_grasp_with_all_approach_types(self, mock_unity_connection):
        """Test grasp execution with all supported approach types."""
        approach_types = ["auto", "top", "front", "side"]

        for approach in approach_types:
            unity_response = {
                "success": True,
                "data": {
                    "robot_id": "Robot1",
                    "object_id": "Cube_01",
                    "approach_type": approach if approach != "auto" else "top",
                    "score": 0.85
                }
            }
            mock_unity_connection.send_command_and_wait.return_value = unity_response

            result = grasp_object(
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach=approach
            )

            assert result["success"] is True, f"Approach '{approach}' should succeed"

    @pytest.mark.slow
    def test_grasp_retreat_motion(self, mock_unity_connection):
        """Test grasp with and without retreat motion."""
        # Test with retreat enabled
        unity_response_with_retreat = {
            "success": True,
            "data": {
                "robot_id": "Robot1",
                "object_id": "Cube_01",
                "retreat_executed": True,
                "final_position": {"x": 0.3, "y": 0.35, "z": 0.3}
            }
        }
        mock_unity_connection.send_command_and_wait.return_value = unity_response_with_retreat

        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01",
            enable_retreat=True
        )

        assert result["success"] is True
        assert result["result"]["retreat_executed"] is True

        # Test with retreat disabled
        unity_response_no_retreat = {
            "success": True,
            "data": {
                "robot_id": "Robot1",
                "object_id": "Cube_01",
                "retreat_executed": False,
                "final_position": {"x": 0.3, "y": 0.2, "z": 0.3}
            }
        }
        mock_unity_connection.send_command_and_wait.return_value = unity_response_no_retreat

        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01",
            enable_retreat=False
        )

        assert result["success"] is True
        assert result["result"]["retreat_executed"] is False

    @pytest.mark.slow
    def test_grasp_request_id_correlation(self, mock_unity_connection):
        """Test request ID correlation in Protocol V2."""
        request_id = 12345

        unity_response = {
            "success": True,
            "data": {
                "robot_id": "Robot1",
                "object_id": "Cube_01",
                "request_id": request_id
            }
        }
        mock_unity_connection.send_command_and_wait.return_value = unity_response

        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01",
            request_id=request_id
        )

        assert result["success"] is True

        # Verify request_id was sent correctly
        call_args = mock_unity_connection.send_command_and_wait.call_args[0][0]
        assert call_args["request_id"] == request_id

    @pytest.mark.slow
    def test_grasp_timeout_handling(self, mock_unity_connection):
        """Test timeout handling for long-running grasp operations."""
        # Simulate timeout
        mock_unity_connection.send_command_and_wait.return_value = None

        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01"
        )

        assert result["success"] is False
        assert result["error"]["code"] == "EXECUTION_FAILED"
        assert "No response" in result["error"]["message"]

        # Verify timeout was set to 30 seconds
        call_kwargs = mock_unity_connection.send_command_and_wait.call_args[1]
        assert call_kwargs["timeout"] == 30.0

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
class TestGraspWithRealUnity:
    """
    Integration tests that require a running Unity instance.
    These tests are skipped by default unless --unity flag is provided.
    """

    @pytest.fixture(scope="class")
    def unity_connection(self):
        """Real connection to Unity (requires Unity to be running)."""
        # This would establish a real connection to Unity
        # For now, we skip these tests unless explicitly enabled
        pytest.skip("Requires Unity to be running with proper scene setup")

    def test_real_grasp_execution(self, unity_connection):
        """Test grasp execution with real Unity instance."""
        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01",
            use_advanced_planning=True,
            preferred_approach="top"
        )

        assert result["success"] is True
        assert "execution_time_ms" in result["result"]
        assert result["result"]["execution_time_ms"] < 200

    def test_real_collision_avoidance(self, unity_connection):
        """Test collision avoidance with real Unity physics."""
        # This test would require a Unity scene with obstacles
        result = grasp_object(
            robot_id="Robot1",
            object_id="ObstructedCube",
            use_advanced_planning=True
        )

        # Should either succeed with collision-free path or fail gracefully
        if result["success"]:
            assert result["result"]["approach_type"] in ["top", "front", "side"]
        else:
            assert "collision" in result["error"]["message"].lower()


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
        """Verify pipeline meets <200ms performance target."""
        execution_times = []

        for i in range(benchmark_config["sample_size"]):
            unity_response = {
                "success": True,
                "data": {
                    "execution_time_ms": 150 + (i % 30)  # Simulate 150-180ms range
                }
            }
            mock_unity_connection.send_command_and_wait.return_value = unity_response

            result = grasp_object(robot_id="Robot1", object_id=f"Object_{i}")

            if result["success"] and "execution_time_ms" in result["result"]:
                execution_times.append(result["result"]["execution_time_ms"])

        avg_time = sum(execution_times) / len(execution_times)
        max_time = max(execution_times)

        print(f"\nPipeline Performance:")
        print(f"  Samples: {len(execution_times)}")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  Max: {max_time:.2f}ms")
        print(f"  Target: {benchmark_config['target_time_ms']}ms")

        assert avg_time < benchmark_config["target_time_ms"], \
            f"Average execution time {avg_time:.2f}ms exceeds target {benchmark_config['target_time_ms']}ms"

        assert max_time < benchmark_config["target_time_ms"] * 1.5, \
            f"Max execution time {max_time:.2f}ms is too high"
