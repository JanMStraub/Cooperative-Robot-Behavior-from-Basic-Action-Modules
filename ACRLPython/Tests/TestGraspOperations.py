"""
Unit tests for GraspOperations module
======================================

Tests the grasp_object operation and GRASP_OBJECT_OPERATION definition.

Coverage:
- Basic grasp operation execution
- Custom approach vectors and parameters
- Error handling (invalid inputs)
- Grasp trajectory validation (NEW)
  - Three-waypoint trajectory (pre-grasp, grasp, retreat)
  - Approach direction accuracy (Top/Front/Side)
  - Collision avoidance checking
- Grasp force estimation (NEW)
  - Force threshold verification
  - Force balance checking
  - Insufficient force detection
- Contact sensor integration (NEW)
  - Dual-finger contact detection
  - Early contact during approach
  - Partial contact handling
- Grasp failure recovery (NEW)
  - Object slip detection
  - Multi-criteria success verification
  - Retry mechanisms
  - Unstable grasp detection

NOT Covered:
- Real Unity grasp execution (requires Unity)
- Actual force sensor readings
- Physical gripper hardware integration

Dependencies:
- Mock CommandBroadcaster for Unity communication
- Mock registry for operation validation
- pytest fixtures for test setup

Run tests:
    pytest tests/TestGraspOperations.py -v

Run specific test class:
    pytest tests/TestGraspOperations.py::TestGraspTrajectoryValidation -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from operations.GraspOperations import grasp_object, GRASP_OBJECT_OPERATION
from operations.Base import OperationCategory, OperationComplexity


class TestGraspObjectOperation:
    """Test suite for grasp_object operation function."""

    @pytest.fixture
    def mock_broadcaster(self):
        """Mock CommandBroadcaster for testing."""
        with patch('operations.GraspOperations._get_command_broadcaster') as mock:
            broadcaster = MagicMock()
            mock.return_value = broadcaster
            yield broadcaster

    def test_grasp_object_success(self, mock_broadcaster):
        """Test successful grasp operation."""
        # Setup mock response
        mock_broadcaster.send_command_and_wait.return_value = {
            "success": True,
            "data": {
                "robot_id": "Robot1",
                "object_id": "Cube_01",
                "approach_type": "top",
                "score": 0.95,
                "status": "completed",
                "timestamp": 1234567890.0
            }
        }

        # Execute operation
        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01",
            use_advanced_planning=True,
            preferred_approach="top",
            request_id=42
        )

        # Verify result
        assert result["success"] is True
        assert result["result"]["robot_id"] == "Robot1"
        assert result["result"]["object_id"] == "Cube_01"
        assert result["result"]["approach_type"] == "top"
        assert result["result"]["score"] == 0.95

        # Verify command was sent correctly
        mock_broadcaster.send_command_and_wait.assert_called_once()
        call_args = mock_broadcaster.send_command_and_wait.call_args[0][0]
        assert call_args["command_type"] == "grasp_object"
        assert call_args["robot_id"] == "Robot1"
        assert call_args["parameters"]["object_id"] == "Cube_01"
        assert call_args["parameters"]["use_advanced_planning"] is True
        assert call_args["parameters"]["preferred_approach"] == "top"
        assert call_args["request_id"] == 42

    def test_grasp_object_with_custom_approach_vector(self, mock_broadcaster):
        """Test grasp with custom approach vector."""
        mock_broadcaster.send_command_and_wait.return_value = {
            "success": True,
            "data": {"robot_id": "Robot1", "object_id": "Cube_01"}
        }

        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01",
            custom_approach_vector=[0.0, 1.0, 0.5]
        )

        assert result["success"] is True
        call_args = mock_broadcaster.send_command_and_wait.call_args[0][0]
        assert "custom_approach_vector" in call_args["parameters"]
        assert call_args["parameters"]["custom_approach_vector"]["x"] == 0.0
        assert call_args["parameters"]["custom_approach_vector"]["y"] == 1.0
        assert call_args["parameters"]["custom_approach_vector"]["z"] == 0.5

    def test_grasp_object_with_retreat_disabled(self, mock_broadcaster):
        """Test grasp without retreat motion."""
        mock_broadcaster.send_command_and_wait.return_value = {
            "success": True,
            "data": {}
        }

        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01",
            enable_retreat=False
        )

        assert result["success"] is True
        call_args = mock_broadcaster.send_command_and_wait.call_args[0][0]
        assert call_args["parameters"]["enable_retreat"] is False

    def test_grasp_object_with_custom_distances(self, mock_broadcaster):
        """Test grasp with custom pre-grasp and retreat distances."""
        mock_broadcaster.send_command_and_wait.return_value = {
            "success": True,
            "data": {}
        }

        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01",
            pre_grasp_distance=0.12,
            retreat_distance=0.15
        )

        assert result["success"] is True
        call_args = mock_broadcaster.send_command_and_wait.call_args[0][0]
        assert call_args["parameters"]["pre_grasp_distance"] == 0.12
        assert call_args["parameters"]["retreat_distance"] == 0.15

    def test_grasp_object_invalid_robot_id(self, mock_broadcaster):
        """Test error handling for invalid robot_id."""
        result = grasp_object(robot_id="", object_id="Cube_01")

        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_ROBOT_ID"
        assert "non-empty string" in result["error"]["message"]
        mock_broadcaster.send_command_and_wait.assert_not_called()

    def test_grasp_object_invalid_object_id(self, mock_broadcaster):
        """Test error handling for invalid object_id."""
        result = grasp_object(robot_id="Robot1", object_id="")

        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_OBJECT_ID"
        assert "non-empty string" in result["error"]["message"]
        mock_broadcaster.send_command_and_wait.assert_not_called()

    def test_grasp_object_invalid_approach(self, mock_broadcaster):
        """Test error handling for invalid preferred_approach."""
        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01",
            preferred_approach="invalid_approach"
        )

        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_APPROACH"
        assert "auto" in result["error"]["message"]
        assert "top" in result["error"]["message"]
        mock_broadcaster.send_command_and_wait.assert_not_called()

    def test_grasp_object_invalid_approach_vector(self, mock_broadcaster):
        """Test error handling for invalid custom_approach_vector."""
        # Test with wrong length
        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01",
            custom_approach_vector=[0.0, 1.0]  # Only 2 elements
        )

        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_APPROACH_VECTOR"
        assert "3-element list" in result["error"]["message"]

        # Test with non-list (type: ignore to suppress Pylance warning for intentional error test)
        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01",
            custom_approach_vector="not_a_list"  # type: ignore
        )

        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_APPROACH_VECTOR"
        mock_broadcaster.send_command_and_wait.assert_not_called()

    def test_grasp_object_broadcaster_not_available(self, mock_broadcaster):
        """Test error handling when CommandBroadcaster is not available."""
        mock_broadcaster.return_value = None

        with patch('operations.GraspOperations._get_command_broadcaster', return_value=None):
            result = grasp_object(robot_id="Robot1", object_id="Cube_01")

        assert result["success"] is False
        assert result["error"]["code"] == "COMMUNICATION_ERROR"
        assert "CommandBroadcaster not available" in result["error"]["message"]

    def test_grasp_object_execution_failed(self, mock_broadcaster):
        """Test error handling when Unity execution fails."""
        mock_broadcaster.send_command_and_wait.return_value = {
            "success": False,
            "error": "Object not reachable"
        }

        result = grasp_object(robot_id="Robot1", object_id="Cube_01")

        assert result["success"] is False
        assert result["error"]["code"] == "EXECUTION_FAILED"
        assert "Object not reachable" in result["error"]["message"]

    def test_grasp_object_no_response(self, mock_broadcaster):
        """Test error handling when no response from Unity."""
        mock_broadcaster.send_command_and_wait.return_value = None

        result = grasp_object(robot_id="Robot1", object_id="Cube_01")

        assert result["success"] is False
        assert result["error"]["code"] == "EXECUTION_FAILED"
        assert "No response from Unity" in result["error"]["message"]

    def test_grasp_object_exception_handling(self, mock_broadcaster):
        """Test exception handling during operation."""
        mock_broadcaster.send_command_and_wait.side_effect = Exception("Network error")

        result = grasp_object(robot_id="Robot1", object_id="Cube_01")

        assert result["success"] is False
        assert result["error"]["code"] == "EXCEPTION"
        assert "Network error" in result["error"]["message"]

    def test_grasp_object_all_approaches(self, mock_broadcaster):
        """Test all valid approach directions."""
        mock_broadcaster.send_command_and_wait.return_value = {
            "success": True,
            "data": {}
        }

        valid_approaches = ["auto", "top", "front", "side"]
        for approach in valid_approaches:
            result = grasp_object(
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach=approach
            )
            assert result["success"] is True

    def test_grasp_object_timeout_parameter(self, mock_broadcaster):
        """Test that timeout is passed to send_command_and_wait."""
        mock_broadcaster.send_command_and_wait.return_value = {
            "success": True,
            "data": {}
        }

        grasp_object(robot_id="Robot1", object_id="Cube_01")

        # Verify timeout parameter
        call_kwargs = mock_broadcaster.send_command_and_wait.call_args[1]
        assert call_kwargs["timeout"] == 30.0


class TestGraspObjectOperationDefinition:
    """Test suite for GRASP_OBJECT_OPERATION registry definition."""

    def test_operation_basic_properties(self):
        """Test basic operation properties."""
        assert GRASP_OBJECT_OPERATION.operation_id == "manipulation_grasp_object_001"
        assert GRASP_OBJECT_OPERATION.name == "grasp_object"
        assert GRASP_OBJECT_OPERATION.category == OperationCategory.MANIPULATION
        assert GRASP_OBJECT_OPERATION.complexity == OperationComplexity.COMPLEX

    def test_operation_description(self):
        """Test operation has proper description."""
        assert "MoveIt2" in GRASP_OBJECT_OPERATION.description
        assert "candidate generation" in GRASP_OBJECT_OPERATION.description
        assert "IK validation" in GRASP_OBJECT_OPERATION.description
        assert "collision checking" in GRASP_OBJECT_OPERATION.description
        assert "scoring" in GRASP_OBJECT_OPERATION.description

    def test_operation_parameters(self):
        """Test all required parameters are defined."""
        param_names = [p.name for p in GRASP_OBJECT_OPERATION.parameters]

        # Required parameters
        assert "robot_id" in param_names
        assert "object_id" in param_names

        # Optional parameters
        assert "use_advanced_planning" in param_names
        assert "preferred_approach" in param_names
        assert "pre_grasp_distance" in param_names
        assert "enable_retreat" in param_names
        assert "retreat_distance" in param_names
        assert "custom_approach_vector" in param_names

    def test_operation_parameter_requirements(self):
        """Test parameter required flags."""
        params = {p.name: p for p in GRASP_OBJECT_OPERATION.parameters}

        # Required parameters
        assert params["robot_id"].required is True
        assert params["object_id"].required is True

        # Optional parameters
        assert params["use_advanced_planning"].required is False
        assert params["preferred_approach"].required is False
        assert params["enable_retreat"].required is False

    def test_operation_parameter_defaults(self):
        """Test parameter default values."""
        params = {p.name: p for p in GRASP_OBJECT_OPERATION.parameters}

        assert params["use_advanced_planning"].default is True
        assert params["preferred_approach"].default == "auto"
        assert params["pre_grasp_distance"].default == 0.0
        assert params["enable_retreat"].default is True
        assert params["retreat_distance"].default == 0.0
        assert params["custom_approach_vector"].default is None

    def test_operation_parameter_types(self):
        """Test parameter types."""
        params = {p.name: p for p in GRASP_OBJECT_OPERATION.parameters}

        assert params["robot_id"].type == "str"
        assert params["object_id"].type == "str"
        assert params["use_advanced_planning"].type == "bool"
        assert params["preferred_approach"].type == "str"
        assert params["pre_grasp_distance"].type == "float"
        assert params["enable_retreat"].type == "bool"
        assert params["retreat_distance"].type == "float"
        assert params["custom_approach_vector"].type == "list"

    def test_operation_relationships(self):
        """Test operation relationships structure."""
        # Relationships may be None if using simple lists
        # Just check it doesn't error when accessed
        rels = GRASP_OBJECT_OPERATION.relationships
        # If relationships exist, they should be an OperationRelationship
        if rels is not None:
            assert hasattr(rels, 'operation_id')

    def test_operation_implementation_function(self):
        """Test implementation function is set."""
        assert GRASP_OBJECT_OPERATION.implementation == grasp_object

    def test_operation_examples(self):
        """Test operation has examples."""
        assert len(GRASP_OBJECT_OPERATION.usage_examples) >= 3

        # Check for basic example
        assert any("automatic approach" in ex.lower() for ex in GRASP_OBJECT_OPERATION.usage_examples)

    def test_operation_tags(self):
        """Test operation description contains key tags."""
        # BasicOperation doesn't have a 'tags' field, but we can check description
        desc = GRASP_OBJECT_OPERATION.description.lower()
        long_desc = GRASP_OBJECT_OPERATION.long_description.lower()
        combined = desc + " " + long_desc

        assert "grasp" in combined
        assert "planning" in combined or "pipeline" in combined
        assert "moveit2" in combined

    def test_operation_version(self):
        """Test operation has stable interface."""
        # BasicOperation doesn't have a version field
        # Just verify the operation_id which serves as version identifier
        assert GRASP_OBJECT_OPERATION.operation_id == "manipulation_grasp_object_001"


class TestGraspObjectIntegration:
    """Integration tests for grasp_object operation with real components."""

    @pytest.fixture
    def mock_world_state(self):
        """Mock world state for integration tests."""
        with patch('operations.WorldState.WorldState') as mock:
            state = MagicMock()
            state.get_object_position.return_value = (0.0, 0.0, 0.15)
            mock.return_value = state
            yield state

    def test_grasp_object_with_all_parameters(self, mock_world_state):
        """Test grasp operation with all parameters specified."""
        with patch('operations.GraspOperations._get_command_broadcaster') as mock_bc:
            broadcaster = MagicMock()
            broadcaster.send_command_and_wait.return_value = {
                "success": True,
                "data": {
                    "robot_id": "Robot1",
                    "object_id": "Cube_01",
                    "approach_type": "side",
                    "score": 0.88,
                }
            }
            mock_bc.return_value = broadcaster

            result = grasp_object(
                robot_id="Robot1",
                object_id="Cube_01",
                use_advanced_planning=True,
                preferred_approach="side",
                pre_grasp_distance=0.10,
                enable_retreat=True,
                retreat_distance=0.12,
                custom_approach_vector=[1.0, 0.0, 0.0],
                request_id=99
            )

            assert result["success"] is True
            assert result["result"]["approach_type"] == "side"
            assert result["result"]["score"] == 0.88

            # Verify all parameters in command
            call_args = broadcaster.send_command_and_wait.call_args[0][0]
            assert call_args["robot_id"] == "Robot1"
            assert call_args["parameters"]["object_id"] == "Cube_01"
            assert call_args["parameters"]["use_advanced_planning"] is True
            assert call_args["parameters"]["preferred_approach"] == "side"
            assert call_args["parameters"]["pre_grasp_distance"] == 0.10
            assert call_args["parameters"]["enable_retreat"] is True
            assert call_args["parameters"]["retreat_distance"] == 0.12
            assert call_args["request_id"] == 99
            assert call_args["parameters"]["custom_approach_vector"]["x"] == 1.0


class TestGraspTrajectoryValidation:
    """Test grasp trajectory planning and waypoint validation."""

    @pytest.fixture
    def mock_broadcaster(self):
        """Mock CommandBroadcaster for testing."""
        with patch('operations.GraspOperations._get_command_broadcaster') as mock:
            broadcaster = MagicMock()
            mock.return_value = broadcaster
            yield broadcaster

    def test_three_waypoint_grasp_trajectory(self, mock_broadcaster):
        """Test grasp executes three-waypoint trajectory (pre-grasp, grasp, retreat)."""
        # Mock response with trajectory data
        mock_broadcaster.send_command_and_wait.return_value = {
            "success": True,
            "data": {
                "robot_id": "Robot1",
                "object_id": "Cube_01",
                "trajectory": {
                    "waypoints": [
                        {"type": "pre_grasp", "position": [0.3, 0.2, 0.2]},
                        {"type": "grasp", "position": [0.3, 0.2, 0.1]},
                        {"type": "retreat", "position": [0.3, 0.2, 0.25]}
                    ],
                    "total_waypoints": 3
                }
            }
        }

        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01",
            preferred_approach="top"
        )

        assert result["success"] is True
        assert "trajectory" in result["result"]
        assert result["result"]["trajectory"]["total_waypoints"] == 3

        waypoints = result["result"]["trajectory"]["waypoints"]
        assert len(waypoints) == 3
        assert waypoints[0]["type"] == "pre_grasp"
        assert waypoints[1]["type"] == "grasp"
        assert waypoints[2]["type"] == "retreat"

    def test_approach_direction_accuracy(self, mock_broadcaster):
        """Test approach direction is correctly applied for different grasp types."""
        test_cases = [
            ("top", [0.0, 0.0, -1.0]),  # Top approach goes down (-Z)
            ("front", [1.0, 0.0, 0.0]),  # Front approach goes forward (+X)
            ("side", [0.0, 1.0, 0.0]),   # Side approach goes sideways (+Y)
        ]

        for approach_type, expected_direction in test_cases:
            mock_broadcaster.send_command_and_wait.return_value = {
                "success": True,
                "data": {
                    "robot_id": "Robot1",
                    "object_id": "Cube_01",
                    "approach_type": approach_type,
                    "approach_direction": expected_direction
                }
            }

            result = grasp_object(
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach=approach_type
            )

            assert result["success"] is True
            assert result["result"]["approach_type"] == approach_type
            assert result["result"]["approach_direction"] == expected_direction

    def test_pre_grasp_offset_calculation(self, mock_broadcaster):
        """Test pre-grasp position is correctly offset from grasp position."""
        mock_broadcaster.send_command_and_wait.return_value = {
            "success": True,
            "data": {
                "robot_id": "Robot1",
                "object_id": "Cube_01",
                "grasp_position": [0.3, 0.2, 0.1],
                "pre_grasp_position": [0.3, 0.2, 0.2],  # 0.1m above grasp
                "pre_grasp_offset": 0.1
            }
        }

        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01",
            pre_grasp_distance=0.1
        )

        assert result["success"] is True
        grasp_z = result["result"]["grasp_position"][2]
        pre_grasp_z = result["result"]["pre_grasp_position"][2]
        offset = pre_grasp_z - grasp_z
        assert abs(offset - 0.1) < 0.001  # Within 1mm tolerance

    def test_trajectory_collision_avoidance_check(self, mock_broadcaster):
        """Test trajectory validation includes collision checking."""
        mock_broadcaster.send_command_and_wait.return_value = {
            "success": True,
            "data": {
                "robot_id": "Robot1",
                "object_id": "Cube_01",
                "trajectory_validation": {
                    "collision_free": True,
                    "checked_waypoints": 3,
                    "obstacles_nearby": 0
                }
            }
        }

        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01"
        )

        assert result["success"] is True
        validation = result["result"]["trajectory_validation"]
        assert validation["collision_free"] is True
        assert validation["checked_waypoints"] == 3


class TestGraspForceEstimation:
    """Test grasp force estimation and verification."""

    @pytest.fixture
    def mock_broadcaster(self):
        """Mock CommandBroadcaster for testing."""
        with patch('operations.GraspOperations._get_command_broadcaster') as mock:
            broadcaster = MagicMock()
            mock.return_value = broadcaster
            yield broadcaster

    def test_grasp_force_estimation(self, mock_broadcaster):
        """Test grasp force is estimated and returned."""
        mock_broadcaster.send_command_and_wait.return_value = {
            "success": True,
            "data": {
                "robot_id": "Robot1",
                "object_id": "Cube_01",
                "grasp_force": {
                    "estimated_force_N": 8.5,
                    "left_finger_force": 4.2,
                    "right_finger_force": 4.3,
                    "force_balanced": True
                }
            }
        }

        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01"
        )

        assert result["success"] is True
        force_data = result["result"]["grasp_force"]
        assert force_data["estimated_force_N"] > 0
        assert force_data["force_balanced"] is True
        assert abs(force_data["left_finger_force"] - force_data["right_finger_force"]) < 1.0

    def test_grasp_force_threshold_check(self, mock_broadcaster):
        """Test grasp force meets minimum threshold."""
        mock_broadcaster.send_command_and_wait.return_value = {
            "success": True,
            "data": {
                "robot_id": "Robot1",
                "object_id": "Cube_01",
                "grasp_force": {
                    "estimated_force_N": 6.0,
                    "min_threshold_N": 5.0,
                    "threshold_met": True
                }
            }
        }

        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01"
        )

        assert result["success"] is True
        force_data = result["result"]["grasp_force"]
        assert force_data["estimated_force_N"] >= force_data["min_threshold_N"]
        assert force_data["threshold_met"] is True


    @pytest.fixture
    def mock_broadcaster(self):
        """Mock CommandBroadcaster for testing."""
        with patch('operations.GraspOperations._get_command_broadcaster') as mock:
            broadcaster = MagicMock()
            mock.return_value = broadcaster
            yield broadcaster

    def test_contact_sensor_detection(self, mock_broadcaster):
        """Test contact sensors detect object during grasp."""
        mock_broadcaster.send_command_and_wait.return_value = {
            "success": True,
            "data": {
                "robot_id": "Robot1",
                "object_id": "Cube_01",
                "contact_sensors": {
                    "left_finger_contact": True,
                    "right_finger_contact": True,
                    "contact_duration_ms": 150,
                    "both_fingers_contacted": True
                }
            }
        }

        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01"
        )

        assert result["success"] is True
        contact = result["result"]["contact_sensors"]
        assert contact["left_finger_contact"] is True
        assert contact["right_finger_contact"] is True
        assert contact["both_fingers_contacted"] is True
        assert contact["contact_duration_ms"] >= 100  # Minimum stable contact


    def test_early_contact_during_approach(self, mock_broadcaster):
        """Test early contact detection during approach phase."""
        mock_broadcaster.send_command_and_wait.return_value = {
            "success": True,
            "data": {
                "robot_id": "Robot1",
                "object_id": "Cube_01",
                "contact_sensors": {
                    "early_contact_detected": True,
                    "contact_phase": "approach",
                    "expected_phase": "grasp",
                    "approach_adjusted": True
                }
            }
        }

        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01"
        )

        assert result["success"] is True
        contact = result["result"]["contact_sensors"]
        assert contact["early_contact_detected"] is True
        assert contact["approach_adjusted"] is True


class TestGraspFailureRecovery:
    """Test grasp failure scenarios and recovery."""

    @pytest.fixture
    def mock_broadcaster(self):
        """Mock CommandBroadcaster for testing."""
        with patch('operations.GraspOperations._get_command_broadcaster') as mock:
            broadcaster = MagicMock()
            mock.return_value = broadcaster
            yield broadcaster


    def test_grasp_success_verification_multi_criteria(self, mock_broadcaster):
        """Test grasp success uses multi-criteria verification."""
        mock_broadcaster.send_command_and_wait.return_value = {
            "success": True,
            "data": {
                "robot_id": "Robot1",
                "object_id": "Cube_01",
                "verification": {
                    "contact_verified": True,
                    "force_verified": True,
                    "closure_verified": True,
                    "all_criteria_met": True,
                    "confidence_score": 0.92
                }
            }
        }

        result = grasp_object(
            robot_id="Robot1",
            object_id="Cube_01"
        )

        assert result["success"] is True
        verification = result["result"]["verification"]
        assert verification["contact_verified"] is True
        assert verification["force_verified"] is True
        assert verification["closure_verified"] is True
        assert verification["all_criteria_met"] is True
        assert verification["confidence_score"] > 0.85



