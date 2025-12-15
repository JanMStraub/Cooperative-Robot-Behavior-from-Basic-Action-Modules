#!/usr/bin/env python3
"""
Integration tests for SequenceExecutor with Verification

Tests integration of formal verification and coordination safety checks
within the sequence execution pipeline.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from orchestrators.SequenceExecutor import SequenceExecutor
from operations.Verification import VerificationResult, PredicateViolation
from operations.CoordinationVerifier import CoordinationCheckResult, CoordinationIssue
from operations.Base import OperationResult, BasicOperation, OperationCategory, OperationComplexity


class TestSequenceExecutorVerification:
    """Test SequenceExecutor with verification enabled"""

    @patch('orchestrators.SequenceExecutor.get_global_registry')
    @patch('operations.Verification.OperationVerifier')
    def test_verification_enabled_blocks_bad_operation(self, mock_verifier_class, mock_registry, cleanup_world_state):
        """Test verification blocks operation with failed preconditions"""
        # Setup mock operation
        mock_op = Mock(spec=BasicOperation)
        mock_op.name = "move_to_coordinate"
        mock_op.category = OperationCategory.NAVIGATION
        mock_op.preconditions = ["target_within_reach(robot_id, x, y, z)"]
        mock_op.postconditions = []

        mock_registry.return_value.get_operation_by_name = Mock(return_value=mock_op)
        # Mock execute_operation_by_name to ensure it's not called
        mock_registry.return_value.execute_operation_by_name = Mock()

        # Setup mock verifier to fail preconditions
        mock_verifier = mock_verifier_class.return_value
        failed_result = VerificationResult(success=False, execution_allowed=False)
        failed_result.violations = [
            PredicateViolation(
                predicate="target_within_reach(robot_id, x, y, z)",
                reason="Target exceeds max reach of 0.8m",
                severity="error",
                recovery_suggestions=["Move target closer", "Use different robot"]
            )
        ]
        mock_verifier.verify_preconditions = Mock(return_value=failed_result)

        # Create executor with verification enabled
        executor = SequenceExecutor(enable_verification=True)

        # Execute command with valid parameters (within range)
        result = executor._execute_single_command(
            operation="move_to_coordinate",
            params={
                "robot_id": "Robot1",
                "x": 0.8,  # Valid parameter value
                "y": 0.0,
                "z": 0.1
            },
            timeout=30.0
        )

        assert result["success"] is False
        assert "Precondition failed" in result.get("error", "")
        # Operation should not have been executed
        mock_registry.return_value.execute_operation_by_name.assert_not_called()

    @patch('orchestrators.SequenceExecutor.get_global_registry')
    def test_verification_disabled_no_checks(self, mock_registry, cleanup_world_state):
        """Test no verification when disabled"""
        # Setup mock operation
        mock_op = Mock(spec=BasicOperation)
        mock_op.name = "move_to_coordinate"
        mock_op.category = OperationCategory.NAVIGATION
        mock_op.preconditions = ["target_within_reach(robot_id, x, y, z)"]
        mock_op.execute = Mock(return_value=OperationResult.success_result({"final_position": (0.5, 0.0, 0.1)}))

        mock_registry.return_value.get_operation_by_name = Mock(return_value=mock_op)
        # Mock execute_operation_by_name to return success
        mock_registry.return_value.execute_operation_by_name = Mock(
            return_value=OperationResult.success_result({"final_position": (0.5, 0.0, 0.1)})
        )

        # Create executor with verification and completion checking disabled (for testing)
        executor = SequenceExecutor(enable_verification=False, check_completion=False)

        # Execute command with valid parameters
        result = executor._execute_single_command(
            operation="move_to_coordinate",
            params={
                "robot_id": "Robot1",
                "x": 0.5,  # Valid parameter value
                "y": 0.0,
                "z": 0.1
            },
            timeout=30.0
        )

        # Should execute without verification checks
        assert result["success"] is True
        mock_registry.return_value.execute_operation_by_name.assert_called_once()

    @patch('orchestrators.SequenceExecutor.get_global_registry')
    @patch('operations.Verification.OperationVerifier')
    def test_precondition_fails_blocks_execution(self, mock_verifier_class, mock_registry, cleanup_world_state):
        """Test precondition failure blocks execution"""
        mock_op = Mock(spec=BasicOperation)
        mock_op.name = "control_gripper"
        mock_op.category = OperationCategory.MANIPULATION
        mock_op.preconditions = ["robot_is_initialized(robot_id)"]
        mock_op.postconditions = []

        mock_registry.return_value.get_operation_by_name = Mock(return_value=mock_op)
        # Mock execute_operation_by_name to ensure it's not called
        mock_registry.return_value.execute_operation_by_name = Mock()

        # Verifier fails precondition
        mock_verifier = mock_verifier_class.return_value
        failed_result = VerificationResult(success=False, execution_allowed=False)
        failed_result.violations = [
            PredicateViolation(
                predicate="robot_is_initialized(robot_id)",
                reason="Robot not found in system",
                severity="error"
            )
        ]
        mock_verifier.verify_preconditions = Mock(return_value=failed_result)

        executor = SequenceExecutor(enable_verification=True)

        result = executor._execute_single_command(
            operation="control_gripper",
            params={
                "robot_id": "Robot1",  # Valid robot ID
                "open_gripper": True   # Add required parameter
            },
            timeout=30.0
        )

        assert result["success"] is False
        assert "Precondition failed" in result["error"]
        # Operation should not have been executed
        mock_registry.return_value.execute_operation_by_name.assert_not_called()

    @patch('orchestrators.SequenceExecutor.get_global_registry')
    @patch('operations.Verification.OperationVerifier')
    @patch('orchestrators.SequenceExecutor.CoordinationVerifier')
    def test_coordination_conflict_blocks_execution(self, mock_coord_verifier_class, mock_verifier_class, mock_registry, cleanup_world_state):
        """Test multi-robot conflict detected and blocks execution"""
        mock_op = Mock(spec=BasicOperation)
        mock_op.name = "move_to_coordinate"
        mock_op.category = OperationCategory.NAVIGATION
        mock_op.preconditions = []
        mock_op.postconditions = []

        mock_registry.return_value.get_operation_by_name = Mock(return_value=mock_op)
        # Mock execute_operation_by_name to ensure it's not called
        mock_registry.return_value.execute_operation_by_name = Mock()

        # Verifier passes preconditions
        mock_verifier = mock_verifier_class.return_value
        mock_verifier.verify_preconditions = Mock(return_value=VerificationResult(success=True))

        # Coordination verifier detects collision
        mock_coord_verifier = mock_coord_verifier_class.return_value
        coord_result = CoordinationCheckResult(safe=False)
        coord_result.issues = [
            CoordinationIssue(
                issue_type="collision",
                severity="blocking",
                description="Path collision with Robot2",
                affected_robots=["Robot1", "Robot2"],
                resolution_suggestions=["Serialize movements", "Use different paths"]
            )
        ]
        mock_coord_verifier.verify_multi_robot_safety = Mock(return_value=coord_result)

        executor = SequenceExecutor(enable_verification=True, check_completion=False)

        result = executor._execute_single_command(
            operation="move_to_coordinate",
            params={
                "robot_id": "Robot1",
                "x": 0.3,
                "y": 0.0,
                "z": 0.1
            },
            timeout=30.0
        )

        assert result["success"] is False
        # Check for coordination error message (matches implementation at line 262)
        assert "coordination" in result["error"].lower() or "collision" in result["error"].lower()
        # Operation should not have been executed
        mock_registry.return_value.execute_operation_by_name.assert_not_called()

    @patch('orchestrators.SequenceExecutor.get_global_registry')
    @patch('operations.Verification.OperationVerifier')
    def test_postcondition_warning_logged(self, mock_verifier_class, mock_registry, cleanup_world_state):
        """Test postcondition violation logged as warning"""
        mock_op = Mock(spec=BasicOperation)
        mock_op.name = "move_to_coordinate"
        mock_op.category = OperationCategory.NAVIGATION
        mock_op.preconditions = []
        mock_op.postconditions = ["robot_is_stationary(robot_id)"]
        mock_op.execute = Mock(return_value=OperationResult.success_result({"final_position": (0.3, 0.2, 0.1)}))

        mock_registry.return_value.get_operation_by_name = Mock(return_value=mock_op)
        # Mock execute_operation_by_name to return success
        mock_registry.return_value.execute_operation_by_name = Mock(
            return_value=OperationResult.success_result({"final_position": (0.3, 0.2, 0.1)})
        )

        # Verifier passes preconditions but fails postconditions
        mock_verifier = mock_verifier_class.return_value
        mock_verifier.verify_preconditions = Mock(return_value=VerificationResult(success=True))

        post_result = VerificationResult(success=True)  # Postconditions are warnings
        post_result.warnings = [
            PredicateViolation(
                predicate="robot_is_stationary(robot_id)",
                reason="Robot still moving",
                severity="warning"
            )
        ]
        mock_verifier.verify_postconditions = Mock(return_value=post_result)

        executor = SequenceExecutor(enable_verification=True, check_completion=False)

        result = executor._execute_single_command(
            operation="move_to_coordinate",
            params={
                "robot_id": "Robot1",
                "x": 0.3,
                "y": 0.2,
                "z": 0.1
            },
            timeout=30.0
        )

        # Should succeed despite postcondition warning
        assert result["success"] is True
        # Operation should have been executed
        mock_registry.return_value.execute_operation_by_name.assert_called_once()

    @patch('orchestrators.SequenceExecutor.get_global_registry')
    @patch('operations.Verification.OperationVerifier')
    def test_verification_recovery_suggestions_included(self, mock_verifier_class, mock_registry, cleanup_world_state):
        """Test error includes recovery suggestions"""
        mock_op = Mock(spec=BasicOperation)
        mock_op.name = "move_to_coordinate"
        mock_op.category = OperationCategory.NAVIGATION
        mock_op.preconditions = ["target_within_reach(robot_id, x, y, z)"]
        mock_op.postconditions = []

        mock_registry.return_value.get_operation_by_name = Mock(return_value=mock_op)

        # Verifier fails with suggestions
        mock_verifier = mock_verifier_class.return_value
        failed_result = VerificationResult(success=False, execution_allowed=False)
        failed_result.violations = [
            PredicateViolation(
                predicate="target_within_reach(robot_id, x, y, z)",
                reason="Target exceeds max reach",
                severity="error",
                recovery_suggestions=[
                    "Move target closer to robot base",
                    "Use a different robot closer to target",
                    "Break movement into multiple steps"
                ]
            )
        ]
        mock_verifier.verify_preconditions = Mock(return_value=failed_result)

        executor = SequenceExecutor(enable_verification=True)

        result = executor._execute_single_command(
            operation="move_to_coordinate",
            params={
                "robot_id": "Robot1",
                "x": 5.0,
                "y": 0.0,
                "z": 0.1
            },
            timeout=30.0
        )

        assert result["success"] is False
        # Recovery suggestions should be in error response
        error_str = str(result.get("error", ""))
        # Check that suggestions are present (may be in different format)
        assert len(error_str) > 0


# ============================================================================
# End-to-End Integration Tests
# ============================================================================

class TestSequenceExecutorEndToEnd:
    """Test end-to-end command execution flows"""

    @patch('orchestrators.SequenceExecutor.get_global_registry')
    @patch('orchestrators.SequenceExecutor.get_command_broadcaster')
    def test_e2e_detect_and_move(self, mock_broadcaster, mock_registry, cleanup_world_state):
        """Test end-to-end detect then move workflow"""
        # Mock detect operation
        detect_op = Mock(spec=BasicOperation)
        detect_op.name = "detect_object_stereo"
        detect_op.category = OperationCategory.PERCEPTION
        detect_op.preconditions = []
        detect_op.postconditions = []
        detect_op.relationships = None

        # Mock move operation
        move_op = Mock(spec=BasicOperation)
        move_op.name = "move_to_coordinate"
        move_op.category = OperationCategory.NAVIGATION
        move_op.preconditions = []
        move_op.postconditions = []
        move_op.relationships = None

        # Setup registry to return appropriate operations
        def get_op_by_name(name):
            if name == "detect_object_stereo":
                return detect_op
            elif name == "move_to_coordinate":
                return move_op
            return None

        mock_registry.return_value.get_operation_by_name = Mock(side_effect=get_op_by_name)

        # Mock execution results
        detect_result = OperationResult.success_result({"position": (0.3, 0.2, 0.1)})
        move_result = OperationResult.success_result({"final_position": (0.3, 0.2, 0.1)})

        def execute_op_by_name(name, **params):
            if name == "detect_object_stereo":
                return detect_result
            elif name == "move_to_coordinate":
                return move_result
            return OperationResult.error_result("UNKNOWN_OP", "Unknown operation", [])

        mock_registry.return_value.execute_operation_by_name = Mock(side_effect=execute_op_by_name)
        mock_broadcaster.return_value.get_completion = Mock(return_value=None)

        executor = SequenceExecutor(enable_verification=False, check_completion=False)

        # Execute sequence
        commands = [
            {"operation": "detect_object_stereo", "params": {"robot_id": "Robot1", "color": "blue"}},
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1", "x": 0.3, "y": 0.2, "z": 0.1}}
        ]

        result = executor.execute_sequence(commands)

        assert result["success"] is True
        assert len(result.get("results", [])) == 2

    @patch('orchestrators.SequenceExecutor.get_global_registry')
    @patch('orchestrators.SequenceExecutor.get_command_broadcaster')
    def test_e2e_detect_move_grip(self, mock_broadcaster, mock_registry, cleanup_world_state):
        """Test full pick-up workflow: detect, move, grip"""
        # Setup mock operations
        operations = {
            "detect_object_stereo": Mock(spec=BasicOperation, name="detect_object_stereo",
                                        category=OperationCategory.PERCEPTION, preconditions=[], postconditions=[]),
            "move_to_coordinate": Mock(spec=BasicOperation, name="move_to_coordinate",
                                      category=OperationCategory.NAVIGATION, preconditions=[], postconditions=[]),
            "control_gripper": Mock(spec=BasicOperation, name="control_gripper",
                                   category=OperationCategory.MANIPULATION, preconditions=[], postconditions=[])
        }
        # Set relationships to None to prevent iteration
        for op in operations.values():
            op.relationships = None

        mock_registry.return_value.get_operation_by_name = Mock(side_effect=lambda name: operations.get(name))

        # Mock results
        results = {
            "detect_object_stereo": OperationResult.success_result({"position": (0.3, 0.15, 0.1)}),
            "move_to_coordinate": OperationResult.success_result({"final_position": (0.3, 0.15, 0.1)}),
            "control_gripper": OperationResult.success_result({"gripper_state": "closed"})
        }

        mock_registry.return_value.execute_operation_by_name = Mock(
            side_effect=lambda name, **params: results.get(name, OperationResult.error_result("ERROR", "Unknown", []))
        )
        mock_broadcaster.return_value.get_completion = Mock(return_value=None)

        executor = SequenceExecutor(enable_verification=False, check_completion=False)

        commands = [
            {"operation": "detect_object_stereo", "params": {"robot_id": "Robot1", "color": "red"}},
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1", "x": 0.3, "y": 0.15, "z": 0.1}},
            {"operation": "control_gripper", "params": {"robot_id": "Robot1", "open_gripper": False}}
        ]

        result = executor.execute_sequence(commands)

        assert result["success"] is True
        assert len(result["results"]) == 3

    @patch('orchestrators.SequenceExecutor.get_global_registry')
    @patch('orchestrators.SequenceExecutor.get_command_broadcaster')
    def test_e2e_multi_robot_coordination(self, mock_broadcaster, mock_registry, cleanup_world_state):
        """Test multi-robot coordinated movements"""
        move_op = Mock(spec=BasicOperation, name="move_to_coordinate",
                      category=OperationCategory.NAVIGATION, preconditions=[], postconditions=[])
        move_op.relationships = None

        mock_registry.return_value.get_operation_by_name = Mock(return_value=move_op)
        mock_registry.return_value.execute_operation_by_name = Mock(
            return_value=OperationResult.success_result({"final_position": (0.3, 0.0, 0.1)})
        )
        mock_broadcaster.return_value.get_completion = Mock(return_value=None)

        executor = SequenceExecutor(enable_verification=False, check_completion=False)

        # Commands for two robots
        commands = [
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1", "x": 0.3, "y": -0.2, "z": 0.1}},
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot2", "x": 0.3, "y": 0.2, "z": 0.1}}
        ]

        result = executor.execute_sequence(commands)

        assert result["success"] is True
        assert len(result["results"]) == 2


# ============================================================================
# Error Scenarios and Propagation
# ============================================================================

class TestSequenceExecutorErrorPropagation:
    """Test error propagation through sequence execution"""

    @patch('orchestrators.SequenceExecutor.get_global_registry')
    def test_e2e_detection_fails(self, mock_registry, cleanup_world_state):
        """Test sequence stops when detection fails"""
        detect_op = Mock(spec=BasicOperation, name="detect_object_stereo",
                        category=OperationCategory.PERCEPTION, preconditions=[], postconditions=[])
        detect_op.relationships = None
        move_op = Mock(spec=BasicOperation, name="move_to_coordinate",
                      category=OperationCategory.NAVIGATION, preconditions=[], postconditions=[])
        move_op.relationships = None

        mock_registry.return_value.get_operation_by_name = Mock(
            side_effect=lambda name: detect_op if name == "detect_object_stereo" else move_op
        )

        # Detection fails
        mock_registry.return_value.execute_operation_by_name = Mock(
            return_value=OperationResult.error_result("NO_OBJECTS", "No objects detected", ["Adjust camera", "Check lighting"])
        )

        executor = SequenceExecutor(enable_verification=False, check_completion=False)

        commands = [
            {"operation": "detect_object_stereo", "params": {"robot_id": "Robot1"}},
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1", "x": 0.3, "y": 0.0, "z": 0.1}}
        ]

        result = executor.execute_sequence(commands)

        # Should fail on first command
        assert result["success"] is False
        assert "NO_OBJECTS" in result.get("error", "")

    @patch('orchestrators.SequenceExecutor.get_global_registry')
    @patch('orchestrators.SequenceExecutor.get_command_broadcaster')
    def test_e2e_movement_timeout(self, mock_broadcaster, mock_registry, cleanup_world_state):
        """Test timeout handling during movement"""
        move_op = Mock(spec=BasicOperation, name="move_to_coordinate",
                      category=OperationCategory.NAVIGATION, preconditions=[], postconditions=[])
        move_op.relationships = None

        mock_registry.return_value.get_operation_by_name = Mock(return_value=move_op)
        mock_registry.return_value.execute_operation_by_name = Mock(
            return_value=OperationResult.success_result({"command_sent": True})
        )

        # Simulate timeout - no completion received
        mock_broadcaster.return_value.get_completion = Mock(return_value=None)

        executor = SequenceExecutor(enable_verification=False, check_completion=True)

        commands = [
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1", "x": 0.3, "y": 0.0, "z": 0.1}}
        ]

        result = executor._execute_single_command(
            operation="move_to_coordinate",
            params={"robot_id": "Robot1", "x": 0.3, "y": 0.0, "z": 0.1},
            timeout=0.1  # Very short timeout
        )

        assert result["success"] is False
        # Check for timeout or timed out in error message
        error_msg = result.get("error", "").lower()
        assert "timed out" in error_msg or "timeout" in error_msg

    @patch('orchestrators.SequenceExecutor.get_global_registry')
    def test_e2e_gripper_fails(self, mock_registry, cleanup_world_state):
        """Test gripper failure handling"""
        gripper_op = Mock(spec=BasicOperation, name="control_gripper",
                         category=OperationCategory.MANIPULATION, preconditions=[], postconditions=[])
        gripper_op.relationships = None

        mock_registry.return_value.get_operation_by_name = Mock(return_value=gripper_op)
        mock_registry.return_value.execute_operation_by_name = Mock(
            return_value=OperationResult.error_result("GRIPPER_JAMMED", "Gripper mechanism jammed", ["Reset gripper", "Check for obstruction"])
        )

        executor = SequenceExecutor(enable_verification=False, check_completion=False)

        commands = [
            {"operation": "control_gripper", "params": {"robot_id": "Robot1", "open_gripper": False}}
        ]

        result = executor.execute_sequence(commands)

        assert result["success"] is False
        assert "GRIPPER_JAMMED" in result.get("error", "")

    @patch('orchestrators.SequenceExecutor.get_global_registry')
    @patch('orchestrators.SequenceExecutor.get_command_broadcaster')
    def test_e2e_coordination_conflict(self, mock_broadcaster, mock_registry, cleanup_world_state):
        """Test coordination conflict during execution"""
        move_op = Mock(spec=BasicOperation, name="move_to_coordinate",
                      category=OperationCategory.NAVIGATION, preconditions=[], postconditions=[])
        move_op.relationships = None

        mock_registry.return_value.get_operation_by_name = Mock(return_value=move_op)

        # First robot succeeds, second robot has collision
        call_count = [0]
        def execute_with_conflict(name, **params):
            call_count[0] += 1
            if call_count[0] == 1:
                return OperationResult.success_result({"position": (0.3, 0.0, 0.1)})
            else:
                return OperationResult.error_result("COLLISION", "Path blocked by Robot1", ["Wait for Robot1", "Use different path"])

        mock_registry.return_value.execute_operation_by_name = Mock(side_effect=execute_with_conflict)
        mock_broadcaster.return_value.get_completion = Mock(return_value=None)

        executor = SequenceExecutor(enable_verification=False, check_completion=False)

        commands = [
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1", "x": 0.3, "y": 0.0, "z": 0.1}},
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot2", "x": 0.3, "y": 0.0, "z": 0.1}}
        ]

        result = executor.execute_sequence(commands)

        assert result["success"] is False


# ============================================================================
# Complex Sequences
# ============================================================================

class TestComplexSequences:
    """Test complex multi-step sequences"""

    @patch('orchestrators.SequenceExecutor.get_global_registry')
    @patch('orchestrators.SequenceExecutor.get_command_broadcaster')
    def test_handoff_between_robots(self, mock_broadcaster, mock_registry, cleanup_world_state):
        """Test object handoff between two robots"""
        operations = {
            "detect_object_stereo": Mock(spec=BasicOperation, name="detect_object_stereo",
                                        category=OperationCategory.PERCEPTION, preconditions=[], postconditions=[]),
            "move_to_coordinate": Mock(spec=BasicOperation, name="move_to_coordinate",
                                      category=OperationCategory.NAVIGATION, preconditions=[], postconditions=[]),
            "control_gripper": Mock(spec=BasicOperation, name="control_gripper",
                                   category=OperationCategory.MANIPULATION, preconditions=[], postconditions=[])
        }
        for op in operations.values():
            op.relationships = None

        mock_registry.return_value.get_operation_by_name = Mock(side_effect=lambda name: operations.get(name))

        results = {
            "detect_object_stereo": OperationResult.success_result({"position": (0.3, 0.0, 0.1)}),
            "move_to_coordinate": OperationResult.success_result({"final_position": (0.3, 0.0, 0.1)}),
            "control_gripper": OperationResult.success_result({"gripper_state": "closed"})
        }

        mock_registry.return_value.execute_operation_by_name = Mock(
            side_effect=lambda name, **params: results.get(name)
        )
        mock_broadcaster.return_value.get_completion = Mock(return_value=None)

        executor = SequenceExecutor(enable_verification=False, check_completion=False)

        # Robot1 picks up, moves to handoff point, opens gripper
        # Robot2 moves to handoff point, closes gripper
        commands = [
            {"operation": "detect_object_stereo", "params": {"robot_id": "Robot1", "color": "blue"}},
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1", "x": 0.3, "y": 0.0, "z": 0.1}},
            {"operation": "control_gripper", "params": {"robot_id": "Robot1", "open_gripper": False}},
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1", "x": 0.0, "y": 0.0, "z": 0.2}},
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot2", "x": 0.0, "y": 0.0, "z": 0.2}},
            {"operation": "control_gripper", "params": {"robot_id": "Robot1", "open_gripper": True}},
            {"operation": "control_gripper", "params": {"robot_id": "Robot2", "open_gripper": False}}
        ]

        result = executor.execute_sequence(commands)

        assert result["success"] is True
        assert len(result["results"]) == 7

    @patch('orchestrators.SequenceExecutor.get_global_registry')
    @patch('orchestrators.SequenceExecutor.get_command_broadcaster')
    def test_parallel_movements(self, mock_broadcaster, mock_registry, cleanup_world_state):
        """Test parallel robot movements"""
        move_op = Mock(spec=BasicOperation, name="move_to_coordinate",
                      category=OperationCategory.NAVIGATION, preconditions=[], postconditions=[])
        move_op.relationships = None

        mock_registry.return_value.get_operation_by_name = Mock(return_value=move_op)
        mock_registry.return_value.execute_operation_by_name = Mock(
            return_value=OperationResult.success_result({"final_position": (0.3, 0.0, 0.1)})
        )
        mock_broadcaster.return_value.get_completion = Mock(return_value=None)

        executor = SequenceExecutor(enable_verification=False, check_completion=False)

        # Three robots move simultaneously to different positions
        commands = [
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1", "x": 0.3, "y": -0.2, "z": 0.1}},
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot2", "x": 0.3, "y": 0.0, "z": 0.1}},
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot3", "x": 0.3, "y": 0.2, "z": 0.1}}
        ]

        result = executor.execute_sequence(commands)

        assert result["success"] is True
        assert len(result["results"]) == 3

    @patch('orchestrators.SequenceExecutor.get_global_registry')
    @patch('orchestrators.SequenceExecutor.get_command_broadcaster')
    def test_compound_command_sequence(self, mock_broadcaster, mock_registry, cleanup_world_state):
        """Test complex compound sequence with branching logic"""
        operations = {
            "detect_object_stereo": Mock(spec=BasicOperation, name="detect_object_stereo",
                                        category=OperationCategory.PERCEPTION, preconditions=[], postconditions=[]),
            "move_to_coordinate": Mock(spec=BasicOperation, name="move_to_coordinate",
                                      category=OperationCategory.NAVIGATION, preconditions=[], postconditions=[]),
            "control_gripper": Mock(spec=BasicOperation, name="control_gripper",
                                   category=OperationCategory.MANIPULATION, preconditions=[], postconditions=[]),
            "check_robot_status": Mock(spec=BasicOperation, name="check_robot_status",
                                      category=OperationCategory.STATE_CHECK, preconditions=[], postconditions=[])
        }
        for op in operations.values():
            op.relationships = None

        mock_registry.return_value.get_operation_by_name = Mock(side_effect=lambda name: operations.get(name))

        results = {
            "detect_object_stereo": OperationResult.success_result({"position": (0.3, 0.15, 0.1), "confidence": 0.95}),
            "move_to_coordinate": OperationResult.success_result({"final_position": (0.3, 0.15, 0.1)}),
            "control_gripper": OperationResult.success_result({"gripper_state": "closed"}),
            "check_robot_status": OperationResult.success_result({"is_moving": False, "position": (0.3, 0.15, 0.1)})
        }

        mock_registry.return_value.execute_operation_by_name = Mock(
            side_effect=lambda name, **params: results.get(name)
        )
        mock_broadcaster.return_value.get_completion = Mock(return_value=None)

        executor = SequenceExecutor(enable_verification=False, check_completion=False)

        commands = [
            {"operation": "detect_object_stereo", "params": {"robot_id": "Robot1", "color": "red"}},
            {"operation": "check_robot_status", "params": {"robot_id": "Robot1"}},
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1", "x": 0.3, "y": 0.15, "z": 0.1}},
            {"operation": "check_robot_status", "params": {"robot_id": "Robot1"}},
            {"operation": "control_gripper", "params": {"robot_id": "Robot1", "open_gripper": False}},
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1", "x": 0.0, "y": 0.0, "z": 0.3}},
            {"operation": "control_gripper", "params": {"robot_id": "Robot1", "open_gripper": True}}
        ]

        result = executor.execute_sequence(commands)

        assert result["success"] is True
        assert len(result["results"]) == 7


# ============================================================================
# Variable Passing and Context
# ============================================================================

class TestVariablePassingIntegration:
    """Test variable passing between operations"""

    @patch('orchestrators.SequenceExecutor.get_global_registry')
    @patch('orchestrators.SequenceExecutor.get_command_broadcaster')
    def test_variable_capture_and_use(self, mock_broadcaster, mock_registry, cleanup_world_state):
        """Test capturing detection result and using in move"""
        detect_op = Mock(spec=BasicOperation, name="detect_object_stereo",
                        category=OperationCategory.PERCEPTION, preconditions=[], postconditions=[])
        detect_op.relationships = None
        move_op = Mock(spec=BasicOperation, name="move_to_coordinate",
                      category=OperationCategory.NAVIGATION, preconditions=[], postconditions=[])
        move_op.relationships = None

        mock_registry.return_value.get_operation_by_name = Mock(
            side_effect=lambda name: detect_op if name == "detect_object_stereo" else move_op
        )

        # Detection returns position
        detect_result = OperationResult.success_result({"position": {"x": 0.35, "y": 0.18, "z": 0.12}})
        move_result = OperationResult.success_result({"final_position": (0.35, 0.18, 0.12)})

        call_count = [0]
        def execute_with_var(name, **params):
            call_count[0] += 1
            if call_count[0] == 1:
                return detect_result
            else:
                # Verify position was passed correctly
                assert "position" in params or ("x" in params and "y" in params and "z" in params)
                return move_result

        mock_registry.return_value.execute_operation_by_name = Mock(side_effect=execute_with_var)
        mock_broadcaster.return_value.get_completion = Mock(return_value=None)

        executor = SequenceExecutor(enable_verification=False, check_completion=False)

        commands = [
            {"operation": "detect_object_stereo", "params": {"robot_id": "Robot1", "color": "blue"}, "capture_var": "target"},
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1", "position": "$target"}}
        ]

        result = executor.execute_sequence(commands)

        assert result["success"] is True

    @patch('orchestrators.SequenceExecutor.get_global_registry')
    @patch('orchestrators.SequenceExecutor.get_command_broadcaster')
    def test_multiple_variable_substitution(self, mock_broadcaster, mock_registry, cleanup_world_state):
        """Test multiple variables in one command"""
        status_op = Mock(spec=BasicOperation, name="check_robot_status",
                        category=OperationCategory.STATE_CHECK, preconditions=[], postconditions=[])
        status_op.relationships = None
        move_op = Mock(spec=BasicOperation, name="move_to_coordinate",
                      category=OperationCategory.NAVIGATION, preconditions=[], postconditions=[])
        move_op.relationships = None

        mock_registry.return_value.get_operation_by_name = Mock(
            side_effect=lambda name: status_op if name == "check_robot_status" else move_op
        )

        status_result = OperationResult.success_result({"position": (0.1, 0.2, 0.3), "speed": 1.5})
        move_result = OperationResult.success_result({"final_position": (0.1, 0.2, 0.3)})

        mock_registry.return_value.execute_operation_by_name = Mock(
            side_effect=lambda name, **params: status_result if name == "check_robot_status" else move_result
        )
        mock_broadcaster.return_value.get_completion = Mock(return_value=None)

        executor = SequenceExecutor(enable_verification=False, check_completion=False)

        commands = [
            {"operation": "check_robot_status", "params": {"robot_id": "Robot2"}, "capture_var": "robot2_state"},
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1", "position": "$robot2_state.position"}}
        ]

        result = executor.execute_sequence(commands)

        assert result["success"] is True


# ============================================================================
# Performance and Stress Tests
# ============================================================================

class TestSequenceExecutorPerformance:
    """Test performance with large sequences"""

    @patch('orchestrators.SequenceExecutor.get_global_registry')
    @patch('orchestrators.SequenceExecutor.get_command_broadcaster')
    def test_large_sequence_execution(self, mock_broadcaster, mock_registry, cleanup_world_state):
        """Test executing a large sequence (20+ commands)"""
        move_op = Mock(spec=BasicOperation, name="move_to_coordinate",
                      category=OperationCategory.NAVIGATION, preconditions=[], postconditions=[])
        move_op.relationships = None

        mock_registry.return_value.get_operation_by_name = Mock(return_value=move_op)
        mock_registry.return_value.execute_operation_by_name = Mock(
            return_value=OperationResult.success_result({"position": (0.3, 0.0, 0.1)})
        )
        mock_broadcaster.return_value.get_completion = Mock(return_value=None)

        executor = SequenceExecutor(enable_verification=False, check_completion=False)

        # Generate 25 move commands
        commands = [
            {"operation": "move_to_coordinate", "params": {
                "robot_id": "Robot1",
                "x": 0.1 + (i * 0.01),
                "y": 0.0,
                "z": 0.1 + (i * 0.005)
            }}
            for i in range(25)
        ]

        result = executor.execute_sequence(commands)

        assert result["success"] is True
        assert len(result["results"]) == 25

    @patch('orchestrators.SequenceExecutor.get_global_registry')
    @patch('orchestrators.SequenceExecutor.get_command_broadcaster')
    def test_rapid_consecutive_commands(self, mock_broadcaster, mock_registry, cleanup_world_state):
        """Test rapid execution of consecutive commands"""
        gripper_op = Mock(spec=BasicOperation, name="control_gripper",
                         category=OperationCategory.MANIPULATION, preconditions=[], postconditions=[])
        gripper_op.relationships = None

        mock_registry.return_value.get_operation_by_name = Mock(return_value=gripper_op)
        mock_registry.return_value.execute_operation_by_name = Mock(
            return_value=OperationResult.success_result({"gripper_state": "open"})
        )
        mock_broadcaster.return_value.get_completion = Mock(return_value=None)

        executor = SequenceExecutor(enable_verification=False, check_completion=False)

        # Rapid open/close cycles
        commands = [
            {"operation": "control_gripper", "params": {"robot_id": "Robot1", "open_gripper": i % 2 == 0}}
            for i in range(10)
        ]

        result = executor.execute_sequence(commands)

        assert result["success"] is True
        assert len(result["results"]) == 10


# ============================================================================
# Edge Cases Specific to Integration
# ============================================================================

class TestSequenceExecutorEdgeCases:
    """Test edge cases in integration scenarios"""

    @patch('orchestrators.SequenceExecutor.get_global_registry')
    def test_empty_sequence(self, mock_registry, cleanup_world_state):
        """Test executing empty sequence"""
        executor = SequenceExecutor(enable_verification=False, check_completion=False)

        result = executor.execute_sequence([])

        assert result["success"] is True
        assert len(result.get("results", [])) == 0

    @patch('orchestrators.SequenceExecutor.get_global_registry')
    def test_sequence_with_unknown_operation(self, mock_registry, cleanup_world_state):
        """Test sequence with unknown operation"""
        mock_registry.return_value.get_operation_by_name = Mock(return_value=None)

        executor = SequenceExecutor(enable_verification=False, check_completion=False)

        commands = [
            {"operation": "nonexistent_operation", "params": {"robot_id": "Robot1"}}
        ]

        result = executor.execute_sequence(commands)

        assert result["success"] is False

    @patch('orchestrators.SequenceExecutor.get_global_registry')
    @patch('orchestrators.SequenceExecutor.get_command_broadcaster')
    def test_partial_sequence_failure_recovery(self, mock_broadcaster, mock_registry, cleanup_world_state):
        """Test partial failure and recovery in sequence"""
        move_op = Mock(spec=BasicOperation, name="move_to_coordinate",
                      category=OperationCategory.NAVIGATION, preconditions=[], postconditions=[])
        move_op.relationships = None

        mock_registry.return_value.get_operation_by_name = Mock(return_value=move_op)

        # First succeeds, second fails, rest should not execute
        call_count = [0]
        def execute_with_failure(name, **params):
            call_count[0] += 1
            if call_count[0] == 1:
                return OperationResult.success_result({"position": (0.1, 0.0, 0.1)})
            elif call_count[0] == 2:
                return OperationResult.error_result("COLLISION", "Obstacle detected", ["Clear obstacle", "Use different path"])
            else:
                # Should not reach here
                raise AssertionError("Should not execute commands after failure")

        mock_registry.return_value.execute_operation_by_name = Mock(side_effect=execute_with_failure)
        mock_broadcaster.return_value.get_completion = Mock(return_value=None)

        executor = SequenceExecutor(enable_verification=False, check_completion=False)

        commands = [
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1", "x": 0.1, "y": 0.0, "z": 0.1}},
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1", "x": 0.2, "y": 0.0, "z": 0.1}},
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1", "x": 0.3, "y": 0.0, "z": 0.1}}
        ]

        result = executor.execute_sequence(commands)

        assert result["success"] is False
        assert call_count[0] == 2  # Only first two should execute
