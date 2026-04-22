#!/usr/bin/env python3
"""
Unit tests for WorkflowPatterns.py — HANDOFF_PATTERN step sequence.

Verifies:
- workflow_handoff_001 uses grasp_object for source robot
- workflow_handoff_001 uses orient_gripper_for_handoff_receive (not receive_handoff)
- return_to_start_position precedes move to presentation position
- signal/wait pair is present
- Step ordering: detect → grasp → return_start → move_present → signal/wait →
                 detect → orient → move_approach → close_gripper → release
"""

from operations.WorkflowPatterns import HANDOFF_PATTERN

ORIENT_OP_ID = "coordination_orient_for_handoff_receive_001"
GRASP_OP_ID = "manipulation_grasp_object_001"
RETURN_START_OP_ID = "motion_return_to_start_001"
MOVE_OP_ID = "motion_move_to_coord_001"
SIGNAL_OP_ID = "sync_signal_001"
WAIT_OP_ID = "sync_wait_for_signal_001"
DETECT_OP_ID = "perception_stereo_detect_001"
CONTROL_GRIPPER_OP_ID = "manipulation_control_gripper_001"
RELEASE_OP_ID = "manipulation_release_object_001"


class TestHandoffPatternSteps:
    """Verify the HANDOFF_PATTERN step IDs and parameter values."""

    def _step_ids(self):
        """Return list of operation_id strings from the pattern's steps."""
        return [step.operation_id for step in HANDOFF_PATTERN.steps]

    def _steps_of(self, op_id):
        """Return all steps with the given operation_id."""
        return [s for s in HANDOFF_PATTERN.steps if s.operation_id == op_id]

    def test_pattern_id(self):
        """Pattern should have the canonical ID."""
        assert HANDOFF_PATTERN.pattern_id == "workflow_handoff_001"

    def test_no_receive_handoff(self):
        """receive_handoff operation must not appear — it was deleted."""
        assert "coordination_receive_handoff_001" not in self._step_ids()

    def test_no_present_for_handoff(self):
        """present_for_handoff operation must not appear — it was deleted."""
        assert "coordination_present_for_handoff_001" not in self._step_ids()

    def test_grasp_step_present(self):
        """grasp_object must be used for source robot."""
        assert GRASP_OP_ID in self._step_ids()

    def test_return_to_start_step_present(self):
        """return_to_start_position must appear for deterministic joint config."""
        assert RETURN_START_OP_ID in self._step_ids()

    def test_orient_step_present(self):
        """orient_gripper_for_handoff_receive must be used for target robot."""
        assert ORIENT_OP_ID in self._step_ids()

    def test_orient_has_required_bindings(self):
        """orient step must pass robot_id, object_id, source_robot_id."""
        orient_step = self._steps_of(ORIENT_OP_ID)[0]
        for key in ("robot_id", "object_id", "source_robot_id"):
            assert key in orient_step.parameter_bindings

    def test_detection_steps_present(self):
        """At least one stereo detection step must be present."""
        assert DETECT_OP_ID in self._step_ids()

    def test_adjust_orientation_step_present(self):
        """adjust_end_effector_orientation must appear to lock wrist after move_to_presentation."""
        ids = self._step_ids()
        assert "motion_adjust_orientation_003" in ids

    def test_adjust_orientation_before_signal(self):
        """Wrist lock must precede the r1_at_handoff signal."""
        ids = self._step_ids()
        orient_lock_idx = ids.index("motion_adjust_orientation_003")
        signal_idx = next(
            i for i, s in enumerate(HANDOFF_PATTERN.steps)
            if s.operation_id == SIGNAL_OP_ID
            and s.parameter_bindings.get("event_name") == "r1_at_handoff"
        )
        assert orient_lock_idx < signal_idx

    def test_signal_and_wait_present(self):
        """signal + wait_for_signal must both appear."""
        ids = self._step_ids()
        assert SIGNAL_OP_ID in ids
        assert WAIT_OP_ID in ids

    def test_close_gripper_step_present(self):
        """control_gripper (close) must appear for target robot."""
        close_steps = [
            s for s in self._steps_of(CONTROL_GRIPPER_OP_ID)
            if str(s.parameter_bindings.get("open_gripper", "")).lower() in ("false", "0", "")
        ]
        assert len(close_steps) >= 1, "No close-gripper step found"

    def test_release_step_present(self):
        """release_object must appear for source robot."""
        assert RELEASE_OP_ID in self._step_ids()

    def test_step_order_grasp_before_return_start(self):
        """grasp_object must precede return_to_start_position."""
        ids = self._step_ids()
        assert ids.index(GRASP_OP_ID) < ids.index(RETURN_START_OP_ID)

    def test_step_order_return_start_before_move_present(self):
        """return_to_start must precede the move to presentation position."""
        ids = self._step_ids()
        first_move_idx = ids.index(MOVE_OP_ID)
        assert ids.index(RETURN_START_OP_ID) < first_move_idx

    def test_step_order_orient_before_close_gripper(self):
        """orient_gripper_for_handoff_receive must precede close-gripper."""
        ids = self._step_ids()
        close_idx = next(
            i for i, s in enumerate(HANDOFF_PATTERN.steps)
            if s.operation_id == CONTROL_GRIPPER_OP_ID
        )
        assert ids.index(ORIENT_OP_ID) < close_idx

    def test_step_order_close_before_release(self):
        """Target gripper close must precede source robot release."""
        ids = self._step_ids()
        close_idx = next(
            i for i, s in enumerate(HANDOFF_PATTERN.steps)
            if s.operation_id == CONTROL_GRIPPER_OP_ID
        )
        assert close_idx < ids.index(RELEASE_OP_ID)

    def test_orient_approach_position_binding(self):
        """Move step after orient must reference approach_position from orient result."""
        ids = self._step_ids()
        orient_idx = ids.index(ORIENT_OP_ID)
        # Find the next move_to_coord after orient
        post_orient_steps = HANDOFF_PATTERN.steps[orient_idx + 1:]
        approach_move = next(
            (s for s in post_orient_steps if s.operation_id == MOVE_OP_ID), None
        )
        assert approach_move is not None, "No move_to_coord step after orient"
        x_binding = approach_move.parameter_bindings.get("x", "")
        assert "approach_position" in str(x_binding), (
            f"move step after orient should reference approach_position, got: {x_binding}"
        )
