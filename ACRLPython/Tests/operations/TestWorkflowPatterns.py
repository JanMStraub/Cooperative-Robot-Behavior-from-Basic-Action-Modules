#!/usr/bin/env python3
"""Unit tests for WorkflowPatterns.py — HANDOFF_PATTERN step sequence"""

from operations.WorkflowPatterns import HANDOFF_PATTERN

RECEIVE_HANDOFF_OP_ID = "coordination_receive_handoff_001"
GRASP_OP_ID = "manipulation_grasp_object_001"
RETURN_START_OP_ID = "motion_return_to_start_001"
MOVE_OP_ID = "motion_move_to_coord_001"
SIGNAL_OP_ID = "sync_signal_001"
WAIT_OP_ID = "sync_wait_for_signal_001"
DETECT_OP_ID = "perception_stereo_detect_001"
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

    def test_no_orient_gripper_for_handoff_receive(self):
        """orient_gripper_for_handoff_receive operation must not appear — it was removed."""
        assert "coordination_orient_for_handoff_receive_001" not in self._step_ids()

    def test_no_present_for_handoff(self):
        """present_for_handoff operation must not appear — it was deleted."""
        assert "coordination_present_for_handoff_001" not in self._step_ids()

    def test_grasp_step_present(self):
        """grasp_object must be used for source robot."""
        assert GRASP_OP_ID in self._step_ids()

    def test_return_to_start_step_present(self):
        """return_to_start_position must appear for deterministic joint config."""
        assert RETURN_START_OP_ID in self._step_ids()

    def test_receive_handoff_step_present(self):
        """receive_handoff must be used for target robot."""
        assert RECEIVE_HANDOFF_OP_ID in self._step_ids()

    def test_receive_handoff_has_required_bindings(self):
        """receive_handoff step must pass robot_id, object_id, source_robot_id."""
        receive_step = self._steps_of(RECEIVE_HANDOFF_OP_ID)[0]
        for key in ("robot_id", "object_id", "source_robot_id"):
            assert key in receive_step.parameter_bindings

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
            i
            for i, s in enumerate(HANDOFF_PATTERN.steps)
            if s.operation_id == SIGNAL_OP_ID
            and s.parameter_bindings.get("event_name") == "r1_at_handoff"
        )
        assert orient_lock_idx < signal_idx

    def test_signal_and_wait_present(self):
        """signal + wait_for_signal must both appear."""
        ids = self._step_ids()
        assert SIGNAL_OP_ID in ids
        assert WAIT_OP_ID in ids

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

    def test_step_order_receive_handoff_before_release(self):
        """receive_handoff must precede source robot release."""
        ids = self._step_ids()
        assert ids.index(RECEIVE_HANDOFF_OP_ID) < ids.index(RELEASE_OP_ID)
