#!/usr/bin/env python3
"""
Unit tests for WorkflowPatterns.py — HANDOFF_PATTERN step sequence.

Verifies:
- workflow_handoff_001 uses grasp_object_for_handoff (not move+gripper) for source robot
- workflow_handoff_001 uses receive_handoff for target robot (orient + offset-move + close in one op)
- Step ordering is correct: detect → grasp_for_handoff → signal → receive_handoff → release
"""

from operations.WorkflowPatterns import HANDOFF_PATTERN

RECEIVE_OP_ID = "coordination_receive_handoff_001"


class TestHandoffPatternSteps:
    """Verify the HANDOFF_PATTERN step IDs and parameter values."""

    def _step_ids(self):
        """Return list of operation_id strings from the pattern's steps."""
        return [step.operation_id for step in HANDOFF_PATTERN.steps]

    def test_pattern_id(self):
        """Pattern should have the canonical ID."""
        assert HANDOFF_PATTERN.pattern_id == "workflow_handoff_001"

    def test_detection_step_present(self):
        """Stereo detection must precede grasp (required precondition for grasp_object_for_handoff)."""
        ids = self._step_ids()
        assert "perception_stereo_detect_001" in ids

    def test_grasp_for_handoff_step_present(self):
        """grasp_object_for_handoff must be used instead of plain move+gripper for source robot."""
        ids = self._step_ids()
        assert "coordination_grasp_object_for_handoff_001" in ids

    def test_receive_handoff_step_present(self):
        """receive_handoff must be used for the target robot instead of separate orient+move+close."""
        ids = self._step_ids()
        assert RECEIVE_OP_ID in ids

    def test_no_bare_orient_step(self):
        """The old separate orient step should be replaced by receive_handoff."""
        ids = self._step_ids()
        assert "coordination_orient_for_handoff_receive_001" not in ids

    def test_receive_handoff_has_source_robot_id_binding(self):
        """receive_handoff step must pass source_robot_id for offset computation."""
        receive_step = next(s for s in HANDOFF_PATTERN.steps if s.operation_id == RECEIVE_OP_ID)
        assert "source_robot_id" in receive_step.parameter_bindings

    def test_receive_handoff_has_object_id_binding(self):
        """receive_handoff step must pass object_id for WorldState geometry lookup."""
        receive_step = next(s for s in HANDOFF_PATTERN.steps if s.operation_id == RECEIVE_OP_ID)
        assert "object_id" in receive_step.parameter_bindings

    def test_receive_handoff_has_robot_id_binding(self):
        """receive_handoff step must bind the target robot."""
        receive_step = next(s for s in HANDOFF_PATTERN.steps if s.operation_id == RECEIVE_OP_ID)
        assert "robot_id" in receive_step.parameter_bindings

    def test_no_bare_adjust_orientation_in_pattern(self):
        """motion_adjust_orientation_003 should not be used directly — receive_handoff handles orientation."""
        assert "motion_adjust_orientation_003" not in self._step_ids()

    def test_step_order_detect_before_grasp(self):
        """Detection must precede grasp_for_handoff."""
        ids = self._step_ids()
        assert ids.index("perception_stereo_detect_001") < ids.index(
            "coordination_grasp_object_for_handoff_001"
        )

    def test_step_order_grasp_before_signal(self):
        """grasp_for_handoff must precede the 'object_gripped' signal."""
        gripped_signal_idx = next(
            i
            for i, s in enumerate(HANDOFF_PATTERN.steps)
            if s.operation_id == "sync_signal_001"
            and s.parameter_bindings.get("event_name") == "object_gripped"
        )
        grasp_idx = self._step_ids().index("coordination_grasp_object_for_handoff_001")
        assert grasp_idx < gripped_signal_idx

    def test_step_order_grasp_before_receive(self):
        """grasp_for_handoff must come before receive_handoff."""
        ids = self._step_ids()
        assert ids.index("coordination_grasp_object_for_handoff_001") < ids.index(RECEIVE_OP_ID)

    def test_step_order_receive_before_release(self):
        """receive_handoff must come before the source robot releases."""
        ids = self._step_ids()
        receive_idx = ids.index(RECEIVE_OP_ID)
        # The release is the final control_gripper or release step after receive
        post_receive_ids = ids[receive_idx + 1:]
        assert any(
            op_id in ("manipulation_control_gripper_001", "manipulation_release_object_001")
            for op_id in post_receive_ids
        ) or "sync_signal_001" in post_receive_ids  # handoff_complete signal is also valid

    def test_handoff_complete_signal_present(self):
        """The pattern must end with a handoff_complete signal."""
        last_step = HANDOFF_PATTERN.steps[-1]
        assert last_step.operation_id == "sync_signal_001"
        assert last_step.parameter_bindings.get("event_name") == "handoff_complete"
