#!/usr/bin/env python3
"""
Unit tests for WorkflowPatterns.py — HANDOFF_PATTERN step sequence.

Verifies:
- workflow_handoff_001 uses grasp_object_for_handoff (not move+gripper) for source robot
- workflow_handoff_001 uses orient_gripper_for_handoff_receive (pitch=90° + axis yaw) for target robot
- Step ordering is correct: detect → grasp_for_handoff → signal → orient_receive → move → exchange
"""

import pytest

from operations.WorkflowPatterns import HANDOFF_PATTERN

ORIENT_OP_ID = "coordination_orient_for_handoff_receive_001"


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

    def test_no_raw_grasp_for_source(self):
        """The old naive pattern (just control_gripper without grasp_for_handoff) should not be used.

        There should be no step that closes the gripper via manipulation_control_gripper_001
        BEFORE the handoff exchange section — the exchange starts at the both_at_handoff signal.
        """
        exchange_signal_idx = next(
            i
            for i, s in enumerate(HANDOFF_PATTERN.steps)
            if s.operation_id == "sync_signal_001"
            and s.parameter_bindings.get("event_name") == "both_at_handoff"
        )
        pre_exchange_ids = [s.operation_id for s in HANDOFF_PATTERN.steps[:exchange_signal_idx]]
        assert "manipulation_control_gripper_001" not in pre_exchange_ids

    def test_orient_for_handoff_receive_step_present(self):
        """Target robot must use orient_gripper_for_handoff_receive (not bare adjust_orientation)."""
        ids = self._step_ids()
        assert ORIENT_OP_ID in ids

    def test_orient_step_has_source_robot_id_binding(self):
        """Orient step must pass source_robot_id so yaw can be computed from object geometry."""
        orient_step = next(s for s in HANDOFF_PATTERN.steps if s.operation_id == ORIENT_OP_ID)
        assert "source_robot_id" in orient_step.parameter_bindings

    def test_orient_step_has_object_id_binding(self):
        """Orient step must pass object_id so WorldState geometry can be looked up."""
        orient_step = next(s for s in HANDOFF_PATTERN.steps if s.operation_id == ORIENT_OP_ID)
        assert "object_id" in orient_step.parameter_bindings

    def test_no_bare_adjust_orientation_in_pattern(self):
        """motion_adjust_orientation_003 should not be used directly — the dedicated
        orient_gripper_for_handoff_receive operation handles both pitch and yaw."""
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

    def test_step_order_orientation_before_target_move(self):
        """orient_gripper_for_handoff_receive must come before the target robot's move to handoff."""
        ids = self._step_ids()
        orient_idx = ids.index(ORIENT_OP_ID)
        post_orient_ids = ids[orient_idx + 1:]
        assert "motion_move_to_coord_001" in post_orient_ids

    def test_handoff_complete_signal_present(self):
        """The pattern must end with a handoff_complete signal."""
        last_step = HANDOFF_PATTERN.steps[-1]
        assert last_step.operation_id == "sync_signal_001"
        assert last_step.parameter_bindings.get("event_name") == "handoff_complete"
