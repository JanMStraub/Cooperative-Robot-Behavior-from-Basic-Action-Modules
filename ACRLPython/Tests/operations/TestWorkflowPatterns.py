#!/usr/bin/env python3
"""
Unit tests for WorkflowPatterns.py — HANDOFF_PATTERN step sequence.

Verifies:
- workflow_handoff_001 uses grasp_object_for_handoff (not move+gripper) for source robot
- workflow_handoff_001 includes an orientation step (pitch=90°) for the target robot
- Step ordering is correct: detect → grasp_for_handoff → signal → orientation → move → exchange
"""

import pytest

from operations.WorkflowPatterns import HANDOFF_PATTERN


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

        There should be no step that closes the gripper (open_gripper=False) via
        manipulation_control_gripper_001 BEFORE the handoff exchange section.
        The only close-gripper step should be in the exchange section (after the signal step).
        """
        exchange_signal_idx = next(
            i
            for i, s in enumerate(HANDOFF_PATTERN.steps)
            if s.operation_id == "sync_signal_001"
            and s.parameter_bindings.get("event_name") == "both_at_handoff"
        )
        pre_exchange_steps = HANDOFF_PATTERN.steps[:exchange_signal_idx]
        pre_exchange_ids = [s.operation_id for s in pre_exchange_steps]
        # manipulation_control_gripper_001 must NOT appear before the exchange signal
        assert "manipulation_control_gripper_001" not in pre_exchange_ids

    def test_orientation_step_present(self):
        """Target robot must get a wrist orientation adjustment before approaching handoff."""
        ids = self._step_ids()
        assert "motion_adjust_orientation_003" in ids

    def test_orientation_step_pitch_value(self):
        """Orientation step must use pitch=90.0 (upward-facing gripper for bottom-approach)."""
        orient_step = next(
            s for s in HANDOFF_PATTERN.steps if s.operation_id == "motion_adjust_orientation_003"
        )
        assert orient_step.parameter_bindings.get("pitch") == 90.0

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
        """Orientation adjustment must come before the target robot's move to handoff position."""
        orient_idx = self._step_ids().index("motion_adjust_orientation_003")
        # The target move step follows orientation — check there is a move_to_coord step after it
        post_orient_ids = self._step_ids()[orient_idx + 1 :]
        assert "motion_move_to_coord_001" in post_orient_ids

    def test_handoff_complete_signal_present(self):
        """The pattern must end with a handoff_complete signal."""
        last_step = HANDOFF_PATTERN.steps[-1]
        assert last_step.operation_id == "sync_signal_001"
        assert last_step.parameter_bindings.get("event_name") == "handoff_complete"
