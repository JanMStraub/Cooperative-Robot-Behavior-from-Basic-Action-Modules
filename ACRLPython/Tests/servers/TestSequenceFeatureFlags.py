from __future__ import annotations

import struct

import config.Servers as srv_cfg
import config.ROS as ros_cfg
import config.Negotiation as neg_cfg
from servers.FeatureFlagContext import FeatureFlagContext
from benchmarks.FeatureFlags import BenchmarkFeatureFlags


def test_no_flags_changes_nothing():
    original_vgn = srv_cfg.VGN_ENABLED
    with FeatureFlagContext(BenchmarkFeatureFlags()):
        assert srv_cfg.VGN_ENABLED == original_vgn
    assert srv_cfg.VGN_ENABLED == original_vgn


def test_vgn_flag_applied_and_restored():
    original = srv_cfg.VGN_ENABLED
    flags = BenchmarkFeatureFlags(use_vgn=not original)
    with FeatureFlagContext(flags):
        assert srv_cfg.VGN_ENABLED == (not original)
    assert srv_cfg.VGN_ENABLED == original


def test_ros_flag_false_applied_and_restored():
    original_enabled = ros_cfg.ROS_ENABLED
    original_mode = ros_cfg.DEFAULT_CONTROL_MODE
    flags = BenchmarkFeatureFlags(use_ros=False)
    with FeatureFlagContext(flags):
        assert ros_cfg.ROS_ENABLED is False
        assert ros_cfg.DEFAULT_CONTROL_MODE == "unity"
    assert ros_cfg.ROS_ENABLED == original_enabled
    assert ros_cfg.DEFAULT_CONTROL_MODE == original_mode


def test_ros_true_sets_ros_mode():
    original_mode = ros_cfg.DEFAULT_CONTROL_MODE
    flags = BenchmarkFeatureFlags(use_ros=True)
    with FeatureFlagContext(flags):
        assert ros_cfg.ROS_ENABLED is True
        assert ros_cfg.DEFAULT_CONTROL_MODE == "ros"
    ros_cfg.DEFAULT_CONTROL_MODE = original_mode


def test_negotiation_flag_applied_and_restored():
    original = neg_cfg.NEGOTIATION_ENABLED
    flags = BenchmarkFeatureFlags(use_negotiation=not original)
    with FeatureFlagContext(flags):
        assert neg_cfg.NEGOTIATION_ENABLED == (not original)
    assert neg_cfg.NEGOTIATION_ENABLED == original


def test_reflection_patches_sequenceexecutor_module():
    import orchestrators.SequenceExecutor as seq_mod

    original = seq_mod.REFLECTION_ENABLED
    flags = BenchmarkFeatureFlags(use_reflection=not original)
    with FeatureFlagContext(flags):
        assert seq_mod.REFLECTION_ENABLED == (not original)
    assert seq_mod.REFLECTION_ENABLED == original


def test_restore_on_exception():
    original_vgn = srv_cfg.VGN_ENABLED
    try:
        with FeatureFlagContext(BenchmarkFeatureFlags(use_vgn=not original_vgn)):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert srv_cfg.VGN_ENABLED == original_vgn


def _make_sequence_msg(command: str, robot_id: str, flags_json: str = "") -> bytes:
    SEQUENCE_QUERY = 0x08
    request_id = 42
    cam = "TableStereoCamera"
    msg = struct.pack("<BI", SEQUENCE_QUERY, request_id)
    for s in (command, robot_id, cam):
        b = s.encode()
        msg += struct.pack("<I", len(b)) + b
    msg += struct.pack("<B", 1)  # auto_execute
    flags_b = flags_json.encode() if flags_json else b""
    msg += struct.pack("<I", len(flags_b)) + flags_b
    return msg


def test_flags_field_parsed_from_message():
    msg = _make_sequence_msg("Robot Robot1: wait 100ms", "Robot1", '{"use_vgn":false}')
    msg_no_flags = _make_sequence_msg("Robot Robot1: wait 100ms", "Robot1")
    assert len(msg) > len(msg_no_flags)


def test_no_flags_field_backward_compatible():
    msg_no_flags = _make_sequence_msg("Robot Robot1: wait 100ms", "Robot1")
    # auto_execute=True is the last byte before flags_len field
    assert msg_no_flags[-5:-4] == b"\x01"  # auto_execute byte before flags_len(4)
