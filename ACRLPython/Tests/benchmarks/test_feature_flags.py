"""Tests for BenchmarkFeatureFlags serialisation and runner message building."""
from __future__ import annotations

import json
import struct

import pytest

from benchmarks.feature_flags import BenchmarkFeatureFlags


def test_default_flags_are_all_none():
    f = BenchmarkFeatureFlags()
    assert f.use_rag is None
    assert f.use_vgn is None
    assert f.use_ros is None
    assert f.use_negotiation is None
    assert f.use_reflexion is None


def test_to_json_excludes_none_fields():
    f = BenchmarkFeatureFlags(use_rag=False, use_vgn=True)
    d = json.loads(f.to_json())
    assert d == {"use_rag": False, "use_vgn": True}


def test_to_json_empty_when_all_none():
    f = BenchmarkFeatureFlags()
    assert f.to_json() == ""


def test_from_json_round_trips():
    f = BenchmarkFeatureFlags(use_rag=False, use_ros=True, use_negotiation=False)
    restored = BenchmarkFeatureFlags.from_json(f.to_json())
    assert restored.use_rag is False
    assert restored.use_ros is True
    assert restored.use_negotiation is False
    assert restored.use_vgn is None


def test_from_json_empty_string_returns_defaults():
    f = BenchmarkFeatureFlags.from_json("")
    assert f.use_rag is None
    assert f.use_vgn is None


def test_from_config_snapshot():
    snapshot = {
        "use_rag": False,
        "use_vgn": False,
        "use_ros_movement": False,
        "use_knowledge_graph": True,
        "reflexion_enabled": True,
    }
    f = BenchmarkFeatureFlags.from_config_snapshot(snapshot)
    assert f.use_rag is False
    assert f.use_vgn is False
    assert f.use_ros is False
    assert f.use_negotiation is None
    assert f.use_reflexion is True


def _parse_flags_from_message(msg: bytes) -> str:
    """Extract flags_json from a raw SEQUENCE_QUERY message."""
    offset = 5  # header: type(1) + request_id(4)
    for _ in range(3):  # command, robot_id, camera_id
        field_len = struct.unpack_from("<I", msg, offset)[0]
        offset += 4 + field_len
    offset += 1  # auto_execute byte
    flags_len = struct.unpack_from("<I", msg, offset)[0]
    offset += 4
    return msg[offset: offset + flags_len].decode()


def test_send_appends_flags_to_message():
    from benchmarks.runner import BenchmarkRunner
    runner = BenchmarkRunner()
    flags = BenchmarkFeatureFlags(use_vgn=False, use_ros=True)
    msg = runner._build_sequence_message("test cmd", "Robot1", flags)
    flags_json = _parse_flags_from_message(msg)
    parsed = BenchmarkFeatureFlags.from_json(flags_json)
    assert parsed.use_vgn is False
    assert parsed.use_ros is True
    assert parsed.use_rag is None


def test_send_empty_flags_when_all_none():
    from benchmarks.runner import BenchmarkRunner
    runner = BenchmarkRunner()
    msg = runner._build_sequence_message("test cmd", "Robot1", BenchmarkFeatureFlags())
    flags_json = _parse_flags_from_message(msg)
    assert flags_json == ""
