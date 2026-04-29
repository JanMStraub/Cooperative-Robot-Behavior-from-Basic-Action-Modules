#!/usr/bin/env python3
"""
Tests for intermediate motion layer (Improvement 4).

Covers:
- _decompose_to_motions() returns list of strings from LLM
- _decompose_to_motions() returns empty list on LLM failure
- _parse_with_motion_layer() augments command with motion strings
- _parse_with_motion_layer() falls back to standard parse when decomposition empty
- parse() honours use_motion_layer=True flag
- USE_MOTION_LAYER config var is readable
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json


@pytest.fixture
def parser():
    from orchestrators.CommandParser import CommandParser

    return CommandParser(use_rag=False, use_cache=False)


def _make_session_response(content, status=200):
    """Build a minimal mock requests.Response."""
    resp = Mock()
    resp.status_code = status
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


# ============================================================================
# _decompose_to_motions()
# ============================================================================


class TestDecomposeToMotions:
    def test_returns_list_of_strings(self, parser):
        """Valid JSON array of strings is returned as-is."""
        motions = ["approach from above", "close gripper", "lift to 0.3m"]
        parser._session = Mock()
        parser._session.post.return_value = _make_session_response(json.dumps(motions))

        result = parser._decompose_to_motions("grasp the red cube", "Robot1")
        assert result == motions

    def test_returns_empty_on_http_error(self, parser):
        """Non-200 response yields empty list."""
        parser._session = Mock()
        parser._session.post.return_value = _make_session_response("", status=500)

        result = parser._decompose_to_motions("grasp the red cube", "Robot1")
        assert result == []

    def test_returns_empty_on_exception(self, parser):
        """Network exception yields empty list."""
        parser._session = Mock()
        parser._session.post.side_effect = ConnectionError("refused")

        result = parser._decompose_to_motions("grasp the red cube", "Robot1")
        assert result == []

    def test_returns_empty_on_non_list_response(self, parser):
        """If LLM returns an object instead of array, empty list is returned."""
        parser._session = Mock()
        parser._session.post.return_value = _make_session_response(
            json.dumps({"motions": []})
        )

        result = parser._decompose_to_motions("grasp the red cube", "Robot1")
        assert result == []

    def test_returns_empty_on_list_of_non_strings(self, parser):
        """List containing non-strings is rejected."""
        parser._session = Mock()
        parser._session.post.return_value = _make_session_response(
            json.dumps([1, 2, 3])
        )

        result = parser._decompose_to_motions("grasp the red cube", "Robot1")
        assert result == []


# ============================================================================
# _parse_with_motion_layer()
# ============================================================================


class TestParseWithMotionLayer:
    def _ok_llm_result(self):
        return {
            "success": True,
            "commands": [
                {"operation": "check_robot_status", "params": {"robot_id": "Robot1"}}
            ],
            "error": None,
        }

    def test_motion_strings_appear_in_llm_prompt(self, parser):
        """Stage 2 prompt (augmented_command) contains the motion strings from Stage 1."""
        motions = ["move above cube", "descend", "close gripper"]
        parser._decompose_to_motions = Mock(return_value=motions)
        # Capture the augmented_command passed to _parse_with_llm (Stage 2)
        captured = {}

        def fake_parse_with_llm(command_text, robot_id):
            captured["command_text"] = command_text
            return self._ok_llm_result()

        parser._parse_with_llm = fake_parse_with_llm

        parser._parse_with_motion_layer("grasp red cube", "Robot1")

        for m in motions:
            assert m in captured["command_text"]

    def test_falls_back_to_standard_parse_when_empty_motions(self, parser):
        """When _decompose_to_motions returns [], _parse_with_llm is called with original text."""
        parser._decompose_to_motions = Mock(return_value=[])
        parser._parse_with_llm = Mock(return_value=self._ok_llm_result())

        parser._parse_with_motion_layer("grasp red cube", "Robot1")

        parser._parse_with_llm.assert_called_once_with("grasp red cube", "Robot1")

    def test_returns_valid_commands(self, parser):
        """_parse_with_motion_layer returns commands on success."""
        motions = ["step 1", "step 2"]
        parser._decompose_to_motions = Mock(return_value=motions)
        parser._parse_with_llm = Mock(return_value=self._ok_llm_result())

        result = parser._parse_with_motion_layer("grasp red cube", "Robot1")
        assert result["success"] is True


# ============================================================================
# parse() use_motion_layer flag
# ============================================================================


class TestParseMotionLayerFlag:
    def test_parse_calls_motion_layer_when_flag_true(self, parser):
        """parse(use_motion_layer=True) calls _parse_with_motion_layer."""
        parser._parse_with_motion_layer = Mock(
            return_value={"success": True, "commands": [], "error": None}
        )
        parser._parse_with_llm = Mock()

        parser.parse("move to (0,0,0)", use_motion_layer=True)

        parser._parse_with_motion_layer.assert_called_once()
        parser._parse_with_llm.assert_not_called()

    def test_parse_skips_motion_layer_when_flag_false(self, parser):
        """parse(use_motion_layer=False) calls _parse_with_llm directly."""
        parser._parse_with_llm = Mock(
            return_value={"success": True, "commands": [], "error": None}
        )
        parser._parse_with_motion_layer = Mock()

        parser.parse("move to (0,0,0)", use_motion_layer=False)

        parser._parse_with_llm.assert_called_once()
        parser._parse_with_motion_layer.assert_not_called()

    def test_parse_defaults_to_config_value(self, parser):
        """parse() without use_motion_layer flag uses USE_MOTION_LAYER config."""
        from config.Servers import USE_MOTION_LAYER

        parser._parse_with_motion_layer = Mock(
            return_value={"success": True, "commands": [], "error": None}
        )
        parser._parse_with_llm = Mock(
            return_value={"success": True, "commands": [], "error": None}
        )

        parser.parse("move to (0,0,0)")

        if USE_MOTION_LAYER:
            parser._parse_with_motion_layer.assert_called_once()
        else:
            parser._parse_with_llm.assert_called_once()


# ============================================================================
# Config
# ============================================================================


class TestMotionLayerConfig:
    def test_use_motion_layer_is_bool(self):
        from config.Servers import USE_MOTION_LAYER

        assert isinstance(USE_MOTION_LAYER, bool)

    def test_use_motion_layer_default_true(self, monkeypatch):
        """Default value is True (enabled)."""
        monkeypatch.delenv("PARSER_USE_MOTION_LAYER", raising=False)
        import config.Servers as srv
        import importlib

        importlib.reload(srv)
        assert srv.USE_MOTION_LAYER is True

    def test_use_motion_layer_env_true(self, monkeypatch):
        """PARSER_USE_MOTION_LAYER=true enables the flag."""
        monkeypatch.setenv("PARSER_USE_MOTION_LAYER", "true")
        import config.Servers as srv
        import importlib

        importlib.reload(srv)
        assert srv.USE_MOTION_LAYER is True


# ============================================================================
# parse_with_hint() motion layer integration
# ============================================================================


class TestParseWithHintMotionLayer:
    def _stub_parser(self, parser, motions):
        """Wire up common mocks for parse_with_hint tests."""
        parser._decompose_to_motions = Mock(return_value=motions)
        parser._do_llm_request = Mock(
            return_value={
                "success": True,
                "parsed": {"commands": [{"operation": "control_gripper", "params": {}}]},
            }
        )
        parser._validate_commands = Mock(
            return_value=[{"operation": "control_gripper", "params": {}}]
        )
        parser._get_spatial_context = Mock(return_value="")
        parser._prompt_builder = Mock()
        parser._prompt_builder.build.return_value = "mocked prompt"

    def test_parse_with_hint_uses_motion_layer_when_flag_true(self, parser):
        """parse_with_hint(use_motion_layer=True) runs Stage 1 before the hint parse."""
        motions = ["approach from above", "close gripper"]
        self._stub_parser(parser, motions)

        result = parser.parse_with_hint(
            "grasp the red cube", robot_id="Robot1", hint="prev error", use_motion_layer=True
        )

        parser._decompose_to_motions.assert_called_once_with("grasp the red cube", "Robot1")
        prompt_call_arg = parser._prompt_builder.build.call_args[0][0]
        assert "approach from above" in prompt_call_arg
        assert result["success"] is True

    def test_parse_with_hint_skips_motion_layer_when_flag_false(self, parser):
        """parse_with_hint(use_motion_layer=False) does not call _decompose_to_motions."""
        self._stub_parser(parser, ["some motion"])

        parser.parse_with_hint(
            "grasp the red cube", robot_id="Robot1", hint="err", use_motion_layer=False
        )

        parser._decompose_to_motions.assert_not_called()

    def test_parse_with_hint_falls_back_when_decompose_empty(self, parser):
        """parse_with_hint with motion layer falls back to original command when Stage 1 empty."""
        self._stub_parser(parser, [])

        parser.parse_with_hint(
            "grasp the red cube", robot_id="Robot1", hint="err", use_motion_layer=True
        )

        prompt_call_arg = parser._prompt_builder.build.call_args[0][0]
        assert "Motion plan" not in prompt_call_arg
        assert "grasp the red cube" in prompt_call_arg
