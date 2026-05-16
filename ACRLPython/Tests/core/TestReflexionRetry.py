#!/usr/bin/env python3
"""Tests for Reflexion retry loop (Improvement 1)"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call

# _PromptBuilder hint injection


class TestPromptBuilderHint:
    @pytest.fixture
    def builder(self):
        from orchestrators.CommandParser import _PromptBuilder

        registry = Mock()
        registry.get_all_operations.return_value = []
        wf_registry = Mock()
        wf_registry.get_pattern_by_name.return_value = None
        return _PromptBuilder(registry, wf_registry, rag=None)

    def test_hint_absent_when_empty(self, builder):
        prompt = builder.build("move to (0,0,0)", "Robot1", hint="")
        assert "REFLECTION" not in prompt

    def test_hint_present_when_set(self, builder):
        prompt = builder.build(
            "move to (0,0,0)", "Robot1", hint="Previous failed: bad coords"
        )
        assert "=== REFLECTION ===" in prompt
        assert "Previous failed: bad coords" in prompt

    def test_hint_does_not_affect_other_sections(self, builder):
        prompt_no_hint = builder.build("move to (0,0,0)", "Robot1")
        prompt_with_hint = builder.build("move to (0,0,0)", "Robot1", hint="some hint")
        # Core sections still present in both
        assert "=== HANDOFF RULE ===" in prompt_no_hint
        assert "=== HANDOFF RULE ===" in prompt_with_hint


# CommandParser.parse_with_hint()


class TestParseWithHint:
    @pytest.fixture
    def parser(self):
        from orchestrators.CommandParser import CommandParser

        return CommandParser(use_rag=False, use_cache=False)

    def _mock_llm_response(self, parser, commands):
        """Patch _do_llm_request to return given commands."""
        parsed = {"commands": commands}
        parser._do_llm_request = Mock(
            return_value={
                "success": True,
                "parsed": parsed,
                "content": "{}",
                "error": None,
            }
        )

    def test_parse_with_hint_returns_commands(self, parser):
        """parse_with_hint succeeds when LLM returns valid commands."""
        cmds = [
            {
                "operation": "move_to_coordinate",
                "params": {"robot_id": "Robot1", "x": 0.1, "y": 0.2, "z": 0.3},
            }
        ]
        self._mock_llm_response(parser, cmds)
        result = parser.parse_with_hint(
            "move to (0.1, 0.2, 0.3)", robot_id="Robot1", hint="retry hint"
        )
        assert result["success"] is True
        assert len(result["commands"]) == 1

    def test_parse_with_hint_injects_hint_into_prompt(self, parser):
        """Hint text appears in the prompt passed to _do_llm_request."""
        cmds = [{"operation": "check_robot_status", "params": {"robot_id": "Robot1"}}]
        self._mock_llm_response(parser, cmds)
        parser.parse_with_hint("check status", robot_id="Robot1", hint="error: timeout")
        prompt_arg = parser._do_llm_request.call_args[0][0]
        assert "error: timeout" in prompt_arg

    def test_parse_with_hint_failure_returns_error(self, parser):
        """parse_with_hint returns failure dict when LLM call fails."""
        parser._do_llm_request = Mock(side_effect=Exception("connection refused"))
        result = parser.parse_with_hint("some command", hint="irrelevant")
        assert result["success"] is False
        assert "error" in result


# SequenceExecutor Reflexion retry loop


class TestSequenceExecutorReflexion:
    @pytest.fixture
    def executor(self):
        from orchestrators.SequenceExecutor import SequenceExecutor

        ex = SequenceExecutor()
        ex.default_timeout = 5.0
        return ex

    def _make_cmd(
        self,
        operation="move_to_coordinate",
        robot_id="Robot1",
        original_text="move to (0,0,0)",
    ):
        return {
            "operation": operation,
            "params": {"robot_id": robot_id},
            "_original_text": original_text,
        }

    def test_reflexion_retries_on_failure_and_succeeds(self, executor):
        """When command fails once, executor retries and records success."""
        fail_result = {
            "success": False,
            "error": "IK solver failed",
            "recovery_suggestions": ["try different approach"],
        }
        ok_result = {"success": True, "result": {"status": "done"}}

        executor._execute_single_command = Mock(side_effect=[fail_result, ok_result])

        retry_parse = {
            "success": True,
            "commands": [
                {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1"}}
            ],
        }

        with patch("orchestrators.CommandParser.get_command_parser") as mock_get_parser:
            mock_parser = Mock()
            mock_parser.parse_with_hint.return_value = retry_parse
            mock_get_parser.return_value = mock_parser

            result = executor.execute_sequence([self._make_cmd()])

        assert result["success"] is True
        assert result["completed_commands"] == 1
        assert result["results"][0]["success"] is True

    def test_reflexion_stops_after_max_retries(self, executor):
        """When all retries fail, sequence reports failure."""
        always_fail = {
            "success": False,
            "error": "persistent error",
            "recovery_suggestions": [],
        }
        executor._execute_single_command = Mock(return_value=always_fail)

        retry_parse = {
            "success": True,
            "commands": [
                {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1"}}
            ],
        }

        with patch("orchestrators.CommandParser.get_command_parser") as mock_get_parser:
            mock_parser = Mock()
            mock_parser.parse_with_hint.return_value = retry_parse
            mock_get_parser.return_value = mock_parser

            from config.Servers import REFLEXION_MAX_RETRIES

            result = executor.execute_sequence([self._make_cmd()])

        assert result["success"] is False
        # Called once for original + REFLEXION_MAX_RETRIES retries
        assert executor._execute_single_command.call_count == 1 + REFLEXION_MAX_RETRIES

    def test_reflexion_skipped_without_original_text(self, executor):
        """When _original_text is absent, no retry is attempted."""
        fail_result = {"success": False, "error": "error", "recovery_suggestions": []}
        executor._execute_single_command = Mock(return_value=fail_result)

        cmd = {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1"}}
        # No _original_text key

        with patch("orchestrators.CommandParser.get_command_parser") as mock_get_parser:
            result = executor.execute_sequence([cmd])

        mock_get_parser.assert_not_called()
        assert result["success"] is False
        assert executor._execute_single_command.call_count == 1

    def test_reflexion_max_retries_config(self):
        """REFLEXION_MAX_RETRIES is a positive integer from config."""
        from config.Servers import REFLEXION_MAX_RETRIES

        assert isinstance(REFLEXION_MAX_RETRIES, int)
        assert REFLEXION_MAX_RETRIES >= 0
