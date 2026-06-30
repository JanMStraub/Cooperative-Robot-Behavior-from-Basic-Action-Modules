import pytest
from unittest.mock import Mock, patch


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
        assert "=== ROBOT WORKSPACE BOUNDARIES ===" in prompt_no_hint
        assert "=== ROBOT WORKSPACE BOUNDARIES ===" in prompt_with_hint


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
            "move to (0.1, 0.2, 0.3)",
            robot_id="Robot1",
            hint="retry hint",
            use_motion_layer=False,
        )
        assert result["success"] is True
        assert len(result["commands"]) == 1

    def test_parse_with_hint_injects_hint_into_prompt(self, parser):
        """Hint text appears in the prompt passed to _do_llm_request."""
        cmds = [{"operation": "check_robot_status", "params": {"robot_id": "Robot1"}}]
        self._mock_llm_response(parser, cmds)
        parser.parse_with_hint(
            "check status",
            robot_id="Robot1",
            hint="error: timeout",
            use_motion_layer=False,
        )
        prompt_arg = parser._do_llm_request.call_args[0][0]
        assert "error: timeout" in prompt_arg

    def test_parse_with_hint_failure_returns_error(self, parser):
        """parse_with_hint returns failure dict when LLM call fails."""
        parser._do_llm_request = Mock(side_effect=Exception("connection refused"))
        result = parser.parse_with_hint(
            "some command", hint="irrelevant", use_motion_layer=False
        )
        assert result["success"] is False
        assert "error" in result


class TestSequenceExecutorReflection:
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

    def test_reflection_retries_on_failure_and_succeeds(self, executor):
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
            mock_parser.generate_reflection = Mock(return_value="")
            mock_get_parser.return_value = mock_parser

            result = executor.execute_sequence([self._make_cmd()])

        assert result["success"] is True
        assert result["completed_commands"] == 1
        assert result["results"][0]["success"] is True

    def test_reflection_stops_after_max_retries(self, executor):
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
            mock_parser.generate_reflection = Mock(return_value="")
            mock_get_parser.return_value = mock_parser

            from config.Servers import REFLECTION_MAX_RETRIES

            result = executor.execute_sequence([self._make_cmd()])

        assert result["success"] is False
        # Called once for original + REFLECTION_MAX_RETRIES retries
        assert executor._execute_single_command.call_count == 1 + REFLECTION_MAX_RETRIES

    def test_reflection_skipped_without_original_text(self, executor):
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

    def test_reflection_max_retries_config(self):
        """REFLECTION_MAX_RETRIES is a positive integer from config."""
        from config.Servers import REFLECTION_MAX_RETRIES

        assert isinstance(REFLECTION_MAX_RETRIES, int)
        assert REFLECTION_MAX_RETRIES >= 0

    def test_reflection_hint_includes_held_objects(self, executor):
        """Hint text includes WorldState held-object info so LLM avoids re-detecting held objects."""
        fail_result = {
            "success": False,
            "error": "not found",
            "recovery_suggestions": [],
        }
        rts_result = {"success": True, "result": {}}
        ok_result = {"success": True, "result": {"x": 0.1, "y": 0.2, "z": 0.3}}

        # Three calls: initial detect (fail), return_to_start (ok), retry detect (ok)
        executor._execute_single_command = Mock(
            side_effect=[fail_result, rts_result, ok_result]
        )

        retry_parse = {
            "success": True,
            "commands": [
                {
                    "operation": "detect_object_stereo",
                    "params": {"robot_id": "Robot1", "color": "red"},
                }
            ],
        }

        # Simulate Robot1 holding the blue cube
        mock_robot_state = Mock()
        mock_robot_state.is_holding_object = True
        mock_robot_state.held_object_id = "blue"
        mock_ws = Mock()
        mock_ws._robot_states = {"Robot1": mock_robot_state}

        captured_hints = []

        with patch("orchestrators.CommandParser.get_command_parser") as mock_get_parser:
            mock_parser = Mock()

            def capture_hint(*args, **kwargs):
                captured_hints.append(kwargs.get("hint", ""))
                return retry_parse

            mock_parser.parse_with_hint.side_effect = capture_hint
            mock_parser.generate_reflection = Mock(return_value="")
            mock_get_parser.return_value = mock_parser

            with patch(
                "orchestrators.SequenceExecutor.get_world_state", return_value=mock_ws
            ):
                executor.execute_sequence(
                    [
                        self._make_cmd(
                            operation="detect_object_stereo",
                            original_text="detect red cube",
                        )
                    ]
                )

        assert captured_hints, "parse_with_hint was never called"
        assert "Robot1 holds 'blue'" in captured_hints[0]
        assert "NOT visible to cameras" in captured_hints[0]

    def test_reflection_perception_retry_calls_return_to_start(self, executor):
        """For PERCEPTION op failures, return_to_start_position is called before each retry."""
        fail_result = {
            "success": False,
            "error": "not found",
            "recovery_suggestions": [],
        }
        ok_result = {"success": True, "result": {"x": 0.1, "y": 0.2, "z": 0.3}}

        calls = []

        def track_calls(operation, params, timeout):
            calls.append(operation)
            if operation == "return_to_start_position":
                return {"success": True, "result": {}}
            if (
                operation == "detect_object_stereo"
                and len([c for c in calls if c == "detect_object_stereo"]) >= 2
            ):
                return ok_result
            return fail_result

        executor._execute_single_command = Mock(side_effect=track_calls)

        retry_parse = {
            "success": True,
            "commands": [
                {
                    "operation": "detect_object_stereo",
                    "params": {"robot_id": "Robot1", "color": "red"},
                }
            ],
        }

        with patch("orchestrators.CommandParser.get_command_parser") as mock_get_parser:
            mock_parser = Mock()
            mock_parser.parse_with_hint.return_value = retry_parse
            mock_parser.generate_reflection = Mock(return_value="")
            mock_get_parser.return_value = mock_parser

            with patch(
                "orchestrators.SequenceExecutor.get_world_state", return_value=None
            ):
                executor.execute_sequence(
                    [
                        self._make_cmd(
                            operation="detect_object_stereo",
                            original_text="detect red cube",
                        )
                    ]
                )

        assert (
            "return_to_start_position" in calls
        ), "return_to_start_position was not called before perception retry"
        rts_idx = calls.index("return_to_start_position")
        detect_retries = [
            i for i, c in enumerate(calls) if c == "detect_object_stereo" and i > 0
        ]
        assert detect_retries, "no detection retry after initial failure"
        assert (
            rts_idx < detect_retries[0]
        ), "return_to_start_position must precede the detection retry"


class TestInferCaptureVars:
    @pytest.fixture
    def executor(self):
        from orchestrators.SequenceExecutor import SequenceExecutor

        return SequenceExecutor()

    def test_infers_capture_var_when_missing(self, executor):
        """Perception op without capture_var gets it inferred from downstream $var reference."""
        commands = [
            {
                "operation": "detect_object_stereo",
                "params": {"robot_id": "Robot1", "color": "red"},
                # no capture_var
            },
            {
                "operation": "move_to_coordinate",
                "params": {
                    "robot_id": "Robot1",
                    "x": "$target.x",
                    "y": "$target.y",
                    "z": "$target.z",
                },
            },
        ]
        result = executor._infer_capture_vars(commands)
        assert result[0].get("capture_var") == "target"

    def test_does_not_overwrite_existing_capture_var(self, executor):
        """Explicit capture_var is preserved."""
        commands = [
            {
                "operation": "detect_object_stereo",
                "params": {"robot_id": "Robot1", "color": "blue"},
                "capture_var": "cube",
            },
            {
                "operation": "move_to_coordinate",
                "params": {
                    "robot_id": "Robot1",
                    "x": "$cube.x",
                    "y": "$cube.y",
                    "z": "$cube.z",
                },
            },
        ]
        result = executor._infer_capture_vars(commands)
        assert result[0].get("capture_var") == "cube"

    def test_infers_separate_vars_for_two_detections(self, executor):
        """Two consecutive detect ops each get the correct downstream variable inferred."""
        commands = [
            {
                "operation": "detect_object_stereo",
                "params": {"robot_id": "Robot1", "color": "red"},
            },
            {
                "operation": "move_to_coordinate",
                "params": {
                    "robot_id": "Robot1",
                    "x": "$target.x",
                    "y": "$target.y",
                    "z": "$target.z",
                },
            },
            {
                "operation": "detect_object_stereo",
                "params": {"robot_id": "Robot1", "color": "yellow"},
            },
            {
                "operation": "move_to_coordinate",
                "params": {
                    "robot_id": "Robot1",
                    "x": "$target.x",
                    "y": "$target.y",
                    "z": "$target.z",
                },
            },
        ]
        result = executor._infer_capture_vars(commands)
        # Both detections should get capture_var="target" (second one picks up the
        # still-unresolved $target reference from cmd 4 when scanning from cmd 3)
        assert result[0].get("capture_var") == "target"
        assert result[2].get("capture_var") == "target"

    def test_no_inference_for_non_perception_op(self, executor):
        """Non-perception ops are not given a capture_var even if downstream references exist."""
        commands = [
            {
                "operation": "move_to_coordinate",
                "params": {"robot_id": "Robot1", "x": 0.1, "y": 0.1, "z": 0.1},
            },
            {
                "operation": "move_to_coordinate",
                "params": {
                    "robot_id": "Robot1",
                    "x": "$target.x",
                    "y": "$target.y",
                    "z": "$target.z",
                },
            },
        ]
        result = executor._infer_capture_vars(commands)
        assert result[0].get("capture_var") is None


class TestGenerateReflection:
    @pytest.fixture
    def parser(self):
        from orchestrators.CommandParser import CommandParser

        return CommandParser(use_rag=False, use_cache=False)

    def _mock_post(self, parser, response_body: dict, status_code: int = 200):
        mock_resp = Mock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": str(response_body)}}]
        }
        # Make content return JSON string so _extract_json_util can parse it
        import json

        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps(response_body)}}]
        }
        parser._session.post = Mock(return_value=mock_resp)
        return mock_resp

    def test_returns_combined_analysis_and_suggestion(self, parser):
        """generate_reflection returns analysis + suggestion joined."""
        self._mock_post(
            parser,
            {"analysis": "bad coords", "suggestion": "try smaller x value"},
        )
        result = parser.generate_reflection(
            "move to (0.5, 0.0, 0.4)", "move_to_coordinate", "IK failed", {}, "Robot1"
        )
        assert "bad coords" in result
        assert "try smaller x value" in result

    def test_returns_empty_on_timeout(self, parser):
        """generate_reflection returns empty string when LLM times out."""
        import requests as req

        parser._session.post = Mock(side_effect=req.exceptions.Timeout)
        result = parser.generate_reflection(
            "move to (0.5, 0.0, 0.4)", "move_to_coordinate", "IK failed", {}, "Robot1"
        )
        assert result == ""

    def test_returns_empty_on_bad_status(self, parser):
        """generate_reflection returns empty string on non-200 response."""
        mock_resp = Mock()
        mock_resp.status_code = 500
        parser._session.post = Mock(return_value=mock_resp)
        result = parser.generate_reflection(
            "move to (0.5, 0.0, 0.4)", "move_to_coordinate", "IK failed", {}, "Robot1"
        )
        assert result == ""

    def test_falls_back_to_raw_content_when_no_json_keys(self, parser):
        """generate_reflection uses raw content (truncated) if JSON has no analysis/suggestion."""
        self._mock_post(parser, {"other_key": "some value"})
        result = parser.generate_reflection(
            "move to (0.1, 0.1, 0.1)", "move_to_coordinate", "error", {}, "Robot1"
        )
        # Should not raise; result is either "" or truncated raw content
        assert isinstance(result, str)


class TestReflectionHintAccumulation:
    @pytest.fixture
    def executor(self):
        from orchestrators.SequenceExecutor import SequenceExecutor

        ex = SequenceExecutor()
        ex.default_timeout = 5.0
        return ex

    def _make_cmd(self, operation="move_to_coordinate", robot_id="Robot1"):
        return {
            "operation": operation,
            "params": {"robot_id": robot_id},
            "_original_text": "move to (0.5, 0.0, 0.4)",
        }

    def test_hint_accumulates_attempt_history_on_second_retry(self, executor):
        """Retry 2's hint includes a record of retry 1's failure."""
        always_fail = {
            "success": False,
            "error": "IK solver failed",
            "recovery_suggestions": [],
        }
        executor._execute_single_command = Mock(return_value=always_fail)

        captured_hints = []

        def capture_hint(*args, **kwargs):
            captured_hints.append(kwargs.get("hint", ""))
            return {
                "success": True,
                "commands": [
                    {
                        "operation": "move_to_coordinate",
                        "params": {"robot_id": "Robot1"},
                    }
                ],
            }

        with patch("orchestrators.CommandParser.get_command_parser") as mock_get_parser:
            mock_parser = Mock()
            mock_parser.parse_with_hint.side_effect = capture_hint
            mock_parser.generate_reflection = Mock(return_value="")
            mock_get_parser.return_value = mock_parser

            with patch(
                "orchestrators.SequenceExecutor.get_world_state", return_value=None
            ):
                with patch(
                    "orchestrators.SequenceExecutor.REFLECTION_SELF_REFLECT_ENABLED",
                    False,
                ):
                    executor.execute_sequence([self._make_cmd()])

        assert len(captured_hints) >= 2, "Expected at least 2 parse_with_hint calls"
        # Second retry's hint must mention "Retry 1"
        assert "Retry 1" in captured_hints[1]
        assert "IK solver failed" in captured_hints[1]

    def test_reflection_text_prepended_to_hint(self, executor):
        """When generate_reflection returns text, it appears before the procedural hint."""
        fail_result = {
            "success": False,
            "error": "IK failed",
            "recovery_suggestions": [],
        }
        ok_result = {"success": True, "result": {}}
        executor._execute_single_command = Mock(side_effect=[fail_result, ok_result])

        captured_hints = []

        def capture_hint(*args, **kwargs):
            captured_hints.append(kwargs.get("hint", ""))
            return {
                "success": True,
                "commands": [
                    {
                        "operation": "move_to_coordinate",
                        "params": {"robot_id": "Robot1"},
                    }
                ],
            }

        with patch("orchestrators.CommandParser.get_command_parser") as mock_get_parser:
            mock_parser = Mock()
            mock_parser.parse_with_hint.side_effect = capture_hint
            mock_parser.generate_reflection = Mock(
                return_value="Arm exceeded workspace boundary. Reduce x by 0.1."
            )
            mock_get_parser.return_value = mock_parser

            with patch(
                "orchestrators.SequenceExecutor.get_world_state", return_value=None
            ):
                with patch(
                    "orchestrators.SequenceExecutor.REFLECTION_SELF_REFLECT_ENABLED",
                    True,
                ):
                    executor.execute_sequence([self._make_cmd()])

        assert captured_hints, "parse_with_hint was never called"
        assert "Arm exceeded workspace boundary" in captured_hints[0]
        # Reflection text should come before the procedural context
        refl_idx = captured_hints[0].index("Arm exceeded workspace boundary")
        proc_idx = captured_hints[0].index("IK failed")
        assert refl_idx < proc_idx

    def test_retry_proceeds_when_reflection_returns_empty(self, executor):
        """When generate_reflection returns '', retry still executes with procedural hint."""
        fail_result = {
            "success": False,
            "error": "IK failed",
            "recovery_suggestions": [],
        }
        ok_result = {"success": True, "result": {}}
        executor._execute_single_command = Mock(side_effect=[fail_result, ok_result])

        with patch("orchestrators.CommandParser.get_command_parser") as mock_get_parser:
            mock_parser = Mock()
            mock_parser.parse_with_hint.return_value = {
                "success": True,
                "commands": [
                    {
                        "operation": "move_to_coordinate",
                        "params": {"robot_id": "Robot1"},
                    }
                ],
            }
            mock_parser.generate_reflection = Mock(return_value="")
            mock_get_parser.return_value = mock_parser

            with patch(
                "orchestrators.SequenceExecutor.get_world_state", return_value=None
            ):
                with patch(
                    "orchestrators.SequenceExecutor.REFLECTION_SELF_REFLECT_ENABLED",
                    True,
                ):
                    result = executor.execute_sequence([self._make_cmd()])

        assert result["success"] is True
        mock_parser.parse_with_hint.assert_called_once()
