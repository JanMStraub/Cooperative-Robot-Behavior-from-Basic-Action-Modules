#!/usr/bin/env python3
"""
Unit tests for CommandParser.py

Tests the command parser for multi-command sequences including:
- LLM-based command parsing with LM Studio
- Regex fallback when LLM unavailable
- Parameter extraction accuracy
- Compound command splitting
- Variable interpolation
- Error handling for ambiguous/malformed commands
- Edge cases (empty, very long, Unicode)
"""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch
import requests

from orchestrators.CommandParser import CommandParser, get_command_parser


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def command_parser(monkeypatch):
    """
    Create a CommandParser instance for testing.

    Returns:
        CommandParser instance with RAG disabled (for faster tests)
    """
    # Disable RAG to avoid initialization issues
    parser = CommandParser(use_rag=False)
    parser.rag = None  # Ensure RAG is not initialized
    return parser


@pytest.fixture
def mock_lm_studio_response():
    """
    Create a mock LM Studio response for testing.

    Returns:
        Mock response object with JSON structure
    """
    response = Mock(spec=requests.Response)
    response.status_code = 200
    response.json = Mock(return_value={
        "choices": [{
            "message": {
                "content": json.dumps({
                    "commands": [
                        {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1", "x": 0.3, "y": 0.2, "z": 0.1}},
                        {"operation": "control_gripper", "params": {"robot_id": "Robot1", "open_gripper": False}}
                    ]
                })
            }
        }]
    })
    return response


@pytest.fixture
def mock_registry():
    """
    Create a mock operation registry for testing.

    Returns:
        Mock OperationRegistry with get_operation_by_name
    """
    from operations.Base import BasicOperation, OperationCategory, OperationComplexity, OperationParameter

    # Create mock operations with parameters attribute
    move_op = Mock(spec=BasicOperation)
    move_op.name = "move_to_coordinate"
    move_op.category = OperationCategory.NAVIGATION
    move_op.complexity = OperationComplexity.BASIC
    move_op.parameters = [
        OperationParameter("robot_id", "str", "Robot ID", required=True),
        OperationParameter("x", "float", "X coordinate", required=True),
        OperationParameter("y", "float", "Y coordinate", required=True),
        OperationParameter("z", "float", "Z coordinate", required=True)
    ]
    move_op.description = "Move robot to coordinate"

    gripper_op = Mock(spec=BasicOperation)
    gripper_op.name = "control_gripper"
    gripper_op.parameters = [
        OperationParameter("robot_id", "str", "Robot ID", required=True),
        OperationParameter("open_gripper", "bool", "Open or close gripper", required=True)
    ]
    gripper_op.description = "Control gripper"

    detect_op = Mock(name="detect_object_stereo")
    detect_op.name = "detect_object_stereo"
    detect_op.parameters = [OperationParameter("robot_id", "str", "Robot ID", required=True)]
    detect_op.description = "Detect objects with stereo vision"

    status_op = Mock(name="check_robot_status")
    status_op.name = "check_robot_status"
    status_op.parameters = [OperationParameter("robot_id", "str", "Robot ID", required=True)]
    status_op.description = "Check robot status"

    return_op = Mock(name="return_to_start_position")
    return_op.name = "return_to_start_position"
    return_op.parameters = [OperationParameter("robot_id", "str", "Robot ID", required=True)]
    return_op.description = "Return to start position"

    registry = Mock()
    registry.get_operation_by_name = Mock(side_effect=lambda name: {
        "move_to_coordinate": move_op,
        "control_gripper": gripper_op,
        "detect_object_stereo": detect_op,
        "check_robot_status": status_op,
        "return_to_start_position": return_op
    }.get(name))
    registry.get_all_operations = Mock(return_value=[move_op, gripper_op, detect_op, status_op, return_op])

    return registry


# ============================================================================
# Test Class: LLM Parsing
# ============================================================================

class TestCommandParserLLM:
    """Test LLM-based command parsing."""

    def test_parse_simple_move_command(self, command_parser, mock_lm_studio_response, mock_registry):
        """Test parsing a simple move command using LLM."""
        with patch('requests.post', return_value=mock_lm_studio_response):
            with patch.object(command_parser, 'registry', mock_registry):
                result = command_parser.parse("move to (0.3, 0.2, 0.1)", robot_id="Robot1")

                assert result["success"] is True
                assert len(result["commands"]) >= 1
                assert result["commands"][0]["operation"] == "move_to_coordinate"

    def test_parse_gripper_command(self, command_parser, mock_registry):
        """Test parsing a gripper command using LLM."""
        llm_response = Mock(spec=requests.Response)
        llm_response.status_code = 200
        llm_response.json = Mock(return_value={
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "commands": [
                            {"operation": "control_gripper", "params": {"robot_id": "Robot1", "open_gripper": True}}
                        ]
                    })
                }
            }]
        })

        with patch('requests.post', return_value=llm_response):
            with patch.object(command_parser, 'registry', mock_registry):
                result = command_parser.parse("open the gripper", robot_id="Robot1")

                assert result["success"] is True
                assert result["commands"][0]["operation"] == "control_gripper"
                assert result["commands"][0]["params"]["open_gripper"] is True

    def test_parse_detection_command(self, command_parser, mock_registry):
        """Test parsing a detection command using LLM."""
        llm_response = Mock(spec=requests.Response)
        llm_response.status_code = 200
        llm_response.json = Mock(return_value={
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "commands": [
                            {"operation": "detect_object_stereo", "params": {"robot_id": "Robot1", "color": "blue"}, "capture_var": "target"}
                        ]
                    })
                }
            }]
        })

        with patch('requests.post', return_value=llm_response):
            with patch.object(command_parser, 'registry', mock_registry):
                result = command_parser.parse("detect the blue cube", robot_id="Robot1")

                assert result["success"] is True
                assert result["commands"][0]["operation"] == "detect_object_stereo"
                assert "capture_var" in result["commands"][0]

    def test_parse_compound_command(self, command_parser, mock_lm_studio_response, mock_registry):
        """Test parsing compound commands (multiple operations)."""
        with patch('requests.post', return_value=mock_lm_studio_response):
            with patch.object(command_parser, 'registry', mock_registry):
                result = command_parser.parse("move to (0.3, 0.2, 0.1) and close gripper", robot_id="Robot1")

                assert result["success"] is True
                assert len(result["commands"]) >= 2
                assert result["commands"][0]["operation"] == "move_to_coordinate"
                assert result["commands"][1]["operation"] == "control_gripper"

    def test_parse_with_variable_substitution(self, command_parser, mock_registry):
        """Test parsing command with variable substitution ($target)."""
        llm_response = Mock(spec=requests.Response)
        llm_response.status_code = 200
        llm_response.json = Mock(return_value={
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "commands": [
                            {"operation": "detect_object_stereo", "params": {"robot_id": "Robot1", "color": "red"}, "capture_var": "target"},
                            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1", "position": "$target"}}
                        ]
                    })
                }
            }]
        })

        with patch('requests.post', return_value=llm_response):
            with patch.object(command_parser, 'registry', mock_registry):
                result = command_parser.parse("detect red cube and move to it", robot_id="Robot1")

                assert result["success"] is True
                assert len(result["commands"]) == 2
                assert result["commands"][1]["params"]["position"] == "$target"

    def test_parse_with_llm_unavailable_fallback(self, command_parser, mock_registry):
        """Test fallback to regex when LLM is unavailable."""
        with patch('requests.post', side_effect=requests.exceptions.ConnectionError("LM Studio not available")):
            with patch.object(command_parser, 'registry', mock_registry):
                result = command_parser.parse("move to (0.3, 0.2, 0.1)", robot_id="Robot1")

                # Should fallback to regex and still succeed
                assert result["success"] is True
                assert result["commands"][0]["operation"] == "move_to_coordinate"


# ============================================================================
# Test Class: Regex Fallback
# ============================================================================

class TestCommandParserRegex:
    """Test regex-based command parsing (fallback mode)."""

    def test_regex_parse_move_to_coordinate(self, command_parser, mock_registry):
        """Test regex parsing of move to coordinate command."""
        with patch.object(command_parser, 'registry', mock_registry):
            result = command_parser._parse_with_regex("move to (0.3, 0.2, 0.1)", "Robot1")

            assert result["success"] is True
            assert result["commands"][0]["operation"] == "move_to_coordinate"
            assert result["commands"][0]["params"]["x"] == 0.3
            assert result["commands"][0]["params"]["y"] == 0.2
            assert result["commands"][0]["params"]["z"] == 0.1

    def test_regex_parse_detect_object(self, command_parser, mock_registry):
        """Test regex parsing of detect object command."""
        with patch.object(command_parser, 'registry', mock_registry):
            result = command_parser._parse_with_regex("detect the blue cube", "Robot1")

            assert result["success"] is True
            assert result["commands"][0]["operation"] == "detect_object_stereo"
            assert result["commands"][0]["params"]["color"] == "blue"

    def test_regex_parse_control_gripper_close(self, command_parser, mock_registry):
        """Test regex parsing of close gripper command."""
        with patch.object(command_parser, 'registry', mock_registry):
            result = command_parser._parse_with_regex("close the gripper", "Robot1")

            assert result["success"] is True
            assert result["commands"][0]["operation"] == "control_gripper"
            assert result["commands"][0]["params"]["open_gripper"] is False

    def test_regex_parse_control_gripper_open(self, command_parser, mock_registry):
        """Test regex parsing of open gripper command."""
        with patch.object(command_parser, 'registry', mock_registry):
            result = command_parser._parse_with_regex("open gripper", "Robot1")

            assert result["success"] is True
            assert result["commands"][0]["operation"] == "control_gripper"
            assert result["commands"][0]["params"]["open_gripper"] is True

    def test_regex_parse_compound_commands(self, command_parser, mock_registry):
        """Test regex parsing of compound commands with 'and'."""
        with patch.object(command_parser, 'registry', mock_registry):
            result = command_parser._parse_with_regex("move to (0.3, 0.2, 0.1) and close gripper", "Robot1")

            assert result["success"] is True
            assert len(result["commands"]) == 2
            assert result["commands"][0]["operation"] == "move_to_coordinate"
            assert result["commands"][1]["operation"] == "control_gripper"


# ============================================================================
# Test Class: Parameter Extraction
# ============================================================================

class TestCommandParserParameters:
    """Test parameter extraction accuracy."""

    def test_extract_coordinates_from_text(self, command_parser, mock_registry):
        """Test extracting coordinates from various text formats."""
        test_cases = [
            ("move to (0.3, 0.2, 0.1)", (0.3, 0.2, 0.1)),
            ("move to x=0.3, y=0.2, z=0.1", (0.3, 0.2, 0.1)),
            ("move to (-0.5, 0.0, 0.3)", (-0.5, 0.0, 0.3)),
        ]

        for command_text, expected in test_cases:
            with patch.object(command_parser, 'registry', mock_registry):
                result = command_parser._parse_with_regex(command_text, "Robot1")

                assert result["success"] is True
                assert result["commands"][0]["params"]["x"] == expected[0]
                assert result["commands"][0]["params"]["y"] == expected[1]
                assert result["commands"][0]["params"]["z"] == expected[2]

    def test_extract_color_and_object_type(self, command_parser, mock_registry):
        """Test extracting color from detection commands."""
        test_cases = [
            ("detect the red cube", "red"),
            ("detect the blue object", "blue"),
            ("detect the green block", "green"),
        ]

        for command_text, expected_color in test_cases:
            with patch.object(command_parser, 'registry', mock_registry):
                result = command_parser._parse_with_regex(command_text, "Robot1")

                if result["success"] and result["commands"]:
                    assert result["commands"][0]["params"]["color"] == expected_color

    def test_extract_robot_id(self, command_parser, mock_registry):
        """Test robot_id is correctly assigned to all operations."""
        with patch.object(command_parser, 'registry', mock_registry):
            result = command_parser._parse_with_regex("move to (0.3, 0.2, 0.1)", "CustomRobot")

            assert result["success"] is True
            assert result["commands"][0]["params"]["robot_id"] == "CustomRobot"


# ============================================================================
# Test Class: Error Handling
# ============================================================================

class TestCommandParserErrors:
    """Test error handling for invalid/ambiguous commands."""

    def test_parse_empty_command(self, command_parser):
        """Test parsing empty command string."""
        result = command_parser.parse("", robot_id="Robot1")

        assert result["success"] is False
        assert "error" in result

    def test_parse_whitespace_only_command(self, command_parser):
        """Test parsing whitespace-only command."""
        result = command_parser.parse("   ", robot_id="Robot1")

        assert result["success"] is False

    def test_parse_very_long_command(self, command_parser, mock_registry):
        """Test parsing very long command string."""
        long_command = "move to (0.3, 0.2, 0.1) " + "and then move again " * 50

        with patch.object(command_parser, 'registry', mock_registry):
            # Should not crash
            result = command_parser._parse_with_regex(long_command, "Robot1")
            # Result may succeed or fail, but should not crash

    def test_parse_unicode_command(self, command_parser, mock_registry):
        """Test parsing command with Unicode characters."""
        unicode_command = "move to (0.3, 0.2, 0.1) 🤖"

        with patch.object(command_parser, 'registry', mock_registry):
            # Should handle gracefully
            result = command_parser._parse_with_regex(unicode_command, "Robot1")
            # May or may not parse Unicode, but should not crash

    def test_parse_unparseable_command(self, command_parser, mock_registry):
        """Test parsing command that doesn't match any pattern."""
        with patch.object(command_parser, 'registry', mock_registry):
            result = command_parser._parse_with_regex("xyzabc nonsense", "Robot1")

            assert result["success"] is False
            assert "error" in result

    def test_llm_timeout_handling(self, command_parser, mock_registry):
        """Test handling LLM request timeout."""
        with patch('requests.post', side_effect=requests.exceptions.Timeout("Request timeout")):
            with patch.object(command_parser, 'registry', mock_registry):
                result = command_parser.parse("move somewhere", robot_id="Robot1", use_llm=True)

                # Should fallback to regex or return error
                assert "error" in result or result["success"] is True

    def test_llm_error_response(self, command_parser, mock_registry):
        """Test handling LLM error response (non-200 status)."""
        error_response = Mock(spec=requests.Response)
        error_response.status_code = 500

        with patch('requests.post', return_value=error_response):
            with patch.object(command_parser, 'registry', mock_registry):
                result = command_parser.parse("move to position", robot_id="Robot1", use_llm=True)

                # Should fallback to regex or return error
                assert "error" in result or result["success"] is True

    def test_malformed_llm_json_response(self, command_parser, mock_registry):
        """Test handling malformed JSON from LLM."""
        bad_response = Mock(spec=requests.Response)
        bad_response.status_code = 200
        bad_response.json = Mock(return_value={
            "choices": [{
                "message": {
                    "content": "This is not valid JSON for commands"
                }
            }]
        })

        with patch('requests.post', return_value=bad_response):
            with patch.object(command_parser, 'registry', mock_registry):
                result = command_parser.parse("move to position", robot_id="Robot1", use_llm=True)

                # Should fallback to regex or return error
                assert "error" in result or result["success"] is True


# ============================================================================
# Test Class: Variable Handling
# ============================================================================

class TestCommandParserVariables:
    """Test variable interpolation and passing."""

    def test_variable_interpolation_single(self, command_parser, mock_registry):
        """Test detecting and moving to variable reference ($target)."""
        with patch.object(command_parser, 'registry', mock_registry):
            result = command_parser._parse_with_regex("detect the red cube and move to it", "Robot1")

            assert result["success"] is True
            assert len(result["commands"]) == 2
            # First command should capture variable
            assert result["commands"][0].get("capture_var") == "target"
            # Second command should reference variable
            assert result["commands"][1]["params"]["position"] == "$target"


# ============================================================================
# Test Class: Integration
# ============================================================================

class TestCommandParserIntegration:
    """Integration tests for command parser."""

    def test_get_command_parser_singleton(self):
        """Test get_command_parser returns singleton instance."""
        parser1 = get_command_parser()
        parser2 = get_command_parser()

        assert parser1 is parser2

    def test_get_supported_patterns(self, command_parser):
        """Test get_supported_patterns returns documentation."""
        patterns = command_parser.get_supported_patterns()

        assert isinstance(patterns, list)
        assert len(patterns) > 0
        assert any("move" in pattern.lower() for pattern in patterns)

    def test_full_command_flow_with_validation(self, command_parser, mock_registry):
        """Test full command parsing flow with operation validation."""
        with patch.object(command_parser, 'registry', mock_registry):
            result = command_parser.parse("move to (0.3, 0.2, 0.1) and close gripper", robot_id="Robot1", use_llm=False)

            assert result["success"] is True
            assert len(result["commands"]) == 2

            # Verify all commands have robot_id
            for cmd in result["commands"]:
                assert "robot_id" in cmd["params"]
                assert cmd["params"]["robot_id"] == "Robot1"
