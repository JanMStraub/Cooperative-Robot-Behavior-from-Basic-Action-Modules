"""
Tests for the Sequence Server system.

Tests the CommandParser, SequenceExecutor, and SequenceServer components.
"""

import pytest
import sys
import os
import inspect

# Add parent directory to path for imports
test_dir = os.path.dirname(os.path.abspath(__file__))
servers_dir = os.path.dirname(test_dir)  # tests/
acrlpython_dir = os.path.dirname(servers_dir)  # ACRLPython/
sys.path.insert(0, acrlpython_dir)

from orchestrators.CommandParser import CommandParser, get_command_parser
from orchestrators.SequenceExecutor import SequenceExecutor
from servers.SequenceServer import SequenceQueryHandler


class TestCommandParser:

    def setup_method(self):
        self.parser = CommandParser(use_rag=False)

    def test_parse_simple_move(self):
        result = self.parser.parse(
            "move to (0.3, 0.2, 0.1)", robot_id="Robot1", use_llm=False
        )

        assert result["success"] is True
        assert len(result["commands"]) == 1
        assert result["commands"][0]["operation"] == "move_to_coordinate"
        assert result["commands"][0]["params"]["x"] == 0.3
        assert result["commands"][0]["params"]["y"] == 0.2
        assert result["commands"][0]["params"]["z"] == 0.1
        assert result["commands"][0]["params"]["robot_id"] == "Robot1"

    def test_parse_close_gripper(self):
        result = self.parser.parse(
            "close the gripper", robot_id="Robot1", use_llm=False
        )

        assert result["success"] is True
        assert len(result["commands"]) == 1
        assert result["commands"][0]["operation"] == "control_gripper"
        assert result["commands"][0]["params"]["open_gripper"] is False

    def test_parse_open_gripper(self):
        result = self.parser.parse("open gripper", robot_id="Robot1", use_llm=False)

        assert result["success"] is True
        assert len(result["commands"]) == 1
        assert result["commands"][0]["operation"] == "control_gripper"
        assert result["commands"][0]["params"]["open_gripper"] is True

    def test_parse_compound_command_with_and(self):
        result = self.parser.parse(
            "move to (0.3, 0.2, 0.1) and close the gripper",
            robot_id="Robot1",
            use_llm=False,
        )

        assert result["success"] is True
        assert len(result["commands"]) == 2
        assert result["commands"][0]["operation"] == "move_to_coordinate"
        assert result["commands"][1]["operation"] == "control_gripper"
        assert result["commands"][1]["params"]["open_gripper"] is False

    def test_parse_compound_command_with_then(self):
        result = self.parser.parse(
            "open gripper then move to (0, 0, 0.3)", robot_id="Robot1", use_llm=False
        )

        assert result["success"] is True
        assert len(result["commands"]) == 2
        assert result["commands"][0]["operation"] == "control_gripper"
        assert result["commands"][0]["params"]["open_gripper"] is True
        assert result["commands"][1]["operation"] == "move_to_coordinate"

    def test_parse_three_commands(self):
        result = self.parser.parse(
            "move to (0.1, 0.2, 0.15), then close gripper, then move to (0, 0, 0.4)",
            robot_id="Robot1",
            use_llm=False,
        )

        assert result["success"] is True
        assert len(result["commands"]) == 3
        assert result["commands"][0]["operation"] == "move_to_coordinate"
        assert result["commands"][1]["operation"] == "control_gripper"
        assert result["commands"][2]["operation"] == "move_to_coordinate"

    def test_parse_alternative_coordinate_format(self):
        result = self.parser.parse(
            "move to x=0.3, y=0.2, z=0.1", robot_id="Robot1", use_llm=False
        )

        assert result["success"] is True
        assert len(result["commands"]) == 1
        assert result["commands"][0]["params"]["x"] == 0.3
        assert result["commands"][0]["params"]["y"] == 0.2
        assert result["commands"][0]["params"]["z"] == 0.1

    def test_parse_negative_coordinates(self):
        result = self.parser.parse(
            "move to (-0.3, 0.2, 0.1)", robot_id="Robot1", use_llm=False
        )

        assert result["success"] is True
        assert result["commands"][0]["params"]["x"] == -0.3

    def test_parse_grasp_synonym(self):
        result = self.parser.parse("grasp", robot_id="Robot1", use_llm=False)

        assert result["success"] is True
        assert result["commands"][0]["operation"] == "control_gripper"
        assert result["commands"][0]["params"]["open_gripper"] is False

    def test_parse_release_synonym(self):
        result = self.parser.parse("release", robot_id="Robot1", use_llm=False)

        assert result["success"] is True
        assert result["commands"][0]["operation"] == "control_gripper"
        assert result["commands"][0]["params"]["open_gripper"] is True

    def test_parse_empty_command(self):
        result = self.parser.parse("", robot_id="Robot1", use_llm=False)

        assert result["success"] is False

    def test_parse_unknown_command(self):
        result = self.parser.parse("dance around", robot_id="Robot1", use_llm=False)

        assert result["success"] is False

    def test_robot_id_propagation(self):
        result = self.parser.parse(
            "move to (0.1, 0.2, 0.3) and close gripper",
            robot_id="MyRobot",
            use_llm=False,
        )

        assert result["success"] is True
        for cmd in result["commands"]:
            assert cmd["params"]["robot_id"] == "MyRobot"


class TestSequenceExecutor:

    def setup_method(self):
        # Disable completion checking for tests (no Unity connection)
        self.executor = SequenceExecutor(check_completion=False)

    def test_execute_empty_sequence(self):
        result = self.executor.execute_sequence([])

        assert result["success"] is True
        assert result["total_commands"] == 0
        assert result["completed_commands"] == 0

    def test_abort_sequence(self):
        self.executor.abort()
        # Verify abort flag is set
        assert self.executor._abort_flag is True


class TestCommandParserSingleton:

    def test_singleton_returns_same_instance(self):
        parser1 = get_command_parser()
        parser2 = get_command_parser()
        assert parser1 is parser2


class TestSequenceQueryHandlerTimeouts:

    def test_execute_sequence_default_timeout_is_120(self):
        sig = inspect.signature(SequenceQueryHandler.execute_sequence)
        assert (
            sig.parameters["timeout"].default == 120.0
        ), f"Expected 120.0, got {sig.parameters['timeout'].default}"

    def test_execute_sequence_inner_default_timeout_is_120(self):
        sig = inspect.signature(SequenceQueryHandler._execute_sequence_inner)
        assert (
            sig.parameters["timeout"].default == 120.0
        ), f"Expected 120.0, got {sig.parameters['timeout'].default}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
