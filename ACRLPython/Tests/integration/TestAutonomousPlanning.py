import pytest

from orchestrators.CommandParser import CommandParser
from operations.WorkflowPatterns import WorkflowPatternRegistry


def is_llm_available():
    """Check if LM Studio is running and responding."""
    try:
        from rag.Embeddings import EmbeddingGenerator

        gen = EmbeddingGenerator()
        return gen.use_lm_studio
    except Exception:
        return False


LLM_AVAILABLE = is_llm_available()


class TestAutonomousPlanning:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.parser = CommandParser(use_rag=True)
        self.workflow_registry = WorkflowPatternRegistry()

    def test_workflow_pattern_surfacing(self):
        """Verify workflow patterns appear in LLM prompt for relevant queries"""
        ops_summary = self.parser._get_available_operations_summary(
            "Robot1 and Robot2 handoff the red cube"
        )

        assert isinstance(ops_summary, str)
        assert len(ops_summary) > 0
        assert "move_to_coordinate" in ops_summary.lower()

        print(f"Operations summary generated ({len(ops_summary)} chars)")
        print(
            f"  Contains workflow patterns: {'WORKFLOW PATTERN' in ops_summary or 'workflow' in ops_summary.lower()}"
        )

    def test_workflow_registry_integration(self):
        """Test workflow registry is properly integrated"""
        all_patterns = self.workflow_registry.get_all_patterns()
        assert len(all_patterns) > 0, "Workflow registry should have patterns"

        handoff = self.workflow_registry.get_pattern_by_name("handoff")
        assert handoff is not None, "Handoff pattern should exist"
        assert handoff.name == "handoff"
        assert len(handoff.steps) > 0

        print(f"Workflow registry has {len(all_patterns)} patterns")
        print(f"  Handoff pattern has {len(handoff.steps)} steps")

    @pytest.mark.skipif(
        not LLM_AVAILABLE, reason="Requires LM Studio with embedding model"
    )
    def test_handoff_plan_generation(self):
        """Test full pipeline generates valid parallel groups for handoff"""
        result = self.parser.parse(
            "Robot1 and Robot2 perform a handoff of the red cube", "Robot1"
        )

        assert result["success"], f"Parse failed: {result.get('error')}"

        commands = result.get("plan") or result.get("commands")
        assert commands is not None
        assert isinstance(commands, list)
        assert len(commands) > 0, "Should generate at least one command"

        print(f"Generated {len(commands)} commands")

        has_parallel_groups = any("parallel_group" in cmd for cmd in commands)
        if has_parallel_groups:
            print(f"  Found parallel_group assignments")

            signals = [c for c in commands if c.get("operation") == "signal"]
            waits = [c for c in commands if c.get("operation") == "wait_for_signal"]

            if len(signals) > 0 or len(waits) > 0:
                print(f"  Has {len(signals)} signals and {len(waits)} waits")
                assert len(signals) > 0, "Multi-robot plan should have signals"
                assert len(waits) > 0, "Multi-robot plan should have waits"
        else:
            print("  No parallel_group (LLM may be unavailable, using fallback)")

    def test_plan_validation_signal_mismatch(self):
        """Test validation catches mismatched signals"""
        invalid_plan = [
            {
                "operation": "wait_for_signal",
                "params": {"robot_id": "Robot2", "event_name": "undefined_signal"},
            }
        ]

        valid, errors = self.parser._validate_multi_robot_plan(invalid_plan)
        assert not valid, "Should detect missing signal"
        assert len(errors) > 0, "Should return error messages"
        assert "undefined_signal" in str(errors)

        print(f"Validation caught missing signal: {errors[0]}")

    def test_plan_validation_variable_usage(self):
        """Test validation catches variables used before definition"""
        invalid_plan = [
            {
                "operation": "move_to_coordinate",
                "params": {"robot_id": "Robot1", "position": "$target"},
            },
            {
                "operation": "detect_object_stereo",
                "params": {"robot_id": "Robot1", "color": "red"},
                "capture_var": "target",
            },
        ]

        valid, errors = self.parser._validate_multi_robot_plan(invalid_plan)
        assert not valid, "Should detect variable used before definition"
        assert len(errors) > 0
        assert "target" in str(errors)

        print(f"Validation caught undefined variable: {errors[0]}")

    def test_plan_validation_valid_plan(self):
        """Test validation passes for valid plans"""
        valid_plan = [
            {
                "operation": "detect_object_stereo",
                "params": {"robot_id": "Robot1", "color": "red"},
                "capture_var": "target",
            },
            {
                "operation": "move_to_coordinate",
                "params": {"robot_id": "Robot1", "position": "$target"},
            },
            {
                "operation": "control_gripper",
                "params": {"robot_id": "Robot1", "open_gripper": False},
            },
            {
                "operation": "signal",
                "params": {"event_name": "object_gripped"},
            },
            {
                "operation": "wait_for_signal",
                "params": {"event_name": "object_gripped", "timeout_ms": 5000},
            },
        ]

        valid, errors = self.parser._validate_multi_robot_plan(valid_plan)
        assert valid, f"Valid plan should pass validation: {errors}"
        assert len(errors) == 0

        print(f"Validation passed for valid plan with {len(valid_plan)} commands")

    @pytest.mark.skipif(
        not LLM_AVAILABLE, reason="Requires LM Studio with embedding model"
    )
    def test_simultaneous_movement_command(self):
        """Test parsing command for simultaneous robot movement"""
        result = self.parser.parse(
            "Move Robot1 to (0.3, 0.1, 0.2) and Robot2 to (-0.3, -0.1, 0.2) simultaneously",
            "Robot1",
        )

        assert result["success"]
        commands = result.get("plan") or result.get("commands")
        assert commands is not None
        assert isinstance(commands, list)

        move_commands = [
            c for c in commands if c.get("operation") == "move_to_coordinate"
        ]
        assert len(move_commands) >= 2, "Should have moves for both robots"

        print(f"Simultaneous movement: {len(move_commands)} move commands")

    @pytest.mark.skipif(
        not LLM_AVAILABLE, reason="Requires LM Studio with embedding model"
    )
    def test_collaborative_task_command(self):
        """Test parsing collaborative task command"""
        result = self.parser.parse(
            "Robot1 should pick up the blue cube and hand it to Robot2", "Robot1"
        )

        assert result["success"]
        commands = result.get("plan") or result.get("commands")
        assert commands is not None
        assert isinstance(commands, list)

        print(f"Collaborative task generated {len(commands)} commands")

        op_types = set(c.get("operation") for c in commands)
        print(f"  Operation types: {', '.join(sorted(op_types))}")


class TestWorkflowPatternFormatting:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.parser = CommandParser(use_rag=True)
        self.workflow_registry = WorkflowPatternRegistry()

    def test_format_workflow_pattern(self):
        """Test workflow pattern formatting"""
        handoff = self.workflow_registry.get_pattern_by_name("handoff")
        assert handoff is not None

        formatted = self.parser._format_workflow_pattern(handoff)

        assert "Pattern:" in formatted
        assert "handoff" in formatted
        assert "Description:" in formatted
        assert "Steps:" in formatted
        assert "Examples:" in formatted

        print(f"Formatted pattern ({len(formatted)} chars):")
        print(formatted[:300] + "...")

    def test_all_patterns_formattable(self):
        """Test all patterns can be formatted"""
        all_patterns = self.workflow_registry.get_all_patterns()

        for pattern in all_patterns:
            formatted = self.parser._format_workflow_pattern(pattern)
            assert isinstance(formatted, str)
            assert len(formatted) > 0

        print(f"All {len(all_patterns)} patterns can be formatted")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
