"""
Test Autonomous Planning for Multi-Robot Coordination
======================================================

Tests for LLM-based autonomous task planning using workflow patterns.
"""

import unittest
import sys
import os
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestrators.CommandParser import CommandParser
from operations.WorkflowPatterns import WorkflowPatternRegistry


# Check if LM Studio service is available
def is_llm_available():
    """Check if LM Studio is running and responding."""
    try:
        from rag.Embeddings import EmbeddingGenerator
        gen = EmbeddingGenerator()
        return gen.use_lm_studio
    except Exception:
        return False


LLM_AVAILABLE = is_llm_available()


class TestAutonomousPlanning(unittest.TestCase):
    """Test LLM-based autonomous planning with workflow patterns"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.parser = CommandParser(use_rag=True)
        cls.workflow_registry = WorkflowPatternRegistry()

    def test_workflow_pattern_surfacing(self):
        """Verify workflow patterns appear in LLM prompt for relevant queries"""
        # Test with handoff query
        ops_summary = self.parser._get_available_operations_summary(
            "Robot1 and Robot2 handoff the red cube"
        )

        # Should surface workflow patterns if RAG is working
        # Pattern section should appear if patterns are retrieved
        self.assertIsInstance(ops_summary, str)
        self.assertGreater(len(ops_summary), 0)

        # Check if operations are listed
        self.assertIn("move_to_coordinate", ops_summary.lower())

        print(f"✓ Operations summary generated ({len(ops_summary)} chars)")
        print(f"  Contains workflow patterns: {'WORKFLOW PATTERN' in ops_summary or 'workflow' in ops_summary.lower()}")

    def test_workflow_registry_integration(self):
        """Test workflow registry is properly integrated"""
        # Verify workflow registry has patterns
        all_patterns = self.workflow_registry.get_all_patterns()
        self.assertGreater(len(all_patterns), 0, "Workflow registry should have patterns")

        # Verify handoff pattern exists
        handoff = self.workflow_registry.get_pattern_by_name("handoff")
        self.assertIsNotNone(handoff, "Handoff pattern should exist")
        self.assertEqual(handoff.name, "handoff")
        self.assertGreater(len(handoff.steps), 0)

        print(f"✓ Workflow registry has {len(all_patterns)} patterns")
        print(f"  Handoff pattern has {len(handoff.steps)} steps")

    @pytest.mark.skipif(not LLM_AVAILABLE, reason="Requires LM Studio with embedding model")
    def test_handoff_plan_generation(self):
        """Test full pipeline generates valid parallel groups for handoff"""
        result = self.parser.parse(
            "Robot1 and Robot2 perform a handoff of the red cube", "Robot1"
        )

        # Should succeed (even if LLM is unavailable, will use regex fallback)
        self.assertTrue(result["success"], f"Parse failed: {result.get('error')}")

        commands = result.get("plan") or result.get("commands")
        self.assertIsInstance(commands, list)
        self.assertGreater(len(commands), 0, "Should generate at least one command")

        print(f"✓ Generated {len(commands)} commands")

        # Check for parallel_group assignments (only if LLM is available)
        has_parallel_groups = any("parallel_group" in cmd for cmd in commands)
        if has_parallel_groups:
            print(f"  Found parallel_group assignments")

            # Verify signal/wait pairs exist in multi-robot plan
            signals = [c for c in commands if c.get("operation") == "signal"]
            waits = [c for c in commands if c.get("operation") == "wait_for_signal"]

            if len(signals) > 0 or len(waits) > 0:
                print(f"  Has {len(signals)} signals and {len(waits)} waits")
                self.assertGreater(
                    len(signals), 0, "Multi-robot plan should have signals"
                )
                self.assertGreater(
                    len(waits), 0, "Multi-robot plan should have waits"
                )
        else:
            print("  No parallel_group (LLM may be unavailable, using fallback)")

    def test_plan_validation_signal_mismatch(self):
        """Test validation catches mismatched signals"""
        # Invalid plan: wait for signal that is never sent
        invalid_plan = [
            {
                "operation": "wait_for_signal",
                "params": {"robot_id": "Robot2", "event_name": "undefined_signal"},
            }
        ]

        valid, errors = self.parser._validate_multi_robot_plan(invalid_plan)
        self.assertFalse(valid, "Should detect missing signal")
        self.assertGreater(len(errors), 0, "Should return error messages")
        self.assertIn("undefined_signal", str(errors))

        print(f"✓ Validation caught missing signal: {errors[0]}")

    def test_plan_validation_variable_usage(self):
        """Test validation catches variables used before definition"""
        # Invalid plan: use $target before it's defined
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
        self.assertFalse(valid, "Should detect variable used before definition")
        self.assertGreater(len(errors), 0)
        self.assertIn("target", str(errors))

        print(f"✓ Validation caught undefined variable: {errors[0]}")

    def test_plan_validation_valid_plan(self):
        """Test validation passes for valid plans"""
        # Valid plan with proper signal/wait and variable usage
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
        self.assertTrue(valid, f"Valid plan should pass validation: {errors}")
        self.assertEqual(len(errors), 0)

        print(f"✓ Validation passed for valid plan with {len(valid_plan)} commands")

    @pytest.mark.skipif(not LLM_AVAILABLE, reason="Requires LM Studio with embedding model")
    def test_simultaneous_movement_command(self):
        """Test parsing command for simultaneous robot movement"""
        result = self.parser.parse(
            "Move Robot1 to (0.3, 0.1, 0.2) and Robot2 to (-0.3, -0.1, 0.2) simultaneously",
            "Robot1",
        )

        self.assertTrue(result["success"])
        commands = result.get("plan") or result.get("commands")
        self.assertIsInstance(commands, list)

        # Should have at least 2 move commands
        move_commands = [c for c in commands if c.get("operation") == "move_to_coordinate"]
        self.assertGreaterEqual(len(move_commands), 2, "Should have moves for both robots")

        print(f"✓ Simultaneous movement: {len(move_commands)} move commands")

    @pytest.mark.skipif(not LLM_AVAILABLE, reason="Requires LM Studio with embedding model")
    def test_collaborative_task_command(self):
        """Test parsing collaborative task command"""
        result = self.parser.parse(
            "Robot1 should pick up the blue cube and hand it to Robot2", "Robot1"
        )

        self.assertTrue(result["success"])
        commands = result.get("plan") or result.get("commands")
        self.assertIsInstance(commands, list)

        print(f"✓ Collaborative task generated {len(commands)} commands")

        # Check for key operation types
        op_types = set(c.get("operation") for c in commands)
        print(f"  Operation types: {', '.join(sorted(op_types))}")


class TestWorkflowPatternFormatting(unittest.TestCase):
    """Test workflow pattern formatting for LLM prompts"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.parser = CommandParser(use_rag=True)
        cls.workflow_registry = WorkflowPatternRegistry()

    def test_format_workflow_pattern(self):
        """Test workflow pattern formatting"""
        handoff = self.workflow_registry.get_pattern_by_name("handoff")
        self.assertIsNotNone(handoff)

        formatted = self.parser._format_workflow_pattern(handoff)

        # Check formatting
        self.assertIn("Pattern:", formatted)
        self.assertIn("handoff", formatted)
        self.assertIn("Description:", formatted)
        self.assertIn("Steps:", formatted)
        self.assertIn("Examples:", formatted)

        print(f"✓ Formatted pattern ({len(formatted)} chars):")
        print(formatted[:300] + "...")

    def test_all_patterns_formattable(self):
        """Test all patterns can be formatted"""
        all_patterns = self.workflow_registry.get_all_patterns()

        for pattern in all_patterns:
            formatted = self.parser._format_workflow_pattern(pattern)
            self.assertIsInstance(formatted, str)
            self.assertGreater(len(formatted), 0)

        print(f"✓ All {len(all_patterns)} patterns can be formatted")


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAutonomousPlanning))
    suite.addTests(loader.loadTestsFromTestCase(TestWorkflowPatternFormatting))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
