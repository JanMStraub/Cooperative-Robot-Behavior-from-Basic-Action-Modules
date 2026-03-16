#!/usr/bin/env python3
"""
Tests for SequenceExecutor Knowledge Graph Integration
=======================================================

Validates:
- _check_spatial_feasibility(): move-op path blocking, grasp reachability guard
- _get_handoff_context(): handoff keyword detection, candidate building
"""

import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helper: minimal op_def stub
# ---------------------------------------------------------------------------


def _op_def(name: str) -> MagicMock:
    """Create a minimal operation definition mock with the given name."""
    od = MagicMock()
    od.name = name
    return od


# ---------------------------------------------------------------------------
# Helper: create SequenceExecutor without real dependencies
# ---------------------------------------------------------------------------


def _make_executor():
    """Instantiate SequenceExecutor bypassing __init__ heavy setup."""
    from orchestrators.SequenceExecutor import SequenceExecutor

    ex = SequenceExecutor.__new__(SequenceExecutor)
    ex._variables = {}
    ex.registry = MagicMock()
    ex.world_state = MagicMock()
    ex.verifier = None
    ex.coordination_verifier = None
    ex.outcome_tracker = None
    return ex


# ===========================================================================
# _check_spatial_feasibility tests
# ===========================================================================


class TestCheckSpatialFeasibility(unittest.TestCase):
    """Tests for SequenceExecutor._check_spatial_feasibility()."""

    def setUp(self):
        self.ex = _make_executor()

    def test_feasibility_returns_safe_when_kg_disabled(self):
        """Returns safe=True immediately when KG is disabled."""
        with patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", False):
            result = self.ex._check_spatial_feasibility(
                _op_def("move_to_coordinate"),
                {"position": [0.1, 0.2, 0.3]},
                "Robot1",
            )
        self.assertTrue(result["safe"])

    def test_feasibility_returns_safe_when_engine_none(self):
        """Returns safe=True when query engine is not available."""
        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch("core.Imports.get_graph_query_engine", return_value=None),
        ):
            result = self.ex._check_spatial_feasibility(
                _op_def("move_to_coordinate"),
                {"position": [0.1, 0.2, 0.3]},
                "Robot1",
            )
        self.assertTrue(result["safe"])

    def test_move_op_safe_when_path_not_blocked(self):
        """Returns safe=True for a move op when is_path_blocked is False."""
        mock_qe = MagicMock()
        mock_qe.is_path_blocked.return_value = False

        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch("core.Imports.get_graph_query_engine", return_value=mock_qe),
        ):
            result = self.ex._check_spatial_feasibility(
                _op_def("move_to_coordinate"),
                {"position": [0.3, 0.2, 0.1]},
                "Robot1",
            )

        self.assertTrue(result["safe"])
        mock_qe.is_path_blocked.assert_called_once_with("Robot1", (0.3, 0.2, 0.1))

    def test_move_op_blocked_when_path_blocked(self):
        """Returns safe=False when is_path_blocked returns True."""
        mock_qe = MagicMock()
        mock_qe.is_path_blocked.return_value = True

        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch("core.Imports.get_graph_query_engine", return_value=mock_qe),
        ):
            result = self.ex._check_spatial_feasibility(
                _op_def("move_to_coordinate"),
                {"position": [0.3, 0.2, 0.1]},
                "Robot1",
            )

        self.assertFalse(result["safe"])
        self.assertIn("blocked", result["warning"].lower())

    def test_move_op_xyz_params(self):
        """Accepts x/y/z individual params in addition to position list."""
        mock_qe = MagicMock()
        mock_qe.is_path_blocked.return_value = False

        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch("core.Imports.get_graph_query_engine", return_value=mock_qe),
        ):
            result = self.ex._check_spatial_feasibility(
                _op_def("move_from_a_to_b"),
                {"x": 0.1, "y": 0.2, "z": 0.3},
                "Robot1",
            )

        mock_qe.is_path_blocked.assert_called_once_with("Robot1", (0.1, 0.2, 0.3))
        self.assertTrue(result["safe"])

    def test_grasp_op_safe_when_robot_in_reachable_list(self):
        """Returns safe=True when the robot is in find_reachable_robots result."""
        mock_qe = MagicMock()
        mock_qe.find_reachable_robots.return_value = ["Robot1", "Robot2"]

        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch("core.Imports.get_graph_query_engine", return_value=mock_qe),
        ):
            result = self.ex._check_spatial_feasibility(
                _op_def("grasp_object"),
                {"object_id": "red_cube", "robot_id": "Robot1"},
                "Robot1",
            )

        self.assertTrue(result["safe"])

    def test_grasp_op_blocked_when_robot_not_in_reachable_list(self):
        """Returns safe=False when reachable list is populated but excludes robot."""
        mock_qe = MagicMock()
        mock_qe.find_reachable_robots.return_value = ["Robot2"]  # only Robot2

        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch("core.Imports.get_graph_query_engine", return_value=mock_qe),
        ):
            result = self.ex._check_spatial_feasibility(
                _op_def("grasp_object"),
                {"object_id": "red_cube", "robot_id": "Robot1"},
                "Robot1",
            )

        self.assertFalse(result["safe"])
        self.assertIn("Robot1", result["warning"])

    def test_grasp_op_safe_when_reachable_list_empty(self):
        """Returns safe=True when the reachable list is empty (KG not populated)."""
        mock_qe = MagicMock()
        mock_qe.find_reachable_robots.return_value = []

        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch("core.Imports.get_graph_query_engine", return_value=mock_qe),
        ):
            result = self.ex._check_spatial_feasibility(
                _op_def("grasp_object"),
                {"object_id": "red_cube", "robot_id": "Robot1"},
                "Robot1",
            )

        self.assertTrue(result["safe"])

    def test_feasibility_safe_on_exception(self):
        """Returns safe=True with warning when an exception occurs."""
        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch(
                "core.Imports.get_graph_query_engine", side_effect=RuntimeError("boom")
            ),
        ):
            result = self.ex._check_spatial_feasibility(
                _op_def("move_to_coordinate"),
                {"position": [0.1, 0.2, 0.3]},
                "Robot1",
            )

        self.assertTrue(result["safe"])
        self.assertIn("skipped", result.get("warning", "").lower())


# ===========================================================================
# _get_handoff_context tests
# ===========================================================================


class TestGetHandoffContext(unittest.TestCase):
    """Tests for SequenceExecutor._get_handoff_context()."""

    def setUp(self):
        self.ex = _make_executor()

    def test_no_context_for_non_handoff_command(self):
        """Returns None when no handoff keyword is present."""
        result = self.ex._get_handoff_context("move red cube to table", "Robot1")
        self.assertIsNone(result)

    def test_returns_none_when_kg_disabled(self):
        """Returns None when KG is disabled even if keyword present."""
        with patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", False):
            result = self.ex._get_handoff_context("hand cube to Robot2", "Robot1")
        self.assertIsNone(result)

    def test_returns_none_when_engine_none(self):
        """Returns None when query engine unavailable."""
        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch("core.Imports.get_graph_query_engine", return_value=None),
        ):
            result = self.ex._get_handoff_context("pass cube to Robot2", "Robot1")
        self.assertIsNone(result)

    def test_returns_none_when_no_matching_object(self):
        """Returns None when no object in KG matches the command text."""
        mock_qe = MagicMock()
        mock_kg = MagicMock()
        mock_kg.get_all_nodes.return_value = ["green_sphere"]  # not in command

        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch("core.Imports.get_graph_query_engine", return_value=mock_qe),
            patch(
                "knowledge_graph._singleton.get_knowledge_graph", return_value=mock_kg
            ),
        ):
            result = self.ex._get_handoff_context(
                "transfer the cube to Robot2", "Robot1"
            )
        self.assertIsNone(result)

    def test_returns_candidates_when_handoff_keyword_present(self):
        """Returns a dict with handoff_candidates when keyword + object match."""
        mock_qe = MagicMock()
        mock_qe.get_handoff_candidates.return_value = [
            {
                "position": (0.0, 0.3, 0.1),
                "region": "shared_zone",
                "r1_distance": 0.4,
                "r2_distance": 0.4,
            }
        ]
        mock_kg = MagicMock()
        mock_kg.get_all_nodes.return_value = ["red_cube"]

        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch("core.Imports.get_graph_query_engine", return_value=mock_qe),
            patch(
                "knowledge_graph._singleton.get_knowledge_graph", return_value=mock_kg
            ),
        ):
            result = self.ex._get_handoff_context("hand red_cube to Robot2", "Robot1")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["handoff_object"], "red_cube")
        self.assertIsInstance(result["handoff_candidates"], list)
        mock_qe.get_handoff_candidates.assert_called_once_with(
            "Robot1", "Robot2", "red_cube"
        )

    def test_robot2_uses_robot1_as_other(self):
        """Robot2 queries handoff candidates against Robot1."""
        mock_qe = MagicMock()
        mock_qe.get_handoff_candidates.return_value = []
        mock_kg = MagicMock()
        mock_kg.get_all_nodes.return_value = ["blue_cube"]

        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch("core.Imports.get_graph_query_engine", return_value=mock_qe),
            patch(
                "knowledge_graph._singleton.get_knowledge_graph", return_value=mock_kg
            ),
        ):
            self.ex._get_handoff_context("give blue_cube to Robot1", "Robot2")

        mock_qe.get_handoff_candidates.assert_called_once_with(
            "Robot2", "Robot1", "blue_cube"
        )

    def test_returns_none_on_exception(self):
        """Returns None when any exception occurs (graceful degrade)."""
        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch(
                "core.Imports.get_graph_query_engine", side_effect=RuntimeError("boom")
            ),
        ):
            result = self.ex._get_handoff_context("transfer cube", "Robot1")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
