#!/usr/bin/env python3
"""
Tests for CommandParser Knowledge Graph Integration
=====================================================

Validates that _get_spatial_context():
- Returns empty string when KG is disabled
- Returns empty string when the query engine is unavailable
- Formats reachable objects and nearby robots correctly
- Caps the reachable object list at 5 entries
- Suppresses all exceptions and degrades gracefully
"""

import unittest
from unittest.mock import MagicMock, patch


class TestCommandParserKG(unittest.TestCase):
    """Tests for CommandParser._get_spatial_context() KG integration."""

    def _make_parser(self):
        """Create a CommandParser with all heavy dependencies mocked out."""
        with patch("orchestrators.CommandParser.RAGSystem", MagicMock()):
            from orchestrators.CommandParser import CommandParser

            parser = CommandParser.__new__(CommandParser)
            parser.rag = None
            return parser

    def test_spatial_context_returns_empty_when_kg_disabled(self):
        """Returns empty string when KNOWLEDGE_GRAPH_ENABLED is False."""
        parser = self._make_parser()
        with patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", False):
            result = parser._get_spatial_context("Robot1")
        self.assertEqual(result, "")

    def test_spatial_context_returns_empty_when_engine_is_none(self):
        """Returns empty string when get_graph_query_engine() returns None."""
        parser = self._make_parser()
        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch("core.Imports.get_graph_query_engine", return_value=None),
        ):
            result = parser._get_spatial_context("Robot1")
        self.assertEqual(result, "")

    def test_spatial_context_formats_reachable_objects(self):
        """Formats reachable objects with distance and color."""
        parser = self._make_parser()
        mock_qe = MagicMock()
        mock_qe.get_objects_in_reach.return_value = [
            {
                "object_id": "red_cube",
                "distance": 0.45,
                "color": "red",
                "grasped_by": None,
            },
        ]
        mock_qe.find_robots_near.return_value = []

        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch("core.Imports.get_graph_query_engine", return_value=mock_qe),
        ):
            result = parser._get_spatial_context("Robot1")

        self.assertIn("red_cube", result)
        self.assertIn("0.45m", result)
        self.assertIn("red", result)
        self.assertIn("SPATIAL CONTEXT", result)

    def test_spatial_context_formats_held_object(self):
        """Appends [held by X] annotation when object is grasped."""
        parser = self._make_parser()
        mock_qe = MagicMock()
        mock_qe.get_objects_in_reach.return_value = [
            {
                "object_id": "blue_cube",
                "distance": 0.3,
                "color": "blue",
                "grasped_by": "Robot2",
            },
        ]
        mock_qe.find_robots_near.return_value = []

        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch("core.Imports.get_graph_query_engine", return_value=mock_qe),
        ):
            result = parser._get_spatial_context("Robot1")

        self.assertIn("held by Robot2", result)

    def test_spatial_context_caps_at_five_objects(self):
        """Only the first 5 reachable objects are included in the context."""
        parser = self._make_parser()
        mock_qe = MagicMock()
        mock_qe.get_objects_in_reach.return_value = [
            {
                "object_id": f"obj_{i}",
                "distance": float(i) * 0.1,
                "color": "red",
                "grasped_by": None,
            }
            for i in range(10)
        ]
        mock_qe.find_robots_near.return_value = []

        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch("core.Imports.get_graph_query_engine", return_value=mock_qe),
        ):
            result = parser._get_spatial_context("Robot1")

        # Only obj_0 through obj_4 should appear
        for i in range(5):
            self.assertIn(f"obj_{i}", result)
        for i in range(5, 10):
            self.assertNotIn(f"obj_{i}", result)

    def test_spatial_context_formats_nearby_robots(self):
        """Formats nearby robots with distance."""
        parser = self._make_parser()
        mock_qe = MagicMock()
        mock_qe.get_objects_in_reach.return_value = []
        mock_qe.find_robots_near.return_value = [
            {"robot_id": "Robot2", "distance": 0.18},
        ]

        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch("core.Imports.get_graph_query_engine", return_value=mock_qe),
        ):
            result = parser._get_spatial_context("Robot1")

        # No objects → header only, so should return empty (no data)
        # But nearby robot was present — expect it in result
        self.assertIn("Robot2", result)
        self.assertIn("0.18m", result)

    def test_spatial_context_suppresses_exceptions(self):
        """Returns empty string when any exception occurs inside the method."""
        parser = self._make_parser()
        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch(
                "core.Imports.get_graph_query_engine", side_effect=RuntimeError("boom")
            ),
        ):
            result = parser._get_spatial_context("Robot1")
        self.assertEqual(result, "")

    def test_spatial_context_returns_empty_when_no_data(self):
        """Returns empty string when KG has no objects and no nearby robots."""
        parser = self._make_parser()
        mock_qe = MagicMock()
        mock_qe.get_objects_in_reach.return_value = []
        mock_qe.find_robots_near.return_value = []

        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch("core.Imports.get_graph_query_engine", return_value=mock_qe),
        ):
            result = parser._get_spatial_context("Robot1")

        self.assertEqual(result, "")

    def test_spatial_context_includes_handoff_candidates(self):
        """Handoff candidates are formatted when both robots can reach the object."""
        parser = self._make_parser()
        mock_qe = MagicMock()
        mock_qe.get_objects_in_reach.return_value = [
            {
                "object_id": "red_cube",
                "distance": 0.4,
                "color": "red",
                "grasped_by": None,
            }
        ]
        mock_qe.find_robots_near.return_value = []
        mock_qe._graph.get_all_nodes.return_value = ["Robot1", "Robot2"]
        mock_qe.get_handoff_candidates.return_value = [
            {
                "position": (0.0, 0.3, 0.0),
                "region": "shared_zone",
                "r1_distance": 0.4,
                "r2_distance": 0.4,
            }
        ]

        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch("core.Imports.get_graph_query_engine", return_value=mock_qe),
        ):
            result = parser._get_spatial_context("Robot1")

        self.assertIn("Handoff red_cube with Robot2", result)
        self.assertIn("r1=0.40m", result)
        self.assertIn("r2=0.40m", result)

    def test_spatial_context_no_handoff_when_empty_candidates(self):
        """No handoff line added when get_handoff_candidates returns empty list."""
        parser = self._make_parser()
        mock_qe = MagicMock()
        mock_qe.get_objects_in_reach.return_value = [
            {
                "object_id": "red_cube",
                "distance": 0.4,
                "color": "red",
                "grasped_by": None,
            }
        ]
        mock_qe.find_robots_near.return_value = []
        mock_qe._graph.get_all_nodes.return_value = ["Robot1", "Robot2"]
        mock_qe.get_handoff_candidates.return_value = []

        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch("core.Imports.get_graph_query_engine", return_value=mock_qe),
        ):
            result = parser._get_spatial_context("Robot1")

        self.assertNotIn("Handoff", result)

    def test_spatial_context_path_clear(self):
        """Path line shows 'clear' when is_path_blocked returns False."""
        parser = self._make_parser()
        mock_qe = MagicMock()
        mock_qe.get_objects_in_reach.return_value = [
            {
                "object_id": "red_cube",
                "distance": 0.4,
                "color": "red",
                "grasped_by": None,
            }
        ]
        mock_qe.find_robots_near.return_value = []
        mock_qe._graph.get_all_nodes.return_value = ["Robot1"]
        mock_qe.get_handoff_candidates.return_value = []
        mock_qe.is_path_blocked.return_value = False

        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch("core.Imports.get_graph_query_engine", return_value=mock_qe),
        ):
            result = parser._get_spatial_context("Robot1", target=(0.3, 0.2, 0.1))

        self.assertIn("Path to target: clear", result)
        mock_qe.is_path_blocked.assert_called_once_with("Robot1", (0.3, 0.2, 0.1))

    def test_spatial_context_path_blocked(self):
        """Path line shows 'BLOCKED' when is_path_blocked returns True."""
        parser = self._make_parser()
        mock_qe = MagicMock()
        mock_qe.get_objects_in_reach.return_value = [
            {
                "object_id": "red_cube",
                "distance": 0.4,
                "color": "red",
                "grasped_by": None,
            }
        ]
        mock_qe.find_robots_near.return_value = []
        mock_qe._graph.get_all_nodes.return_value = ["Robot1"]
        mock_qe.get_handoff_candidates.return_value = []
        mock_qe.is_path_blocked.return_value = True

        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch("core.Imports.get_graph_query_engine", return_value=mock_qe),
        ):
            result = parser._get_spatial_context("Robot1", target=(0.3, 0.2, 0.1))

        self.assertIn("Path to target: BLOCKED", result)

    def test_spatial_context_no_path_check_when_no_target(self):
        """No path line when target is not provided."""
        parser = self._make_parser()
        mock_qe = MagicMock()
        mock_qe.get_objects_in_reach.return_value = [
            {
                "object_id": "red_cube",
                "distance": 0.4,
                "color": "red",
                "grasped_by": None,
            }
        ]
        mock_qe.find_robots_near.return_value = []
        mock_qe._graph.get_all_nodes.return_value = ["Robot1"]
        mock_qe.get_handoff_candidates.return_value = []

        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch("core.Imports.get_graph_query_engine", return_value=mock_qe),
        ):
            result = parser._get_spatial_context("Robot1")

        self.assertNotIn("Path to target", result)
        mock_qe.is_path_blocked.assert_not_called()


if __name__ == "__main__":
    unittest.main()
