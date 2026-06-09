from unittest.mock import MagicMock, patch


class TestCommandParserKG:

    def _make_parser(self):
        with patch("orchestrators.CommandParser.RAGSystem", MagicMock()):
            from orchestrators.CommandParser import CommandParser

            parser = CommandParser.__new__(CommandParser)
            parser.rag = None
            return parser

    def test_spatial_context_returns_empty_when_kg_disabled(self):
        parser = self._make_parser()
        with patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", False):
            result = parser._get_spatial_context("Robot1")
        assert result == ""

    def test_spatial_context_returns_empty_when_engine_is_none(self):
        parser = self._make_parser()
        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch("core.Imports.get_graph_query_engine", return_value=None),
        ):
            result = parser._get_spatial_context("Robot1")
        assert result == ""

    def test_spatial_context_formats_reachable_objects(self):
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

        assert "red_cube" in result
        assert "0.45m" in result
        assert "red" in result
        assert "SPATIAL CONTEXT" in result

    def test_spatial_context_formats_held_object(self):
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

        assert "held by Robot2" in result

    def test_spatial_context_caps_at_five_objects(self):
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

        for i in range(5):
            assert f"obj_{i}" in result
        for i in range(5, 10):
            assert f"obj_{i}" not in result

    def test_spatial_context_formats_nearby_robots(self):
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

        assert "Robot2" in result
        assert "0.18m" in result

    def test_spatial_context_suppresses_exceptions(self):
        parser = self._make_parser()
        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch(
                "core.Imports.get_graph_query_engine", side_effect=RuntimeError("boom")
            ),
        ):
            result = parser._get_spatial_context("Robot1")
        assert result == ""

    def test_spatial_context_returns_empty_when_no_data(self):
        parser = self._make_parser()
        mock_qe = MagicMock()
        mock_qe.get_objects_in_reach.return_value = []
        mock_qe.find_robots_near.return_value = []

        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch("core.Imports.get_graph_query_engine", return_value=mock_qe),
        ):
            result = parser._get_spatial_context("Robot1")

        assert result == ""

    def test_spatial_context_includes_handoff_candidates(self):
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
            result = parser._get_spatial_context(
                "Robot1", command_text="handoff red_cube to Robot2"
            )

        assert "Handoff red_cube with Robot2" in result
        assert "r1=0.40m" in result
        assert "r2=0.40m" in result

    def test_spatial_context_no_handoff_when_empty_candidates(self):
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

        assert "Handoff" not in result

    def test_spatial_context_path_clear(self):
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

        assert "Path to target: clear" in result
        mock_qe.is_path_blocked.assert_called_once_with("Robot1", (0.3, 0.2, 0.1))

    def test_spatial_context_path_blocked(self):
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

        assert "Path to target: BLOCKED" in result

    def test_spatial_context_no_path_check_when_no_target(self):
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

        assert "Path to target" not in result
        mock_qe.is_path_blocked.assert_not_called()


class TestCommandParserBetweenPrompt:

    def _get_prompt(self, robot_id="Robot1"):
        with patch("orchestrators.CommandParser.RAGSystem", MagicMock()):
            from orchestrators.CommandParser import _PromptBuilder

            builder = _PromptBuilder(
                registry=MagicMock(), workflow_registry=MagicMock(), rag=None
            )
            return builder.build("place between blue and red cube", robot_id)

    def test_between_section_header_present(self):
        prompt = self._get_prompt()
        assert "BETWEEN PLACEMENT" in prompt

    def test_between_prompt_mentions_arithmetic(self):
        prompt = self._get_prompt()
        assert "$blue_obj" in prompt
        assert "$red_obj" in prompt

    def test_between_prompt_has_midpoint_example(self):
        prompt = self._get_prompt()
        assert "blue_obj.x + $red_obj.x" in prompt
        assert "/ 2" in prompt

    def test_between_prompt_mentions_parallel_groups(self):
        prompt = self._get_prompt()
        assert "parallel_group" in prompt

    def test_between_prompt_uses_two_capture_vars(self):
        prompt = self._get_prompt()
        assert "blue_obj" in prompt
        assert "red_obj" in prompt
