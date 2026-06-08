import pytest
from unittest.mock import MagicMock, patch


def _op_def(name: str) -> MagicMock:
    od = MagicMock()
    od.name = name
    return od


def _make_executor():
    from orchestrators.SequenceExecutor import SequenceExecutor

    ex = SequenceExecutor.__new__(SequenceExecutor)
    ex._variables = {}
    ex.registry = MagicMock()
    ex.world_state = MagicMock()
    ex.verifier = None
    ex.coordination_verifier = None
    ex.outcome_tracker = None
    return ex


class TestCheckSpatialFeasibility:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ex = _make_executor()

    def test_feasibility_returns_safe_when_kg_disabled(self):
        with patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", False):
            result = self.ex._check_spatial_feasibility(
                _op_def("move_to_coordinate"),
                {"position": [0.1, 0.2, 0.3]},
                "Robot1",
            )
        assert result["safe"]

    def test_feasibility_returns_safe_when_engine_none(self):
        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch("core.Imports.get_graph_query_engine", return_value=None),
        ):
            result = self.ex._check_spatial_feasibility(
                _op_def("move_to_coordinate"),
                {"position": [0.1, 0.2, 0.3]},
                "Robot1",
            )
        assert result["safe"]

    def test_move_op_safe_when_path_not_blocked(self):
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

        assert result["safe"]
        mock_qe.is_path_blocked.assert_called_once_with("Robot1", (0.3, 0.2, 0.1))

    def test_move_op_blocked_when_path_blocked(self):
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

        assert not result["safe"]
        assert "blocked" in result["warning"].lower()

    def test_move_op_xyz_params(self):
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
        assert result["safe"]

    def test_grasp_op_safe_when_robot_in_reachable_list(self):
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

        assert result["safe"]

    def test_grasp_op_blocked_when_robot_not_in_reachable_list(self):
        mock_qe = MagicMock()
        mock_qe.find_reachable_robots.return_value = ["Robot2"]

        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch("core.Imports.get_graph_query_engine", return_value=mock_qe),
        ):
            result = self.ex._check_spatial_feasibility(
                _op_def("grasp_object"),
                {"object_id": "red_cube", "robot_id": "Robot1"},
                "Robot1",
            )

        assert not result["safe"]
        assert "Robot1" in result["warning"]

    def test_grasp_op_safe_when_reachable_list_empty(self):
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

        assert result["safe"]

    def test_feasibility_safe_on_exception(self):
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

        assert result["safe"]
        assert "skipped" in result.get("warning", "").lower()


class TestGetHandoffContext:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ex = _make_executor()

    def test_no_context_for_non_handoff_command(self):
        result = self.ex._get_handoff_context("move red cube to table", "Robot1")
        assert result is None

    def test_returns_none_when_kg_disabled(self):
        with patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", False):
            result = self.ex._get_handoff_context("hand cube to Robot2", "Robot1")
        assert result is None

    def test_returns_none_when_engine_none(self):
        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch("core.Imports.get_graph_query_engine", return_value=None),
        ):
            result = self.ex._get_handoff_context("pass cube to Robot2", "Robot1")
        assert result is None

    def test_returns_none_when_no_matching_object(self):
        mock_qe = MagicMock()
        mock_kg = MagicMock()
        mock_kg.get_all_nodes.return_value = ["green_sphere"]

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
        assert result is None

    def test_returns_candidates_when_handoff_keyword_present(self):
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

        assert result is not None
        assert result["handoff_object"] == "red_cube"
        assert isinstance(result["handoff_candidates"], list)
        mock_qe.get_handoff_candidates.assert_called_once_with(
            "Robot1", "Robot2", "red_cube"
        )

    def test_robot2_uses_robot1_as_other(self):
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
        with (
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", True),
            patch(
                "core.Imports.get_graph_query_engine", side_effect=RuntimeError("boom")
            ),
        ):
            result = self.ex._get_handoff_context("transfer cube", "Robot1")
        assert result is None
