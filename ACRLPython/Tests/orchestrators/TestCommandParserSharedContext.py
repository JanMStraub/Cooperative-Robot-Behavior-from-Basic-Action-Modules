from unittest.mock import MagicMock, patch
import pytest


def _make_parser(mock_ws=None):
    """Build a CommandParser._get_peer_context caller with a mocked WorldState."""
    ws = mock_ws or MagicMock()
    with patch("core.Imports.get_world_state", return_value=ws):
        from orchestrators.CommandParser import CommandParser

        parser = CommandParser.__new__(CommandParser)
        parser._world_state = ws  # type: ignore[attr-defined]
        return parser, ws


class TestPeerContext:
    def test_peer_context_includes_peer_robot(self):
        r1 = MagicMock()
        r1.robot_id = "Robot1"
        r1.moving_toward_object = None

        r2 = MagicMock()
        r2.robot_id = "Robot2"
        r2.moving_toward_object = None

        mock_ws = MagicMock()
        mock_ws.get_all_robots.return_value = [r1, r2]
        mock_ws.get_robot_intents.return_value = {}
        mock_ws.get_world_context_string.side_effect = lambda rid: f"{rid} context"

        parser, _ = _make_parser(mock_ws)

        with patch("core.Imports.get_world_state", return_value=mock_ws):
            ctx = parser._get_peer_context("Robot1")

        assert "Robot2" in ctx
        # Robot1 should not appear (it's the planning robot)
        assert "Robot1 context" not in ctx

    def test_peer_context_includes_intent_warning(self):
        mock_ws = MagicMock()
        mock_ws.get_all_robots.return_value = []
        mock_ws.get_robot_intents.return_value = {"Robot2": "red_cube"}
        mock_ws.get_world_context_string.return_value = ""

        parser, _ = _make_parser(mock_ws)

        with patch("core.Imports.get_world_state", return_value=mock_ws):
            ctx = parser._get_peer_context("Robot1")

        assert "Robot2" in ctx
        assert "red_cube" in ctx
        assert "WARNING" in ctx

    def test_peer_context_empty_when_no_peers(self):
        mock_ws = MagicMock()
        mock_ws.get_all_robots.return_value = []
        mock_ws.get_robot_intents.return_value = {}
        mock_ws.get_world_context_string.return_value = ""

        parser, _ = _make_parser(mock_ws)

        with patch("core.Imports.get_world_state", return_value=mock_ws):
            ctx = parser._get_peer_context("Robot1")

        assert ctx == ""

    def test_peer_context_empty_when_worldstate_none(self):
        parser, _ = _make_parser()

        with patch("core.Imports.get_world_state", return_value=None):
            ctx = parser._get_peer_context("Robot1")

        assert ctx == ""

    def test_peer_context_does_not_raise_on_exception(self):
        mock_ws = MagicMock()
        mock_ws.get_all_robots.side_effect = RuntimeError("boom")

        parser, _ = _make_parser(mock_ws)

        with patch("core.Imports.get_world_state", return_value=mock_ws):
            ctx = parser._get_peer_context("Robot1")

        assert isinstance(ctx, str)  # graceful degrade
