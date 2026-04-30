# ACRLPython/tests/benchmarks/test_server_manager.py
from unittest.mock import MagicMock, patch
import pytest
from benchmarks.server_manager import ServerManager


def test_wait_for_ports_success():
    """port_open returns True twice → ready immediately."""
    mgr = ServerManager(startup_timeout=5.0)
    with patch("benchmarks.server_manager.port_open", return_value=True):
        assert mgr._wait_for_ports() is True


def test_wait_for_ports_timeout():
    """port_open always False → times out and returns False."""
    mgr = ServerManager(startup_timeout=0.1, poll_interval=0.05)
    with patch("benchmarks.server_manager.port_open", return_value=False):
        assert mgr._wait_for_ports() is False


def test_start_spawns_subprocess():
    """start() calls subprocess.Popen with correct args."""
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # still alive
    with patch("benchmarks.server_manager.subprocess.Popen", return_value=mock_proc) as mock_popen, \
         patch("benchmarks.server_manager.port_open", return_value=True):
        mgr = ServerManager()
        mgr.start()
        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        assert "RunRobotController" in " ".join(args)
        mgr.stop()


def test_stop_terminates_process():
    """stop() sends SIGTERM to process group."""
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_proc.poll.return_value = None
    with patch("benchmarks.server_manager.subprocess.Popen", return_value=mock_proc), \
         patch("benchmarks.server_manager.port_open", return_value=True), \
         patch("benchmarks.server_manager.os.killpg") as mock_kill:
        mgr = ServerManager()
        mgr.start()
        mgr.stop()
        mock_kill.assert_called_once()


def test_context_manager_stops_on_exit():
    """Exiting context manager calls stop() even on exception."""
    mock_proc = MagicMock()
    mock_proc.pid = 99
    mock_proc.poll.return_value = None
    with patch("benchmarks.server_manager.subprocess.Popen", return_value=mock_proc), \
         patch("benchmarks.server_manager.port_open", return_value=True), \
         patch("benchmarks.server_manager.os.killpg"):
        mgr = ServerManager()
        with pytest.raises(RuntimeError):
            with mgr:
                raise RuntimeError("test error")
        mock_proc.wait.assert_called()
