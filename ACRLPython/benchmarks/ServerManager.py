#!/usr/bin/env python3
"""
ServerManager — subprocess lifecycle for live benchmark runs.

Spawns RunRobotController as a child process group, polls ports 5007+5008
until ready, and tears down on stop() or context manager exit.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import Optional


def port_open(port: int, timeout: float = 1.0) -> bool:
    """Return True if a TCP server is accepting connections on port."""
    import socket

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex(("localhost", port))
        sock.close()
        return result == 0
    except Exception:
        return False


_READINESS_PORTS = (5007, 5008)


class ServerManager:
    """
    Manage RunRobotController subprocess lifetime for live benchmarks.

    Usage::

        with ServerManager(startup_timeout=60.0) as mgr:
            runner.run(benchmark_id, cfg)
    """

    def __init__(
        self,
        startup_timeout: float = 60.0,
        poll_interval: float = 1.0,
        extra_args: Optional[list] = None,
    ) -> None:
        """
        Configure server manager.
        """
        self._startup_timeout = startup_timeout
        self._poll_interval = poll_interval
        self._extra_args = extra_args or []
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        """
        Spawn RunRobotController and block until ports 5007+5008 are open.

        Raises:
            RuntimeError: If the process exits early or ports don't open
                          within startup_timeout.
        """
        cmd = [
            sys.executable,
            "-m",
            "orchestrators.RunRobotController",
            "--no-web",
            *self._extra_args,
        ]
        self._proc = subprocess.Popen(
            cmd,
            start_new_session=True,  # own process group → clean SIGTERM
        )

        if not self._wait_for_ports():
            self.stop()
            raise RuntimeError(
                f"Servers did not become ready within {self._startup_timeout}s. "
                "Check that Unity is running and connected."
            )

    def stop(self) -> None:
        """
        Terminate the server process group and wait for it to exit.

        Safe to call multiple times.
        """
        if self._proc is None:
            return
        pid = self._proc.pid
        try:
            pgid = os.getpgid(pid)
        except Exception:
            pgid = pid  # fallback: treat pid as its own group
        try:
            os.killpg(pgid, signal.SIGTERM)
        except Exception:
            pass
        try:
            self._proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            try:
                pgid = os.getpgid(pid)
            except Exception:
                pgid = pid
            try:
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                pass
            self._proc.wait()
        self._proc = None

    def __enter__(self) -> "ServerManager":
        """Start servers on context entry."""
        self.start()
        return self

    def __exit__(self, *_) -> None:
        """Stop servers on context exit (even on exception)."""
        self.stop()

    def _wait_for_ports(self) -> bool:
        """
        Poll _READINESS_PORTS until all open or timeout.
        """
        deadline = time.monotonic() + self._startup_timeout
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                return False  # process died
            if all(port_open(p) for p in _READINESS_PORTS):
                return True
            time.sleep(self._poll_interval)
        return False
