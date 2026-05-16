#!/usr/bin/env python3
"""
Backend Client Helper
=====================

Shared Protocol V2 TCP client for integration tests that communicate with
the live ACRL Python backend over the SequenceServer (port 5008).

This module is the single source of truth for the BackendClient class.
Both TestUnityIntegration.py and TestAllOperations.py import from here to
avoid divergent copies of the same Protocol V2 framing logic.

Why route through the SequenceServer (not direct operation calls)?
  Importing Python operations directly in a test process would instantiate
  an uninitialised CommandBroadcaster singleton in that test process, which
  has no active Unity connection.  By sending commands over the network to
  the already-running backend process we re-use its correctly-initialised
  singletons (CommandBroadcaster, WorldStateManager, etc.).

Protocol V2 framing (little-endian):
    Request:  [type:1 = 0x08][request_id:4][cmd_len:4][cmd:N]
              [robot_id_len:4][robot_id:N][camera_id_len:4][camera_id:N]
              [auto_execute:1]
    Response: [type:1 = 0x02][request_id:4][json_len:4][json:N]
"""

import json
import socket
import struct
from typing import Any, Dict

# Port availability helpers


def port_open(port: int, timeout: float = 2.0) -> bool:
    """Return True if a TCP server is accepting connections on *port*."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex(("localhost", port))
        sock.close()
        return result == 0
    except Exception:
        return False


def reset_simulation(timeout: float = 20.0) -> Dict[str, Any]:
    """
    Send reset_simulation to the backend and wait for completion.

    Resets all robots to start positions, releases grasped objects, and
    restores all dynamic scene objects to their initial positions.

    """
    import random

    with BackendClient(timeout=timeout) as client:
        return client.send_command(
            command="EXEC:"
            + json.dumps([{"operation": "reset_simulation", "params": {}}]),
            robot_id="system",
            request_id=random.randint(1, 0xFFFFFFFF),
        )


def backend_available() -> bool:
    """
    Return True when both the CommandServer (5007) and SequenceServer (5008)
    are reachable.

    We probe port 5007 as a proxy for Unity being connected — that port is
    only active once Unity has registered with the backend.  Port 5008 is
    the SequenceServer that tests actually send commands to.

    """
    return port_open(5007) and port_open(5008)


# Protocol V2 client


class BackendClient:
    """
    Minimal Protocol V2 TCP client that talks to the SequenceServer (port 5008).

    The SequenceServer receives natural-language or structured commands,
    executes them through the full operations pipeline
    (CommandParser → SequenceExecutor → Operations → CommandBroadcaster → Unity),
    and returns a JSON result.

    Usage::

        with BackendClient(timeout=30.0) as client:
            result = client.send_command(
                command="check robot status for Robot1",
                robot_id="Robot1",
            )
        assert result["success"] is True
    """

    SEQUENCE_QUERY: int = 0x08
    RESULT: int = 0x02
    PORT: int = 5008

    def __init__(self, timeout: float = 30.0) -> None:
        """Connect to the SequenceServer."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(timeout)
        self._sock.connect(("localhost", self.PORT))

    # Context-manager support

    def __enter__(self) -> "BackendClient":
        """Support usage as a context manager."""
        return self

    def __exit__(self, *_: Any) -> None:
        """Close the connection when exiting the context block."""
        self.close()

    def close(self) -> None:
        """Close the underlying TCP socket (idempotent)."""
        try:
            self._sock.close()
        except Exception:
            pass

    # Public API

    def send_command(
        self,
        command: str,
        robot_id: str = "Robot1",
        camera_id: str = "TableStereoCamera",
        auto_execute: bool = True,
        request_id: int = 1,
    ) -> Dict[str, Any]:
        """Send a command to the backend SequenceServer and return the JSON response."""
        self._send(command, robot_id, camera_id, auto_execute, request_id)
        return self._recv(request_id)

    # Private helpers

    def _encode_str(self, s: str) -> bytes:
        """Encode *s* as [len:4 LE][utf-8 bytes]."""
        encoded = s.encode("utf-8")
        return struct.pack("<I", len(encoded)) + encoded

    def _send(
        self,
        command: str,
        robot_id: str,
        camera_id: str,
        auto_execute: bool,
        request_id: int,
    ) -> None:
        """Build and transmit a SEQUENCE_QUERY message."""
        header = struct.pack("B", self.SEQUENCE_QUERY)  # type byte
        header += struct.pack("<I", request_id)  # request_id (4 bytes LE)
        body = (
            self._encode_str(command)
            + self._encode_str(robot_id)
            + self._encode_str(camera_id)
            + struct.pack("B", 1 if auto_execute else 0)  # auto_execute flag
            + struct.pack("<I", 0)  # flags_len = 0 (no feature-flag overrides)
        )
        self._sock.sendall(header + body)

    def _recv_exact(self, n: int) -> bytes:
        """Read exactly *n* bytes from the socket."""
        data = b""
        while len(data) < n:
            chunk = self._sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Connection closed by backend")
            data += chunk
        return data

    def _recv(self, expected_request_id: int) -> Dict[str, Any]:
        """Read a RESULT response frame and decode the JSON payload."""
        # Header: [type:1][request_id:4]
        header = self._recv_exact(5)
        msg_type = header[0]
        if msg_type != self.RESULT:
            raise ValueError(f"Unexpected response type: {msg_type:#04x}")

        # JSON payload: [json_len:4][json:N]
        json_len = struct.unpack("<I", self._recv_exact(4))[0]
        json_bytes = self._recv_exact(json_len)
        return json.loads(json_bytes.decode("utf-8"))
