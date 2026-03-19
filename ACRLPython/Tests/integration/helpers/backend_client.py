#!/usr/bin/env python3
"""
Backend Client Helper
=====================

Shared Protocol V2 TCP client for integration tests that communicate with
the live ACRL Python backend over the SequenceServer (port 5011).

This module is the single source of truth for the BackendClient class.
Both TestUnityIntegration.py and TestAllOperations.py import from here to
avoid divergent copies of the same Protocol V2 framing logic.

Why route through the SequenceServer (not direct operation calls)?
  Importing Python operations directly in a test process would instantiate
  an uninitialised CommandBroadcaster singleton in that test process, which
  has no active Unity connection.  By sending commands over the network to
  the already-running backend process we re-use its correctly-initialised
  singletons (CommandBroadcaster, WorldStateManager, OutcomeTracker, etc.).

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


# ---------------------------------------------------------------------------
# Port availability helpers
# ---------------------------------------------------------------------------

def port_open(port: int, timeout: float = 2.0) -> bool:
    """
    Return True if a TCP server is accepting connections on *port*.

    Args:
        port: TCP port number to probe.
        timeout: Socket connect timeout in seconds.

    Returns:
        True if the port is open and accepting connections, False otherwise.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex(("localhost", port))
        sock.close()
        return result == 0
    except Exception:
        return False


def backend_available() -> bool:
    """
    Return True when both the CommandServer (5007) and SequenceServer (5011)
    are reachable.

    We probe port 5007 as a proxy for Unity being connected — that port is
    only active once Unity has registered with the backend.  Port 5011 is
    the SequenceServer that tests actually send commands to.

    Returns:
        True if both ports are reachable, False otherwise.
    """
    return port_open(5007) and port_open(5011)


# ---------------------------------------------------------------------------
# Protocol V2 client
# ---------------------------------------------------------------------------

class BackendClient:
    """
    Minimal Protocol V2 TCP client that talks to the SequenceServer (port 5011).

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
    PORT: int = 5011

    def __init__(self, timeout: float = 30.0) -> None:
        """
        Connect to the SequenceServer.

        Args:
            timeout: Socket timeout in seconds.  Choose based on the slowest
                     operation category you are testing:
                     - Status / sync / gripper : 15 s
                     - Navigation              : 30 s
                     - Grasp (full pipeline)   : 60 s
                     - Multi-robot / collab    : 120 s
        """
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(timeout)
        self._sock.connect(("localhost", self.PORT))

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_command(
        self,
        command: str,
        robot_id: str = "Robot1",
        camera_id: str = "TableStereoCamera",
        auto_execute: bool = True,
        request_id: int = 1,
    ) -> Dict[str, Any]:
        """
        Send a command to the backend SequenceServer and return the JSON response.

        Args:
            command: Command string forwarded to CommandParser.  Use explicit
                     structured phrasing (e.g. "check robot status for Robot1")
                     rather than vague natural language to avoid triggering
                     dynamic operation generation (RAG score < 0.4).
            robot_id: Target robot identifier (e.g. "Robot1", "Robot2").
            camera_id: Camera identifier for vision operations.
                       Use "TableStereoCamera" for stereo / field ops.
            auto_execute: When True the backend executes the parsed ops
                          immediately; when False it only parses them.
            request_id: Correlation ID for Protocol V2 request/response
                        matching.  Must be unique per open connection.

        Returns:
            Decoded JSON response dict.  Always contains a "success" key
            (bool).  On failure also contains an "error" key with "code"
            and "message" sub-keys.
        """
        self._send(command, robot_id, camera_id, auto_execute, request_id)
        return self._recv(request_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_str(self, s: str) -> bytes:
        """
        Encode *s* as [len:4 LE][utf-8 bytes].

        Args:
            s: String to encode.

        Returns:
            Bytes representing the length-prefixed string.
        """
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
        """
        Build and transmit a SEQUENCE_QUERY message.

        Args:
            command: Command string.
            robot_id: Robot identifier.
            camera_id: Camera identifier.
            auto_execute: Execution flag.
            request_id: Protocol V2 correlation ID.
        """
        header = struct.pack("B", self.SEQUENCE_QUERY)   # type byte
        header += struct.pack("<I", request_id)           # request_id (4 bytes LE)
        body = (
            self._encode_str(command)
            + self._encode_str(robot_id)
            + self._encode_str(camera_id)
            + struct.pack("B", 1 if auto_execute else 0)  # auto_execute flag
        )
        self._sock.sendall(header + body)

    def _recv_exact(self, n: int) -> bytes:
        """
        Read exactly *n* bytes from the socket.

        Args:
            n: Number of bytes to read.

        Returns:
            Exactly *n* bytes of data.

        Raises:
            ConnectionError: If the remote side closes the connection early.
        """
        data = b""
        while len(data) < n:
            chunk = self._sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Connection closed by backend")
            data += chunk
        return data

    def _recv(self, expected_request_id: int) -> Dict[str, Any]:
        """
        Read a RESULT response frame and decode the JSON payload.

        Args:
            expected_request_id: The request_id we are waiting for.
                                 Currently used only for validation logging;
                                 the SequenceServer is single-request-per-
                                 connection so no multiplexing is needed.

        Returns:
            Decoded JSON response dict.

        Raises:
            ValueError: If the response message type byte is not 0x02 (RESULT).
            ConnectionError: If the connection drops mid-read.
        """
        # Header: [type:1][request_id:4]
        header = self._recv_exact(5)
        msg_type = header[0]
        if msg_type != self.RESULT:
            raise ValueError(f"Unexpected response type: {msg_type:#04x}")

        # JSON payload: [json_len:4][json:N]
        json_len = struct.unpack("<I", self._recv_exact(4))[0]
        json_bytes = self._recv_exact(json_len)
        return json.loads(json_bytes.decode("utf-8"))
