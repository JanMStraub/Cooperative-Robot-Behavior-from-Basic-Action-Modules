#!/usr/bin/env python3
"""
SequenceServer.py - TCP server for multi-command sequence execution

Receives compound commands from Unity, parses them into operation sequences,
and executes them sequentially with completion tracking.

Port: 5008 (SEQUENCE_SERVER_PORT)

Protocol V2:
    Query (Unity → Python):
        [type:1][request_id:4][command_len:4][command_text:N][robot_id_len:4][robot_id:N][camera_id_len:4][camera_id:N][auto_execute:1]

    Response (Python → Unity):
        [type:1][request_id:4][response_len:4][response_json:N]
"""

import socket
import struct
import json
from typing import Dict, Any, Optional, Union, TYPE_CHECKING

# Handle both direct execution and package import
try:
    from ..core.TCPServerBase import TCPServerBase, ServerConfig
    from ..core.LoggingSetup import get_logger
    from ..core.UnityProtocol import MessageType, UnityProtocol
    from ..core.SingletonBase import SingletonBase

    # NOTE: CommandParser and SequenceExecutor imported lazily in initialize() to avoid circular dependency
except ImportError:
    from core.TCPServerBase import TCPServerBase, ServerConfig
    from core.LoggingSetup import get_logger
    from core.UnityProtocol import MessageType, UnityProtocol
    from core.SingletonBase import SingletonBase

    # NOTE: CommandParser and SequenceExecutor imported lazily in initialize() to avoid circular dependency

if TYPE_CHECKING:
    try:
        from ..orchestrators.CommandParser import CommandParser
        from ..orchestrators.SequenceExecutor import SequenceExecutor
    except ImportError:
        from orchestrators.CommandParser import CommandParser
        from orchestrators.SequenceExecutor import SequenceExecutor

try:
    from config.Servers import (
        DEFAULT_HOST,
        SEQUENCE_SERVER_PORT,
        MAX_STRING_LENGTH,
        DEFAULT_LMSTUDIO_MODEL,
    )
    from config.Vision import DEFAULT_CAMERA_ID
except ImportError:
    from ..config.Servers import (
        DEFAULT_HOST,
        SEQUENCE_SERVER_PORT,
        MAX_STRING_LENGTH,
        DEFAULT_LMSTUDIO_MODEL,
    )
    from ..config.Vision import DEFAULT_CAMERA_ID

logger = get_logger(__name__)


class SequenceQueryHandler(SingletonBase):
    """Singleton that owns CommandParser + SequenceExecutor for sequence query processing."""

    def _singleton_init(self):
        self._parser: Optional[CommandParser] = None
        self._executor: Optional[SequenceExecutor] = None

    def initialize(
        self,
        lm_studio_url: Optional[str] = None,
        model: Optional[str] = None,
        check_completion: bool = True,  # Enabled - Unity sends completion signals
    ) -> bool:
        try:
            from core.Imports import get_command_parser, get_sequence_executor

            self._parser = get_command_parser(lm_studio_url=lm_studio_url, model=model)
            self._executor = get_sequence_executor(
                check_completion=check_completion, default_timeout=180.0
            )

            return True
        except Exception as e:
            logger.error(f"Failed to initialize SequenceQueryHandler: {e}")
            return False

    # Prefix used by the test scripts (tools/send_command.py etc.) to send a
    # pre-parsed operation list without going through the LLM.
    # Format: "EXEC:" + JSON-encoded list of {"operation": str, "params": dict}
    DIRECT_EXEC_PREFIX = "EXEC:"

    def execute_sequence(
        self,
        command_text: str,
        robot_id: str = "Robot1",
        camera_id: str = DEFAULT_CAMERA_ID,
        auto_execute: bool = True,
        timeout: float = 120.0,
        flags_json: str = "",
    ) -> Dict[str, Any]:
        """
        Parse and execute a command sequence.

        When command_text starts with ``EXEC:`` the remainder is treated as a
        JSON-encoded list of already-parsed commands
        (``[{"operation": str, "params": dict}, ...]``) and is passed directly
        to the executor, bypassing the LLM entirely.  This is used by test
        scripts (tools/send_command.py) to drive Unity without LLM latency.

        When flags_json is non-empty, feature overrides are applied for the
        duration of this sequence and restored afterwards via FeatureFlagContext.

        timeout default of 120 s covers concurrent dual-robot ops where place_object can take ~55 s.
        """
        from benchmarks.FeatureFlags import BenchmarkFeatureFlags
        from servers.FeatureFlagContext import FeatureFlagContext

        flags = BenchmarkFeatureFlags.from_json(flags_json)
        with FeatureFlagContext(flags):
            return self._execute_sequence_inner(
                command_text, robot_id, camera_id, auto_execute, timeout
            )

    def _execute_sequence_inner(
        self,
        command_text: str,
        robot_id: str = "Robot1",
        camera_id: str = DEFAULT_CAMERA_ID,
        auto_execute: bool = True,
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        if not self._parser or not self._executor:
            return {"success": False, "error": "SequenceQueryHandler not initialized"}

        # ── Direct execution path (no LLM) ────────────────────────────────────
        if command_text.startswith(self.DIRECT_EXEC_PREFIX):
            raw = command_text[len(self.DIRECT_EXEC_PREFIX) :]
            try:
                commands = json.loads(raw)
                if not isinstance(commands, list):
                    return {
                        "success": False,
                        "error": "EXEC: payload must be a JSON array",
                    }
            except json.JSONDecodeError as e:
                return {"success": False, "error": f"EXEC: invalid JSON - {e}"}

            logger.info(
                f"Direct execution (no LLM): {len(commands)} command(s) for {robot_id}"
            )

            if not auto_execute:
                return {
                    "success": True,
                    "parsed_commands": commands,
                    "original_command": command_text,
                    "auto_execute": False,
                    "total_commands": len(commands),
                    "completed_commands": 0,
                    "results": [],
                    "total_duration_ms": 0,
                }

            exec_result = self._executor.execute_sequence(
                commands, timeout_per_command=timeout
            )
            exec_result["parsed_commands"] = commands
            exec_result["original_command"] = command_text
            exec_result["direct_exec"] = True
            return exec_result

        # ── Normal path: LLM parsing ───────────────────────────────────────────
        if self._executor and auto_execute:
            negotiated = self._executor.negotiate_if_needed(command_text, robot_id)
            if negotiated is not None:
                exec_result = self._executor.execute_sequence(
                    negotiated, timeout_per_command=timeout
                )
                exec_result["negotiated"] = True
                exec_result["original_command"] = command_text
                exec_result["parsed_commands"] = negotiated
                try:
                    from core.Imports import get_negotiation_hub

                    _hub = get_negotiation_hub()
                    exec_result["negotiation_rounds"] = (
                        _hub.get_last_round_count() if _hub else 0
                    )
                except Exception:
                    exec_result["negotiation_rounds"] = 0
                return exec_result

        parse_result = self._parser.parse(command_text, robot_id)

        if not parse_result["success"]:
            return {
                "success": False,
                "error": f"Parse failed: {parse_result.get('error')}",
                "commands": [],
            }

        commands = parse_result["commands"]
        if not commands:
            return {
                "success": False,
                "error": "No commands parsed from input",
                "commands": [],
            }

        # Tag each command with the original text so SequenceExecutor can
        # re-parse it with Reflection context on failure.
        for cmd in commands:
            cmd["_original_text"] = command_text

        # Add camera_id to commands that need it (perception operations).
        # detect_objects is intentionally excluded: it reads from single-camera
        # storage (port 5005) using its own "main" default, not the stereo camera.
        perception_ops = [
            "detect_object_stereo",  # Stereo detection with depth (port 5006)
            "analyze_scene",  # LLM vision analysis (single camera)
        ]
        for cmd in commands:
            if cmd.get("operation") in perception_ops:
                if "params" not in cmd:
                    cmd["params"] = {}
                cmd["params"]["camera_id"] = camera_id

        if not auto_execute:
            return {
                "success": True,
                "parsed_commands": commands,
                "original_command": command_text,
                "auto_execute": False,
                "total_commands": len(commands),
                "completed_commands": 0,
                "results": [],
                "total_duration_ms": 0,
            }

        exec_result = self._executor.execute_sequence(
            commands, timeout_per_command=timeout
        )

        exec_result["parsed_commands"] = commands
        exec_result["original_command"] = command_text
        exec_result["camera_id"] = camera_id

        return exec_result

    def is_ready(self) -> bool:
        """Check if handler is ready for queries."""
        return self._parser is not None and self._executor is not None


class SequenceServer(TCPServerBase):
    """TCP server for command sequence execution (port 5008)."""

    def __init__(self, config: Optional[ServerConfig] = None):
        if config is None:
            config = ServerConfig(host=DEFAULT_HOST, port=SEQUENCE_SERVER_PORT)
        super().__init__(config)

    def _pre_connection_setup(self, client: socket.socket, _address: tuple) -> None:
        # Enable TCP keepalives so the OS sends probes during long-running sequence
        # executions (grasp can take 60+ seconds). Without this, Unity's receive
        # thread sees no data for >5s and raises a WouldBlock/socket error, causing
        # it to incorrectly treat the idle-but-alive connection as dropped.
        client.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        try:
            # Linux-specific: start probes after 10s idle, then every 5s, 3 attempts
            # TCP_KEEPIDLE is Linux-only; getattr avoids AttributeError on macOS/Windows
            TCP_KEEPIDLE = getattr(socket, "TCP_KEEPIDLE", None)
            TCP_KEEPINTVL = getattr(socket, "TCP_KEEPINTVL", None)
            TCP_KEEPCNT = getattr(socket, "TCP_KEEPCNT", None)
            if TCP_KEEPIDLE is not None:
                client.setsockopt(socket.IPPROTO_TCP, TCP_KEEPIDLE, 10)
                client.setsockopt(socket.IPPROTO_TCP, TCP_KEEPINTVL, 5)  # type: ignore[arg-type]
                client.setsockopt(socket.IPPROTO_TCP, TCP_KEEPCNT, 3)  # type: ignore[arg-type]
        except OSError:
            pass  # Not available on all platforms (e.g. macOS uses different names)

    def _handle_message(self, client: socket.socket, _address: tuple) -> None:
        # Protocol V2: Read header (type:1 + request_id:4)
        # Note: Uses little-endian per UnityProtocol.py specification
        header_bytes = self._recv_exact(client, 5)
        if not header_bytes:
            raise ConnectionResetError("Connection closed by client")

        msg_type = header_bytes[0]
        request_id = struct.unpack("<I", header_bytes[1:5])[0]

        # Route to appropriate handler based on message type
        if msg_type == MessageType.AUTORT_COMMAND:
            # Handle AutoRT command - pass header bytes for complete message reading
            self._handle_autort_command(client, header_bytes)
            return
        elif msg_type != MessageType.SEQUENCE_QUERY:
            self._logger.error(
                f"Invalid message type: {msg_type} (expected {MessageType.SEQUENCE_QUERY} or {MessageType.AUTORT_COMMAND})"
            )
            self._send_error(client, request_id, f"Invalid message type: {msg_type}")
            return

        cmd_len_bytes = self._recv_exact(client, 4)
        if not cmd_len_bytes:
            raise ConnectionResetError("Connection closed reading command length")
        cmd_len = struct.unpack("<I", cmd_len_bytes)[0]

        if cmd_len > MAX_STRING_LENGTH * 10:  # Allow longer commands
            self._logger.error(f"Command too long: {cmd_len}")
            self._send_error(client, request_id, "Command too long")
            return

        command_bytes = self._recv_exact(client, cmd_len)
        if not command_bytes:
            raise ConnectionResetError("Connection closed reading command text")
        command_text = command_bytes.decode("utf-8")

        robot_id_len_bytes = self._recv_exact(client, 4)
        if not robot_id_len_bytes:
            raise ConnectionResetError("Connection closed reading robot_id length")
        robot_id_len = struct.unpack("<I", robot_id_len_bytes)[0]

        robot_id = "Robot1"
        if robot_id_len > 0:
            robot_id_bytes = self._recv_exact(client, robot_id_len)
            if robot_id_bytes:
                robot_id = robot_id_bytes.decode("utf-8")

        camera_id_len_bytes = self._recv_exact(client, 4)
        if not camera_id_len_bytes:
            raise ConnectionResetError("Connection closed reading camera_id length")
        camera_id_len = struct.unpack("<I", camera_id_len_bytes)[0]

        # Fall back to configured default when Unity sends length=0
        camera_id = DEFAULT_CAMERA_ID
        if camera_id_len > 0:
            camera_id_bytes = self._recv_exact(client, camera_id_len)
            if camera_id_bytes:
                camera_id = camera_id_bytes.decode("utf-8")

        auto_execute_bytes = self._recv_exact(client, 1)
        if not auto_execute_bytes:
            raise ConnectionResetError("Connection closed reading auto_execute flag")
        auto_execute = auto_execute_bytes[0] == 1

        # Read optional feature-flag overrides (benchmark runner only).
        # flags_len=0 or missing (Unity / legacy clients) → no overrides.
        flags_json = ""
        flags_len_bytes = self._recv_exact(client, 4)
        if flags_len_bytes:
            flags_len = struct.unpack("<I", flags_len_bytes)[0]
            if flags_len > 0:
                flag_bytes = self._recv_exact(client, flags_len)
                if flag_bytes:
                    flags_json = flag_bytes.decode("utf-8")

        self._logger.info(
            f"Received sequence query (id={request_id}): {command_text} (camera={camera_id}, auto_execute={auto_execute})"
        )

        handler = SequenceQueryHandler()
        try:
            result = handler.execute_sequence(
                command_text,
                robot_id,
                camera_id,
                auto_execute,
                flags_json=flags_json,
            )
        except Exception as exc:
            self._logger.exception(
                f"Unhandled exception in execute_sequence (id={request_id}): {exc}"
            )
            self._send_error(client, request_id, f"Internal server error: {exc}")
            return

        # Send response
        self._send_response(client, request_id, result)

    def _handle_autort_command(self, client: socket.socket, header_bytes: bytes):
        # Extract request_id from the already-read header so it is always
        # bound, even if decode_autort_command() raises before assigning it.
        request_id = struct.unpack("<I", header_bytes[1:5])[0]
        try:
            try:
                from ..servers.AutoRTIntegration import AutoRTHandler
            except ImportError:
                from servers.AutoRTIntegration import AutoRTHandler

            # Receive complete message using TCPServerBase helper
            complete_message = self._receive_complete_autort_command(
                client, header_bytes
            )
            if not complete_message:
                logger.error("Failed to receive complete AutoRT command")
                return

            # Decode using UnityProtocol (now accepts only bytes)
            request_id, command_type, params = UnityProtocol.decode_autort_command(
                complete_message
            )

            logger.info(f"AutoRT command received: {command_type} (params={params})")

            # Get AutoRT handler singleton
            handler = AutoRTHandler.get_instance()

            # Route command to appropriate method
            if command_type == "generate":
                result = handler.generate_tasks(
                    num_tasks=params.get("num_tasks"),
                    robot_ids=params.get("robot_ids"),
                    strategy=params.get("strategy", "balanced"),
                )
            elif command_type == "start_loop":
                result = handler.start_loop(
                    loop_delay=params.get("loop_delay"),
                    robot_ids=params.get("robot_ids"),
                    strategy=params.get("strategy", "balanced"),
                )
            elif command_type == "stop_loop":
                result = handler.stop_loop()
            elif command_type == "execute_task":
                task_id = params.get("task_id")
                if not task_id:
                    result = {"success": False, "error": "Missing task_id parameter"}
                else:
                    result = handler.execute_task(task_id)
            elif command_type == "get_status":
                result = handler.get_status()
            else:
                result = {
                    "success": False,
                    "error": f"Unknown command type: {command_type}",
                }

            # Send AUTORT_RESPONSE
            self._send_autort_response(client, request_id, result)

        except Exception as e:
            logger.error(f"AutoRT command handling failed: {e}", exc_info=True)
            error_result = {
                "success": False,
                "tasks": [],
                "loop_running": False,
                "error": str(e),
            }
            self._send_autort_response(client, request_id, error_result)

    def _send_autort_response(
        self, client: socket.socket, request_id: int, result: Dict[str, Any]
    ):
        try:
            # Encode using UnityProtocol
            response_bytes = UnityProtocol.encode_autort_response(result, request_id)
            client.sendall(response_bytes)
            logger.info(
                f"Sent AutoRT response for request {request_id}: success={result.get('success')}, status={result.get('status')}"
            )

        except Exception as e:
            logger.error(f"Failed to send AutoRT response: {e}")

    def _send_response(
        self, client: socket.socket, request_id: int, result: Dict[str, Any]
    ):
        """
        Send response to client.

        """
        try:
            # Add request_id to result for Protocol V2 correlation
            result["request_id"] = request_id

            # Encode result as JSON
            result_json = json.dumps(result).encode("utf-8")

            # Note: Uses little-endian per UnityProtocol.py specification
            response = struct.pack("B", MessageType.RESULT)  # type (1 byte)
            response += struct.pack(
                "<I", request_id
            )  # request_id (4 bytes, little-endian)
            response += struct.pack(
                "<I", len(result_json)
            )  # json_len (4 bytes, little-endian)
            response += result_json  # json data

            client.sendall(response)
            logger.debug(f"Sent response for request {request_id}")

        except Exception as e:
            logger.error(f"Failed to send response: {e}")

    def _send_error(self, client: socket.socket, request_id: int, error_message: str):
        """
        Send error response to client.

        """
        result = {"success": False, "error": error_message}
        self._send_response(client, request_id, result)


def run_sequence_server_background(
    config: Optional[Union[ServerConfig, Dict[str, Any]]] = None,
    lm_studio_url: Optional[str] = None,
    model: Optional[str] = None,
    check_completion: bool = True,
) -> SequenceServer:
    """
    Start the SequenceServer in the background.


    """
    handler = SequenceQueryHandler()
    handler.initialize(
        lm_studio_url=lm_studio_url, model=model, check_completion=check_completion
    )

    if config:
        # Handle both ServerConfig objects and dicts
        if isinstance(config, ServerConfig):
            server_config = config
        else:
            server_config = ServerConfig(
                host=config.get("host", DEFAULT_HOST),
                port=config.get("port", SEQUENCE_SERVER_PORT),
            )
    else:
        server_config = ServerConfig(host=DEFAULT_HOST, port=SEQUENCE_SERVER_PORT)

    server = SequenceServer(server_config)
    server.start()

    return server


if __name__ == "__main__":
    import argparse
    import signal
    import time

    parser = argparse.ArgumentParser(
        description="Run SequenceServer for multi-command execution"
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Server host")
    parser.add_argument(
        "--port", type=int, default=SEQUENCE_SERVER_PORT, help="Server port"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LMSTUDIO_MODEL,
        help="LM Studio model for parsing",
    )
    args = parser.parse_args()

    handler = SequenceQueryHandler()
    handler.initialize(model=args.model)

    config = ServerConfig(host=args.host, port=args.port)
    server = SequenceServer(config)

    # Handle shutdown
    def signal_handler(_signum, _frame):
        logger.info("Shutdown signal received")
        server.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info(f"Starting SequenceServer on {args.host}:{args.port}")
    server.start()

    try:
        while server.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
