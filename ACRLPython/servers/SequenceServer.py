#!/usr/bin/env python3
"""
SequenceServer.py - TCP server for multi-command sequence execution

Receives compound commands from Unity, parses them into operation sequences,
and executes them sequentially with completion tracking.

Port: 5013 (SEQUENCE_SERVER_PORT)

Protocol:
    Query (Unity → Python):
        [request_id:4][command_len:4][command_text:N][robot_id_len:4][robot_id:N][camera_id_len:4][camera_id:N][auto_execute:1]

    Response (Python → Unity):
        [request_id:4][response_len:4][response_json:N]
"""

import socket
import struct
import json
import logging
import threading
from typing import Dict, Any, Optional

from core.TCPServerBase import TCPServerBase, ServerConfig
from orchestrators.CommandParser import CommandParser
from orchestrators.SequenceExecutor import SequenceExecutor

# Import config
try:
    import LLMConfig as cfg
except ImportError:
    from .. import LLMConfig as cfg

logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=cfg.LOG_FORMAT)
logger = logging.getLogger(__name__)


class SequenceQueryHandler:
    """
    Singleton handler for sequence queries.

    Manages the CommandParser and SequenceExecutor instances and handles
    query processing.
    """

    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self._parser: Optional[CommandParser] = None
            self._executor: Optional[SequenceExecutor] = None
            self._initialized = True

    def initialize(
        self,
        lm_studio_url: Optional[str] = None,
        model: Optional[str] = None,
        check_completion: bool = True  # Enabled - Unity sends completion signals
    ) -> bool:
        """
        Initialize the parser and executor.

        Args:
            lm_studio_url: LM Studio URL for parsing
            model: Model name for parsing
            check_completion: Whether to check for operation completion

        Returns:
            True if initialization succeeded
        """
        try:
            self._parser = CommandParser(
                lm_studio_url=lm_studio_url,
                model=model
            )
            self._executor = SequenceExecutor(
                check_completion=check_completion
            )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize SequenceQueryHandler: {e}")
            return False

    def execute_sequence(
        self,
        command_text: str,
        robot_id: str = "Robot1",
        camera_id: str = "TableStereoCamera",
        auto_execute: bool = True,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Parse and execute a command sequence.

        Args:
            command_text: Natural language command
            robot_id: Default robot ID
            camera_id: Camera ID for perception operations (depth detection)
            auto_execute: Whether to automatically execute parsed operations
            timeout: Timeout per command

        Returns:
            Execution result dictionary
        """
        if not self._parser or not self._executor:
            return {
                "success": False,
                "error": "SequenceQueryHandler not initialized"
            }

        # Parse the command
        parse_result = self._parser.parse(command_text, robot_id)

        if not parse_result["success"]:
            return {
                "success": False,
                "error": f"Parse failed: {parse_result.get('error')}",
                "commands": []
            }

        commands = parse_result["commands"]
        if not commands:
            return {
                "success": False,
                "error": "No commands parsed from input",
                "commands": []
            }

        # Add camera_id to commands that need it (perception operations)
        for cmd in commands:
            if cmd.get("operation") in ["detect_objects", "get_depth", "stereo_detect"]:
                cmd["camera_id"] = camera_id

        # If auto_execute is False, just return parsed commands without executing
        if not auto_execute:
            return {
                "success": True,
                "parsed_commands": commands,
                "original_command": command_text,
                "auto_execute": False,
                "total_commands": len(commands),
                "completed_commands": 0,
                "results": [],
                "total_duration_ms": 0
            }

        # Execute the sequence
        exec_result = self._executor.execute_sequence(
            commands,
            timeout_per_command=timeout
        )

        # Add parsed commands to result
        exec_result["parsed_commands"] = commands
        exec_result["original_command"] = command_text
        exec_result["camera_id"] = camera_id

        return exec_result

    def is_ready(self) -> bool:
        """Check if handler is ready for queries."""
        return self._parser is not None and self._executor is not None


class SequenceServer(TCPServerBase):
    """
    TCP server for receiving and executing command sequences from Unity.

    Listens on port 5013 for sequence queries and returns execution results.
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        """
        Initialize the SequenceServer.

        Args:
            config: Server configuration (uses default if None)
        """
        if config is None:
            config = ServerConfig(
                host=cfg.DEFAULT_HOST,
                port=cfg.SEQUENCE_SERVER_PORT
            )
        super().__init__(config)

    def handle_client_connection(self, client: socket.socket, address: tuple):
        """
        Handle a client connection.

        Receives sequence queries and returns execution results.

        Args:
            client: Client socket
            address: Client address
        """
        logger.info(f"Sequence client connected from {address}")

        # Set socket timeout for persistent connection
        client.settimeout(5.0)

        while self._running:
            try:
                # Read request_id (4 bytes)
                request_id_bytes = self._recv_exact(client, 4)
                if not request_id_bytes:
                    break
                request_id = struct.unpack("!I", request_id_bytes)[0]

                # Read command length (4 bytes)
                cmd_len_bytes = self._recv_exact(client, 4)
                if not cmd_len_bytes:
                    break
                cmd_len = struct.unpack("!I", cmd_len_bytes)[0]

                if cmd_len > cfg.MAX_STRING_LENGTH * 10:  # Allow longer commands
                    logger.error(f"Command too long: {cmd_len}")
                    self._send_error(client, request_id, "Command too long")
                    continue

                # Read command text
                command_bytes = self._recv_exact(client, cmd_len)
                if not command_bytes:
                    break
                command_text = command_bytes.decode('utf-8')

                # Read robot_id length (4 bytes)
                robot_id_len_bytes = self._recv_exact(client, 4)
                if not robot_id_len_bytes:
                    break
                robot_id_len = struct.unpack("!I", robot_id_len_bytes)[0]

                # Read robot_id
                robot_id = "Robot1"
                if robot_id_len > 0:
                    robot_id_bytes = self._recv_exact(client, robot_id_len)
                    if robot_id_bytes:
                        robot_id = robot_id_bytes.decode('utf-8')

                # Read camera_id length (4 bytes)
                camera_id_len_bytes = self._recv_exact(client, 4)
                if not camera_id_len_bytes:
                    break
                camera_id_len = struct.unpack("!I", camera_id_len_bytes)[0]

                # Read camera_id
                camera_id = "TableStereoCamera"
                if camera_id_len > 0:
                    camera_id_bytes = self._recv_exact(client, camera_id_len)
                    if camera_id_bytes:
                        camera_id = camera_id_bytes.decode('utf-8')

                # Read auto_execute flag (1 byte)
                auto_execute_bytes = self._recv_exact(client, 1)
                if not auto_execute_bytes:
                    break
                auto_execute = auto_execute_bytes[0] == 1

                logger.info(f"Received sequence query (id={request_id}): {command_text[:100]} (camera={camera_id}, auto_execute={auto_execute})")

                # Execute the sequence
                handler = SequenceQueryHandler()
                result = handler.execute_sequence(command_text, robot_id, camera_id, auto_execute)

                # Send response
                self._send_response(client, request_id, result)

            except socket.timeout:
                # Expected for persistent connections
                continue
            except Exception as e:
                is_fatal, desc = self._is_connection_error_fatal(e)
                if is_fatal:
                    logger.info(f"Client {address} disconnected: {desc}")
                    break
                else:
                    logger.warning(f"Non-fatal error with {address}: {desc}")

        logger.info(f"Sequence client disconnected from {address}")

    def _recv_exact(self, client: socket.socket, num_bytes: int) -> Optional[bytes]:
        """
        Receive exact number of bytes from client.

        Args:
            client: Client socket
            num_bytes: Number of bytes to receive

        Returns:
            Received bytes or None if connection closed
        """
        data = b''
        while len(data) < num_bytes:
            chunk = client.recv(num_bytes - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def _send_response(self, client: socket.socket, request_id: int, result: Dict[str, Any]):
        """
        Send response to client.

        Args:
            client: Client socket
            request_id: Request ID for correlation
            result: Result dictionary to send
        """
        try:
            # Encode result as JSON
            result_json = json.dumps(result).encode('utf-8')

            # Build response: [request_id:4][result_len:4][result_json:N]
            response = struct.pack("!I", request_id)
            response += struct.pack("!I", len(result_json))
            response += result_json

            client.sendall(response)
            logger.debug(f"Sent response for request {request_id}")

        except Exception as e:
            logger.error(f"Failed to send response: {e}")

    def _send_error(self, client: socket.socket, request_id: int, error_message: str):
        """
        Send error response to client.

        Args:
            client: Client socket
            request_id: Request ID
            error_message: Error message
        """
        result = {
            "success": False,
            "error": error_message
        }
        self._send_response(client, request_id, result)


def run_sequence_server_background(
    config: Optional[Dict[str, Any]] = None,
    lm_studio_url: Optional[str] = None,
    model: Optional[str] = None,
    setup_signals: bool = False,
    check_completion: bool = True
) -> SequenceServer:
    """
    Start the SequenceServer in the background.

    Args:
        config: Server configuration dictionary
        lm_studio_url: LM Studio URL for parsing
        model: Model name for parsing
        setup_signals: Whether to set up signal handlers (False for threads)
        check_completion: Whether to wait for Unity completion signals

    Returns:
        SequenceServer instance
    """
    # Initialize the query handler
    handler = SequenceQueryHandler()
    handler.initialize(lm_studio_url=lm_studio_url, model=model, check_completion=check_completion)

    # Create server config
    if config:
        server_config = ServerConfig(
            host=config.get("host", cfg.DEFAULT_HOST),
            port=config.get("port", cfg.SEQUENCE_SERVER_PORT)
        )
    else:
        server_config = ServerConfig(
            host=cfg.DEFAULT_HOST,
            port=cfg.SEQUENCE_SERVER_PORT
        )

    # Create and start server
    server = SequenceServer(server_config)
    server.start()

    return server


if __name__ == "__main__":
    import argparse
    import signal
    import time

    parser = argparse.ArgumentParser(description="Run SequenceServer for multi-command execution")
    parser.add_argument("--host", default=cfg.DEFAULT_HOST, help="Server host")
    parser.add_argument("--port", type=int, default=cfg.SEQUENCE_SERVER_PORT, help="Server port")
    parser.add_argument("--model", default=cfg.DEFAULT_LMSTUDIO_MODEL, help="LM Studio model for parsing")
    args = parser.parse_args()

    # Initialize handler
    handler = SequenceQueryHandler()
    handler.initialize(model=args.model)

    # Create and start server
    config = ServerConfig(host=args.host, port=args.port)
    server = SequenceServer(config)

    # Handle shutdown
    def signal_handler(signum, frame):
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
