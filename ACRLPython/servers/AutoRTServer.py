#!/usr/bin/env python3
"""Dedicated TCP server for AutoRT Unity integration (AUTORT_SERVER_PORT)."""

import logging
import socket
import struct
from typing import Any, Dict, Optional

try:
    from core.LoggingSetup import setup_logging

    setup_logging(__name__)
except ImportError:
    from ..core.LoggingSetup import setup_logging

    setup_logging(__name__)

logger = logging.getLogger(__name__)

try:
    from core.TCPServerBase import TCPServerBase, ServerConfig
    from core.UnityProtocol import UnityProtocol, MessageType
    from config.Servers import DEFAULT_HOST, AUTORT_SERVER_PORT
except ImportError:
    from ..core.TCPServerBase import TCPServerBase, ServerConfig
    from ..core.UnityProtocol import UnityProtocol, MessageType
    from ..config.Servers import DEFAULT_HOST, AUTORT_SERVER_PORT

try:
    from servers.AutoRTIntegration import AutoRTHandler
except ImportError:
    from .AutoRTIntegration import AutoRTHandler

MAX_STRING_LENGTH = 1024 * 1024  # 1MB


class AutoRTServer(TCPServerBase):
    """TCP server for AutoRT commands from Unity (port AUTORT_SERVER_PORT)."""

    def __init__(self, config: Optional[ServerConfig] = None):
        if config is None:
            config = ServerConfig(host=DEFAULT_HOST, port=AUTORT_SERVER_PORT)
        super().__init__(config)

    def handle_client_connection(self, client: socket.socket, address: tuple):
        logger.info(f"AutoRT client connected from {address}")
        client.settimeout(5.0)

        while self._running:
            try:
                header_bytes = self._recv_exact(client, 5)
                if not header_bytes:
                    break

                msg_type = header_bytes[0]
                request_id = struct.unpack("<I", header_bytes[1:5])[0]

                # Validate message type
                if msg_type != MessageType.AUTORT_COMMAND:
                    logger.error(
                        f"Invalid message type: {msg_type} (expected {MessageType.AUTORT_COMMAND})"
                    )
                    self._send_error(
                        client, request_id, f"Invalid message type: {msg_type}"
                    )
                    continue

                # Handle AutoRT command
                self._handle_autort_command(client, header_bytes, request_id)

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

        logger.info(f"AutoRT client disconnected from {address}")

    def _handle_autort_command(
        self, client: socket.socket, header_bytes: bytes, request_id: int
    ):
        try:
            complete_message = self._receive_complete_autort_command(
                client, header_bytes
            )
            if not complete_message:
                logger.error("Failed to receive complete AutoRT command")
                self._send_error(
                    client, request_id, "Failed to receive complete command"
                )
                return

            request_id, command_type, params = UnityProtocol.decode_autort_command(
                complete_message
            )

            logger.info(f"AutoRT command received: {command_type} (params={params})")

            handler = AutoRTHandler.get_instance()

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
                    result = {
                        "success": False,
                        "error": "Missing task_id parameter",
                    }
                else:
                    result = handler.execute_task(task_id)
            elif command_type == "get_status":
                result = handler.get_status()
            else:
                result = {
                    "success": False,
                    "error": f"Unknown command type: {command_type}",
                }

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
            response_bytes = UnityProtocol.encode_autort_response(result, request_id)
            client.sendall(response_bytes)
            logger.info(
                f"Sent AutoRT response for request {request_id}: "
                f"success={result.get('success')}, status={result.get('status')}"
            )

        except Exception as e:
            logger.error(f"Failed to send AutoRT response: {e}")

    def _send_error(self, client: socket.socket, request_id: int, error_msg: str):
        error_result = {
            "success": False,
            "tasks": [],
            "loop_running": False,
            "error": error_msg,
        }
        self._send_autort_response(client, request_id, error_result)

    def _recv_exact(self, client: socket.socket, num_bytes: int) -> Optional[bytes]:
        """Raises socket.timeout on read timeout — caller handles for persistent connections."""
        data = b""
        while len(data) < num_bytes:
            try:
                chunk = client.recv(num_bytes - len(data))
                if not chunk:
                    # Connection closed by peer (recv returned empty bytes)
                    logger.debug(
                        f"Connection closed by peer (_recv_exact got empty chunk)"
                    )
                    return None
                data += chunk
            except socket.timeout:
                # Socket timeout - for persistent connections, this should propagate
                # so the caller can decide whether to continue waiting or close
                raise
        return data


# Standalone server entry point
if __name__ == "__main__":
    import time

    logger.info("Starting AutoRT Server...")
    server = AutoRTServer()
    server.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
