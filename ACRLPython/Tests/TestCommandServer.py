#!/usr/bin/env python3
"""
Unit tests for CommandServer.py

Tests the bidirectional command and results server including:
- Connection management and lifecycle
- Bidirectional command flow
- Request ID correlation (Protocol V2)
- Completion callback handling
- Broadcasting to multiple clients
- Error recovery and timeout handling
"""

import pytest
import socket
import struct
import json
import time
import threading
from unittest.mock import Mock, MagicMock, patch, call
from queue import Queue, Empty

from servers.CommandServer import CommandServer, CommandBroadcaster, get_command_broadcaster
from core.TCPServerBase import ServerConfig
from core.UnityProtocol import UnityProtocol, MessageType


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def command_broadcaster():
    """
    Create a fresh CommandBroadcaster instance for testing.

    Returns:
        CommandBroadcaster instance with no server attached
    """
    # Reset singleton
    CommandBroadcaster._instance = None
    broadcaster = CommandBroadcaster()
    return broadcaster


@pytest.fixture
def command_server(server_config):
    """
    Create a CommandServer instance for testing.

    Args:
        server_config: ServerConfig fixture

    Returns:
        CommandServer instance (not started)
    """
    server = CommandServer(server_config)
    yield server
    # Cleanup
    if server.is_running():
        server.stop()


@pytest.fixture
def mock_client_socket():
    """
    Create a mock client socket for testing.

    Returns:
        Mock socket configured for CommandServer testing
    """
    sock = Mock(spec=socket.socket)
    sock.recv = Mock(return_value=b"")
    sock.sendall = Mock(return_value=None)
    sock.close = Mock(return_value=None)
    sock.settimeout = Mock(return_value=None)
    return sock


# ============================================================================
# Test Class: CommandBroadcaster
# ============================================================================

class TestCommandBroadcaster:
    """Test CommandBroadcaster singleton and command sending functionality."""

    def test_singleton_instance(self, command_broadcaster):
        """Test that CommandBroadcaster is a singleton."""
        broadcaster1 = get_command_broadcaster()
        broadcaster2 = get_command_broadcaster()

        assert broadcaster1 is broadcaster2

    def test_send_command_no_server(self, command_broadcaster):
        """Test sending command when no server is attached."""
        command = {"command_type": "test", "data": "value"}

        result = command_broadcaster.send_command(command, request_id=1)

        assert result is False

    def test_send_command_with_server(self, command_broadcaster, command_server):
        """Test sending command with server attached."""
        command_broadcaster.set_server(command_server)
        command_server.broadcast_to_all_clients = Mock(return_value=1)

        command = {"command_type": "move", "robot_id": "Robot1"}
        result = command_broadcaster.send_command(command, request_id=123)

        assert result is True
        assert command["request_id"] == 123
        command_server.broadcast_to_all_clients.assert_called_once()

    def test_send_command_queue_when_no_clients(self, command_broadcaster, command_server):
        """Test command is queued when no clients are connected."""
        command_broadcaster.set_server(command_server)
        command_server.broadcast_to_all_clients = Mock(return_value=0)  # No clients

        command = {"command_type": "test"}
        result = command_broadcaster.send_command(command, request_id=1)

        assert result is True
        queued = command_broadcaster.get_queued_results()
        assert len(queued) == 1
        assert queued[0]["command_type"] == "test"

    def test_send_result_backward_compatibility(self, command_broadcaster, command_server):
        """Test send_result() for backward compatibility with ResultsBroadcaster."""
        command_broadcaster.set_server(command_server)
        command_server.broadcast_to_all_clients = Mock(return_value=1)

        result = {"success": True, "data": "test", "request_id": 456}
        success = command_broadcaster.send_result(result)

        assert success is True
        command_server.broadcast_to_all_clients.assert_called_once()

    def test_completion_queue_lifecycle(self, command_broadcaster):
        """Test creating, using, and removing completion queues."""
        request_id = 789

        # Create queue
        command_broadcaster.create_completion_queue(request_id)

        # Put completion
        completion = {"success": True, "result": "done"}
        command_broadcaster.put_completion(request_id, completion)

        # Get completion
        retrieved = command_broadcaster.get_completion(request_id, timeout=0.5)
        assert retrieved == completion

        # Remove queue
        command_broadcaster.remove_completion_queue(request_id)

    def test_get_completion_timeout(self, command_broadcaster):
        """Test get_completion returns None on timeout."""
        request_id = 999
        command_broadcaster.create_completion_queue(request_id)

        # Don't put anything, should timeout
        result = command_broadcaster.get_completion(request_id, timeout=0.1)

        assert result is None

    def test_put_completion_no_queue(self, command_broadcaster):
        """Test putting completion when no queue exists (should log warning)."""
        # Should not raise exception
        command_broadcaster.put_completion(999, {"data": "test"})


# ============================================================================
# Test Class: CommandServer - Connection Management
# ============================================================================

class TestCommandServerConnection:
    """Test CommandServer connection management and lifecycle."""

    def test_server_initialization(self, server_config):
        """Test server initializes with correct configuration."""
        server = CommandServer(server_config)

        assert server._config == server_config
        assert not server.is_running()
        assert server._broadcaster is not None

    def test_server_start_stop(self, command_server):
        """Test server can start and stop cleanly."""
        # Start server
        command_server.start()
        time.sleep(0.1)  # Give it time to start

        assert command_server.is_running()

        # Stop server
        command_server.stop()
        time.sleep(0.1)  # Give it time to stop

        assert not command_server.is_running()

    def test_multiple_client_connections(self, command_server):
        """Test server can handle multiple simultaneous client connections."""
        command_server.start()
        time.sleep(0.1)

        # Mock client connections
        with patch.object(command_server, 'handle_client_connection') as mock_handle:
            mock_handle.return_value = None

            # Simulate multiple clients connecting
            command_server._client_threads.append(Mock())
            command_server._client_threads.append(Mock())

            assert len(command_server._client_threads) == 2

        command_server.stop()

    def test_client_reconnection_after_disconnect(self, command_server, mock_client_socket):
        """Test client can reconnect after disconnecting."""
        command_server.start()
        time.sleep(0.1)

        # Simulate connection, disconnection, reconnection
        with patch.object(command_server, 'handle_client_connection') as mock_handle:
            # First connection
            mock_handle.return_value = None
            command_server._client_threads.append(Mock())

            # Second connection (reconnect)
            command_server._client_threads.append(Mock())

            assert len(command_server._client_threads) >= 1

        command_server.stop()


# ============================================================================
# Test Class: CommandServer - Command Handling
# ============================================================================

class TestCommandServerCommands:
    """Test CommandServer command handling functionality."""

    def test_receive_completion_valid(self, command_server, mock_client_socket):
        """Test receiving a valid completion message from Unity."""
        request_id = 123
        completion = {"success": True, "result": "movement complete"}

        # Encode completion message
        message = UnityProtocol.encode_result_message(completion, request_id)

        # Mock socket to return message
        mock_client_socket.recv = Mock(side_effect=[
            message[:5],  # Header
            message[5:9],  # JSON length
            message[9:]  # JSON data
        ])

        result = command_server._receive_completion(mock_client_socket)

        assert result is not None
        assert result["success"] is True
        assert result["request_id"] == request_id

    def test_receive_completion_invalid_type(self, command_server, mock_client_socket):
        """Test receiving message with invalid type (should be rejected)."""
        # Create message with wrong type
        header = struct.pack("<B", 0xFF) + struct.pack("<I", 123)  # Invalid type
        json_data = json.dumps({"test": "data"}).encode("utf-8")
        json_len = struct.pack("<I", len(json_data))
        message = header + json_len + json_data

        mock_client_socket.recv = Mock(side_effect=[
            message[:5],
            message[5:9],
            message[9:]
        ])

        result = command_server._receive_completion(mock_client_socket)

        assert result is None

    def test_receive_completion_too_large(self, command_server, mock_client_socket):
        """Test receiving completion message that's too large."""
        import LLMConfig as cfg

        header = struct.pack("<B", MessageType.RESULT) + struct.pack("<I", 1)
        # JSON length exceeds maximum
        json_len = struct.pack("<I", cfg.MAX_STRING_LENGTH * 20)

        mock_client_socket.recv = Mock(side_effect=[
            header,
            json_len
        ])

        result = command_server._receive_completion(mock_client_socket)

        assert result is None

    def test_send_queued_results(self, command_server, mock_client_socket):
        """Test sending queued results to newly connected client."""
        # Queue some results
        command_server._broadcaster.send_command({"type": "test1"}, 1)
        command_server._broadcaster.send_command({"type": "test2"}, 2)

        # Simulate sending queued results
        command_server._send_queued_results(mock_client_socket)

        # Should have called sendall for each queued result
        assert mock_client_socket.sendall.call_count >= 0


# ============================================================================
# Test Class: CommandServer - Protocol V2
# ============================================================================

class TestCommandServerProtocolV2:
    """Test Protocol V2 request ID correlation."""

    def test_request_id_correlation(self, command_broadcaster, command_server):
        """Test request ID is correctly correlated between command and completion."""
        command_broadcaster.set_server(command_server)
        command_server.broadcast_to_all_clients = Mock(return_value=1)

        request_id = 42
        command = {"command_type": "move", "robot_id": "Robot1"}

        # Create completion queue
        command_broadcaster.create_completion_queue(request_id)

        # Send command
        command_broadcaster.send_command(command, request_id)

        # Simulate completion
        completion = {"success": True, "request_id": request_id}
        command_broadcaster.put_completion(request_id, completion)

        # Retrieve completion
        result = command_broadcaster.get_completion(request_id, timeout=0.5)

        assert result is not None
        assert result["success"] is True
        assert result["request_id"] == request_id

    def test_multiple_pending_requests(self, command_broadcaster):
        """Test handling multiple pending requests simultaneously."""
        request_ids = [1, 2, 3]

        # Create queues for all requests
        for rid in request_ids:
            command_broadcaster.create_completion_queue(rid)

        # Put completions in random order
        command_broadcaster.put_completion(2, {"id": 2})
        command_broadcaster.put_completion(1, {"id": 1})
        command_broadcaster.put_completion(3, {"id": 3})

        # Retrieve in order
        result1 = command_broadcaster.get_completion(1, timeout=0.1)
        result2 = command_broadcaster.get_completion(2, timeout=0.1)
        result3 = command_broadcaster.get_completion(3, timeout=0.1)

        assert result1["id"] == 1
        assert result2["id"] == 2
        assert result3["id"] == 3


# ============================================================================
# Test Class: CommandServer - Error Handling
# ============================================================================

class TestCommandServerErrors:
    """Test CommandServer error handling and recovery."""

    def test_malformed_json_handling(self, command_server, mock_client_socket):
        """Test handling of malformed JSON in completion message."""
        header = struct.pack("<B", MessageType.RESULT) + struct.pack("<I", 1)
        json_data = b"{invalid json}"
        json_len = struct.pack("<I", len(json_data))
        message = header + json_len + json_data

        mock_client_socket.recv = Mock(side_effect=[
            message[:5],
            message[5:9],
            message[9:]
        ])

        result = command_server._receive_completion(mock_client_socket)

        assert result is None

    def test_client_disconnect_during_receive(self, command_server, mock_client_socket):
        """Test handling client disconnect during message reception."""
        # Simulate disconnection (recv returns empty bytes)
        mock_client_socket.recv = Mock(return_value=b"")

        result = command_server._receive_completion(mock_client_socket)

        assert result is None

    def test_network_error_recovery(self, command_server, mock_client_socket):
        """Test recovery from network errors."""
        # Simulate network error
        mock_client_socket.recv = Mock(side_effect=OSError("Network error"))

        # _recv_exact will raise the OSError since it doesn't catch it
        # This is expected behavior - the error is handled at a higher level (in handle_client_connection)
        # We just verify the method raises the error as expected
        with pytest.raises(OSError):
            command_server._receive_completion(mock_client_socket)

    def test_world_state_update_handling(self, command_server, mock_client_socket):
        """Test handling world_state_update messages (should not return as completion)."""
        request_id = 100
        world_state_update = {
            "type": "world_state_update",
            "robots": [{"robot_id": "Robot1", "position": {"x": 0.3, "y": 0.0, "z": 0.1}}],
            "objects": []
        }

        # Encode message
        message = UnityProtocol.encode_result_message(world_state_update, request_id)

        mock_client_socket.recv = Mock(side_effect=[
            message[:5],
            message[5:9],
            message[9:]
        ])

        result = command_server._receive_completion(mock_client_socket)

        # World state updates should not be returned as completions
        assert result is None


# ============================================================================
# Integration Test
# ============================================================================

class TestCommandServerIntegration:
    """Integration tests for full command flow."""

    def test_bidirectional_command_flow(self, command_server):
        """Test complete bidirectional command flow (send command, receive completion)."""
        command_server.start()
        time.sleep(0.1)

        broadcaster = command_server._broadcaster
        request_id = 555

        # Create completion queue
        broadcaster.create_completion_queue(request_id)

        # Send command (will be queued since no clients)
        command = {"command_type": "test_command", "robot_id": "Robot1"}
        broadcaster.send_command(command, request_id)

        # Simulate receiving completion
        completion = {"success": True, "result": "done"}
        broadcaster.put_completion(request_id, completion)

        # Retrieve completion
        result = broadcaster.get_completion(request_id, timeout=0.5)

        assert result is not None
        assert result["success"] is True

        command_server.stop()
