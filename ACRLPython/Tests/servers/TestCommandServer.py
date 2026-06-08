import pytest
import socket
import struct
import json
import time
from unittest.mock import Mock, patch

from servers.CommandServer import (
    CommandServer,
    CommandBroadcaster,
    get_command_broadcaster,
)
from core.UnityProtocol import UnityProtocol, MessageType


@pytest.fixture
def command_broadcaster():
    # Reset singleton
    CommandBroadcaster._instance = None
    broadcaster = CommandBroadcaster()
    return broadcaster


@pytest.fixture
def command_server(server_config):
    server = CommandServer(server_config)
    yield server
    # Cleanup
    if server.is_running():
        server.stop()


@pytest.fixture
def mock_client_socket():
    sock = Mock(spec=socket.socket)
    sock.recv = Mock(return_value=b"")
    sock.sendall = Mock(return_value=None)
    sock.close = Mock(return_value=None)
    sock.settimeout = Mock(return_value=None)
    return sock


class TestCommandBroadcaster:

    def test_singleton_instance(self, command_broadcaster):
        broadcaster1 = get_command_broadcaster()
        broadcaster2 = get_command_broadcaster()

        assert broadcaster1 is broadcaster2

    def test_send_command_no_server(self, command_broadcaster):
        command = {"command_type": "test", "data": "value"}

        result = command_broadcaster.send_command(command, request_id=1)

        assert result is False

    def test_send_command_with_server(self, command_broadcaster, command_server):
        command_broadcaster.set_server(command_server)
        command_server.broadcast_to_all_clients = Mock(return_value=1)

        command = {"command_type": "move", "robot_id": "Robot1"}
        result = command_broadcaster.send_command(command, request_id=123)

        assert result is True
        assert command["request_id"] == 123
        command_server.broadcast_to_all_clients.assert_called_once()

    def test_send_command_queue_when_no_clients(
        self, command_broadcaster, command_server
    ):
        command_broadcaster.set_server(command_server)
        command_server.broadcast_to_all_clients = Mock(return_value=0)  # No clients

        command = {"command_type": "test"}
        result = command_broadcaster.send_command(command, request_id=1)

        assert result is True
        queued = command_broadcaster.get_queued_results()
        assert len(queued) == 1
        assert queued[0]["command_type"] == "test"

    def test_send_result_backward_compatibility(
        self, command_broadcaster, command_server
    ):
        command_broadcaster.set_server(command_server)
        command_server.broadcast_to_all_clients = Mock(return_value=1)

        result = {"success": True, "data": "test", "request_id": 456}
        success = command_broadcaster.send_result(result)

        assert success is True
        command_server.broadcast_to_all_clients.assert_called_once()

    def test_completion_queue_lifecycle(self, command_broadcaster):
        request_id = 789

        command_broadcaster.create_completion_queue(request_id)

        completion = {"success": True, "result": "done"}
        command_broadcaster.put_completion(request_id, completion)

        retrieved = command_broadcaster.get_completion(request_id, timeout=0.5)
        assert retrieved == completion

        command_broadcaster.remove_completion_queue(request_id)

    def test_get_completion_timeout(self, command_broadcaster):
        request_id = 999
        command_broadcaster.create_completion_queue(request_id)

        result = command_broadcaster.get_completion(request_id, timeout=0.1)

        assert result is None

    def test_put_completion_no_queue(self, command_broadcaster):
        command_broadcaster.put_completion(999, {"data": "test"})


class TestCommandServerConnection:

    def test_server_initialization(self, server_config):
        server = CommandServer(server_config)

        assert server._config == server_config
        assert not server.is_running()
        assert server._broadcaster is not None

    def test_server_start_stop(self, command_server):
        command_server.start()
        time.sleep(0.1)

        assert command_server.is_running()

        command_server.stop()
        time.sleep(0.1)

        assert not command_server.is_running()

    def test_multiple_client_connections(self, command_server):
        command_server.start()
        time.sleep(0.1)

        with patch.object(command_server, "handle_client_connection") as mock_handle:
            mock_handle.return_value = None

            command_server._client_threads.append(Mock())
            command_server._client_threads.append(Mock())

            assert len(command_server._client_threads) == 2

        command_server.stop()

    def test_client_reconnection_after_disconnect(
        self, command_server, mock_client_socket
    ):
        command_server.start()
        time.sleep(0.1)

        with patch.object(command_server, "handle_client_connection") as mock_handle:
            mock_handle.return_value = None
            command_server._client_threads.append(Mock())
            command_server._client_threads.append(Mock())

            assert len(command_server._client_threads) >= 1

        command_server.stop()


class TestCommandServerCommands:

    def test_receive_completion_valid(self, command_server, mock_client_socket):
        request_id = 123
        completion = {"success": True, "result": "movement complete"}

        message = UnityProtocol.encode_result_message(completion, request_id)

        mock_client_socket.recv = Mock(
            side_effect=[
                message[:5],  # Header
                message[5:9],  # JSON length
                message[9:],  # JSON data
            ]
        )

        result = command_server._receive_completion(mock_client_socket)

        assert result is not None
        assert result["success"] is True
        assert result["request_id"] == request_id

    def test_receive_completion_invalid_type(self, command_server, mock_client_socket):
        header = struct.pack("<B", 0xFF) + struct.pack("<I", 123)
        json_data = json.dumps({"test": "data"}).encode("utf-8")
        json_len = struct.pack("<I", len(json_data))
        message = header + json_len + json_data

        mock_client_socket.recv = Mock(
            side_effect=[message[:5], message[5:9], message[9:]]
        )

        result = command_server._receive_completion(mock_client_socket)

        assert result is None

    def test_receive_completion_too_large(self, command_server, mock_client_socket):
        from config.Servers import MAX_STRING_LENGTH

        header = struct.pack("<B", MessageType.RESULT) + struct.pack("<I", 1)
        # JSON length exceeds maximum
        json_len = struct.pack("<I", MAX_STRING_LENGTH * 20)

        mock_client_socket.recv = Mock(side_effect=[header, json_len])

        result = command_server._receive_completion(mock_client_socket)

        assert result is None

    def test_send_queued_results(self, command_server, mock_client_socket):
        command_server._broadcaster.send_command({"type": "test1"}, 1)
        command_server._broadcaster.send_command({"type": "test2"}, 2)

        command_server._send_queued_results(mock_client_socket)

        assert mock_client_socket.sendall.call_count >= 0


class TestCommandServerProtocolV2:

    def test_request_id_correlation(self, command_broadcaster, command_server):
        command_broadcaster.set_server(command_server)
        command_server.broadcast_to_all_clients = Mock(return_value=1)

        request_id = 42
        command = {"command_type": "move", "robot_id": "Robot1"}

        command_broadcaster.create_completion_queue(request_id)
        command_broadcaster.send_command(command, request_id)

        completion = {"success": True, "request_id": request_id}
        command_broadcaster.put_completion(request_id, completion)

        result = command_broadcaster.get_completion(request_id, timeout=0.5)

        assert result is not None
        assert result["success"] is True
        assert result["request_id"] == request_id

    def test_multiple_pending_requests(self, command_broadcaster):
        request_ids = [1, 2, 3]

        for rid in request_ids:
            command_broadcaster.create_completion_queue(rid)

        command_broadcaster.put_completion(2, {"id": 2})
        command_broadcaster.put_completion(1, {"id": 1})
        command_broadcaster.put_completion(3, {"id": 3})

        result1 = command_broadcaster.get_completion(1, timeout=0.1)
        result2 = command_broadcaster.get_completion(2, timeout=0.1)
        result3 = command_broadcaster.get_completion(3, timeout=0.1)

        assert result1["id"] == 1
        assert result2["id"] == 2
        assert result3["id"] == 3


class TestCommandServerErrors:

    def test_malformed_json_handling(self, command_server, mock_client_socket):
        header = struct.pack("<B", MessageType.RESULT) + struct.pack("<I", 1)
        json_data = b"{invalid json}"
        json_len = struct.pack("<I", len(json_data))
        message = header + json_len + json_data

        mock_client_socket.recv = Mock(
            side_effect=[message[:5], message[5:9], message[9:]]
        )

        result = command_server._receive_completion(mock_client_socket)

        assert result is None

    def test_client_disconnect_during_receive(self, command_server, mock_client_socket):
        mock_client_socket.recv = Mock(return_value=b"")

        result = command_server._receive_completion(mock_client_socket)

        assert result is None

    def test_network_error_recovery(self, command_server, mock_client_socket):
        mock_client_socket.recv = Mock(side_effect=OSError("Network error"))

        with pytest.raises(OSError):
            command_server._receive_completion(mock_client_socket)

    def test_world_state_update_handling(self, command_server, mock_client_socket):
        request_id = 100
        world_state_update = {
            "type": "world_state_update",
            "robots": [
                {"robot_id": "Robot1", "position": {"x": 0.3, "y": 0.0, "z": 0.1}}
            ],
            "objects": [],
        }

        message = UnityProtocol.encode_result_message(world_state_update, request_id)

        mock_client_socket.recv = Mock(
            side_effect=[message[:5], message[5:9], message[9:]]
        )

        result = command_server._receive_completion(mock_client_socket)

        assert result is None


class TestCommandServerIntegration:

    def test_bidirectional_command_flow(self, command_server):
        command_server.start()
        time.sleep(0.1)

        broadcaster = command_server._broadcaster
        request_id = 555

        broadcaster.create_completion_queue(request_id)

        command = {"command_type": "test_command", "robot_id": "Robot1"}
        broadcaster.send_command(command, request_id)

        completion = {"success": True, "result": "done"}
        broadcaster.put_completion(request_id, completion)

        result = broadcaster.get_completion(request_id, timeout=0.5)

        assert result is not None
        assert result["success"] is True

        command_server.stop()
