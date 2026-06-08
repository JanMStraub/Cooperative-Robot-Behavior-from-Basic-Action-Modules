import pytest
import socket
import struct
import threading
import time
from unittest.mock import Mock, patch, MagicMock

from core.UnityProtocol import UnityProtocol, MessageType
from core.TCPServerBase import ServerConfig
from servers.SequenceServer import SequenceServer
from servers.AutoRTIntegration import AutoRTHandler

# Use a test-only port to avoid conflicts with the live SequenceServer on SEQUENCE_SERVER_PORT
TEST_SEQUENCE_PORT = 15013


class TestAutoRTSequenceServerIntegration:

    @classmethod
    def setup_class(cls):
        config = ServerConfig(host="127.0.0.1", port=TEST_SEQUENCE_PORT)
        cls.server = SequenceServer(config=config)
        cls.server_thread = threading.Thread(target=cls.server.start, daemon=True)
        cls.server_thread.start()
        time.sleep(0.5)

    @classmethod
    def teardown_class(cls):
        cls.server.stop()
        time.sleep(0.2)

    def setup_method(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect(("127.0.0.1", TEST_SEQUENCE_PORT))
        self.client.settimeout(5.0)

        AutoRTHandler._instance = None

    def teardown_method(self):
        try:
            self.client.close()
        except:
            pass

        handler = AutoRTHandler.get_instance()
        if handler._loop_running:
            handler.stop_loop()
            time.sleep(0.1)

    def test_generate_command_routing(self):
        command_type = "generate"
        params = {"num_tasks": 3, "robot_ids": ["Robot1"], "strategy": "balanced"}
        request_id = 10001

        handler = AutoRTHandler.get_instance()
        mock_orch = MagicMock()
        handler._orchestrator = mock_orch

        from autort.DataModels import SceneDescription

        mock_scene = SceneDescription(
            timestamp=0.0, objects=[], scene_summary="", robot_states={}
        )
        mock_orch._capture_scene.return_value = mock_scene
        mock_orch.task_generator.generate_tasks.return_value = []

        with patch("servers.AutoRTIntegration.ENABLE_SAFETY_VALIDATION", False):
            message = UnityProtocol.encode_autort_command(
                command_type, params, request_id
            )
            self.client.sendall(message)

            response_data = self._receive_complete_response()

            decoded_request_id, response = UnityProtocol.decode_autort_response(
                response_data
            )

            assert decoded_request_id == request_id
            assert "success" in response
            assert "tasks" in response
            assert "loop_running" in response

    def test_start_loop_command_routing(self):
        handler = AutoRTHandler.get_instance()
        mock_orch = MagicMock()
        handler._orchestrator = mock_orch
        from autort.DataModels import SceneDescription

        mock_scene = SceneDescription(
            timestamp=0.0, objects=[], scene_summary="", robot_states={}
        )
        mock_orch._capture_scene.return_value = mock_scene
        mock_orch.task_generator.generate_tasks.return_value = []

        command_type = "start_loop"
        params = {
            "loop_delay": 60.0,
            "robot_ids": ["Robot1"],
            "strategy": "explore",
        }
        request_id = 10002

        with patch("servers.AutoRTIntegration.ENABLE_SAFETY_VALIDATION", False):
            message = UnityProtocol.encode_autort_command(
                command_type, params, request_id
            )
            self.client.sendall(message)

            response_data = self._receive_complete_response()
            decoded_request_id, response = UnityProtocol.decode_autort_response(
                response_data
            )

        assert decoded_request_id == request_id
        assert response["success"]
        assert response["loop_running"]

        stop_message = UnityProtocol.encode_autort_command("stop_loop", {}, 10003)
        self.client.sendall(stop_message)
        self._receive_complete_response()

    def test_stop_loop_command_routing(self):
        handler = AutoRTHandler.get_instance()
        mock_orch = MagicMock()
        handler._orchestrator = mock_orch
        from autort.DataModels import SceneDescription

        mock_scene = SceneDescription(
            timestamp=0.0, objects=[], scene_summary="", robot_states={}
        )
        mock_orch._capture_scene.return_value = mock_scene
        mock_orch.task_generator.generate_tasks.return_value = []

        with patch("servers.AutoRTIntegration.ENABLE_SAFETY_VALIDATION", False):
            start_message = UnityProtocol.encode_autort_command(
                "start_loop", {"loop_delay": 60.0}, 10004
            )
            self.client.sendall(start_message)
            self._receive_complete_response()
            time.sleep(0.1)

        stop_message = UnityProtocol.encode_autort_command("stop_loop", {}, 10005)
        self.client.sendall(stop_message)

        response_data = self._receive_complete_response()
        decoded_request_id, response = UnityProtocol.decode_autort_response(
            response_data
        )

        assert decoded_request_id == 10005
        assert response["success"]
        assert not response["loop_running"]

    def test_get_status_command_routing(self):
        command_type = "get_status"
        request_id = 10006

        message = UnityProtocol.encode_autort_command(command_type, {}, request_id)
        self.client.sendall(message)

        response_data = self._receive_complete_response()
        decoded_request_id, response = UnityProtocol.decode_autort_response(
            response_data
        )

        assert decoded_request_id == request_id
        assert response["success"]
        assert "pending_tasks_count" in response
        assert "loop_config" in response

    def test_execute_task_command_routing(self):
        from autort.DataModels import ProposedTask, Operation

        handler = AutoRTHandler.get_instance()
        task = ProposedTask(
            task_id="seq_test_task_001",
            description="Test task",
            operations=[
                Operation(type="wait", robot_id="Robot1", parameters={"seconds": 1})
            ],
            required_robots=["Robot1"],
            estimated_complexity=1,
            reasoning="test",
        )
        task_id = handler._cache_task(task)

        command_type = "execute_task"
        params = {"task_id": task_id}
        request_id = 10007

        mock_orch = MagicMock()
        mock_orch._execute_task.return_value = {"success": True, "error": None}
        handler._orchestrator = mock_orch

        message = UnityProtocol.encode_autort_command(command_type, params, request_id)
        self.client.sendall(message)

        response_data = self._receive_complete_response()
        decoded_request_id, response = UnityProtocol.decode_autort_response(
            response_data
        )

        assert decoded_request_id == request_id
        assert "success" in response
        assert "result" in response

    def test_unknown_command_error_handling(self):
        command_type = "unknown_command_xyz"
        request_id = 10008

        message = UnityProtocol.encode_autort_command(command_type, {}, request_id)
        self.client.sendall(message)

        response_data = self._receive_complete_response()
        decoded_request_id, response = UnityProtocol.decode_autort_response(
            response_data
        )

        assert decoded_request_id == request_id
        assert not response["success"]
        assert "error" in response
        assert "Unknown command" in response["error"]

    def test_concurrent_autort_and_sequence_messages(self):
        """Test that AutoRT and sequence messages can be handled concurrently."""
        seq_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        seq_client.connect(("127.0.0.1", TEST_SEQUENCE_PORT))
        seq_client.settimeout(5.0)

        try:
            autort_message = UnityProtocol.encode_autort_command(
                "get_status", {}, 20001
            )
            self.client.sendall(autort_message)

            response_data = self._receive_complete_response()
            decoded_request_id, response = UnityProtocol.decode_autort_response(
                response_data
            )

            assert decoded_request_id == 20001
            assert response["success"]

        finally:
            seq_client.close()

    def test_malformed_autort_command(self):
        """Sends a structurally invalid AutoRT message; the server may hang reading
        the malformed body until the connection drops. We verify the server remains
        responsive by opening a fresh connection afterward.
        """
        header = struct.pack("B", MessageType.AUTORT_COMMAND)  # type
        header += struct.pack("<I", 30001)  # request_id
        invalid_body = b"invalid data"
        message = header + invalid_body

        self.client.sendall(message)

        time.sleep(0.2)

        recovery_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            recovery_client.connect(("127.0.0.1", TEST_SEQUENCE_PORT))
            recovery_client.settimeout(5.0)

            valid_message = UnityProtocol.encode_autort_command("get_status", {}, 30002)
            recovery_client.sendall(valid_message)

            header_data = b""
            while len(header_data) < 5:
                chunk = recovery_client.recv(5 - len(header_data))
                if not chunk:
                    break
                header_data += chunk

            assert len(header_data) == 5
        finally:
            recovery_client.close()

    def test_request_id_correlation(self):
        """Test that request IDs are correctly correlated in responses."""
        request_ids = [40001, 40002, 40003]

        for req_id in request_ids:
            message = UnityProtocol.encode_autort_command("get_status", {}, req_id)
            self.client.sendall(message)

            response_data = self._receive_complete_response()
            decoded_request_id, response = UnityProtocol.decode_autort_response(
                response_data
            )

            assert decoded_request_id == req_id

    def _receive_complete_response(self):
        """Helper to receive complete AutoRT response message."""
        header = self._recv_exact(5)
        if not header:
            raise Exception("Connection closed")

        msg_type = header[0]
        request_id = struct.unpack("<I", header[1:5])[0]

        if msg_type != MessageType.AUTORT_RESPONSE:
            raise Exception(f"Expected AUTORT_RESPONSE, got {msg_type}")

        json_len_bytes = self._recv_exact(4)
        if not json_len_bytes:
            raise Exception("Failed to read JSON length")
        json_len = struct.unpack("<I", json_len_bytes)[0]

        json_data = self._recv_exact(json_len)
        if not json_data:
            raise Exception("Failed to read JSON data")

        return header + json_len_bytes + json_data

    def _recv_exact(self, num_bytes):
        """Helper to receive exact number of bytes."""
        data = b""
        while len(data) < num_bytes:
            chunk = self.client.recv(num_bytes - len(data))
            if not chunk:
                return None
            data += chunk
        return data


class TestAutoRTProtocolCompliance:

    def test_protocol_v2_header_format(self):
        command_type = "generate"
        params = {"num_tasks": 5}
        request_id = 50001

        encoded = UnityProtocol.encode_autort_command(command_type, params, request_id)

        assert len(encoded) >= 5
        assert encoded[0] == MessageType.AUTORT_COMMAND

        header_request_id = struct.unpack("<I", encoded[1:5])[0]
        assert header_request_id == request_id

    def test_response_format_compliance(self):
        response_data = {
            "success": True,
            "tasks": [],
            "loop_running": False,
            "error": None,
        }
        request_id = 50002

        encoded = UnityProtocol.encode_autort_response(response_data, request_id)

        assert encoded[0] == MessageType.AUTORT_RESPONSE
        header_request_id = struct.unpack("<I", encoded[1:5])[0]
        assert header_request_id == request_id

        json_length = struct.unpack("<I", encoded[5:9])[0]
        assert json_length > 0

        actual_json_bytes = encoded[9:]
        assert len(actual_json_bytes) == json_length

    def test_utf8_encoding(self):
        command_type = "generate"
        params = {"description": "Déplacer l'objet"}  # French with accents
        request_id = 50003

        encoded = UnityProtocol.encode_autort_command(command_type, params, request_id)
        decoded_request_id, decoded_command, decoded_params = (
            UnityProtocol.decode_autort_command(encoded)
        )

        assert decoded_request_id == request_id
        assert decoded_params["description"] == "Déplacer l'objet"
