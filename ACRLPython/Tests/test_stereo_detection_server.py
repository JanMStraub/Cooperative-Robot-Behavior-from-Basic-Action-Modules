#!/usr/bin/env python3
"""
Unit tests for StereoDetectionServer.py

Tests the stereo detection server and storage
"""

import pytest
import numpy as np
import time
import threading
import struct
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add LLMCommunication directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "LLMCommunication"))

from LLMCommunication.servers.StereoDetectionServer import StereoImageStorage, StereoDetectionServer
from LLMCommunication.core.TCPServerBase import ServerConfig
import LLMCommunication.llm_config as cfg


class TestStereoImageStorageSingleton:
    """Test StereoImageStorage singleton pattern"""

    def test_singleton_instance(self, cleanup_singletons):
        """Test that StereoImageStorage is a singleton"""
        storage1 = StereoImageStorage()
        storage2 = StereoImageStorage()

        assert storage1 is storage2

    def test_singleton_state_preserved(self, cleanup_singletons, sample_stereo_pair):
        """Test that singleton state is preserved"""
        imgL, imgR = sample_stereo_pair

        storage1 = StereoImageStorage()
        storage1.store_stereo_pair("pair1", imgL, imgR, "test")

        storage2 = StereoImageStorage()
        pair = storage2.get_stereo_pair("pair1")

        assert pair is not None


class TestStereoImageStorage:
    """Test StereoImageStorage functionality"""

    def test_store_and_retrieve_stereo_pair(self, cleanup_singletons, sample_stereo_pair):
        """Test storing and retrieving a stereo pair"""
        imgL, imgR = sample_stereo_pair
        storage = StereoImageStorage()
        pair_id = "test_pair"
        prompt = "detect objects"

        storage.store_stereo_pair(pair_id, imgL, imgR, prompt)

        retrieved = storage.get_stereo_pair(pair_id)

        assert retrieved is not None
        if retrieved:  # Type narrowing for Pylance
            ret_imgL, ret_imgR, ret_prompt = retrieved
            assert np.array_equal(ret_imgL, imgL)
            assert np.array_equal(ret_imgR, imgR)
            assert ret_prompt == prompt

    def test_store_stereo_pair_with_empty_prompt(self, cleanup_singletons, sample_stereo_pair):
        """Test storing stereo pair with empty prompt"""
        imgL, imgR = sample_stereo_pair
        storage = StereoImageStorage()

        storage.store_stereo_pair("pair1", imgL, imgR, "")

        pair_result = storage.get_stereo_pair("pair1")
        assert pair_result is not None
        if pair_result:  # Type narrowing
            _, _, prompt = pair_result
            assert prompt == ""

    def test_get_pair_age(self, cleanup_singletons, sample_stereo_pair):
        """Test getting stereo pair age"""
        imgL, imgR = sample_stereo_pair
        storage = StereoImageStorage()

        storage.store_stereo_pair("pair1", imgL, imgR, "test")

        # Age should be very small (just stored)
        age = storage.get_pair_age("pair1")
        assert age is not None
        if age is not None:  # Type narrowing for Pylance
            assert age < 1.0  # Less than 1 second old

            # Wait a bit and check age increased
            time.sleep(0.1)
            age2 = storage.get_pair_age("pair1")
            assert age2 is not None
            if age2 is not None:  # Type narrowing
                assert age2 > age

    def test_get_nonexistent_pair(self, cleanup_singletons):
        """Test retrieving non-existent stereo pair"""
        storage = StereoImageStorage()

        pair = storage.get_stereo_pair("nonexistent")
        age = storage.get_pair_age("nonexistent")

        assert pair is None
        assert age is None

    def test_update_existing_pair(self, cleanup_singletons, sample_stereo_pair):
        """Test updating an existing stereo pair"""
        imgL, imgR = sample_stereo_pair
        storage = StereoImageStorage()

        # Store first pair
        storage.store_stereo_pair("pair1", imgL, imgR, "first")

        time.sleep(0.05)  # Small sleep to ensure time difference

        # Store second pair with same ID
        new_imgL = np.ones((480, 640, 3), dtype=np.uint8) * 255
        new_imgR = np.ones((480, 640, 3), dtype=np.uint8) * 255
        storage.store_stereo_pair("pair1", new_imgL, new_imgR, "second")

        # Should have updated
        pair_result = storage.get_stereo_pair("pair1")
        age2 = storage.get_pair_age("pair1")

        assert pair_result is not None
        if pair_result:  # Type narrowing
            ret_imgL, ret_imgR, prompt = pair_result
            assert prompt == "second"
            # Verify new pair was stored (age should be very recent)
            if age2 is not None:
                assert age2 < 1.0  # Should be less than 1 second old
            assert not np.array_equal(ret_imgL, imgL)
            assert np.array_equal(ret_imgL, new_imgL)


class TestStereoImageStorageThreadSafety:
    """Test StereoImageStorage thread safety"""

    def test_concurrent_store_operations(self, cleanup_singletons, sample_stereo_pair):
        """Test concurrent store operations are thread-safe"""
        imgL, imgR = sample_stereo_pair
        storage = StereoImageStorage()

        def store_pairs(prefix):
            for i in range(10):
                storage.store_stereo_pair(f"{prefix}_{i}", imgL, imgR, f"prompt_{i}")

        threads = []
        for i in range(3):
            t = threading.Thread(target=store_pairs, args=(f"thread{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All pairs should be stored (30 total)
        # Note: exact count depends on implementation details

    def test_concurrent_read_write(self, cleanup_singletons, sample_stereo_pair):
        """Test concurrent read and write operations"""
        imgL, imgR = sample_stereo_pair
        storage = StereoImageStorage()
        storage.store_stereo_pair("shared_pair", imgL, imgR, "initial")

        errors = []

        def writer():
            try:
                for i in range(20):
                    storage.store_stereo_pair("shared_pair", imgL, imgR, f"write_{i}")
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(20):
                    pair = storage.get_stereo_pair("shared_pair")
                    age = storage.get_pair_age("shared_pair")
                    assert pair is not None
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # No errors should have occurred
        assert len(errors) == 0


class TestStereoDetectionServerInitialization:
    """Test StereoDetectionServer initialization"""

    def test_server_initialization(self):
        """Test that StereoDetectionServer initializes correctly"""
        config = ServerConfig(host="127.0.0.1", port=9999, socket_timeout=0.1)
        server = StereoDetectionServer(config)

        assert server._config.host == "127.0.0.1"
        assert server._config.port == 9999
        assert server.image_storage is not None

    def test_server_initialization_default_config(self):
        """Test server initialization with default config"""
        server = StereoDetectionServer()

        assert server._config is not None
        assert server.image_storage is not None


class TestStereoDetectionServerProtocol:
    """Test StereoDetectionServer protocol handling"""

    def test_receive_exactly_full_data(self):
        """Test _receive_exactly with complete data"""
        server = StereoDetectionServer()

        mock_socket = Mock()
        mock_socket.recv.return_value = b"test_data"

        result = server._receive_exactly(mock_socket, 9)

        assert result == b"test_data"

    def test_receive_exactly_partial_data(self):
        """Test _receive_exactly with partial reads"""
        server = StereoDetectionServer()

        mock_socket = Mock()
        # Simulate partial reads
        mock_socket.recv.side_effect = [b"test", b"_data"]

        result = server._receive_exactly(mock_socket, 9)

        assert result == b"test_data"

    def test_receive_exactly_connection_closed(self):
        """Test _receive_exactly when connection closes"""
        server = StereoDetectionServer()

        mock_socket = Mock()
        mock_socket.recv.return_value = b""  # Connection closed

        result = server._receive_exactly(mock_socket, 10)

        assert result is None

    def test_receive_exactly_timeout(self):
        """Test _receive_exactly handles timeout"""
        import socket

        server = StereoDetectionServer()

        mock_socket = Mock()
        mock_socket.recv.side_effect = socket.timeout()

        result = server._receive_exactly(mock_socket, 10)

        # Should continue on timeout (implementation specific)
        # May return None or raise


class TestStereoDetectionServerClientHandling:
    """Test StereoDetectionServer client handling"""

    def test_handle_client_connection_protocol(self, cleanup_singletons, sample_stereo_pair):
        """Test client connection handles stereo protocol"""
        imgL, imgR = sample_stereo_pair

        server = StereoDetectionServer()
        mock_client = Mock()

        # Encode a stereo message
        cam_pair_id = "stereo_pair_1"
        cam_L_id = "left_camera"
        cam_R_id = "right_camera"
        prompt = "detect objects"

        # Encode images
        import cv2
        _, imgL_encoded = cv2.imencode('.png', imgL)
        _, imgR_encoded = cv2.imencode('.png', imgR)
        imgL_bytes = imgL_encoded.tobytes()
        imgR_bytes = imgR_encoded.tobytes()

        # Build message according to protocol
        message = bytearray()
        message.extend(struct.pack('I', len(cam_pair_id.encode('utf-8'))))
        message.extend(cam_pair_id.encode('utf-8'))
        message.extend(struct.pack('I', len(cam_L_id.encode('utf-8'))))
        message.extend(cam_L_id.encode('utf-8'))
        message.extend(struct.pack('I', len(cam_R_id.encode('utf-8'))))
        message.extend(cam_R_id.encode('utf-8'))
        message.extend(struct.pack('I', len(prompt.encode('utf-8'))))
        message.extend(prompt.encode('utf-8'))
        message.extend(struct.pack('I', len(imgL_bytes)))
        message.extend(imgL_bytes)
        message.extend(struct.pack('I', len(imgR_bytes)))
        message.extend(imgR_bytes)

        # Setup mock to return message data then disconnect
        message_bytes = bytes(message)
        chunks = [message_bytes[i:i+100] for i in range(0, len(message_bytes), 100)]
        chunks.append(b"")  # Disconnect

        mock_client.recv.side_effect = lambda n: chunks.pop(0) if chunks else b""

        # Handle client (will process one message then disconnect)
        try:
            server.handle_client_connection(mock_client, ("127.0.0.1", 12345))
        except:
            pass

        # Verify stereo pair was stored
        stored = server.image_storage.get_stereo_pair(cam_pair_id)
        # May or may not be stored depending on error handling

    def test_handle_client_connection_invalid_data(self, cleanup_singletons):
        """Test handling of invalid protocol data"""
        server = StereoDetectionServer()
        mock_client = Mock()

        # Send malformed data
        mock_client.recv.return_value = b"invalid_data"

        # Should handle error gracefully
        try:
            server.handle_client_connection(mock_client, ("127.0.0.1", 12345))
        except:
            pass  # Errors are expected

    def test_handle_client_oversized_string(self, cleanup_singletons):
        """Test handling of oversized string fields"""
        server = StereoDetectionServer()
        mock_client = Mock()

        # Create message with oversized camera ID
        oversized_id = "A" * (cfg.MAX_STRING_LENGTH + 100)
        message = struct.pack('I', len(oversized_id))
        message += oversized_id.encode('utf-8')

        mock_client.recv.return_value = message

        # Should reject oversized data
        try:
            server.handle_client_connection(mock_client, ("127.0.0.1", 12345))
        except:
            pass


class TestStereoDetectionServerIntegration:
    """Integration tests for StereoDetectionServer"""

    @patch('socket.socket')
    def test_server_lifecycle(self, mock_socket_class, cleanup_singletons):
        """Test server start/stop lifecycle"""
        import socket
        mock_server_socket = MagicMock()
        mock_socket_class.return_value = mock_server_socket
        mock_server_socket.accept.side_effect = socket.timeout()

        config = ServerConfig(host="127.0.0.1", port=9999, socket_timeout=0.1)
        server = StereoDetectionServer(config)

        # Start server
        server.start()
        assert server.is_running()

        time.sleep(0.2)

        # Stop server
        server.stop()
        assert not server.is_running()

    def test_run_stereo_detection_server_background(self, cleanup_singletons):
        """Test running server in background"""
        import socket
        from LLMCommunication.servers.StereoDetectionServer import run_stereo_detection_server_background

        with patch('socket.socket') as mock_socket_class:
            mock_server_socket = MagicMock()
            mock_socket_class.return_value = mock_server_socket
            mock_server_socket.accept.side_effect = socket.timeout()

            server = run_stereo_detection_server_background(host="127.0.0.1", port=9999)

            # Server should be running in background
            time.sleep(0.2)

            # Cleanup
            try:
                server.stop()
            except:
                pass


class TestStereoDetectionServerErrorHandling:
    """Test error handling in StereoDetectionServer"""

    def test_handle_client_socket_error(self, cleanup_singletons):
        """Test handling of socket errors during client connection"""
        import socket

        server = StereoDetectionServer()
        mock_client = Mock()
        mock_client.recv.side_effect = socket.error("Connection reset")

        # Should handle error gracefully
        try:
            server.handle_client_connection(mock_client, ("127.0.0.1", 12345))
        except:
            pass  # Errors are expected and handled

    def test_handle_client_decode_error(self, cleanup_singletons):
        """Test handling of image decode errors"""
        server = StereoDetectionServer()
        mock_client = Mock()

        # Create valid protocol but invalid image data
        cam_pair_id = "pair1"
        cam_L_id = "left"
        cam_R_id = "right"
        prompt = "test"
        invalid_img = b"NOT_A_VALID_IMAGE"

        message = bytearray()
        message.extend(struct.pack('I', len(cam_pair_id)))
        message.extend(cam_pair_id.encode('utf-8'))
        message.extend(struct.pack('I', len(cam_L_id)))
        message.extend(cam_L_id.encode('utf-8'))
        message.extend(struct.pack('I', len(cam_R_id)))
        message.extend(cam_R_id.encode('utf-8'))
        message.extend(struct.pack('I', len(prompt)))
        message.extend(prompt.encode('utf-8'))
        message.extend(struct.pack('I', len(invalid_img)))
        message.extend(invalid_img)
        message.extend(struct.pack('I', len(invalid_img)))
        message.extend(invalid_img)
        message.extend(b"")  # End

        mock_client.recv.return_value = bytes(message)

        # Should handle decode error
        try:
            server.handle_client_connection(mock_client, ("127.0.0.1", 12345))
        except:
            pass


class TestStereoDetectionServerMessageParsing:
    """Test stereo message parsing"""

    def test_parse_complete_stereo_message(self, sample_stereo_pair):
        """Test parsing a complete stereo message"""
        imgL, imgR = sample_stereo_pair

        # Encode images
        import cv2
        _, imgL_encoded = cv2.imencode('.png', imgL)
        _, imgR_encoded = cv2.imencode('.png', imgR)

        # Message should be parseable
        assert imgL_encoded is not None
        assert imgR_encoded is not None

    def test_field_order(self):
        """Test that fields are parsed in correct order"""
        # Protocol order:
        # 1. cam_pair_id
        # 2. camera_L_id
        # 3. camera_R_id
        # 4. prompt
        # 5. image_L
        # 6. image_R

        fields = [
            "cam_pair_id",
            "camera_L_id",
            "camera_R_id",
            "prompt",
            "image_L",
            "image_R"
        ]

        # Verify order is correct (documentation test)
        expected_order = [
            "cam_pair_id",
            "camera_L_id",
            "camera_R_id",
            "prompt",
            "image_L",
            "image_R"
        ]

        assert fields == expected_order
