#!/usr/bin/env python3
"""
Unit tests for StreamingServer.py

Tests the image streaming server and storage
"""

import numpy as np
import time
import threading
from unittest.mock import Mock, patch, MagicMock

from servers.StreamingServer import ImageStorage, StreamingServer
from core.TCPServerBase import ServerConfig

class TestImageStorageSingleton:
    """Test ImageStorage singleton pattern"""

    def test_singleton_instance(self, cleanup_singletons):
        """Test that ImageStorage is a singleton"""
        storage1 = ImageStorage.get_instance()
        storage2 = ImageStorage.get_instance()

        assert storage1 is storage2

    def test_singleton_state_preserved(self, cleanup_singletons, sample_image):
        """Test that singleton state is preserved across get_instance calls"""
        storage1 = ImageStorage.get_instance()
        storage1.store_image("test_cam", sample_image, "test prompt")

        storage2 = ImageStorage.get_instance()
        image = storage2.get_camera_image("test_cam")

        assert image is not None
        assert np.array_equal(image, sample_image)


class TestImageStorage:
    """Test ImageStorage functionality"""

    def test_store_and_retrieve_image(self, cleanup_singletons, sample_image):
        """Test storing and retrieving an image"""
        storage = ImageStorage.get_instance()
        camera_id = "test_camera"
        prompt = "test prompt"

        storage.store_image(camera_id, sample_image, prompt)

        retrieved = storage.get_camera_image(camera_id)

        assert retrieved is not None
        assert np.array_equal(retrieved, sample_image)

    def test_store_image_with_empty_prompt(self, cleanup_singletons, sample_image):
        """Test storing image with empty prompt"""
        storage = ImageStorage.get_instance()
        camera_id = "test_camera"

        storage.store_image(camera_id, sample_image, "")

        retrieved_prompt = storage.get_camera_prompt(camera_id)
        assert retrieved_prompt == ""

    def test_get_camera_prompt(self, cleanup_singletons, sample_image):
        """Test retrieving camera prompt"""
        storage = ImageStorage.get_instance()
        camera_id = "test_camera"
        prompt = "What do you see?"

        storage.store_image(camera_id, sample_image, prompt)

        retrieved_prompt = storage.get_camera_prompt(camera_id)
        assert retrieved_prompt == prompt

    def test_get_camera_age(self, cleanup_singletons, sample_image):
        """Test getting image age"""
        storage = ImageStorage.get_instance()
        camera_id = "test_camera"

        storage.store_image(camera_id, sample_image, "")

        # Age should be very small (just stored)
        age = storage.get_camera_age(camera_id)
        assert age is not None
        if age is not None:  # Type narrowing for Pylance
            assert age < 1.0  # Less than 1 second old

            # Wait a bit and check age increased
            time.sleep(0.1)
            age2 = storage.get_camera_age(camera_id)
            assert age2 is not None
            if age2 is not None:  # Type narrowing
                assert age2 > age

    def test_get_nonexistent_camera(self, cleanup_singletons):
        """Test retrieving data for non-existent camera"""
        storage = ImageStorage.get_instance()

        image = storage.get_camera_image("nonexistent")
        prompt = storage.get_camera_prompt("nonexistent")
        age = storage.get_camera_age("nonexistent")

        assert image is None
        assert prompt is None
        assert age is None

    def test_get_all_camera_ids(self, cleanup_singletons, sample_image):
        """Test getting list of all camera IDs"""
        storage = ImageStorage.get_instance()

        # Initially empty
        cameras = storage.get_all_camera_ids()
        assert len(cameras) == 0

        # Add some cameras
        storage.store_image("camera1", sample_image, "")
        storage.store_image("camera2", sample_image, "")
        storage.store_image("camera3", sample_image, "")

        cameras = storage.get_all_camera_ids()
        assert len(cameras) == 3
        assert "camera1" in cameras
        assert "camera2" in cameras
        assert "camera3" in cameras

    def test_update_existing_camera(self, cleanup_singletons, sample_image):
        """Test updating an existing camera's image"""
        storage = ImageStorage.get_instance()
        camera_id = "test_camera"

        # Store first image
        storage.store_image(camera_id, sample_image, "first")
        first_timestamp = time.time()

        time.sleep(0.05)  # Small sleep to ensure time difference

        # Store second image
        new_image = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White image
        storage.store_image(camera_id, new_image, "second")
        second_timestamp = time.time()

        # Should have updated
        retrieved = storage.get_camera_image(camera_id)
        prompt = storage.get_camera_prompt(camera_id)
        age2 = storage.get_camera_age(camera_id)

        assert prompt == "second"
        # Verify new image was stored (age should be very recent)
        if age2 is not None:
            assert age2 < 1.0  # Should be less than 1 second old
        assert retrieved is not None
        if retrieved is not None:  # Type narrowing
            assert not np.array_equal(retrieved, sample_image)
            assert np.array_equal(retrieved, new_image)

    def test_get_camera_image_returns_copy(self, cleanup_singletons, sample_image):
        """Test that get_camera_image returns a copy, not reference"""
        storage = ImageStorage.get_instance()
        camera_id = "test_camera"

        storage.store_image(camera_id, sample_image, "")

        retrieved1 = storage.get_camera_image(camera_id)
        retrieved2 = storage.get_camera_image(camera_id)

        assert retrieved1 is not None
        assert retrieved2 is not None

        if retrieved1 is not None and retrieved2 is not None:  # Type narrowing
            # Should be equal but not the same object
            assert np.array_equal(retrieved1, retrieved2)
            assert retrieved1 is not retrieved2

            # Modifying retrieved should not affect storage
            retrieved1[0, 0, 0] = 255
            retrieved2_check = storage.get_camera_image(camera_id)
            assert retrieved2_check is not None
            if retrieved2_check is not None:  # Type narrowing
                assert not np.array_equal(retrieved1, retrieved2_check)


class TestImageStorageThreadSafety:
    """Test ImageStorage thread safety"""

    def test_concurrent_store_operations(self, cleanup_singletons, sample_image):
        """Test concurrent store operations are thread-safe"""
        storage = ImageStorage.get_instance()

        def store_images(camera_prefix):
            for i in range(10):
                storage.store_image(f"{camera_prefix}_{i}", sample_image, f"prompt_{i}")

        threads = []
        for i in range(5):
            t = threading.Thread(target=store_images, args=(f"thread{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All cameras should be stored
        cameras = storage.get_all_camera_ids()
        assert len(cameras) == 50  # 5 threads * 10 cameras each

    def test_concurrent_read_write(self, cleanup_singletons, sample_image):
        """Test concurrent read and write operations"""
        storage = ImageStorage.get_instance()
        storage.store_image("shared_camera", sample_image, "initial")

        errors = []

        def writer():
            try:
                for i in range(20):
                    storage.store_image("shared_camera", sample_image, f"write_{i}")
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(20):
                    image = storage.get_camera_image("shared_camera")
                    prompt = storage.get_camera_prompt("shared_camera")
                    age = storage.get_camera_age("shared_camera")
                    assert image is not None
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # No errors should have occurred
        assert len(errors) == 0


class TestStreamingServerInitialization:
    """Test StreamingServer initialization"""

    def test_server_initialization(self, server_config):
        """Test that StreamingServer initializes correctly"""
        server = StreamingServer(server_config)

        assert server._config == server_config
        assert server._storage is not None
        assert isinstance(server._storage, ImageStorage)

    def test_server_initialization_default_config(self):
        """Test server initialization with default config"""
        config = ServerConfig(host="127.0.0.1", port=5005, socket_timeout=1.0)
        server = StreamingServer(config)

        assert server._config is not None
        assert server._storage is not None


class TestStreamingServerProtocol:
    """Test StreamingServer protocol handling"""

    def test_recv_exactly_full_data(self):
        """Test _recv_exactly with complete data"""
        config = ServerConfig(host="127.0.0.1", port=5005, socket_timeout=1.0)
        server = StreamingServer(config)

        mock_socket = Mock()
        mock_socket.recv.return_value = b"test_data"

        result = server._recv_exactly(mock_socket, 9)

        assert result == b"test_data"

    def test_recv_exactly_partial_data(self):
        """Test _recv_exactly with partial reads"""
        config = ServerConfig(host="127.0.0.1", port=5005, socket_timeout=1.0)
        server = StreamingServer(config)

        mock_socket = Mock()
        # Simulate partial reads
        mock_socket.recv.side_effect = [b"test", b"_data"]

        result = server._recv_exactly(mock_socket, 9)

        assert result == b"test_data"

    def test_recv_exactly_connection_closed(self):
        """Test _recv_exactly when connection closes"""
        config = ServerConfig(host="127.0.0.1", port=5005, socket_timeout=1.0)
        server = StreamingServer(config)

        mock_socket = Mock()
        mock_socket.recv.return_value = b""  # Connection closed

        result = server._recv_exactly(mock_socket, 10)

        assert result is None

    def test_recv_exactly_timeout(self):
        """Test _recv_exactly handles timeout"""
        import socket

        config = ServerConfig(host="127.0.0.1", port=5005, socket_timeout=1.0)
        server = StreamingServer(config)

        mock_socket = Mock()
        mock_socket.recv.side_effect = socket.timeout()

        result = server._recv_exactly(mock_socket, 10)

        assert result is None


class TestStreamingServerIntegration:
    """Integration tests for StreamingServer"""

    @patch("socket.socket")
    def test_server_stores_received_images(self, mock_socket_class, cleanup_singletons):
        """Test that server correctly stores received images"""
        # This is a simplified test - full integration would require
        # actual socket communication which is complex to test

        server_config = ServerConfig(host="127.0.0.1", port=9999, socket_timeout=0.1)
        server = StreamingServer(server_config)

        # Verify storage is accessible
        storage = ImageStorage.get_instance()
        assert storage is server._storage
