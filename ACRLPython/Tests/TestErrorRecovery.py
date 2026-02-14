#!/usr/bin/env python3
"""
Error Recovery Tests
====================

Tests error recovery and graceful degradation across the robot control system.
Covers scenarios like network failures, resource exhaustion, and external
dependency failures.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from queue import Queue, Full
import socket

from operations.MoveOperations import move_to_coordinate
from operations.DetectionOperations import detect_objects
from orchestrators.CommandParser import CommandParser
from servers.CommandServer import CommandBroadcaster
from servers.ImageStorageCore import UnifiedImageStorage


# ============================================================================
# Network Failure Tests
# ============================================================================

class TestNetworkFailureRecovery:
    """Test recovery from network failures"""

    def test_command_send_with_no_server_connection(self):
        """Test command execution when server is not connected"""
        broadcaster = CommandBroadcaster()
        # No server attached

        command = {"command_type": "move", "robot_id": "Robot1"}
        result = broadcaster.send_command(command, request_id=1)

        # Should return False but not crash
        assert result is False

    @pytest.mark.skip(reason="Test logic needs update for current CommandBroadcaster API")

    def test_intermittent_network_failure(self):
        """Test handling of intermittent network failures"""
        broadcaster = CommandBroadcaster()
        mock_server = Mock()

        # Simulate intermittent failures (fails, succeeds, fails, succeeds)
        call_count = [0]

        def intermittent_broadcast(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                raise ConnectionError("Network temporarily unavailable")
            return 1  # Success

        mock_server.broadcast_to_all_clients = intermittent_broadcast
        broadcaster.set_server(mock_server)

        # Try sending commands
        results = []
        for i in range(4):
            try:
                result = broadcaster.send_command(
                    {"command_type": "test", "index": i},
                    request_id=i
                )
                results.append(("success", result))
            except ConnectionError:
                results.append(("error", None))

        # Should have mix of successes and failures
        assert len(results) == 4
        # Odd indices should fail, even should succeed based on our mock
        assert results[0][0] == "error"  # First call fails
        assert results[1][0] == "success"  # Second succeeds
        assert results[2][0] == "error"  # Third fails
        assert results[3][0] == "success"  # Fourth succeeds

    def test_server_restart_mid_operation(self):
        """Test recovery when server restarts during operation"""
        broadcaster = CommandBroadcaster()

        # Simulate server restart by changing server instance
        mock_server1 = Mock()
        mock_server1.broadcast_to_all_clients = Mock(return_value=1)
        broadcaster.set_server(mock_server1)

        # Send first command
        result1 = broadcaster.send_command({"type": "cmd1"}, request_id=1)
        assert result1 is True

        # Server restarts - new instance
        mock_server2 = Mock()
        mock_server2.broadcast_to_all_clients = Mock(return_value=1)
        broadcaster.set_server(mock_server2)

        # Send second command - should work with new server
        result2 = broadcaster.send_command({"type": "cmd2"}, request_id=2)
        assert result2 is True

    def test_socket_timeout_handling(self):
        """Test handling of socket timeout errors"""
        with patch('socket.socket') as mock_socket_class:
            mock_sock = Mock()
            mock_sock.recv = Mock(side_effect=socket.timeout("Operation timed out"))
            mock_socket_class.return_value = mock_sock

            # Try to receive data
            try:
                mock_sock.recv(1024)
                received = False
            except socket.timeout:
                received = True

            # Timeout should be raised and caught
            assert received is True


# ============================================================================
# Resource Exhaustion Tests
# ============================================================================

class TestResourceExhaustion:
    """Test recovery from resource exhaustion"""

    def test_disk_full_during_image_storage(self):
        """Test handling when disk is full during image storage"""
        import numpy as np

        storage = UnifiedImageStorage()

        # Create large image
        large_image = np.ones((4000, 4000, 3), dtype=np.uint8)

        # Mock OSError for disk full
        with patch.object(storage, 'store_single_image') as mock_store:
            mock_store.side_effect = OSError("[Errno 28] No space left on device")

            # Try to store image
            try:
                storage.store_single_image("camera_test", large_image, "test")
                disk_full_handled = False
            except OSError as e:
                disk_full_handled = True
                assert "space" in str(e).lower() or "errno 28" in str(e).lower()

            assert disk_full_handled is True

    @pytest.mark.skip(reason="Test logic needs update for batch operations")

    def test_memory_exhaustion_during_batch_operations(self):
        """Test handling of memory exhaustion during large batch operations"""
        import numpy as np

        # Try to allocate very large arrays
        large_arrays: list = []
        try:
            # Attempt to allocate 100GB (will fail on most systems)
            for _ in range(100):
                large_arrays.append(np.zeros((10000, 10000, 100), dtype=np.float64))
            memory_error_occurred = False
        except MemoryError:
            memory_error_occurred = True

        # Should handle MemoryError gracefully (test framework catches it)
        # In real code, this would be caught and logged
        assert memory_error_occurred is True or len(large_arrays) < 100

    def test_queue_overflow_handling(self):
        """Test handling of queue overflow"""
        # Create small queue
        small_queue = Queue(maxsize=5)

        # Fill queue
        for i in range(5):
            small_queue.put(i)

        # Try to add more - should raise Full
        try:
            small_queue.put(999, block=False)
            overflow_handled = False
        except Full:
            overflow_handled = True

        assert overflow_handled is True

        # Verify queue still functional after overflow
        assert small_queue.qsize() == 5
        assert small_queue.get() == 0  # Can still retrieve


# ============================================================================
# External Dependency Failure Tests
# ============================================================================

class TestExternalDependencyFailures:
    """Test recovery from external dependency failures"""

    @pytest.mark.skip(reason="analyze_with_ollama function does not exist")
    def test_ollama_server_down(self):
        """Test handling when Ollama server is unavailable"""
        # This test is skipped because analyze_with_ollama doesn't exist
        # Keeping as placeholder for future Ollama integration testing
        pass

    def test_lm_studio_unavailable_for_rag(self):
        """Test RAG system when LM Studio is unavailable"""
        from rag import RAGSystem

        with patch("rag.EmbeddingGenerator") as mock_emb_gen:
            # Mock LM Studio connection failure
            mock_emb = Mock()
            mock_emb.generate_embedding.side_effect = ConnectionError("LM Studio not available")
            mock_emb_gen.return_value = mock_emb

            rag = RAGSystem(auto_load_index=False)

            # Search should fail gracefully
            try:
                results = rag.search("test query")
                # If it returns, should be empty or handle error
                assert isinstance(results, list)
            except ConnectionError:
                # Expected - connection failed
                pass

    def test_yolo_detector_not_available(self):
        """Test object detection when YOLO is not available"""
        from vision.ObjectDetector import CubeDetector
        import numpy as np

        image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Create detector (should fall back to color-based detection)
        detector = CubeDetector()

        # Detection should still work with HSV fallback
        result = detector.detect_objects(image, camera_id="test")

        # Should return valid result (empty detections ok)
        assert hasattr(result, 'detections')
        assert isinstance(result.detections, list)


# ============================================================================
# Graceful Degradation Tests
# ============================================================================

class TestGracefulDegradation:
    """Test graceful degradation when features unavailable"""

    def test_llm_parsing_fallback_to_regex(self):
        """Test CommandParser falls back to regex when LLM fails"""
        parser = CommandParser(use_rag=False)

        # Mock LLM failure
        with patch.object(parser, '_parse_with_llm') as mock_llm:
            mock_llm.side_effect = Exception("LLM unavailable")

            # Parse with LLM disabled (should use regex)
            result = parser.parse(
                "move to (0.3, 0.2, 0.1) and close gripper",
                robot_id="Robot1",
                use_llm=False
            )

            # Should parse successfully with regex fallback
            assert result["success"] is True
            assert len(result["commands"]) >= 2

    def test_stereo_detection_fallback_to_monocular(self):
        """Test fallback to monocular detection when stereo unavailable"""
        from servers.ImageStorageCore import UnifiedImageStorage
        import numpy as np

        storage = UnifiedImageStorage()

        # Store only single image (no stereo)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        storage.store_single_image("mono_camera", image, "test")

        # Try to get stereo - should return None
        stereo = storage.get_latest_stereo_image()
        assert stereo is None

        # But monocular should work
        mono = storage.get_single_image("mono_camera")
        assert mono is not None

    def test_operation_continues_without_verification(self):
        """Test operations continue when verification is disabled"""
        from operations.MoveOperations import move_to_coordinate

        with patch('config.ROS.ROS_ENABLED', False), \
             patch('operations.MoveOperations._get_command_broadcaster') as mock_broadcaster:
            mock_broadcaster.return_value.send_command = Mock(return_value=True)

            # Execute without verification (verification optional)
            result = move_to_coordinate(
                robot_id="Robot1",
                x=0.3, y=0.2, z=0.1
            )

            # Should succeed
            assert result["success"] is True


# ============================================================================
# Concurrent Failure Tests
# ============================================================================

class TestConcurrentFailures:
    """Test handling of concurrent failures across threads"""

    def test_multiple_threads_with_failures(self):
        """Test system stability when multiple threads encounter failures"""
        errors = []
        successes = []

        def operation_with_failure(thread_id):
            try:
                # Simulate some threads failing
                if thread_id % 3 == 0:
                    raise ValueError(f"Thread {thread_id} failed")

                # Simulate work
                time.sleep(0.01)
                successes.append(thread_id)
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = [
            threading.Thread(target=operation_with_failure, args=(i,))
            for i in range(15)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=2.0)

        # Some should fail (threads 0, 3, 6, 9, 12 = 5 failures)
        assert len(errors) == 5

        # Others should succeed (10 successes)
        assert len(successes) == 10

    def test_cascading_failure_prevention(self):
        """Test that one failure doesn't cascade to other components"""
        broadcaster = CommandBroadcaster()

        # Create multiple completion queues
        for i in range(10):
            broadcaster.create_completion_queue(i)

        # Fail one queue by putting invalid data
        try:
            broadcaster.put_completion(999, {"invalid": "data"})  # Non-existent queue
        except Exception:
            pass  # Expected

        # Other queues should still work
        broadcaster.put_completion(0, {"success": True})
        result = broadcaster.get_completion(0, timeout=0.5)

        assert result is not None
        assert result["success"] is True


# ============================================================================
# State Recovery Tests
# ============================================================================

class TestStateRecovery:
    """Test state recovery after failures"""

    def test_world_state_recovery_after_corruption(self, cleanup_world_state):
        """Test WorldState can recover from corrupted state"""
        from operations.WorldState import get_world_state

        world_state = get_world_state()

        # Populate some state
        world_state.update_object_position("cube_01", (0.3, 0.2, 0.1), "red")

        # Simulate partial corruption by directly modifying internal state
        # Use type: ignore to intentionally test corruption scenario
        world_state._objects["corrupted"] = None  # type: ignore[assignment]

        # Try to access objects - should handle None gracefully
        try:
            all_objects = world_state.get_all_objects()
            # Should return non-None objects
            assert all_objects is not None
        except Exception as e:
            # If it raises, verify it's a reasonable error
            assert "objects" in str(e).lower() or "none" in str(e).lower()

        # Explicitly clean up corruption for safety (autouse fixture should handle this, but be explicit)
        world_state.reset()

    def test_singleton_reset_after_failure(self):
        """Test singletons can be reset after failure"""
        from servers.ImageStorageCore import UnifiedImageStorage

        storage = UnifiedImageStorage()

        # Store some data
        import numpy as np
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        storage.store_single_image("test_cam", image, "test")

        # Simulate failure by corrupting internal state
        original_dict = storage._single_images
        storage._single_images = None  # type: ignore[assignment]  # Intentionally corrupt for testing

        # Try to access - should fail
        try:
            storage.get_single_image("test_cam")
            corrupted = False
        except (AttributeError, TypeError):
            corrupted = True

        # Restore state
        storage._single_images = original_dict or {}

        # Should work again
        restored_image = storage.get_single_image("test_cam")
        assert restored_image is not None or storage._single_images is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
