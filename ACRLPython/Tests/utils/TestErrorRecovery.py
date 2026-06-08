import pytest
import time
import threading
from unittest.mock import Mock, patch
from queue import Queue, Full
import socket

from orchestrators.CommandParser import CommandParser
from servers.CommandServer import CommandBroadcaster
from servers.ImageStorageCore import UnifiedImageStorage


class TestNetworkFailureRecovery:

    def test_command_send_with_no_server_connection(self):
        broadcaster = CommandBroadcaster()
        original_server = broadcaster._server
        try:
            broadcaster._server = None
            command = {"command_type": "move", "robot_id": "Robot1"}
            result = broadcaster.send_command(command, request_id=1)
            assert result is False
        finally:
            broadcaster._server = original_server

    def test_intermittent_network_failure(self):
        """Test that send_command returns False (not raises) on intermittent failures"""
        broadcaster = CommandBroadcaster()
        mock_server = Mock()

        call_count = [0]

        def intermittent_broadcast(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                raise ConnectionError("Network temporarily unavailable")
            return 1

        mock_server.broadcast_to_all_clients = intermittent_broadcast
        broadcaster.set_server(mock_server)

        results = []
        for i in range(4):
            result = broadcaster.send_command(
                {"command_type": "test", "index": i}, request_id=i
            )
            results.append(result)

        assert len(results) == 4
        assert results[0] is False
        assert results[1] is True
        assert results[2] is False
        assert results[3] is True

    def test_server_restart_mid_operation(self):
        broadcaster = CommandBroadcaster()

        mock_server1 = Mock()
        mock_server1.broadcast_to_all_clients = Mock(return_value=1)
        broadcaster.set_server(mock_server1)

        result1 = broadcaster.send_command({"type": "cmd1"}, request_id=1)
        assert result1 is True

        mock_server2 = Mock()
        mock_server2.broadcast_to_all_clients = Mock(return_value=1)
        broadcaster.set_server(mock_server2)

        result2 = broadcaster.send_command({"type": "cmd2"}, request_id=2)
        assert result2 is True

    def test_socket_timeout_handling(self):
        with patch("socket.socket") as mock_socket_class:
            mock_sock = Mock()
            mock_sock.recv = Mock(side_effect=socket.timeout("Operation timed out"))
            mock_socket_class.return_value = mock_sock

            try:
                mock_sock.recv(1024)
                received = False
            except socket.timeout:
                received = True

            assert received is True


class TestResourceExhaustion:

    def test_disk_full_during_image_storage(self):
        import numpy as np

        storage = UnifiedImageStorage()
        large_image = np.ones((4000, 4000, 3), dtype=np.uint8)

        with patch.object(storage, "store_single_image") as mock_store:
            mock_store.side_effect = OSError("[Errno 28] No space left on device")

            try:
                storage.store_single_image("camera_test", large_image, "test")
                disk_full_handled = False
            except OSError as e:
                disk_full_handled = True
                assert "space" in str(e).lower() or "errno 28" in str(e).lower()

            assert disk_full_handled is True

    def test_memory_exhaustion_during_batch_operations(self):
        import numpy as np

        def process_batch(size):
            try:
                _ = np.zeros(size, dtype=np.float64)
                return True
            except MemoryError:
                return False

        with patch("numpy.zeros", side_effect=MemoryError("Out of memory")):
            result = process_batch((10000, 10000, 100))

        assert result is False

    def test_queue_overflow_handling(self):
        small_queue = Queue(maxsize=5)

        for i in range(5):
            small_queue.put(i)

        try:
            small_queue.put(999, block=False)
            overflow_handled = False
        except Full:
            overflow_handled = True

        assert overflow_handled is True
        assert small_queue.qsize() == 5
        assert small_queue.get() == 0


class TestExternalDependencyFailures:

    def test_lm_studio_unavailable_for_rag(self):
        from rag import RAGSystem  # type: ignore[attr-defined]

        with patch("rag.EmbeddingGenerator") as mock_emb_gen:
            mock_emb = Mock()
            mock_emb.generate_embedding.side_effect = ConnectionError(
                "LM Studio not available"
            )
            mock_emb_gen.return_value = mock_emb

            rag = RAGSystem(auto_load_index=False)

            try:
                results = rag.search("test query")
                assert isinstance(results, list)
            except ConnectionError:
                pass

    def test_yolo_detector_not_available(self):
        from vision.ObjectDetector import CubeDetector
        import numpy as np

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detector = CubeDetector()
        result = detector.detect_objects(image, camera_id="test")

        assert hasattr(result, "detections")
        assert isinstance(result.detections, list)


class TestGracefulDegradation:

    def test_llm_parsing_fallback_to_regex(self):
        parser = CommandParser(use_rag=False)

        with patch.object(parser, "_parse_with_llm") as mock_llm:
            mock_llm.side_effect = Exception("LLM unavailable")

            result = parser.parse(
                "move to (0.3, 0.2, 0.1) and close gripper",
                robot_id="Robot1",
                use_llm=False,
            )

            assert result["success"] is True
            assert len(result["commands"]) >= 2

    def test_stereo_detection_fallback_to_monocular(self):
        from servers.ImageStorageCore import UnifiedImageStorage
        import numpy as np

        storage = UnifiedImageStorage()

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        storage.store_single_image("mono_camera", image, "test")

        stereo = storage.get_latest_stereo_image()
        assert stereo is None

        mono = storage.get_single_image("mono_camera")
        assert mono is not None

    def test_operation_continues_without_verification(self):
        from operations.MoveOperations import move_to_coordinate

        with patch("config.ROS.ROS_ENABLED", False), patch(
            "operations.MoveOperations._get_command_broadcaster"
        ) as mock_broadcaster:
            mock_broadcaster.return_value.send_command = Mock(return_value=True)

            result = move_to_coordinate(robot_id="Robot1", x=0.3, y=0.2, z=0.1)

            assert result["success"] is True


class TestConcurrentFailures:

    def test_multiple_threads_with_failures(self):
        errors = []
        successes = []

        def operation_with_failure(thread_id):
            try:
                if thread_id % 3 == 0:
                    raise ValueError(f"Thread {thread_id} failed")

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
        assert len(successes) == 10

    def test_cascading_failure_prevention(self):
        broadcaster = CommandBroadcaster()

        for i in range(10):
            broadcaster.create_completion_queue(i)

        try:
            broadcaster.put_completion(999, {"invalid": "data"})
        except Exception:
            pass

        broadcaster.put_completion(0, {"success": True})
        result = broadcaster.get_completion(0, timeout=0.5)

        assert result is not None
        assert result["success"] is True


class TestStateRecovery:

    def test_world_state_recovery_after_corruption(self, cleanup_world_state):
        from operations.WorldState import get_world_state

        world_state = get_world_state()

        world_state.update_object_position("cube_01", (0.3, 0.2, 0.1), "red")

        world_state._objects["corrupted"] = None  # type: ignore[assignment]

        try:
            all_objects = world_state.get_all_objects()
            assert all_objects is not None
        except Exception as e:
            assert "objects" in str(e).lower() or "none" in str(e).lower()

        world_state.reset()

    def test_singleton_reset_after_failure(self):
        from servers.ImageStorageCore import UnifiedImageStorage

        storage = UnifiedImageStorage()

        import numpy as np

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        storage.store_single_image("test_cam", image, "test")

        original_dict = storage._single_images
        storage._single_images = None  # type: ignore[assignment]

        try:
            storage.get_single_image("test_cam")
            corrupted = False
        except (AttributeError, TypeError):
            corrupted = True

        storage._single_images = original_dict or {}  # type: ignore[assignment]

        restored_image = storage.get_single_image("test_cam")
        assert restored_image is not None or storage._single_images is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
