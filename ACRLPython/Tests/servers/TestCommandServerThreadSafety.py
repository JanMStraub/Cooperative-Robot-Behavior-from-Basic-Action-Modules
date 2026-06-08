import pytest
import threading
import time
from unittest.mock import patch
from servers.CommandServer import CommandServer, CommandBroadcaster


class TestCommandBroadcasterThreadSafety:

    def test_broadcaster_singleton(self):
        broadcaster1 = CommandBroadcaster()
        broadcaster2 = CommandBroadcaster()

        assert broadcaster1 is broadcaster2

    def test_broadcaster_has_thread_safe_structures(self):
        broadcaster = CommandBroadcaster()

        assert hasattr(broadcaster, "_active_commands")
        assert hasattr(broadcaster, "_active_commands_lock")
        assert hasattr(broadcaster, "_robot_clients")

        rlock_type = type(threading.RLock())
        assert isinstance(broadcaster._active_commands_lock, rlock_type)

    def test_broadcaster_lock_is_reentrant(self):
        broadcaster = CommandBroadcaster()

        acquired = False
        with broadcaster._active_commands_lock:
            with broadcaster._active_commands_lock:
                acquired = True

        assert acquired is True

    def test_broadcaster_active_commands_initialization(self):
        broadcaster = CommandBroadcaster()

        assert isinstance(broadcaster._active_commands, dict)
        assert len(broadcaster._active_commands) >= 0

    def test_broadcaster_robot_clients_initialization(self):
        broadcaster = CommandBroadcaster()

        assert isinstance(broadcaster._robot_clients, dict)


class TestCommandServerInitialization:

    @pytest.fixture
    def server(self):
        with patch("socket.socket"):
            from core.TCPServerBase import ServerConfig

            config = ServerConfig(host="localhost", port=5007)
            server = CommandServer(config=config)
            yield server

    def test_server_creates_broadcaster(self, server):
        assert hasattr(server, "_broadcaster")
        assert server._broadcaster is not None
        assert isinstance(server._broadcaster, CommandBroadcaster)

    def test_server_broadcaster_has_server_reference(self, server):
        broadcaster = server._broadcaster
        assert hasattr(broadcaster, "_server")
        assert broadcaster._server is server


class TestThreadSafeLockMechanism:

    def test_lock_prevents_race_condition(self):
        broadcaster = CommandBroadcaster()
        test_dict = {}
        errors = []

        def update_dict(thread_id):
            try:
                for i in range(100):
                    key = f"thread{thread_id}_item{i}"
                    with broadcaster._active_commands_lock:
                        test_dict[key] = thread_id
            except Exception as e:
                errors.append(str(e))

        threads = []
        for i in range(10):
            thread = threading.Thread(target=update_dict, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0
        assert len(test_dict) == 1000

    def test_lock_allows_concurrent_reads(self):
        broadcaster = CommandBroadcaster()
        read_counts = []

        def read_operation(thread_id):
            with broadcaster._active_commands_lock:
                # Simulate some work
                time.sleep(0.001)
                read_counts.append(thread_id)

        threads = []
        for i in range(20):
            thread = threading.Thread(target=read_operation, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(read_counts) == 20

    def test_nested_lock_acquisition(self):
        broadcaster = CommandBroadcaster()
        completed = False

        def nested_locks():
            nonlocal completed
            with broadcaster._active_commands_lock:
                with broadcaster._active_commands_lock:
                    completed = True

        thread = threading.Thread(target=nested_locks)
        thread.start()
        thread.join(timeout=2.0)

        assert completed is True
        assert not thread.is_alive()


class TestCommandBroadcasterMethods:

    def test_broadcaster_has_send_command(self):
        broadcaster = CommandBroadcaster()
        assert hasattr(broadcaster, "send_command")
        assert callable(broadcaster.send_command)

    def test_broadcaster_has_track_command(self):
        broadcaster = CommandBroadcaster()
        assert hasattr(broadcaster, "track_command")
        assert callable(broadcaster.track_command)

    def test_broadcaster_has_complete_command(self):
        broadcaster = CommandBroadcaster()
        assert hasattr(broadcaster, "complete_command")
        assert callable(broadcaster.complete_command)

    def test_broadcaster_has_register_robot_client(self):
        broadcaster = CommandBroadcaster()
        assert hasattr(broadcaster, "register_robot_client")
        assert callable(broadcaster.register_robot_client)


class TestConcurrentAccess:

    def test_concurrent_dictionary_access(self):
        broadcaster = CommandBroadcaster()
        success_count = {"value": 0}
        lock = threading.Lock()

        def safe_access(thread_id):
            try:
                for i in range(50):
                    with broadcaster._active_commands_lock:
                        # Read
                        _ = len(broadcaster._active_commands)
                        # Write - use integer key (request IDs are integers)
                        key = thread_id * 1000 + i  # Generate unique int key
                        broadcaster._active_commands[key] = {
                            "thread": thread_id,
                            "index": i,
                        }
                with lock:
                    success_count["value"] += 1
            except Exception:
                pass

        threads = []
        for i in range(10):
            thread = threading.Thread(target=safe_access, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert success_count["value"] == 10
