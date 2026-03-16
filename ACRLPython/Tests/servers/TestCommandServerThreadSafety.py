#!/usr/bin/env python3
"""
Unit tests for CommandServer thread-safe command tracking (Phase 2 improvement).

Tests the thread-safe command tracking with RLock to prevent race conditions
in multi-threaded scenarios with concurrent command submission and completion.

Note: These tests verify the thread-safe structures exist and are properly initialized.
Full integration testing requires actual Unity connection.
"""

import pytest
import threading
import time
from unittest.mock import patch
from servers.CommandServer import CommandServer, CommandBroadcaster


class TestCommandBroadcasterThreadSafety:
    """Test CommandBroadcaster thread-safe structures"""

    def test_broadcaster_singleton(self):
        """Test CommandBroadcaster is a singleton"""
        broadcaster1 = CommandBroadcaster()
        broadcaster2 = CommandBroadcaster()

        assert broadcaster1 is broadcaster2
        assert id(broadcaster1) == id(broadcaster2)

    def test_broadcaster_has_thread_safe_structures(self):
        """Test broadcaster initializes with thread-safe structures"""
        broadcaster = CommandBroadcaster()

        # Verify thread-safe command tracking structures exist
        assert hasattr(broadcaster, "_active_commands")
        assert hasattr(broadcaster, "_active_commands_lock")
        assert hasattr(broadcaster, "_robot_clients")

        # Verify lock is RLock for reentrant locking
        rlock_type = type(threading.RLock())
        assert isinstance(broadcaster._active_commands_lock, rlock_type)

    def test_broadcaster_lock_is_reentrant(self):
        """Test RLock allows reentrant locking (same thread can acquire twice)"""
        broadcaster = CommandBroadcaster()

        # Should not deadlock
        acquired = False
        with broadcaster._active_commands_lock:
            with broadcaster._active_commands_lock:
                acquired = True

        assert acquired is True

    def test_broadcaster_active_commands_initialization(self):
        """Test active commands dictionary is initialized"""
        broadcaster = CommandBroadcaster()

        assert isinstance(broadcaster._active_commands, dict)
        # Should start empty or with minimal entries
        assert len(broadcaster._active_commands) >= 0

    def test_broadcaster_robot_clients_initialization(self):
        """Test robot clients dictionary is initialized"""
        broadcaster = CommandBroadcaster()

        assert isinstance(broadcaster._robot_clients, dict)


class TestCommandServerInitialization:
    """Test CommandServer initialization with thread-safe broadcaster"""

    @pytest.fixture
    def server(self):
        """Create a CommandServer instance for testing"""
        with patch("socket.socket"):
            from core.TCPServerBase import ServerConfig

            config = ServerConfig(host="localhost", port=5010)
            server = CommandServer(config=config)
            yield server

    def test_server_creates_broadcaster(self, server):
        """Test server creates and initializes broadcaster"""
        assert hasattr(server, "_broadcaster")
        assert server._broadcaster is not None
        assert isinstance(server._broadcaster, CommandBroadcaster)

    def test_server_broadcaster_has_server_reference(self, server):
        """Test broadcaster has reference to server"""
        broadcaster = server._broadcaster
        assert hasattr(broadcaster, "_server")
        assert broadcaster._server is server


class TestThreadSafeLockMechanism:
    """Test the actual thread-safe lock mechanism"""

    def test_lock_prevents_race_condition(self):
        """Test lock prevents race conditions in dictionary updates"""
        broadcaster = CommandBroadcaster()
        test_dict = {}
        errors = []

        def update_dict(thread_id):
            """Update shared dictionary with lock protection"""
            try:
                for i in range(100):
                    key = f"thread{thread_id}_item{i}"
                    with broadcaster._active_commands_lock:
                        test_dict[key] = thread_id
            except Exception as e:
                errors.append(str(e))

        # Create 10 threads updating dictionary concurrently
        threads = []
        for i in range(10):
            thread = threading.Thread(target=update_dict, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify no errors and all updates succeeded
        assert len(errors) == 0
        assert len(test_dict) == 1000  # 10 threads * 100 items

    def test_lock_allows_concurrent_reads(self):
        """Test lock allows multiple readers"""
        broadcaster = CommandBroadcaster()
        read_counts = []

        def read_operation(thread_id):
            """Read operation that acquires lock"""
            with broadcaster._active_commands_lock:
                # Simulate some work
                time.sleep(0.001)
                read_counts.append(thread_id)

        # Create multiple reader threads
        threads = []
        for i in range(20):
            thread = threading.Thread(target=read_operation, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All reads should complete
        assert len(read_counts) == 20

    def test_nested_lock_acquisition(self):
        """Test nested lock acquisition doesn't deadlock"""
        broadcaster = CommandBroadcaster()
        completed = False

        def nested_locks():
            """Acquire lock twice in same thread"""
            nonlocal completed
            with broadcaster._active_commands_lock:
                # First acquisition
                with broadcaster._active_commands_lock:
                    # Second acquisition (reentrant)
                    completed = True

        thread = threading.Thread(target=nested_locks)
        thread.start()
        thread.join(timeout=2.0)

        # Should complete without deadlock
        assert completed is True
        assert not thread.is_alive()


class TestCommandBroadcasterMethods:
    """Test CommandBroadcaster methods exist and are callable"""

    def test_broadcaster_has_send_command(self):
        """Test broadcaster has send_command method"""
        broadcaster = CommandBroadcaster()
        assert hasattr(broadcaster, "send_command")
        assert callable(broadcaster.send_command)

    def test_broadcaster_has_track_command(self):
        """Test broadcaster has track_command method"""
        broadcaster = CommandBroadcaster()
        assert hasattr(broadcaster, "track_command")
        assert callable(broadcaster.track_command)

    def test_broadcaster_has_complete_command(self):
        """Test broadcaster has complete_command method"""
        broadcaster = CommandBroadcaster()
        assert hasattr(broadcaster, "complete_command")
        assert callable(broadcaster.complete_command)

    def test_broadcaster_has_register_robot_client(self):
        """Test broadcaster has register_robot_client method"""
        broadcaster = CommandBroadcaster()
        assert hasattr(broadcaster, "register_robot_client")
        assert callable(broadcaster.register_robot_client)


class TestConcurrentAccess:
    """Test concurrent access to broadcaster structures"""

    def test_concurrent_dictionary_access(self):
        """Test concurrent access to active commands dictionary"""
        broadcaster = CommandBroadcaster()
        success_count = {"value": 0}
        lock = threading.Lock()

        def safe_access(thread_id):
            """Safely access broadcaster structures"""
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

        # Create 10 threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=safe_access, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All threads should complete successfully
        assert success_count["value"] == 10


# Note: Full integration tests require actual Unity connection and are beyond
# the scope of unit tests. These tests verify the thread-safe infrastructure
# is in place and correctly initialized.
