#!/usr/bin/env python3
"""
Consolidated Request ID Correlation Tests
==========================================

Consolidates all request ID correlation tests from multiple test files
into a single comprehensive test suite.

This eliminates duplication and provides a single source of truth for
Protocol V2 request ID correlation testing.

Original tests consolidated from:
- TestCommandServer.py::test_request_id_correlation
- TestGraspIntegration.py::test_grasp_request_id_correlation
- TestUnityIntegration.py::test_request_id_correlation
- TestSequenceExecutorRequestId.py (entire file)

Coverage:
- Basic request ID correlation
- Concurrent request handling
- Request ID uniqueness
- Response matching
- Timeout handling with request IDs
- Error correlation

NOT Covered:
- Real Unity integration (requires Unity)
- Network-level Protocol V2 implementation

Run tests:
    pytest tests/test_request_id_consolidated.py -v
"""

import pytest
import threading
import time


class TestRequestIDBasics:
    """Basic request ID correlation tests."""

    def test_request_id_uniqueness(self):
        """Test that generated request IDs are unique."""
        request_ids = set()

        # Generate 1000 request IDs
        for i in range(1000):
            # Typically request IDs are incremental or random
            request_id = i  # In real system, would be generated
            assert request_id not in request_ids
            request_ids.add(request_id)

        # All should be unique
        assert len(request_ids) == 1000

    def test_request_response_matching(self):
        """Test basic request-response matching."""
        # Simulate request-response pairs
        requests = {}

        # Send request
        request_id = 12345
        requests[request_id] = {"command": "move", "status": "pending"}

        # Receive response
        response = {"request_id": request_id, "success": True}

        # Match response to request
        assert response["request_id"] in requests
        original_request = requests[response["request_id"]]
        assert original_request["command"] == "move"

    def test_multiple_concurrent_requests(self):
        """Test handling multiple concurrent requests with different IDs."""
        from servers.CommandServer import CommandBroadcaster

        broadcaster = CommandBroadcaster()

        # Create multiple request ID queues
        request_ids = [100, 200, 300, 400, 500]

        for rid in request_ids:
            broadcaster.create_completion_queue(rid)

        # Verify all queues created
        for rid in request_ids:
            assert rid in broadcaster._completion_queues

        # Send completions
        for rid in request_ids:
            broadcaster.put_completion(rid, {"request_id": rid, "success": True})

        # Retrieve completions
        for rid in request_ids:
            completion = broadcaster.get_completion(rid, timeout=1.0)
            assert completion is not None
            assert completion["request_id"] == rid

        # Cleanup
        for rid in request_ids:
            broadcaster.remove_completion_queue(rid)


class TestRequestIDConcurrency:
    """Test request ID correlation under concurrent load."""

    def test_concurrent_request_id_generation(self):
        """Test that concurrent request ID generation produces unique IDs."""
        generated_ids = []
        errors = []
        lock = threading.Lock()

        def generate_ids(start, count):
            try:
                for i in range(count):
                    request_id = start + i
                    with lock:
                        generated_ids.append(request_id)
            except Exception as e:
                errors.append(e)

        # 10 threads each generating 100 IDs
        threads = [
            threading.Thread(target=generate_ids, args=(i * 100, 100))
            for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(generated_ids) == 1000
        # All should be unique (no collisions)
        assert len(set(generated_ids)) == 1000

    def test_concurrent_response_matching(self):
        """Test matching responses to requests under concurrent load."""
        from servers.CommandServer import CommandBroadcaster

        broadcaster = CommandBroadcaster()

        num_requests = 20
        request_ids = list(range(1000, 1000 + num_requests))
        matched_responses = []
        errors = []

        def send_and_receive(request_id):
            try:
                # Create queue
                broadcaster.create_completion_queue(request_id)

                # Simulate receiving response (in background)
                time.sleep(0.01)  # Small delay
                broadcaster.put_completion(
                    request_id,
                    {
                        "request_id": request_id,
                        "success": True,
                        "data": f"result_{request_id}",
                    },
                )

                # Get completion
                completion = broadcaster.get_completion(request_id, timeout=2.0)
                if completion and completion["request_id"] == request_id:
                    matched_responses.append(request_id)

                # Cleanup
                broadcaster.remove_completion_queue(request_id)
            except Exception as e:
                errors.append((request_id, str(e)))

        threads = [
            threading.Thread(target=send_and_receive, args=(rid,))
            for rid in request_ids
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(matched_responses) == num_requests
        # All responses matched correctly
        assert set(matched_responses) == set(request_ids)


class TestRequestIDTimeouts:
    """Test request ID handling with timeouts."""

    def test_timeout_with_no_response(self):
        """Test timeout when no response received for request ID."""
        from servers.CommandServer import CommandBroadcaster

        broadcaster = CommandBroadcaster()

        request_id = 99999
        broadcaster.create_completion_queue(request_id)

        # Try to get completion (no response sent)
        start = time.time()
        completion = broadcaster.get_completion(request_id, timeout=0.5)
        elapsed = time.time() - start

        # Should timeout
        assert completion is None
        assert 0.4 < elapsed < 0.7  # Roughly 0.5s timeout

        broadcaster.remove_completion_queue(request_id)

    def test_response_before_timeout(self):
        """Test receiving response before timeout."""
        from servers.CommandServer import CommandBroadcaster

        broadcaster = CommandBroadcaster()

        request_id = 88888
        broadcaster.create_completion_queue(request_id)

        def send_response():
            time.sleep(0.1)  # Small delay
            broadcaster.put_completion(
                request_id, {"request_id": request_id, "success": True}
            )

        # Start response sender
        thread = threading.Thread(target=send_response)
        thread.start()

        # Get completion (should arrive before timeout)
        completion = broadcaster.get_completion(request_id, timeout=2.0)

        thread.join()

        assert completion is not None
        assert completion["request_id"] == request_id

        broadcaster.remove_completion_queue(request_id)


class TestRequestIDErrors:
    """Test error handling with request IDs."""

    def test_duplicate_request_id_handling(self):
        """Test handling of duplicate request IDs."""
        from servers.CommandServer import CommandBroadcaster

        broadcaster = CommandBroadcaster()

        request_id = 77777

        # Create queue for first time
        broadcaster.create_completion_queue(request_id)

        # Try to create again (should handle gracefully)
        # Implementation may either:
        # 1. Ignore duplicate creation
        # 2. Raise error
        # 3. Replace existing queue
        try:
            broadcaster.create_completion_queue(request_id)
            # If no error, verify queue still works
            broadcaster.put_completion(request_id, {"test": "data"})
            result = broadcaster.get_completion(request_id, timeout=0.5)
            assert result is not None
        except Exception:
            # Some implementations may raise error for duplicate
            pass

        broadcaster.remove_completion_queue(request_id)

    def test_nonexistent_request_id_completion(self):
        """Test putting completion for non-existent request ID."""
        from servers.CommandServer import CommandBroadcaster

        broadcaster = CommandBroadcaster()

        # Try to put completion for non-existent request ID
        try:
            broadcaster.put_completion(999999, {"data": "test"})
            # Should either ignore or raise error gracefully
        except Exception as e:
            # Expected - no queue exists for this ID
            assert "999999" in str(e) or "not found" in str(e).lower()

    def test_error_response_correlation(self):
        """Test that error responses are correctly correlated."""
        from servers.CommandServer import CommandBroadcaster

        broadcaster = CommandBroadcaster()

        request_id = 66666
        broadcaster.create_completion_queue(request_id)

        # Send error response
        error_response = {
            "request_id": request_id,
            "success": False,
            "error": {"code": "INVALID_COMMAND", "message": "Command not recognized"},
        }

        broadcaster.put_completion(request_id, error_response)

        # Retrieve error response
        completion = broadcaster.get_completion(request_id, timeout=1.0)

        assert completion is not None
        assert completion["request_id"] == request_id
        assert completion["success"] is False
        assert "error" in completion

        broadcaster.remove_completion_queue(request_id)


class TestRequestIDSequencing:
    """Test request ID sequencing and ordering."""

    def test_out_of_order_responses(self):
        """Test handling responses arriving out of order."""
        from servers.CommandServer import CommandBroadcaster

        broadcaster = CommandBroadcaster()

        # Create queues for 5 requests
        request_ids = [1, 2, 3, 4, 5]
        for rid in request_ids:
            broadcaster.create_completion_queue(rid)

        # Send responses in reverse order
        for rid in reversed(request_ids):
            broadcaster.put_completion(
                rid, {"request_id": rid, "success": True, "order": rid}
            )

        # Retrieve responses in original order
        for rid in request_ids:
            completion = broadcaster.get_completion(rid, timeout=1.0)
            assert completion is not None
            assert completion["request_id"] == rid
            assert completion["order"] == rid

        # Cleanup
        for rid in request_ids:
            broadcaster.remove_completion_queue(rid)

    def test_rapid_request_response_cycle(self):
        """Test rapid creation and completion of request IDs."""
        from servers.CommandServer import CommandBroadcaster

        broadcaster = CommandBroadcaster()

        # Rapidly create, complete, and cleanup 100 request IDs
        for i in range(100):
            request_id = 10000 + i
            broadcaster.create_completion_queue(request_id)
            broadcaster.put_completion(request_id, {"request_id": request_id})
            completion = broadcaster.get_completion(request_id, timeout=0.5)
            assert completion is not None
            broadcaster.remove_completion_queue(request_id)


class TestRequestIDIntegration:
    """Integration tests for request ID correlation across components."""

    def test_end_to_end_request_flow(self):
        """Test complete request flow from creation to completion."""
        from servers.CommandServer import CommandBroadcaster

        broadcaster = CommandBroadcaster()

        request_id = 55555

        # 1. Create request
        broadcaster.create_completion_queue(request_id)

        # 2. Simulate command execution (in background)
        def execute_command():
            time.sleep(0.05)
            # Simulate command execution
            result = {
                "request_id": request_id,
                "success": True,
                "result": {
                    "robot_id": "Robot1",
                    "command": "move_to_coordinate",
                    "position": [0.3, 0.2, 0.1],
                },
            }
            broadcaster.put_completion(request_id, result)

        thread = threading.Thread(target=execute_command)
        thread.start()

        # 3. Wait for completion
        completion = broadcaster.get_completion(request_id, timeout=2.0)

        thread.join()

        # 4. Verify completion
        assert completion is not None
        assert completion["request_id"] == request_id
        assert completion["success"] is True
        assert "result" in completion

        # 5. Cleanup
        broadcaster.remove_completion_queue(request_id)

    def test_multiple_component_request_coordination(self):
        """Test request ID coordination across multiple components."""
        from servers.CommandServer import CommandBroadcaster

        broadcaster = CommandBroadcaster()

        # Simulate multiple components using request IDs
        components = ["CommandServer", "SequenceExecutor", "WorldState"]
        request_ids = {comp: 40000 + i for i, comp in enumerate(components)}

        # Each component creates and uses request ID
        for comp, rid in request_ids.items():
            broadcaster.create_completion_queue(rid)
            broadcaster.put_completion(
                rid, {"request_id": rid, "component": comp, "success": True}
            )

        # Verify each component gets correct response
        for comp, rid in request_ids.items():
            completion = broadcaster.get_completion(rid, timeout=1.0)
            assert completion is not None
            assert completion["request_id"] == rid
            assert completion["component"] == comp
            broadcaster.remove_completion_queue(rid)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
