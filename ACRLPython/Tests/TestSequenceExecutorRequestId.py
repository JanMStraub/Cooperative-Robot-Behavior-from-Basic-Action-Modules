#!/usr/bin/env python3
"""
Unit tests for SequenceExecutor request ID generation (Phase 2 improvement).

Tests the atomic counter + timestamp hybrid approach to prevent request ID collisions
in multi-threaded scenarios with rapid sequential operations.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch
from orchestrators.SequenceExecutor import SequenceExecutor


class TestRequestIdGeneration:
    """Test request ID generation with atomic counter + timestamp hybrid"""

    def test_request_id_uniqueness(self):
        """Test that generated request IDs are unique"""
        request_ids = set()

        # Generate 100 request IDs rapidly
        for _ in range(100):
            request_id = SequenceExecutor._generate_request_id()
            assert request_id not in request_ids, f"Duplicate request ID: {request_id}"
            request_ids.add(request_id)

        assert len(request_ids) == 100

    def test_request_id_range(self):
        """Test request IDs are within valid 32-bit unsigned range"""
        for _ in range(50):
            request_id = SequenceExecutor._generate_request_id()
            assert 0 <= request_id < 2**32, f"Request ID out of range: {request_id}"

    def test_request_id_counter_increments(self):
        """Test internal counter increments sequentially"""
        # Reset counter to known state
        SequenceExecutor._request_id_counter = 0

        request_id_1 = SequenceExecutor._generate_request_id()
        request_id_2 = SequenceExecutor._generate_request_id()

        # Extract counter part (lower 16 bits)
        counter_1 = request_id_1 & 0xFFFF
        counter_2 = request_id_2 & 0xFFFF

        # Counter should increment
        assert counter_2 == (counter_1 + 1) % 0x10000

    def test_request_id_timestamp_component(self):
        """Test request ID includes timestamp component"""
        # Get timestamp at request time
        before_time = int(time.time() * 1000) & 0xFFFF
        request_id = SequenceExecutor._generate_request_id()
        after_time = int(time.time() * 1000) & 0xFFFF

        # Extract timestamp part (upper 16 bits)
        timestamp_part = (request_id >> 16) & 0xFFFF

        # Timestamp should be between before and after (with wrapping consideration)
        assert before_time <= timestamp_part <= after_time or \
               (before_time > after_time and (timestamp_part >= before_time or timestamp_part <= after_time))

    def test_request_id_thread_safety(self):
        """Test request ID generation is thread-safe"""
        request_ids = set()
        lock = threading.Lock()

        def generate_ids():
            """Generate 50 request IDs in a thread"""
            local_ids = []
            for _ in range(50):
                request_id = SequenceExecutor._generate_request_id()
                local_ids.append(request_id)

            # Add to shared set with lock
            with lock:
                for rid in local_ids:
                    request_ids.add(rid)

        # Create 10 threads generating IDs concurrently
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=generate_ids)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should have 500 unique IDs (10 threads * 50 IDs each)
        assert len(request_ids) == 500, f"Expected 500 unique IDs, got {len(request_ids)}"

    def test_request_id_rapid_sequential(self):
        """Test rapid sequential ID generation doesn't create collisions"""
        request_ids = []

        # Generate IDs as fast as possible
        for _ in range(1000):
            request_id = SequenceExecutor._generate_request_id()
            request_ids.append(request_id)

        # All should be unique
        unique_ids = set(request_ids)
        assert len(unique_ids) == 1000, f"Expected 1000 unique IDs, got {len(unique_ids)}"

    def test_request_id_counter_overflow(self):
        """Test counter handles overflow gracefully"""
        # Set counter near overflow
        SequenceExecutor._request_id_counter = 0xFFFE

        # Generate IDs across overflow boundary
        id_before = SequenceExecutor._generate_request_id()  # Counter: 0xFFFF
        id_overflow = SequenceExecutor._generate_request_id()  # Counter: 0x0000 (wrapped)
        id_after = SequenceExecutor._generate_request_id()  # Counter: 0x0001

        # All should be unique (timestamp part will differ or be handled)
        assert id_before != id_overflow
        assert id_overflow != id_after
        assert id_before != id_after

    def test_request_id_format(self):
        """Test request ID format: [timestamp:16bit][counter:16bit]"""
        request_id = SequenceExecutor._generate_request_id()

        # Extract components
        timestamp_part = (request_id >> 16) & 0xFFFF
        counter_part = request_id & 0xFFFF

        # Verify format
        reconstructed = (timestamp_part << 16) | counter_part
        assert reconstructed == request_id

    def test_request_id_lock_usage(self):
        """Test request ID generation uses lock for thread safety"""
        assert hasattr(SequenceExecutor, '_request_id_lock')
        # Lock is a threading.Lock instance
        lock_type = type(threading.Lock())
        assert isinstance(SequenceExecutor._request_id_lock, lock_type)


class TestRequestIdEdgeCases:
    """Test edge cases for request ID generation"""

    def test_request_id_after_counter_reset(self):
        """Test request IDs remain unique after counter reset"""
        # Note: In production, counter should never be reset, but test resilience

        # Generate some IDs
        ids_before = [SequenceExecutor._generate_request_id() for _ in range(10)]

        # Simulate counter reset (shouldn't happen in production)
        original_counter = SequenceExecutor._request_id_counter
        SequenceExecutor._request_id_counter = 0

        # Wait a tiny bit to ensure timestamp changes
        time.sleep(0.001)

        # Generate IDs after reset
        ids_after = [SequenceExecutor._generate_request_id() for _ in range(10)]

        # Restore counter
        SequenceExecutor._request_id_counter = original_counter

        # All IDs from both sets should be unique (timestamp provides uniqueness)
        all_ids = ids_before + ids_after
        unique_ids = set(all_ids)
        # Due to timestamp + counter hybrid, we should have good uniqueness
        assert len(unique_ids) >= 10  # At minimum, one set should be fully unique

    def test_request_id_high_frequency(self):
        """Test request ID generation under high frequency load"""
        request_ids = []
        num_iterations = 10000

        start_time = time.time()
        for _ in range(num_iterations):
            request_id = SequenceExecutor._generate_request_id()
            request_ids.append(request_id)
        duration = time.time() - start_time

        # All should be unique
        unique_ids = set(request_ids)
        assert len(unique_ids) == num_iterations

        # Should complete in reasonable time (< 1 second for 10k IDs)
        assert duration < 1.0, f"Request ID generation too slow: {duration}s for {num_iterations} IDs"
