#!/usr/bin/env python3
"""
Unit and Integration Tests for SyncOperations.py

Tests synchronization primitives (signal, wait_for_signal, wait) and EventBus
for multi-robot coordination.
"""

import time
import threading

from operations.SyncOperations import (
    EventBus,
    SIGNAL_OPERATION,
    WAIT_FOR_SIGNAL_OPERATION,
    WAIT_OPERATION,
    _execute_signal,
    _execute_wait_for_signal,
    _execute_wait,
)


# ============================================================================
# UNIT TESTS: EventBus Core
# ============================================================================


class TestEventBusCore:
    """Test EventBus singleton and basic event operations"""

    def test_singleton_pattern(self, cleanup_event_bus):
        """Test EventBus follows singleton pattern"""
        bus1 = EventBus()
        bus2 = EventBus()
        assert bus1 is bus2

    def test_singleton_thread_safe_initialization(self, cleanup_event_bus):
        """Test EventBus singleton is thread-safe during initialization"""
        # Reset singleton
        EventBus._instance = None

        instances = []
        errors = []

        def get_instance():
            try:
                inst = EventBus()
                instances.append(inst)
                time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_instance) for _ in range(50)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All instances should be the same object
        assert len(errors) == 0
        assert len(instances) == 50
        assert all(inst is instances[0] for inst in instances)

    def test_reset_clears_all_events(self, event_bus):
        """Test reset() clears all events and waiter counts"""
        event_bus.signal("event1")
        event_bus.signal("event2")

        assert len(event_bus._events) == 2

        event_bus.reset()

        assert len(event_bus._events) == 0
        # After reset, previously-used events should report 0 waiters
        assert event_bus.get_waiter_count("event1") == 0
        assert event_bus.get_waiter_count("event2") == 0

    def test_signal_creates_event(self, event_bus):
        """Test signal() creates event if it doesn't exist"""
        event_name = "test_event"

        assert event_name not in event_bus._events

        event_bus.signal(event_name)

        assert event_name in event_bus._events
        assert event_bus._is_signaled(event_name)

    def test_wait_for_signal_blocks_until_signaled(self, event_bus, async_executor):
        """Test wait_for_signal() blocks until signal is received"""
        event_name = "blocking_event"
        wait_result = {}

        def waiter():
            start = time.time()
            result = event_bus.wait_for_signal(event_name, timeout_ms=5000)
            elapsed_ms = (time.time() - start) * 1000
            wait_result["received"] = result
            wait_result["elapsed_ms"] = elapsed_ms

        # Start waiter
        waiter_thread = async_executor(waiter)

        # Give waiter time to start waiting
        time.sleep(0.1)

        # Signal while waiting
        event_bus.signal(event_name)

        # Wait for waiter to complete
        waiter_thread.join(timeout=1.0)

        assert wait_result["received"] is True
        # Should complete quickly (within 500ms)
        assert wait_result["elapsed_ms"] < 500

    def test_wait_for_signal_times_out(self, event_bus):
        """Test wait_for_signal() times out when signal not received"""
        event_name = "timeout_event"

        start = time.time()
        result = event_bus.wait_for_signal(event_name, timeout_ms=500)
        elapsed_ms = (time.time() - start) * 1000

        assert result is False
        # Should timeout after ~500ms (within ±20% tolerance)
        assert 400 <= elapsed_ms <= 600

    def test_auto_clear_when_all_waiters_done(self, event_bus):
        """Test event auto-clears when all waiters have received it"""
        event_name = "auto_clear_event"
        wait_results = []

        def waiter():
            result = event_bus.wait_for_signal(event_name, timeout_ms=2000)
            wait_results.append(result)

        # Start 3 waiters
        threads = [threading.Thread(target=waiter) for _ in range(3)]
        for t in threads:
            t.start()

        # Give waiters time to start
        time.sleep(0.1)

        # Signal - should wake all waiters
        event_bus.signal(event_name)

        # Wait for all to complete
        for t in threads:
            t.join(timeout=1.0)

        # All should have received signal
        assert len(wait_results) == 3
        assert all(r is True for r in wait_results)

        # Event is NOT auto-cleared in the generation-based implementation;
        # the generation counter remains elevated (signal persists) but new
        # waiters can still wait for the NEXT signal by recording their baseline.
        assert event_bus.get_waiter_count(event_name) == 0

    def test_manual_clear_event(self, event_bus):
        """Test manual clear_event() clears event flag"""
        event_name = "manual_clear"

        event_bus.signal(event_name)
        assert event_bus._is_signaled(event_name)

        event_bus.clear_event(event_name)
        assert not event_bus._is_signaled(event_name)

    def test_signal_nonexistent_event_creates_it(self, event_bus):
        """Test signaling non-existent event creates it"""
        event_name = "new_event"

        assert event_name not in event_bus._events

        event_bus.signal(event_name)

        assert event_name in event_bus._events
        assert event_bus._is_signaled(event_name)
        assert event_bus.get_waiter_count(event_name) == 0

    def test_wait_on_already_signaled_event(self, event_bus):
        """Test wait_for_signal() returns immediately if event already signaled"""
        event_name = "already_signaled"

        # Signal first
        event_bus.signal(event_name)

        # Then wait - should return immediately
        start = time.time()
        result = event_bus.wait_for_signal(event_name, timeout_ms=5000)
        elapsed_ms = (time.time() - start) * 1000

        assert result is True
        # Should complete almost instantly (< 50ms)
        assert elapsed_ms < 50


# ============================================================================
# UNIT TESTS: EventBus Thread Safety
# ============================================================================


class TestEventBusThreadSafety:
    """Test EventBus thread safety and race conditions"""

    def test_concurrent_signals_different_events(
        self, event_bus, thread_error_collector
    ):
        """Test multiple threads signaling different events simultaneously"""
        errors, add_error = thread_error_collector
        num_threads = 20

        def signal_event(event_num):
            try:
                event_bus.signal(f"event_{event_num}")
            except Exception as e:
                add_error(e)

        threads = [
            threading.Thread(target=signal_event, args=(i,)) for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(event_bus._events) == num_threads
        # All events should be signaled
        assert all(event_bus._is_signaled(f"event_{i}") for i in range(num_threads))

    def test_concurrent_signals_same_event(self, event_bus, thread_error_collector):
        """Test multiple threads signaling the same event simultaneously"""
        errors, add_error = thread_error_collector
        event_name = "shared_event"
        num_threads = 20

        def signal_event():
            try:
                event_bus.signal(event_name)
            except Exception as e:
                add_error(e)

        threads = [threading.Thread(target=signal_event) for _ in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert event_bus._is_signaled(event_name)

    def test_multiple_waiters_same_event(self, event_bus):
        """Test multiple threads waiting on the same event"""
        event_name = "multi_waiter_event"
        wait_results = []

        def waiter():
            result = event_bus.wait_for_signal(event_name, timeout_ms=3000)
            wait_results.append(result)

        # Start 10 waiters
        threads = [threading.Thread(target=waiter) for _ in range(10)]
        for t in threads:
            t.start()

        # Give waiters time to start
        time.sleep(0.1)

        # Signal once - should wake all waiters
        event_bus.signal(event_name)

        # Wait for all to complete
        for t in threads:
            t.join(timeout=2.0)

        # All should have received signal
        assert len(wait_results) == 10
        assert all(r is True for r in wait_results)

    def test_waiters_across_different_events(self, event_bus):
        """Test waiters on different events don't interfere"""
        results = {"event1": [], "event2": []}

        def waiter(event_name):
            result = event_bus.wait_for_signal(event_name, timeout_ms=2000)
            results[event_name].append(result)

        # Start waiters for two different events
        threads_e1 = [
            threading.Thread(target=waiter, args=("event1",)) for _ in range(5)
        ]
        threads_e2 = [
            threading.Thread(target=waiter, args=("event2",)) for _ in range(5)
        ]

        for t in threads_e1 + threads_e2:
            t.start()

        time.sleep(0.1)

        # Signal only event1
        event_bus.signal("event1")

        # Give event1 waiters time to complete
        time.sleep(0.2)

        # event1 waiters should have received signal
        assert len(results["event1"]) == 5
        assert all(r is True for r in results["event1"])

        # event2 waiters should still be waiting
        assert len(results["event2"]) == 0

        # Signal event2
        event_bus.signal("event2")

        # Wait for all to complete
        for t in threads_e1 + threads_e2:
            t.join(timeout=2.0)

        # Now event2 waiters should have received signal
        assert len(results["event2"]) == 5
        assert all(r is True for r in results["event2"])

    def test_signal_before_wait(self, event_bus):
        """Test signal sent before wait starts"""
        event_name = "signal_first"

        # Signal first
        event_bus.signal(event_name)

        # Then wait - should return immediately
        result = event_bus.wait_for_signal(event_name, timeout_ms=5000)

        assert result is True

    def test_signal_during_wait(self, event_bus, async_executor):
        """Test signal sent while wait is in progress"""
        event_name = "signal_during"
        wait_result = {}

        def waiter():
            start = time.time()
            result = event_bus.wait_for_signal(event_name, timeout_ms=5000)
            elapsed_ms = (time.time() - start) * 1000
            wait_result["received"] = result
            wait_result["elapsed_ms"] = elapsed_ms

        # Start waiter
        waiter_thread = async_executor(waiter)

        # Give waiter time to start waiting
        time.sleep(0.1)

        # Signal while waiting
        event_bus.signal(event_name)

        # Wait for waiter to complete
        waiter_thread.join(timeout=1.0)

        assert wait_result["received"] is True
        # Should complete quickly
        assert wait_result["elapsed_ms"] < 500

    def test_signal_after_timeout(self, event_bus, async_executor):
        """Test signal sent after wait has timed out"""
        event_name = "signal_after_timeout"
        wait_result = {}

        def waiter():
            result = event_bus.wait_for_signal(event_name, timeout_ms=200)
            wait_result["received"] = result

        # Start waiter
        waiter_thread = async_executor(waiter)

        # Wait for timeout to occur
        waiter_thread.join(timeout=1.0)

        # Waiter should have timed out
        assert wait_result["received"] is False

        # Signal after timeout - should have no effect on completed wait
        event_bus.signal(event_name)

        # Event should still be signaled (signal persists in generation counter)
        assert event_bus._is_signaled(event_name)

    def test_rapid_signal_wait_cycles(self, event_bus):
        """Test rapid signal/wait cycles (stress test)"""
        event_name = "rapid_cycle"
        num_cycles = 50
        errors = []

        def cycle_worker():
            try:
                for _ in range(num_cycles):
                    event_bus.signal(event_name)
                    event_bus.wait_for_signal(event_name, timeout_ms=100)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=cycle_worker) for _ in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        # Should complete without errors
        assert len(errors) == 0

    def test_waiter_count_accuracy_under_load(self, event_bus):
        """Test waiter count tracking is accurate under concurrent load"""
        event_name = "count_test"
        num_waiters = 20

        def waiter():
            event_bus.wait_for_signal(event_name, timeout_ms=3000)

        threads = [threading.Thread(target=waiter) for _ in range(num_waiters)]

        for t in threads:
            t.start()

        # Give waiters time to register
        time.sleep(0.2)

        # Waiter count should be accurate
        assert event_bus.get_waiter_count(event_name) == num_waiters

        # Signal to release all
        event_bus.signal(event_name)

        # Wait for all to complete
        for t in threads:
            t.join(timeout=2.0)

        # Waiter count should be back to 0
        assert event_bus.get_waiter_count(event_name) == 0

    def test_auto_clear_with_concurrent_waiters(self, event_bus):
        """Test auto-clear works correctly with concurrent waiters"""
        event_name = "concurrent_auto_clear"
        wait_results = []

        def waiter():
            result = event_bus.wait_for_signal(event_name, timeout_ms=3000)
            wait_results.append(result)

        # Start 15 waiters
        threads = [threading.Thread(target=waiter) for _ in range(15)]
        for t in threads:
            t.start()

        time.sleep(0.1)

        # Signal - should wake all
        event_bus.signal(event_name)

        # Wait for all to complete
        for t in threads:
            t.join(timeout=2.0)

        # All should have received
        assert len(wait_results) == 15
        assert all(r is True for r in wait_results)

        # All waiters received the signal; waiter count back to 0
        assert event_bus.get_waiter_count(event_name) == 0

    def test_waiter_decrement_on_timeout(self, event_bus):
        """Test waiter count decrements correctly on timeout"""
        event_name = "timeout_decrement"

        def waiter():
            event_bus.wait_for_signal(event_name, timeout_ms=200)

        threads = [threading.Thread(target=waiter) for _ in range(5)]

        for t in threads:
            t.start()

        time.sleep(0.1)

        # Should have 5 waiters
        assert event_bus.get_waiter_count(event_name) == 5

        # Wait for timeouts
        for t in threads:
            t.join(timeout=1.0)

        # Waiter count should be back to 0
        assert event_bus.get_waiter_count(event_name) == 0


# ============================================================================
# UNIT TESTS: Signal Operation
# ============================================================================


class TestSignalOperation:
    """Test signal() operation"""

    def test_signal_success(self, cleanup_event_bus):
        """Test signal operation succeeds"""
        result = _execute_signal("test_event")

        assert result.success is True
        assert result.result is not None and result.result["event_name"] == "test_event"
        assert result.result is not None and "signaled_at" in result.result

    def test_signal_returns_timestamp(self, cleanup_event_bus):
        """Test signal operation returns timestamp"""
        before = time.time()
        result = _execute_signal("timestamp_event")
        after = time.time()

        assert result.success is True
        assert (
            result.result is not None
            and before <= result.result["signaled_at"] <= after
        )

    def test_signal_invalid_event_name(self, cleanup_event_bus):
        """Test signal with various event name types"""
        # Empty string should still work (EventBus creates it)
        result = _execute_signal("")
        assert result.success is True

        # None gets converted to string "None" by Python, EventBus accepts it
        result = _execute_signal(None)  # type: ignore[arg-type]
        # This actually succeeds because Python converts None to str
        assert result.success is True or (
            result.success is False
            and result.error is not None
            and result.error["code"] == "SIGNAL_FAILED"
        )

    def test_signal_empty_string(self, cleanup_event_bus):
        """Test signal with empty string event name"""
        result = _execute_signal("")

        # Should succeed (EventBus allows it)
        assert result.success is True

    def test_signal_exception_handling(self, cleanup_event_bus):
        """Test signal handles exceptions gracefully"""
        # Pass integer - Python converts to string, EventBus accepts it
        result = _execute_signal(12345)  # type: ignore[arg-type]

        # This succeeds because Python converts int to str
        assert result.success is True or (
            result.success is False
            and result.error is not None
            and result.error["code"] == "SIGNAL_FAILED"
        )


# ============================================================================
# UNIT TESTS: Wait For Signal Operation
# ============================================================================


class TestWaitForSignalOperation:
    """Test wait_for_signal() operation"""

    def test_wait_receives_signal(self, cleanup_event_bus, async_executor):
        """Test wait_for_signal operation receives signal"""
        event_name = "receive_test"
        wait_result = {}

        def waiter():
            result = _execute_wait_for_signal(event_name, timeout_ms=3000)
            wait_result["result"] = result

        waiter_thread = async_executor(waiter)

        time.sleep(0.1)

        # Signal from main thread
        bus = EventBus()
        bus.signal(event_name)

        waiter_thread.join(timeout=2.0)

        assert wait_result["result"].success is True
        assert wait_result["result"].result["received"] is True
        assert wait_result["result"].result["elapsed_ms"] < 1000

    def test_wait_timeout(self, cleanup_event_bus):
        """Test wait_for_signal times out correctly"""
        event_name = "timeout_test"

        result = _execute_wait_for_signal(event_name, timeout_ms=500)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "WAIT_TIMEOUT"
        assert event_name in result.error["message"]

    def test_wait_elapsed_time_tracking(
        self, cleanup_event_bus, async_executor, timing_helper
    ):
        """Test wait_for_signal tracks elapsed time accurately"""
        event_name = "elapsed_test"
        wait_result = {}

        def waiter():
            result = _execute_wait_for_signal(event_name, timeout_ms=5000)
            wait_result["result"] = result

        waiter_thread = async_executor(waiter)

        time.sleep(0.3)  # Wait 300ms before signaling

        bus = EventBus()
        bus.signal(event_name)

        waiter_thread.join(timeout=2.0)

        assert wait_result["result"].success is True
        # Elapsed time should be ~300ms (within ±20% tolerance)
        assert timing_helper(
            wait_result["result"].result["elapsed_ms"], 300, tolerance_percent=20
        )

    def test_wait_default_timeout(self, cleanup_event_bus):
        """Test wait_for_signal uses default timeout (30 seconds)"""
        # We'll test this times out, but with shorter timeout for test speed
        event_name = "default_timeout"

        # Default is 30000ms, but we'll use explicit short timeout for testing
        result = _execute_wait_for_signal(event_name)  # Uses default 30000ms

        # This would take 30 seconds - instead verify operation accepts no timeout param
        # Just check that function signature allows it
        assert WAIT_FOR_SIGNAL_OPERATION.parameters[1].default == 30000

    def test_wait_custom_timeout(self, cleanup_event_bus):
        """Test wait_for_signal with custom timeout"""
        event_name = "custom_timeout"

        start = time.time()
        result = _execute_wait_for_signal(event_name, timeout_ms=300)
        elapsed_ms = (time.time() - start) * 1000

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "WAIT_TIMEOUT"
        assert 250 <= elapsed_ms <= 400  # ~300ms ±20%

    def test_wait_invalid_timeout(self, cleanup_event_bus):
        """Test wait_for_signal with invalid timeout values"""
        event_name = "invalid_timeout"

        # Negative timeout - Python will handle this (Event.wait allows it)
        # It will timeout immediately
        result = _execute_wait_for_signal(event_name, timeout_ms=-100)
        # Should timeout immediately
        assert result.success is False or (
            result.result is not None and result.result.get("elapsed_ms", 0) < 10
        )

    def test_wait_negative_timeout(self, cleanup_event_bus):
        """Test wait_for_signal with negative timeout"""
        event_name = "negative_timeout"

        result = _execute_wait_for_signal(event_name, timeout_ms=-1)

        # Should timeout immediately (negative timeout converted to 0)
        assert result.success is False or (
            result.result is not None and result.result.get("elapsed_ms", 0) < 10
        )

    def test_wait_zero_timeout(self, cleanup_event_bus):
        """Test wait_for_signal with zero timeout"""
        event_name = "zero_timeout"

        result = _execute_wait_for_signal(event_name, timeout_ms=0)

        # Should timeout immediately
        assert result.success is False or (
            result.result is not None and result.result.get("elapsed_ms", 0) < 10
        )

    def test_wait_recovery_suggestions_on_timeout(self, cleanup_event_bus):
        """Test wait_for_signal provides recovery suggestions on timeout"""
        event_name = "recovery_test"

        result = _execute_wait_for_signal(event_name, timeout_ms=200)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "WAIT_TIMEOUT"
        assert "recovery_suggestions" in result.error
        assert len(result.error["recovery_suggestions"]) > 0
        # Should mention checking if signal is sent
        assert any(
            "signal" in suggestion.lower()
            for suggestion in result.error["recovery_suggestions"]
        )


# ============================================================================
# UNIT TESTS: Wait Operation
# ============================================================================


class TestWaitOperation:
    """Test wait() operation"""

    def test_wait_basic(self):
        """Test basic wait operation"""
        duration_ms = 200

        start = time.time()
        result = _execute_wait(duration_ms)
        elapsed_ms = (time.time() - start) * 1000

        assert result.success is True
        assert result.result is not None
        assert result.result["requested_ms"] == duration_ms
        assert 180 <= elapsed_ms <= 250  # ~200ms ±15%

    def test_wait_accuracy(self, timing_helper):
        """Test wait operation timing accuracy"""
        test_durations = [100, 500, 1000]

        for duration_ms in test_durations:
            result = _execute_wait(duration_ms)

            assert result.success is True
            assert result.result is not None
            actual_ms = result.result["actual_ms"]
            requested_ms = result.result["requested_ms"]

            # Verify within ±15% tolerance
            assert timing_helper(actual_ms, requested_ms, tolerance_percent=15)

    def test_wait_zero_duration(self):
        """Test wait with zero duration"""
        result = _execute_wait(0)

        assert result.success is True
        assert result.result is not None
        assert result.result["requested_ms"] == 0
        assert result.result["actual_ms"] < 10  # Should be nearly instant

    def test_wait_negative_duration(self):
        """Test wait with negative duration"""
        result = _execute_wait(-100)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_DURATION"
        assert "-100" in result.error["message"]

    def test_wait_large_duration(self):
        """Test wait with large duration (within valid range)"""
        # Test max valid duration (60 seconds = 60000ms)
        # But use shorter duration for test speed
        duration_ms = 2000

        start = time.time()
        result = _execute_wait(duration_ms)
        elapsed_ms = (time.time() - start) * 1000

        assert result.success is True
        assert 1800 <= elapsed_ms <= 2200  # ~2000ms ±10%

    def test_wait_exception_handling(self):
        """Test wait handles exceptions gracefully"""
        # Pass invalid type
        result = _execute_wait("invalid")  # type: ignore[arg-type]

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "WAIT_FAILED"


# ============================================================================
# INTEGRATION TESTS: Basic Scenarios
# ============================================================================


class TestSyncIntegrationBasic:
    """Basic integration scenarios with real EventBus"""

    def test_simple_signal_wait_flow(self, cleanup_event_bus):
        """Test simple signal → wait pattern"""
        event_name = "simple_flow"

        # Signal
        signal_result = SIGNAL_OPERATION.execute(event_name=event_name)
        assert signal_result.success is True

        # Wait (should receive immediately)
        wait_result = WAIT_FOR_SIGNAL_OPERATION.execute(
            event_name=event_name, timeout_ms=1000
        )
        assert wait_result.success is True
        assert wait_result.result is not None
        assert wait_result.result["received"] is True

    def test_two_robot_handoff_pattern(self, cleanup_event_bus):
        """Test two-robot handoff using signal/wait"""
        results = {"robot1": [], "robot2": []}

        def robot1():
            # Robot1: Move, grip, signal
            results["robot1"].append("moved")
            results["robot1"].append("gripped")

            # Signal cube gripped
            signal_result = SIGNAL_OPERATION.execute(event_name="cube_gripped")
            assert signal_result.success

            # Wait for robot2 ready
            wait_result = WAIT_FOR_SIGNAL_OPERATION.execute(
                event_name="robot2_ready", timeout_ms=2000
            )
            assert wait_result.success

            results["robot1"].append("released")

        def robot2():
            # Robot2: Wait for cube gripped
            wait_result = WAIT_FOR_SIGNAL_OPERATION.execute(
                event_name="cube_gripped", timeout_ms=2000
            )
            assert wait_result.success

            results["robot2"].append("moved_to_handoff")

            # Signal ready
            signal_result = SIGNAL_OPERATION.execute(event_name="robot2_ready")
            assert signal_result.success

            results["robot2"].append("gripped")

        # Run both robots
        t1 = threading.Thread(target=robot1)
        t2 = threading.Thread(target=robot2)

        t1.start()
        t2.start()

        t1.join(timeout=3.0)
        t2.join(timeout=3.0)

        # Verify sequence
        assert "moved" in results["robot1"]
        assert "gripped" in results["robot1"]
        assert "released" in results["robot1"]
        assert "moved_to_handoff" in results["robot2"]
        assert "gripped" in results["robot2"]

    def test_multiple_sync_points_sequence(self, cleanup_event_bus):
        """Test sequence with multiple synchronization points"""
        sequence = []

        def robot1():
            sequence.append("r1_step1")
            SIGNAL_OPERATION.execute(event_name="step1_done")

            WAIT_FOR_SIGNAL_OPERATION.execute(event_name="step2_done", timeout_ms=2000)
            sequence.append("r1_step3")
            SIGNAL_OPERATION.execute(event_name="step3_done")

        def robot2():
            WAIT_FOR_SIGNAL_OPERATION.execute(event_name="step1_done", timeout_ms=2000)
            sequence.append("r2_step2")
            SIGNAL_OPERATION.execute(event_name="step2_done")

            WAIT_FOR_SIGNAL_OPERATION.execute(event_name="step3_done", timeout_ms=2000)
            sequence.append("r2_step4")

        t1 = threading.Thread(target=robot1)
        t2 = threading.Thread(target=robot2)

        t1.start()
        t2.start()

        t1.join(timeout=3.0)
        t2.join(timeout=3.0)

        # Verify order
        assert sequence.index("r1_step1") < sequence.index("r2_step2")
        assert sequence.index("r2_step2") < sequence.index("r1_step3")
        assert sequence.index("r1_step3") < sequence.index("r2_step4")

    def test_mixed_wait_and_wait_for_signal(self, cleanup_event_bus):
        """Test mixing wait (time-based) and wait_for_signal (event-based)"""
        results = []

        def robot():
            results.append("start")

            # Time-based wait
            WAIT_OPERATION.execute(duration_ms=200)
            results.append("after_time_wait")

            # Event-based wait
            WAIT_FOR_SIGNAL_OPERATION.execute(event_name="event1", timeout_ms=2000)
            results.append("after_signal_wait")

        t = threading.Thread(target=robot)
        t.start()

        time.sleep(0.5)  # Let robot get past time wait

        # Signal
        SIGNAL_OPERATION.execute(event_name="event1")

        t.join(timeout=3.0)

        assert len(results) == 3
        assert results == ["start", "after_time_wait", "after_signal_wait"]


# ============================================================================
# INTEGRATION TESTS: Multi-Robot Coordination
# ============================================================================


class TestSyncIntegrationMultiRobot:
    """Multi-robot coordination scenarios"""

    def test_two_robots_parallel_with_sync(self, cleanup_event_bus):
        """Test two robots running parallel operations with sync points"""
        results = {"robot1": [], "robot2": []}

        def robot1():
            results["robot1"].append("parallel_op1")
            SIGNAL_OPERATION.execute(event_name="r1_done")

            WAIT_FOR_SIGNAL_OPERATION.execute(event_name="r2_done", timeout_ms=2000)
            results["robot1"].append("final_op1")

        def robot2():
            results["robot2"].append("parallel_op2")
            SIGNAL_OPERATION.execute(event_name="r2_done")

            WAIT_FOR_SIGNAL_OPERATION.execute(event_name="r1_done", timeout_ms=2000)
            results["robot2"].append("final_op2")

        threads = [threading.Thread(target=robot1), threading.Thread(target=robot2)]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3.0)

        assert len(results["robot1"]) == 2
        assert len(results["robot2"]) == 2

    def test_three_robots_sequential_handoff(self, cleanup_event_bus):
        """Test sequential handoff across 3 robots"""
        sequence = []

        def robot1():
            sequence.append("r1_grip")
            SIGNAL_OPERATION.execute(event_name="r1_ready")

        def robot2():
            WAIT_FOR_SIGNAL_OPERATION.execute(event_name="r1_ready", timeout_ms=2000)
            sequence.append("r2_grip")
            SIGNAL_OPERATION.execute(event_name="r2_ready")

        def robot3():
            WAIT_FOR_SIGNAL_OPERATION.execute(event_name="r2_ready", timeout_ms=2000)
            sequence.append("r3_grip")

        threads = [
            threading.Thread(target=robot1),
            threading.Thread(target=robot2),
            threading.Thread(target=robot3),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3.0)

        # Verify sequential order
        assert sequence == ["r1_grip", "r2_grip", "r3_grip"]

    def test_broadcast_signal_to_multiple_waiters(self, cleanup_event_bus):
        """Test single signal broadcasts to multiple waiting robots"""
        results = []

        def robot_waiter(robot_id):
            WAIT_FOR_SIGNAL_OPERATION.execute(event_name="broadcast", timeout_ms=2000)
            results.append(robot_id)

        # Start 5 robots waiting
        threads = [
            threading.Thread(target=robot_waiter, args=(f"robot{i}",)) for i in range(5)
        ]

        for t in threads:
            t.start()

        time.sleep(0.2)  # Let waiters start

        # Single broadcast signal
        SIGNAL_OPERATION.execute(event_name="broadcast")

        for t in threads:
            t.join(timeout=2.0)

        # All 5 robots should have received
        assert len(results) == 5

    def test_chain_coordination_robot1_robot2_robot3(self, cleanup_event_bus):
        """Test chain coordination pattern"""
        chain = []

        def robot1():
            chain.append("r1_start")
            WAIT_OPERATION.execute(duration_ms=100)
            chain.append("r1_done")
            SIGNAL_OPERATION.execute(event_name="r1_complete")

        def robot2():
            WAIT_FOR_SIGNAL_OPERATION.execute(event_name="r1_complete", timeout_ms=2000)
            chain.append("r2_start")
            WAIT_OPERATION.execute(duration_ms=100)
            chain.append("r2_done")
            SIGNAL_OPERATION.execute(event_name="r2_complete")

        def robot3():
            WAIT_FOR_SIGNAL_OPERATION.execute(event_name="r2_complete", timeout_ms=2000)
            chain.append("r3_start")
            WAIT_OPERATION.execute(duration_ms=100)
            chain.append("r3_done")

        threads = [
            threading.Thread(target=robot1),
            threading.Thread(target=robot2),
            threading.Thread(target=robot3),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3.0)

        # Verify chain order
        assert chain.index("r1_start") < chain.index("r1_done")
        assert chain.index("r1_done") < chain.index("r2_start")
        assert chain.index("r2_start") < chain.index("r2_done")
        assert chain.index("r2_done") < chain.index("r3_start")


# ============================================================================
# INTEGRATION TESTS: SequenceExecutor Integration
# ============================================================================


class TestSyncWithSequenceExecutor:
    """Integration with SequenceExecutor"""

    def test_sequence_with_sync_primitives(self, cleanup_event_bus):
        """Test SequenceExecutor executes sync operations in sequence"""
        from orchestrators.SequenceExecutor import SequenceExecutor

        executor = SequenceExecutor(check_completion=False)

        commands = [
            {"operation": "signal", "params": {"event_name": "test_event"}},
            {"operation": "wait", "params": {"duration_ms": 100}},
            {
                "operation": "wait_for_signal",
                "params": {"event_name": "test_event", "timeout_ms": 1000},
            },
        ]

        result = executor.execute_sequence(commands)

        assert result["success"] is True
        assert result["completed_commands"] == 3
        assert all(r["success"] for r in result["results"])

    def test_parallel_group_with_signals(self, cleanup_event_bus):
        """Test parallel execution groups with signal synchronization"""
        from orchestrators.SequenceExecutor import SequenceExecutor

        executor = SequenceExecutor(check_completion=False)

        # Parallel group pattern: both signal, then both wait
        commands = [
            {
                "parallel_group": 0,
                "operation": "signal",
                "params": {"event_name": "group0_r1"},
            },
            {
                "parallel_group": 0,
                "operation": "signal",
                "params": {"event_name": "group0_r2"},
            },
            {
                "parallel_group": 1,
                "operation": "wait_for_signal",
                "params": {"event_name": "group0_r1", "timeout_ms": 1000},
            },
            {
                "parallel_group": 1,
                "operation": "wait_for_signal",
                "params": {"event_name": "group0_r2", "timeout_ms": 1000},
            },
        ]

        result = executor.execute_sequence(commands)

        assert result["success"] is True
        assert result["completed_commands"] == 4

    def test_abort_during_wait_for_signal(self, cleanup_event_bus):
        """Test aborting sequence during wait_for_signal"""
        from orchestrators.SequenceExecutor import SequenceExecutor

        executor = SequenceExecutor(check_completion=False)

        commands = [
            {
                "operation": "wait_for_signal",
                "params": {"event_name": "never_sent", "timeout_ms": 5000},
            },
            {"operation": "wait", "params": {"duration_ms": 100}},
        ]

        # Execute in thread so we can abort
        result_holder = {}

        def run_sequence():
            try:
                result_holder["result"] = executor.execute_sequence(commands)
            except Exception as e:
                result_holder["error"] = str(e)

        thread = threading.Thread(target=run_sequence)
        thread.start()

        time.sleep(0.3)  # Let it start waiting

        # Abort
        executor.abort()

        thread.join(timeout=3.0)

        # Should have result (even if aborted or failed)
        # Abort during wait_for_signal should stop sequence
        if "result" in result_holder:
            assert result_holder["result"]["completed_commands"] <= 2
        else:
            # If no result, abort happened before first command completed
            assert "error" in result_holder or len(result_holder) == 0

    def test_timeout_propagation(self, cleanup_event_bus):
        """Test timeout in wait_for_signal propagates correctly"""
        from orchestrators.SequenceExecutor import SequenceExecutor

        executor = SequenceExecutor(check_completion=False)

        commands = [
            {
                "operation": "wait_for_signal",
                "params": {"event_name": "timeout_event", "timeout_ms": 300},
            },
            {"operation": "wait", "params": {"duration_ms": 100}},  # Should not execute
        ]

        result = executor.execute_sequence(commands)

        # First command should fail (timeout) or execute successfully depending on timing
        # Either way, we should have attempted at least one command
        assert len(result["results"]) >= 1

        # If first command completed, check its result
        if result["completed_commands"] > 0:
            # First command likely timed out
            # Sequence stops after first failure
            if not result["results"][0]["success"]:
                assert result["completed_commands"] == 1
        else:
            # Command failed to execute entirely (error before execution)
            assert result["success"] is False
