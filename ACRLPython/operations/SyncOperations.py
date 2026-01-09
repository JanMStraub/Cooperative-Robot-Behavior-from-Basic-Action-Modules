"""
Synchronization Operations
===========================

Provides pub/sub synchronization primitives for multi-robot coordination.
Enables LLM-driven coordination without hardcoded coordination operations.

Operations:
- signal: Emit named event for other robots to wait on
- wait_for_signal: Block until named event is received
- wait: Simple time-based pause
"""

import time
import threading
from typing import Dict, Optional

from .Base import (
    BasicOperation,
    OperationParameter,
    OperationCategory,
    OperationComplexity,
    OperationResult,
)


class EventBus:
    """
    Thread-safe event bus for robot-to-robot signaling.

    Singleton pattern ensures all operations share the same event state.
    """

    _instance: Optional["EventBus"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._events: Dict[str, threading.Event] = {}
        self._event_lock = threading.Lock()
        self._waiter_counts: Dict[str, int] = {}
        self._initialized = True

    def signal(self, event_name: str) -> None:
        """
        Set an event flag, waking all waiting threads.

        Args:
            event_name: Name of the event to signal
        """
        with self._event_lock:
            if event_name not in self._events:
                self._events[event_name] = threading.Event()
                self._waiter_counts[event_name] = 0

            self._events[event_name].set()

    def wait_for_signal(self, event_name: str, timeout_ms: int = 30000) -> bool:
        """
        Wait for an event to be signaled.

        Args:
            event_name: Name of the event to wait for
            timeout_ms: Maximum wait time in milliseconds

        Returns:
            True if event was signaled, False if timeout
        """
        with self._event_lock:
            if event_name not in self._events:
                self._events[event_name] = threading.Event()
                self._waiter_counts[event_name] = 0

            self._waiter_counts[event_name] += 1
            event = self._events[event_name]

        # Wait outside the lock
        timeout_sec = timeout_ms / 1000.0
        result = event.wait(timeout=timeout_sec)

        # Decrement waiter count and potentially clear event
        # Handle case where event was reset/cleared during wait (e.g., test cleanup)
        with self._event_lock:
            if event_name in self._waiter_counts:
                self._waiter_counts[event_name] -= 1

                # Auto-clear event when all waiters have received it
                if self._waiter_counts[event_name] == 0 and event.is_set():
                    event.clear()

        return result

    def clear_event(self, event_name: str) -> None:
        """
        Manually clear an event (usually not needed due to auto-clear).

        Args:
            event_name: Name of the event to clear
        """
        with self._event_lock:
            if event_name in self._events:
                self._events[event_name].clear()

    def reset(self) -> None:
        """Clear all events (useful for testing)."""
        with self._event_lock:
            self._events.clear()
            self._waiter_counts.clear()


# ============================================================================
# SIGNAL OPERATION
# ============================================================================


def _execute_signal(
    event_name: str, request_id: Optional[int] = None, robot_id: Optional[str] = None
) -> OperationResult:
    """
    Emit a named event for other robots to wait on.

    Args:
        event_name: Name of the event to signal
        request_id: Optional request ID for tracking (ignored for sync operations)
        robot_id: Optional robot ID (ignored, sync operations are global)

    Returns:
        OperationResult with success status
    """
    try:
        event_bus = EventBus()
        event_bus.signal(event_name)

        return OperationResult.success_result(
            {"event_name": event_name, "signaled_at": time.time()}
        )
    except Exception as e:
        return OperationResult.error_result(
            error_code="SIGNAL_FAILED",
            message=f"Failed to signal event '{event_name}': {str(e)}",
            recovery_suggestions=["Verify event_name is a valid string"],
        )


SIGNAL_OPERATION = BasicOperation(
    operation_id="sync_signal_001",
    name="signal",
    category=OperationCategory.SYNC,
    complexity=OperationComplexity.ATOMIC,
    description="Emit named event for other robots to wait on",
    long_description=(
        "Signals a named event that other robots can wait for using wait_for_signal. "
        "This enables flexible synchronization between robots without hardcoded coordination patterns. "
        "The event persists until all waiting robots have received it, then auto-clears."
    ),
    parameters=[
        OperationParameter(
            name="event_name",
            type="str",
            description="Name of the event to signal (e.g., 'cube_gripped', 'robot1_ready')",
            required=True,
        ),
    ],
    preconditions=["Event name is valid string"],
    postconditions=["Event is set", "All waiters are notified"],
    average_duration_ms=1,
    success_rate=99.9,
    failure_modes=["Invalid event name"],
    usage_examples=[
        "signal('cube_gripped') - Signal that cube has been gripped",
        "signal('robot1_at_handoff') - Signal robot reached handoff position",
    ],
    relationships=None,
    implementation=_execute_signal,
)


# ============================================================================
# WAIT_FOR_SIGNAL OPERATION
# ============================================================================


def _execute_wait_for_signal(
    event_name: str, timeout_ms: int = 30000, request_id: Optional[int] = None, robot_id: Optional[str] = None
) -> OperationResult:
    """
    Block until a named event is received.

    Args:
        event_name: Name of the event to wait for
        timeout_ms: Maximum wait time in milliseconds (default 30 seconds)
        request_id: Optional request ID for tracking (ignored for sync operations)
        robot_id: Optional robot ID (ignored, sync operations are global)

    Returns:
        OperationResult with success if event received, error if timeout
    """
    try:
        event_bus = EventBus()
        start_time = time.time()

        received = event_bus.wait_for_signal(event_name, timeout_ms)

        elapsed_ms = (time.time() - start_time) * 1000

        if received:
            return OperationResult.success_result(
                {"event_name": event_name, "received": True, "elapsed_ms": elapsed_ms}
            )
        else:
            return OperationResult.error_result(
                error_code="WAIT_TIMEOUT",
                message=f"Timeout waiting for event '{event_name}' after {timeout_ms}ms",
                recovery_suggestions=[
                    f"Check if the signaling robot is executing signal('{event_name}')",
                    "Increase timeout_ms if operation takes longer than expected",
                    "Verify execution order - signal must come after wait_for_signal starts",
                ],
            )
    except Exception as e:
        return OperationResult.error_result(
            error_code="WAIT_FAILED",
            message=f"Failed to wait for event '{event_name}': {str(e)}",
            recovery_suggestions=[
                "Check event_name is a valid string",
                "Verify timeout_ms is a positive integer",
            ],
        )


WAIT_FOR_SIGNAL_OPERATION = BasicOperation(
    operation_id="sync_wait_for_signal_001",
    name="wait_for_signal",
    category=OperationCategory.SYNC,
    complexity=OperationComplexity.ATOMIC,
    description="Block until named event is received",
    long_description=(
        "Waits for another robot to signal a named event. Blocks execution until the event "
        "is signaled or timeout is reached. Use this to synchronize multi-robot tasks. "
        "Common pattern: Robot2 waits for 'cube_gripped' while Robot1 detects and grips cube."
    ),
    parameters=[
        OperationParameter(
            name="event_name",
            type="str",
            description="Name of the event to wait for (must match signal)",
            required=True,
        ),
        OperationParameter(
            name="timeout_ms",
            type="int",
            description="Maximum wait time in milliseconds",
            required=False,
            default=30000,
            valid_range=(100, 300000),  # 100ms to 5 minutes
        ),
    ],
    preconditions=["Event name is valid string", "Timeout is positive"],
    postconditions=["Event has been received OR timeout reached"],
    average_duration_ms=5000,  # Depends on when signal is sent
    success_rate=95.0,
    failure_modes=["Timeout reached", "Signal never sent", "Event name mismatch"],
    usage_examples=[
        "wait_for_signal('cube_gripped') - Wait for another robot to grip cube",
        "wait_for_signal('robot1_ready', timeout_ms=10000) - Wait up to 10 seconds",
    ],
    relationships=None,
    implementation=_execute_wait_for_signal,
)


# ============================================================================
# WAIT OPERATION
# ============================================================================


def _execute_wait(
    duration_ms: int, request_id: Optional[int] = None, robot_id: Optional[str] = None
) -> OperationResult:
    """
    Pause execution for specified duration.

    Args:
        duration_ms: Time to wait in milliseconds
        request_id: Optional request ID for tracking (ignored for sync operations)
        robot_id: Optional robot ID (ignored, sync operations are global)

    Returns:
        OperationResult with success status
    """
    try:
        if duration_ms < 0:
            return OperationResult.error_result(
                error_code="INVALID_DURATION",
                message=f"Duration must be non-negative, got {duration_ms}ms",
                recovery_suggestions=["Provide a non-negative duration_ms value"],
            )

        start_time = time.time()
        time.sleep(duration_ms / 1000.0)
        actual_ms = (time.time() - start_time) * 1000

        return OperationResult.success_result(
            {"requested_ms": duration_ms, "actual_ms": actual_ms}
        )
    except Exception as e:
        return OperationResult.error_result(
            error_code="WAIT_FAILED",
            message=f"Failed to wait for {duration_ms}ms: {str(e)}",
            recovery_suggestions=["Verify duration_ms is a valid integer"],
        )


WAIT_OPERATION = BasicOperation(
    operation_id="sync_wait_001",
    name="wait",
    category=OperationCategory.SYNC,
    complexity=OperationComplexity.ATOMIC,
    description="Pause execution for specified duration",
    long_description=(
        "Simple time-based pause in execution. Use this for timing coordination "
        "or allowing time for physical processes to complete (e.g., gripper closing, "
        "settling after movement). For robot-to-robot synchronization, prefer wait_for_signal."
    ),
    parameters=[
        OperationParameter(
            name="duration_ms",
            type="int",
            description="Time to wait in milliseconds",
            required=True,
            valid_range=(0, 60000),  # 0 to 60 seconds
        ),
    ],
    preconditions=["Duration is non-negative"],
    postconditions=["Specified time has elapsed"],
    average_duration_ms=1000,  # Depends on parameter
    success_rate=99.9,
    failure_modes=["Invalid duration"],
    usage_examples=[
        "wait(500) - Wait 0.5 seconds for gripper to close",
        "wait(2000) - Wait 2 seconds for object to settle",
    ],
    relationships=None,
    implementation=_execute_wait,
)
