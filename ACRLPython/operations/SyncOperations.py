#!/usr/bin/env python3
"""Pub/sub synchronization primitives for multi-robot coordination (signal, wait_for_signal, wait)."""

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
from ._imports import get_command_broadcaster as _get_command_broadcaster
from core.SingletonBase import SingletonBase


class EventBus(SingletonBase):
    """
    Thread-safe event bus for robot-to-robot signaling.

    Uses threading.Condition with a generation counter to fix the race condition
    present in the original threading.Event approach.

    **The original bug**: Robot1 and Robot2 both wait for "event_X". Signal fires.
    Robot1 wakes, decrements waiter count to 0, and clears the event. Robot2 has
    not yet been scheduled. When Robot2 runs, the event is already cleared — it
    blocks forever.

    **The fix**: Use a Condition with a predicate. Each waiter records `generation_before`
    = the generation at the time it starts waiting. signal() increments the generation
    and calls notify_all(). Each waiter uses `wait_for(lambda: gen > baseline)` which:
    - Returns immediately if gen was already > baseline (signal fired before wait)
    - Returns immediately when any signal fires while waiting
    - No event to clear — the generation counter accumulates, so late-arriving waiters
      for a given signal round must record their baseline BEFORE the signal they want.

    **Usage pattern**: For each coordination round, robots record their baseline first,
    then wait. The signal() increments the generation, waking all current waiters.
    Between rounds, call reset() to restore fresh state (or waiters will record a
    higher baseline and correctly wait for the NEXT signal).

    Singleton pattern ensures all operations share the same event state.
    """

    def _singleton_init(self):
        """Initialize the event bus (called once by SingletonBase)."""
        self._conditions: Dict[str, threading.Condition] = {}
        # Maps event_name -> monotonic signal generation counter.
        # Incremented on each signal(). Waiters compare against a baseline snapshot
        # taken before waiting to detect signals — including pre-wait signals.
        self._generations: Dict[str, int] = {}
        # Maps event_name -> active waiter count (for monitoring/tests).
        self._waiter_counts: Dict[str, int] = {}
        self._event_lock = threading.Lock()

    def _ensure_event(self, event_name: str) -> None:
        """Ensure the Condition, generation counter, and waiter count exist. Must hold _event_lock."""
        if event_name not in self._conditions:
            self._conditions[event_name] = threading.Condition(threading.Lock())
            self._generations[event_name] = 0
            self._waiter_counts[event_name] = 0

    @property
    def _events(self) -> Dict[str, threading.Condition]:
        """Backward-compatible accessor. Tests: use _is_signaled() not _events[name].is_set() (Condition has no is_set)."""
        return self._conditions

    def _is_signaled(self, event_name: str) -> bool:
        """True if event signaled at least once since last reset/clear (generation > 0)."""
        return self._generations.get(event_name, 0) > 0

    def signal(self, event_name: str) -> None:
        """Increment generation counter and wake all waiting threads (notify_all under Condition lock)."""
        with self._event_lock:
            self._ensure_event(event_name)
            cond = self._conditions[event_name]

        with cond:
            self._generations[event_name] += 1
            cond.notify_all()

    def wait_for_signal(self, event_name: str, timeout_ms: int = 30000) -> bool:
        """Wait for event. Returns True immediately if already signaled (generation snapshot before sleep)."""
        with self._event_lock:
            self._ensure_event(event_name)
            cond = self._conditions[event_name]
            self._waiter_counts[event_name] += 1

        timeout_sec = timeout_ms / 1000.0
        try:
            with cond:
                # Predicate: True if event has been signaled at least once since last
                # reset/clear. Using generation > 0 means any waiter returns True
                # immediately if the event was already signaled, matching the
                # threading.Event.is_set() semantic. Cleared events reset to 0,
                # so post-clear waiters correctly wait for the next signal.
                received = cond.wait_for(
                    lambda: self._generations[event_name] > 0,
                    timeout=timeout_sec,
                )
        finally:
            with self._event_lock:
                if (
                    event_name in self._waiter_counts
                    and self._waiter_counts[event_name] > 0
                ):
                    self._waiter_counts[event_name] -= 1

        return received

    def clear_event(self, event_name: str) -> None:
        """Reset generation counter to 0 (un-signaled). In-progress waits are unaffected."""
        with self._event_lock:
            if event_name in self._generations:
                self._generations[event_name] = 0

    def reset(self) -> None:
        """Clear all events and waiter counts (useful for testing and between task rounds)."""
        with self._event_lock:
            self._conditions.clear()
            self._generations.clear()
            self._waiter_counts.clear()

    # ------------------------------------------------------------------
    # Public inspection helpers — for testing without accessing internals
    # ------------------------------------------------------------------

    def is_event_signaled(self, event_name: str) -> bool:
        """True if event generation counter > 0 (stable public API for tests)."""
        return self._generations.get(event_name, 0) > 0

    def get_waiter_count(self, event_name: str) -> int:
        """Number of threads currently waiting on event (stable public API for tests)."""
        return self._waiter_counts.get(event_name, 0)


def _execute_signal(
    event_name: str,
    request_id: Optional[int] = None,
    robot_id: Optional[str] = None,
    use_ros: bool = False,
) -> OperationResult:
    """Emit a named event. request_id/robot_id/use_ros ignored for sync ops."""
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
    preconditions=[],
    postconditions=[],
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


def _execute_wait_for_signal(
    event_name: str,
    timeout_ms: int = 30000,
    request_id: Optional[int] = None,
    robot_id: Optional[str] = None,
    use_ros: bool = False,
) -> OperationResult:
    """Block until named event received. request_id/robot_id/use_ros ignored."""
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
    preconditions=[],
    postconditions=[],
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


def _execute_wait(
    duration_ms: int,
    request_id: Optional[int] = None,
    robot_id: Optional[str] = None,
    use_ros: bool = False,
) -> OperationResult:
    """Pause execution for duration_ms milliseconds."""
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
    preconditions=[],
    postconditions=[],
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


def _execute_reset_simulation(**kwargs) -> OperationResult:
    """Send reset_simulation command to Unity and wait for completion."""
    command = {
        "command_type": "reset_simulation",
        "robot_id": "system",
    }

    broadcaster = _get_command_broadcaster()
    completion = broadcaster.send_command_and_wait(command, timeout=15.0)

    if completion is None:
        return OperationResult.error_result(
            "COMMUNICATION_FAILED",
            "Failed to send reset_simulation to Unity or timed out",
            ["Ensure Unity is running and connected to CommandServer (port 5007)"],
        )

    if not completion.get("success", False):
        return OperationResult.error_result(
            "RESET_FAILED",
            "Simulation reset did not complete successfully",
            ["Check Unity console for SimulationManager errors"],
        )

    return OperationResult.success_result({"reset": True})


RESET_SIMULATION_OPERATION = BasicOperation(
    operation_id="sync_reset_001",
    name="reset_simulation",
    category=OperationCategory.SYNC,
    complexity=OperationComplexity.ATOMIC,
    description="Reset simulation to initial state (robots, objects, scene)",
    long_description=(
        "Triggers a full simulation reset: robots return to start positions, "
        "all scene objects return to their initial positions. "
        "Used between benchmark runs or after errors."
    ),
    parameters=[],
    preconditions=[],
    postconditions=[],
    average_duration_ms=3000,
    success_rate=99.0,
    failure_modes=["Unity not connected", "SimulationManager not found"],
    usage_examples=["reset_simulation() - Reset scene between benchmark runs"],
    relationships=None,
    implementation=_execute_reset_simulation,
)
