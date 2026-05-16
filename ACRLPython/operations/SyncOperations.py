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
    """Thread-safe robot-to-robot signaling via generation counters.

    Uses Condition + generation counter instead of threading.Event to avoid a race: with Event,
    Robot1 can clear the event before Robot2 wakes, causing Robot2 to block forever. Generation
    counter never clears — late waiters record their baseline before waiting and return when
    gen > baseline, so they can't miss a signal that fired while they were scheduling.
    """

    def _singleton_init(self):
        self._conditions: Dict[str, threading.Condition] = {}
        # monotonic counter per event; waiters snapshot this before sleeping
        self._generations: Dict[str, int] = {}
        self._waiter_counts: Dict[str, int] = {}
        self._event_lock = threading.Lock()

    def _ensure_event(self, event_name: str) -> None:
        """Init Condition + counters for event_name if not yet present. Must hold _event_lock."""
        if event_name not in self._conditions:
            self._conditions[event_name] = threading.Condition(threading.Lock())
            self._generations[event_name] = 0
            self._waiter_counts[event_name] = 0

    @property
    def _events(self) -> Dict[str, threading.Condition]:
        """Backward-compat alias. Tests: use _is_signaled() — Conditions don't have is_set()."""
        return self._conditions

    def _is_signaled(self, event_name: str) -> bool:
        return self._generations.get(event_name, 0) > 0

    def signal(self, event_name: str) -> None:
        with self._event_lock:
            self._ensure_event(event_name)
            cond = self._conditions[event_name]

        with cond:
            self._generations[event_name] += 1
            cond.notify_all()

    def wait_for_signal(self, event_name: str, timeout_ms: int = 30000) -> bool:
        with self._event_lock:
            self._ensure_event(event_name)
            cond = self._conditions[event_name]
            self._waiter_counts[event_name] += 1

        timeout_sec = timeout_ms / 1000.0
        try:
            with cond:
                # gen > 0 → already signaled; returns immediately (mirrors Event.is_set() semantic)
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
        """Reset to 0 (un-signaled). In-progress waits are unaffected."""
        with self._event_lock:
            if event_name in self._generations:
                self._generations[event_name] = 0

    def reset(self) -> None:
        """Clear all events and waiter counts — use between task rounds or in test teardown."""
        with self._event_lock:
            self._conditions.clear()
            self._generations.clear()
            self._waiter_counts.clear()

    def is_event_signaled(self, event_name: str) -> bool:
        return self._generations.get(event_name, 0) > 0

    def get_waiter_count(self, event_name: str) -> int:
        return self._waiter_counts.get(event_name, 0)


def _execute_signal(
    event_name: str,
    request_id: Optional[int] = None,
    robot_id: Optional[str] = None,
    _use_ros: bool = False,
) -> OperationResult:
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
    _use_ros: bool = False,
) -> OperationResult:
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
    _use_ros: bool = False,
) -> OperationResult:
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


def _execute_reset_simulation(**_kwargs) -> OperationResult:
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
