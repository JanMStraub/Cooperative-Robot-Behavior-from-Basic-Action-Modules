"""
Sequence Executor for Multi-Command Operations
===============================================

This module executes parsed command sequences sequentially, waiting for each
operation to complete before proceeding to the next.

Example:
    >>> executor = SequenceExecutor()
    >>> commands = [
    ...     {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1", "x": 0.3, "y": 0.2, "z": 0.1}},
    ...     {"operation": "control_gripper", "params": {"robot_id": "Robot1", "open_gripper": False}}
    ... ]
    >>> result = executor.execute_sequence(commands)
"""

from typing import Dict, Any, List, Optional, Callable
import time
import logging
import threading

from operations.Registry import get_global_registry
from servers.StatusServer import StatusResponseHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SequenceExecutor:
    """
    Executes command sequences with completion tracking and error handling.

    Each command is executed and the executor waits for completion before
    proceeding to the next command in the sequence.
    """

    def __init__(
        self,
        default_timeout: float = 60.0,
        check_completion: bool = True
    ):
        """
        Initialize the SequenceExecutor.

        Args:
            default_timeout: Default timeout in seconds for each operation (60s for movements)
            check_completion: Whether to check for operation completion via StatusServer
        """
        self.registry = get_global_registry()
        self.default_timeout = default_timeout
        self.check_completion = check_completion
        self._abort_flag = False
        self._current_sequence_id: Optional[str] = None
        self._progress_callbacks: List[Callable] = []

    def execute_sequence(
        self,
        commands: List[Dict[str, Any]],
        sequence_id: Optional[str] = None,
        timeout_per_command: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute a sequence of commands sequentially.

        Args:
            commands: List of commands to execute
            sequence_id: Optional ID for this sequence
            timeout_per_command: Timeout per command in seconds

        Returns:
            Dict with structure:
            {
                "success": bool,
                "sequence_id": str,
                "total_commands": int,
                "completed_commands": int,
                "results": [
                    {
                        "index": int,
                        "operation": str,
                        "success": bool,
                        "result": dict or None,
                        "error": str or None,
                        "duration_ms": float
                    },
                    ...
                ],
                "total_duration_ms": float,
                "error": str or None
            }
        """
        self._abort_flag = False
        self._current_sequence_id = sequence_id or f"seq_{int(time.time() * 1000)}"
        timeout = timeout_per_command or self.default_timeout

        start_time = time.time()
        results = []
        completed = 0

        logger.info(f"Starting sequence {self._current_sequence_id} with {len(commands)} commands")

        for i, cmd in enumerate(commands):
            if self._abort_flag:
                logger.warning(f"Sequence {self._current_sequence_id} aborted at command {i}")
                break

            operation = cmd.get("operation", "")
            params = cmd.get("params", {})

            logger.info(f"Executing command {i + 1}/{len(commands)}: {operation}")
            self._notify_progress(i, len(commands), operation, "executing")

            cmd_start = time.time()
            cmd_result = self._execute_single_command(operation, params, timeout)
            cmd_duration = (time.time() - cmd_start) * 1000

            result_entry = {
                "index": i,
                "operation": operation,
                "success": cmd_result["success"],
                "result": cmd_result.get("result"),
                "error": cmd_result.get("error"),
                "duration_ms": cmd_duration
            }
            results.append(result_entry)

            if cmd_result["success"]:
                completed += 1
                self._notify_progress(i, len(commands), operation, "completed")
                logger.info(f"Command {i + 1} completed successfully in {cmd_duration:.0f}ms")
            else:
                self._notify_progress(i, len(commands), operation, "failed")
                logger.error(f"Command {i + 1} failed: {cmd_result.get('error')}")
                # Stop sequence on first failure
                break

        total_duration = (time.time() - start_time) * 1000
        success = completed == len(commands)

        result = {
            "success": success,
            "sequence_id": self._current_sequence_id,
            "total_commands": len(commands),
            "completed_commands": completed,
            "results": results,
            "total_duration_ms": total_duration,
            "error": None if success else f"Sequence failed at command {completed}"
        }

        logger.info(
            f"Sequence {self._current_sequence_id} finished: "
            f"{completed}/{len(commands)} commands in {total_duration:.0f}ms"
        )

        return result

    def _execute_single_command(
        self,
        operation: str,
        params: Dict[str, Any],
        timeout: float
    ) -> Dict[str, Any]:
        """
        Execute a single command and wait for completion.

        Args:
            operation: Operation name
            params: Operation parameters
            timeout: Timeout in seconds

        Returns:
            Operation result
        """
        # Generate request_id for this operation (for completion tracking)
        request_id = int(time.time() * 1000) % (2**32)

        # If completion checking is enabled, create queue before sending command
        if self.check_completion:
            StatusResponseHandler.create_request_queue(request_id)
            logger.debug(f"Created completion queue for request_id {request_id}")

        try:
            # Add request_id to params so operation includes it in command to Unity
            params_with_request_id = {**params, "request_id": request_id}

            # Execute the operation
            op_result = self.registry.execute_operation_by_name(operation, **params_with_request_id)

            if not op_result.success:
                return {
                    "success": False,
                    "result": None,
                    "error": op_result.error.get("message") if op_result.error else "Unknown error"
                }

            # If completion checking is disabled, return immediately
            if not self.check_completion:
                return {
                    "success": True,
                    "result": op_result.result,
                    "error": None
                }

            # Wait for completion using the same request_id
            completed = self._wait_for_completion(operation, request_id, timeout)

            if not completed:
                return {
                    "success": False,
                    "result": op_result.result,
                    "error": f"Operation timed out after {timeout}s"
                }

            return {
                "success": True,
                "result": op_result.result,
                "error": None
            }
        finally:
            # Always clean up the queue to prevent accumulation
            if self.check_completion:
                StatusResponseHandler.remove_request_queue(request_id)
                logger.debug(f"Removed completion queue for request_id {request_id}")

    def _wait_for_completion(
        self,
        operation: str,
        request_id: int,
        timeout: float
    ) -> bool:
        """
        Wait for an operation to complete.

        Waits for a completion signal from Unity via StatusServer.
        Unity sends a "command_completion" message when the operation finishes.

        Args:
            operation: Operation name
            request_id: Request ID for completion tracking (must match ID sent to Unity)
            timeout: Timeout in seconds

        Returns:
            True if completed, False if timed out
        """
        start_time = time.time()
        poll_interval = 0.1  # Poll every 100ms

        # Queue was already created in _execute_single_command

        while time.time() - start_time < timeout:
            if self._abort_flag:
                return False

            try:
                # Wait for completion signal from Unity
                response = StatusResponseHandler.get_response(request_id, timeout=poll_interval)

                if response:
                    # Check if this is a completion signal
                    response_type = response.get("type", "")
                    if response_type == "command_completion":
                        success = response.get("success", False)
                        completed_cmd = response.get("command_type", "")
                        logger.debug(f"Received completion for {completed_cmd}: {success}")
                        return success

                    # Also check status-based completion (fallback)
                    status = response.get("status", response)
                    if self._is_operation_complete(operation, status):
                        return True

            except Exception as e:
                logger.debug(f"Completion wait error: {e}")

            time.sleep(poll_interval)

        logger.warning(f"Operation {operation} timed out after {timeout}s")
        return False

    def _is_operation_complete(self, operation: str, status: Dict[str, Any]) -> bool:
        """
        Determine if an operation is complete based on robot status.

        Args:
            operation: Operation name
            status: Robot status data

        Returns:
            True if operation is complete
        """
        if operation == "move_to_coordinate":
            # Check if robot is no longer moving
            is_moving = status.get("is_moving", True)
            return not is_moving

        elif operation == "return_to_start_position":
            # Check if robot is no longer moving (similar to move_to_coordinate)
            is_moving = status.get("is_moving", True)
            return not is_moving

        elif operation == "control_gripper":
            # Gripper operations are fast, consider complete after short delay
            # Could also check gripper state in status
            return True

        elif operation == "check_robot_status":
            # Status check is immediate
            return True

        elif operation == "calculate_object_coordinates":
            # Detection is immediate (results sent via separate channel)
            return True

        # Default: assume complete
        return True

    def abort(self):
        """Abort the current sequence execution."""
        self._abort_flag = True
        logger.warning(f"Abort requested for sequence {self._current_sequence_id}")

    def add_progress_callback(self, callback: Callable):
        """
        Add a callback for progress updates.

        Callback signature: callback(index: int, total: int, operation: str, status: str)
        """
        self._progress_callbacks.append(callback)

    def _notify_progress(self, index: int, total: int, operation: str, status: str):
        """Notify all progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(index, total, operation, status)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")


class AsyncSequenceExecutor:
    """
    Asynchronous wrapper for SequenceExecutor that runs in a background thread.
    """

    def __init__(self, executor: Optional[SequenceExecutor] = None):
        """
        Initialize the async executor.

        Args:
            executor: SequenceExecutor instance to use
        """
        self.executor = executor or SequenceExecutor()
        self._current_thread: Optional[threading.Thread] = None
        self._result: Optional[Dict[str, Any]] = None
        self._completion_callbacks: List[Callable] = []

    def execute_async(
        self,
        commands: List[Dict[str, Any]],
        sequence_id: Optional[str] = None,
        timeout_per_command: Optional[float] = None
    ) -> str:
        """
        Execute a sequence in the background.

        Args:
            commands: List of commands to execute
            sequence_id: Optional sequence ID
            timeout_per_command: Timeout per command

        Returns:
            Sequence ID
        """
        seq_id = sequence_id or f"seq_{int(time.time() * 1000)}"

        def run():
            self._result = self.executor.execute_sequence(
                commands, seq_id, timeout_per_command
            )
            self._notify_completion()

        self._current_thread = threading.Thread(target=run, daemon=True)
        self._current_thread.start()

        return seq_id

    def is_running(self) -> bool:
        """Check if a sequence is currently running."""
        return self._current_thread is not None and self._current_thread.is_alive()

    def get_result(self) -> Optional[Dict[str, Any]]:
        """Get the result of the last executed sequence."""
        return self._result

    def wait_for_completion(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Wait for the current sequence to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            Sequence result or None if timed out
        """
        if self._current_thread is not None:
            self._current_thread.join(timeout=timeout)
        return self._result

    def abort(self):
        """Abort the current sequence."""
        self.executor.abort()

    def add_completion_callback(self, callback: Callable):
        """
        Add a callback for sequence completion.

        Callback signature: callback(result: Dict[str, Any])
        """
        self._completion_callbacks.append(callback)

    def _notify_completion(self):
        """Notify all completion callbacks."""
        for callback in self._completion_callbacks:
            try:
                callback(self._result)
            except Exception as e:
                logger.error(f"Completion callback error: {e}")


# Singleton instances
_executor_instance: Optional[SequenceExecutor] = None
_async_executor_instance: Optional[AsyncSequenceExecutor] = None


def get_sequence_executor() -> SequenceExecutor:
    """Get the global SequenceExecutor singleton."""
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = SequenceExecutor()
    return _executor_instance


def get_async_executor() -> AsyncSequenceExecutor:
    """Get the global AsyncSequenceExecutor singleton."""
    global _async_executor_instance
    if _async_executor_instance is None:
        _async_executor_instance = AsyncSequenceExecutor()
    return _async_executor_instance
