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

# Lazy imports to avoid circular dependency
# Handle both direct execution and package import
try:
    # from ..operations.Registry import get_global_registry
    from ..operations.Base import OperationCategory
    from ..operations.Verification import OperationVerifier
    from ..operations.CoordinationVerifier import CoordinationVerifier
    from ..core.Imports import get_world_state
    # from ..servers.CommandServer import get_command_broadcaster
except ImportError:
    # from operations.Registry import get_global_registry
    from operations.Base import OperationCategory
    from operations.Verification import OperationVerifier
    from operations.CoordinationVerifier import CoordinationVerifier
    from core.Imports import get_world_state
    # from servers.CommandServer import get_command_broadcaster

# Configure logging with safe handler for background threads
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SafeStreamHandler(logging.StreamHandler):
    """
    StreamHandler that silently ignores I/O errors from closed streams.

    This prevents logging errors when pytest closes log handlers before
    background threads finish executing.
    """

    def emit(self, record):
        """
        Emit a record, catching I/O errors from closed streams.

        Args:
            record: Log record to emit
        """
        try:
            super().emit(record)
        except (ValueError, OSError):
            # Stream closed (e.g., during pytest teardown)
            # Silently ignore - this is expected in test scenarios
            pass

    def handleError(self, record):
        """
        Handle errors during logging, suppressing I/O errors from closed streams.

        Args:
            record: Log record that caused the error
        """
        import sys

        if sys.exc_info()[0] in (ValueError, OSError):
            # Stream closed - silently ignore
            pass
        else:
            # Other errors - use default handling
            super().handleError(record)


_patched_handlers = set()  # Track which handlers have been patched


def _safe_log(log_func: Callable, message: str, *args, **kwargs):
    """
    Safely log a message, catching I/O errors from closed streams.

    This prevents logging errors when pytest closes log handlers before
    background threads finish executing.

    Args:
        log_func: Logger function (logger.info, logger.error, etc.)
        message: Log message
        *args: Additional positional arguments for log function
        **kwargs: Additional keyword arguments for log function
    """
    # Lazily patch any new handlers (pytest adds handlers after module import)
    for handler in logging.root.handlers + logger.handlers:
        if id(handler) not in _patched_handlers:
            _make_handler_safe(handler)
            _patched_handlers.add(id(handler))

    try:
        log_func(message, *args, **kwargs)
    except (ValueError, OSError):
        # Stream closed (e.g., during pytest teardown)
        # Silently ignore - this is expected in test scenarios
        pass


# Patch all existing handlers to safely handle closed streams
def _make_handler_safe(handler):
    """Patch a handler's emit method to catch I/O errors from closed streams"""
    # Store original unbound method to avoid double-wrapping
    if not hasattr(handler, "_original_emit"):
        handler._original_emit = handler.__class__.emit
        handler._original_handleError = handler.__class__.handleError

    def safe_emit(record):
        try:
            handler._original_emit(handler, record)
        except (ValueError, OSError, RuntimeError):
            # Stream closed (e.g., during pytest teardown) or reentrant call during shutdown
            # Silently ignore - these are expected during signal handler shutdown
            pass

    def safe_handleError(record):
        """Override handleError to suppress I/O error diagnostics"""
        import sys

        exc_type, exc_val, exc_tb = sys.exc_info()
        if exc_type in (ValueError, OSError, RuntimeError):
            # Stream closed or reentrant call - silently ignore diagnostics
            pass
        else:
            # Other errors - use default handling
            handler._original_handleError(handler, record)

    handler.emit = safe_emit
    handler.handleError = safe_handleError
    return handler


# Apply safe patching to all handlers (including pytest's handlers)
for handler in logger.handlers[:]:  # Create copy to avoid modification during iteration
    _make_handler_safe(handler)

# Also patch root logger handlers (pytest uses these)
for handler in logging.root.handlers[:]:
    _make_handler_safe(handler)


class SequenceExecutor:
    """
    Executes command sequences with completion tracking and error handling.

    Each command is executed and the executor waits for completion before
    proceeding to the next command in the sequence.

    Supports variable passing between operations:
        detect_object -> $target
        move_to_coordinate with position=$target
    """

    # Class-level atomic counter for request IDs (shared across all instances)
    _request_id_counter = 0
    _request_id_lock = threading.Lock()

    def __init__(
        self,
        default_timeout: float = 90.0,  # Increased from 60s for complex movements
        check_completion: bool = True,
        enable_verification: bool = True,
    ):
        """
        Initialize the SequenceExecutor.

        Args:
            default_timeout: Default timeout in seconds for each operation (90s for complex movements)
            check_completion: Whether to check for operation completion via StatusServer
            enable_verification: Whether to enable formal verification (preconditions/postconditions)
        """
        # Import from centralized lazy import system (prevents circular dependencies)
        from core.Imports import get_global_registry

        self.registry = get_global_registry()
        self.default_timeout = default_timeout
        self.check_completion = check_completion
        self.enable_verification = enable_verification
        self._abort_flag = False
        self._current_sequence_id: Optional[str] = None
        self._progress_callbacks: List[Callable] = []
        self._variables: Dict[str, Any] = (
            {}
        )  # Variable storage for passing results between operations

        # Operation metrics — lightweight runtime counters.
        # avg_duration_ms uses Welford's online update: O(1) memory, no list accumulation.
        self._metrics_lock = threading.Lock()
        self._ops_executed: int = 0
        self._ops_succeeded: int = 0
        self._ops_failed: int = 0
        self._avg_duration_ms: float = 0.0

        # Initialize verification components
        if enable_verification:
            self.verifier = OperationVerifier()
            self.coordination_verifier = CoordinationVerifier()
            self.world_state = get_world_state()
        else:
            self.verifier = None
            self.coordination_verifier = None
            self.world_state = None

    def _get_command_broadcaster(self):
        """Get command broadcaster using centralized lazy import system"""
        from core.Imports import get_command_broadcaster
        return get_command_broadcaster()

    @classmethod
    def _generate_request_id(cls) -> int:
        """
        Generate a unique request ID using atomic counter + timestamp hybrid.
        Prevents collisions in multi-threaded scenarios.

        Returns:
            Unique request ID (32-bit unsigned integer)
        """
        with cls._request_id_lock:
            cls._request_id_counter += 1
            # Hybrid approach: timestamp in upper bits, counter in lower bits
            timestamp_part = (int(time.time() * 1000) & 0xFFFF) << 16
            counter_part = cls._request_id_counter & 0xFFFF
            request_id = (timestamp_part | counter_part) % (2**32)
            return request_id

    def execute_sequence(
        self,
        commands: List[Dict[str, Any]],
        sequence_id: Optional[str] = None,
        timeout_per_command: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute a sequence of commands sequentially or in parallel groups.

        Supports parallel execution: commands with the same parallel_group number
        execute concurrently, then wait for all to complete before next group.

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

        # Clear variables for new sequence
        self._variables = {}

        start_time = time.time()
        results = []
        completed = 0

        logger.info(
            f"Starting sequence {self._current_sequence_id} with {len(commands)} commands"
        )

        # Check if commands use parallel_group
        has_parallel_groups = any("parallel_group" in cmd for cmd in commands)

        if has_parallel_groups:
            # Execute with parallel group support
            logger.info("Parallel execution mode enabled (parallel_group detected)")
            group_results, group_completed = self._execute_parallel_groups(
                commands, timeout
            )
            results = group_results
            completed = group_completed
        else:
            # Sequential execution (legacy mode)
            logger.info("Sequential execution mode (no parallel_group)")
            for i, cmd in enumerate(commands):
                if self._abort_flag:
                    logger.warning(
                        f"Sequence {self._current_sequence_id} aborted at command {i}"
                    )
                    break

                operation = cmd.get("operation", "")
                params = cmd.get("params", {})
                capture_var = cmd.get("capture_var")  # Variable name to capture result

                # === PRIORITY 2: Automatic Parameter Flow ===
                # Auto-inject parameters from previous operations based on ParameterFlow definitions
                params = self._auto_inject_parameters(operation, params)

                # Resolve variable references in params (manual $ references)
                params = self._resolve_variables(params)

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
                    "error_code": cmd_result.get("error_code"),
                    "duration_ms": cmd_duration,
                }
                results.append(result_entry)

                if cmd_result["success"]:
                    completed += 1
                    self._notify_progress(i, len(commands), operation, "completed")
                    logger.info(
                        f"Command {i + 1} completed successfully in {cmd_duration:.0f}ms"
                    )

                    # Capture result to variable if specified (manual capture)
                    if capture_var and cmd_result.get("result"):
                        self._variables[capture_var] = cmd_result["result"]
                        logger.debug(f"Captured result to ${capture_var}")

                    # === PRIORITY 2: Automatic Parameter Flow ===
                    # Automatically capture outputs based on ParameterFlow definitions
                    self._auto_capture_outputs(operation, cmd_result.get("result", {}))
                else:
                    self._notify_progress(i, len(commands), operation, "failed")
                    _safe_log(
                        logger.error,
                        f"Command {i + 1} failed: {cmd_result.get('error')}",
                    )
                    # Stop sequence on first failure
                    break

        total_duration = (time.time() - start_time) * 1000
        success = completed == len(commands)

        # Get error details from first failed command
        error_message = None
        if not success and results:
            # Find the first failed result
            failed_result = next(
                (r for r in results if r is not None and not r.get("success", False)),
                None,
            )
            if failed_result:
                # Include error_code if present, otherwise just the error message
                error_code = failed_result.get("error_code")
                error_msg = failed_result.get("error", "Unknown error")
                if error_code:
                    error_message = f"{error_code}: {error_msg}"
                else:
                    error_message = error_msg
            else:
                error_message = f"Sequence failed at command {completed}"

        result = {
            "success": success,
            "sequence_id": self._current_sequence_id,
            "total_commands": len(commands),
            "completed_commands": completed,
            "results": results,
            "total_duration_ms": total_duration,
            "error": error_message,
        }

        _safe_log(
            logger.info,
            f"Sequence {self._current_sequence_id} finished: "
            f"{completed}/{len(commands)} commands in {total_duration:.0f}ms",
        )

        return result

    def _execute_parallel_groups(
        self, commands: List[Dict[str, Any]], timeout: float
    ) -> tuple[List[Optional[Dict[str, Any]]], int]:
        """
        Execute commands grouped by parallel_group number.

        Commands with the same parallel_group execute concurrently,
        then wait for all to complete before proceeding to next group.

        Args:
            commands: List of commands with parallel_group field
            timeout: Timeout per command in seconds

        Returns:
            (results, completed_count)
        """
        from collections import defaultdict

        # Group commands by parallel_group number
        groups = defaultdict(list)
        for i, cmd in enumerate(commands):
            group_num = cmd.get(
                "parallel_group", i
            )  # Default: each cmd is its own group
            groups[group_num].append((i, cmd))

        # Sort groups by group number
        sorted_groups = sorted(groups.items())

        results: List[Optional[Dict[str, Any]]] = [None] * len(
            commands
        )  # Pre-allocate results array
        completed = 0

        for group_num, group_commands in sorted_groups:
            if self._abort_flag:
                logger.warning(f"Parallel execution aborted at group {group_num}")
                break

            logger.info(
                f"Executing parallel group {group_num} with {len(group_commands)} commands"
            )

            # Execute all commands in this group concurrently
            threads = []
            thread_results = {}
            result_lock = threading.Lock()

            def execute_command_thread(index, cmd):
                """Thread function to execute a single command"""
                operation = cmd.get("operation", "")
                params = cmd.get("params", {})
                capture_var = cmd.get("capture_var")

                # Parameter injection and resolution (thread-safe for reads)
                params = self._auto_inject_parameters(operation, params)
                params = self._resolve_variables(params)

                logger.info(f"[Group {group_num}] Executing: {operation}")
                self._notify_progress(index, len(commands), operation, "executing")

                cmd_start = time.time()
                cmd_result = self._execute_single_command(operation, params, timeout)
                cmd_duration = (time.time() - cmd_start) * 1000

                result_entry = {
                    "index": index,
                    "operation": operation,
                    "success": cmd_result["success"],
                    "result": cmd_result.get("result"),
                    "error": cmd_result.get("error"),
                    "duration_ms": cmd_duration,
                    "parallel_group": group_num,
                }

                # Store result (thread-safe)
                with result_lock:
                    thread_results[index] = (result_entry, cmd_result, capture_var)

            # Launch all threads for this group
            for idx, cmd in group_commands:
                thread = threading.Thread(
                    target=execute_command_thread, args=(idx, cmd), daemon=True
                )
                threads.append(thread)
                thread.start()

            # Wait for all threads in this group to complete
            # Track which threads actually completed vs timed out
            thread_completed = {}
            for i, thread in enumerate(threads):
                thread.join(timeout=timeout + 5.0)  # Add 5s buffer
                thread_completed[i] = not thread.is_alive()
                if thread.is_alive():
                    logger.warning(f"[Group {group_num}] Thread {i} still running after join timeout")

            # Process results from this group
            group_success = True
            for thread_idx, (idx, cmd) in enumerate(group_commands):
                if idx in thread_results:
                    result_entry, cmd_result, capture_var = thread_results[idx]
                    results[idx] = result_entry

                    if cmd_result["success"]:
                        completed += 1
                        self._notify_progress(
                            idx, len(commands), result_entry["operation"], "completed"
                        )
                        logger.info(
                            f"[Group {group_num}] Command {idx} completed in {result_entry['duration_ms']:.0f}ms"
                        )

                        # Capture variables (thread-safe write)
                        if capture_var and cmd_result.get("result"):
                            self._variables[capture_var] = cmd_result["result"]
                            logger.debug(f"Captured result to ${capture_var}")

                        # Auto-capture outputs
                        self._auto_capture_outputs(
                            result_entry["operation"], cmd_result.get("result", {})
                        )
                    else:
                        group_success = False
                        self._notify_progress(
                            idx, len(commands), result_entry["operation"], "failed"
                        )
                        logger.error(
                            f"[Group {group_num}] Command {idx} failed: {cmd_result.get('error')}"
                        )
                else:
                    # Thread didn't complete
                    group_success = False
                    logger.error(
                        f"[Group {group_num}] Command {idx} did not complete (thread timeout)"
                    )
                    results[idx] = {
                        "index": idx,
                        "operation": cmd.get("operation", ""),
                        "success": False,
                        "result": None,
                        "error": "Thread execution timeout",
                        "duration_ms": timeout * 1000,
                        "parallel_group": group_num,
                    }

            # Stop if any command in the group failed
            if not group_success:
                logger.error(
                    f"Parallel group {group_num} had failures, stopping sequence"
                )
                break

        # Fill in any None results (shouldn't happen, but safety check)
        for i in range(len(results)):
            if results[i] is None:
                results[i] = {
                    "index": i,
                    "operation": commands[i].get("operation", ""),
                    "success": False,
                    "result": None,
                    "error": "Command not executed",
                    "duration_ms": 0,
                }

        return results, completed

    def _record_metric(self, success: bool, duration_ms: float):
        """
        Update operation metrics counters in a thread-safe manner.

        Uses Welford's online algorithm for the running average so no sample
        list needs to be maintained.

        Args:
            success: Whether the operation succeeded
            duration_ms: Wall-clock duration of the operation in milliseconds
        """
        with self._metrics_lock:
            self._ops_executed += 1
            if success:
                self._ops_succeeded += 1
            else:
                self._ops_failed += 1
            # Welford online mean: avg += (x - avg) / n
            self._avg_duration_ms += (duration_ms - self._avg_duration_ms) / self._ops_executed

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return a snapshot of operation metrics for this executor.

        Returns:
            Dict with keys: ops_executed, ops_succeeded, ops_failed,
            ops_success_rate (0.0–1.0), avg_duration_ms.
        """
        with self._metrics_lock:
            executed = self._ops_executed
            succeeded = self._ops_succeeded
            failed = self._ops_failed
            avg_ms = self._avg_duration_ms

        success_rate = (succeeded / executed) if executed > 0 else 0.0
        return {
            "ops_executed": executed,
            "ops_succeeded": succeeded,
            "ops_failed": failed,
            "ops_success_rate": round(success_rate, 4),
            "avg_duration_ms": round(avg_ms, 1),
        }

    def reset_metrics(self):
        """Reset all operation metrics counters to zero."""
        with self._metrics_lock:
            self._ops_executed = 0
            self._ops_succeeded = 0
            self._ops_failed = 0
            self._avg_duration_ms = 0.0

    def _execute_single_command(
        self, operation: str, params: Dict[str, Any], timeout: float
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
        # Generate unique request_id using atomic counter + timestamp hybrid
        request_id = self._generate_request_id()
        _cmd_start = time.time()

        # If completion checking is enabled, create queue before sending command
        if self.check_completion:
            self._get_command_broadcaster().create_completion_queue(request_id)
            logger.debug(f"Created completion queue for request_id {request_id}")

        _result: Dict[str, Any] = {"success": False, "result": None, "error": "internal"}
        try:
            # Get operation definition for verification
            op_def = self.registry.get_operation_by_name(operation)
            if op_def is None:
                _result = {
                    "success": False,
                    "result": None,
                    "error": f"Operation '{operation}' not found in registry",
                }
                return _result

            # === PRIORITY 3: Unified Verification ===
            if self.enable_verification and self.verifier:
                verification_result = self._verify_operation_safety(op_def, params)

                if not verification_result["safe"]:
                    _result = {
                        "success": False,
                        "result": None,
                        "error": verification_result["error"],
                        "verification_details": verification_result["details"],
                    }
                    return _result

                # Log any warnings
                if verification_result["warnings"]:
                    for warning in verification_result["warnings"]:
                        logger.warning(f"Verification warning: {warning}")

            # Add request_id to params so operation includes it in command to Unity
            params_with_request_id = {**params, "request_id": request_id}

            # Inject use_ros from config only for operations whose implementation
            # signature explicitly accepts it.  Injecting it blindly into every call
            # crashes operations like detect_field that have no such parameter.
            if "use_ros" not in params_with_request_id and op_def is not None:
                try:
                    from config.ROS import ROS_ENABLED, DEFAULT_CONTROL_MODE
                    if ROS_ENABLED and DEFAULT_CONTROL_MODE in ("ros", "hybrid"):
                        import inspect
                        impl = op_def.implementation
                        sig = inspect.signature(impl)
                        if "use_ros" in sig.parameters or any(
                            p.kind == inspect.Parameter.VAR_KEYWORD
                            for p in sig.parameters.values()
                        ):
                            params_with_request_id["use_ros"] = True
                except ImportError:
                    pass

            # Execute the operation
            op_result = self.registry.execute_operation_by_name(
                operation, **params_with_request_id
            )

            # === PHASE 3: Postcondition Verification ===
            if self.enable_verification and self.verifier and op_result.success:
                logger.debug(f"Verifying postconditions for {operation}")
                post_result = self.verifier.verify_postconditions(
                    op_def, op_result, params, self.world_state
                )

                # Postcondition failures are warnings, not blockers
                if not post_result.success:
                    violation_msgs = [
                        f"{v.predicate}: {v.reason}" for v in post_result.violations
                    ]
                    logger.warning(
                        f"Postcondition verification failed: {violation_msgs}"
                    )

            if not op_result.success:
                error_msg = (
                    op_result.error.get("message")
                    if op_result.error
                    else "Unknown error"
                )
                error_code = op_result.error.get("code") if op_result.error else None
                _result = {
                    "success": False,
                    "result": None,
                    "error": error_msg,
                    "error_code": error_code,
                }
                return _result

            # If completion checking is disabled, return immediately
            if not self.check_completion:
                _result = {"success": True, "result": op_result.result, "error": None}
                return _result

            # Skip completion waiting for operations that executed via ROS
            # (Unity never received the command, so it won't send completion)
            if op_result.result and op_result.result.get("status") in (
                "ros_executed", "ros_command_sent"
            ):
                logger.debug(
                    f"Skipping completion wait for ROS-executed operation: {operation}"
                )
                _result = {"success": True, "result": op_result.result, "error": None}
                return _result

            # Skip completion waiting for operations that execute in Python only
            op_def = self.registry.get_operation_by_name(operation)
            if op_def and op_def.category in (OperationCategory.PERCEPTION, OperationCategory.SYNC):
                logger.debug(
                    f"Skipping completion wait for {op_def.category.value} operation: {operation}"
                )
                _result = {"success": True, "result": op_result.result, "error": None}
                return _result

            # Wait for completion using the same request_id
            completed = self._wait_for_completion(operation, request_id, timeout)

            if not completed:
                _result = {
                    "success": False,
                    "result": op_result.result,
                    "error": f"Operation timed out after {timeout}s",
                }
                return _result

            _result = {"success": True, "result": op_result.result, "error": None}
            return _result
        finally:
            # Always clean up the queue and record metrics
            if self.check_completion:
                self._get_command_broadcaster().remove_completion_queue(request_id)
                logger.debug(f"Removed completion queue for request_id {request_id}")
            self._record_metric(
                success=_result.get("success", False),
                duration_ms=(time.time() - _cmd_start) * 1000,
            )

    def _wait_for_completion(
        self, operation: str, request_id: int, timeout: float
    ) -> bool:
        """
        Wait for an operation to complete with adaptive polling.

        Waits for a completion signal from Unity via StatusServer.
        Unity sends a "command_completion" message when the operation finishes.

        Implements adaptive polling: starts at 50ms, gradually increases to 500ms
        to reduce CPU usage for long-running operations.

        Args:
            operation: Operation name
            request_id: Request ID for completion tracking (must match ID sent to Unity)
            timeout: Timeout in seconds

        Returns:
            True if completed, False if timed out
        """
        start_time = time.time()

        # Adaptive polling parameters
        min_poll_interval = 0.05  # Start at 50ms
        max_poll_interval = 0.5   # Max 500ms
        poll_increase_rate = 1.1  # Increase by 10% each iteration
        current_poll_interval = min_poll_interval

        # Queue was already created in _execute_single_command

        while time.time() - start_time < timeout:
            if self._abort_flag:
                return False

            try:
                # Wait for completion signal from Unity
                response = self._get_command_broadcaster().get_completion(
                    request_id, timeout=current_poll_interval
                )

                if response:
                    # Check if this is a completion signal
                    response_type = response.get("type", "")
                    if response_type == "command_completion":
                        success = response.get("success", False)
                        completed_cmd = response.get("command_type", "")
                        elapsed = time.time() - start_time
                        logger.debug(
                            f"Received completion for {completed_cmd}: {success} (elapsed: {elapsed:.2f}s)"
                        )
                        return success

                    # Also check status-based completion (fallback)
                    status = response.get("status", response)
                    if self._is_operation_complete(operation, status):
                        return True

            except Exception as e:
                logger.debug(f"Completion wait error: {e}")

            # Adaptive polling: gradually increase poll interval
            time.sleep(current_poll_interval)
            current_poll_interval = min(
                current_poll_interval * poll_increase_rate, max_poll_interval
            )

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

        elif operation == "detect_object_stereo":
            # Stereo detection is immediate (results sent via separate channel)
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

    def _verify_operation_safety(
        self, op_def, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Unified verification combining operation preconditions and multi-robot coordination checks.

        This is PRIORITY 3: Single pre-execution safety check that combines both verifiers.

        Args:
            op_def: Operation definition from registry
            params: Operation parameters

        Returns:
            Dict with structure:
            {
                "safe": bool,  # True if all checks passed
                "error": str or None,  # Error message if not safe
                "warnings": List[str],  # Non-critical warnings
                "details": {
                    "precondition_check": dict,
                    "coordination_check": dict
                }
            }
        """
        warnings = []
        details = {}

        # Sanity check - this method should only be called when verification is enabled
        if not self.verifier:
            return {"safe": True, "error": None, "warnings": [], "details": {}}

        # === Step 1: Verify operation preconditions ===
        logger.debug(f"Running unified verification for {op_def.name}")
        pre_result = self.verifier.verify_preconditions(
            op_def, params, self.world_state
        )
        details["precondition_check"] = pre_result.to_dict()

        if not pre_result.execution_allowed:
            violation_msgs = [
                f"{v.predicate}: {v.reason}" for v in pre_result.violations
            ]
            logger.error(f"Precondition verification failed: {violation_msgs}")
            return {
                "safe": False,
                "error": f"Precondition failed: {'; '.join(violation_msgs)}",
                "warnings": warnings,
                "details": details,
            }

        # Collect precondition warnings
        if pre_result.warnings:
            warnings.extend(
                [
                    f"Precondition - {w.predicate}: {w.reason}"
                    for w in pre_result.warnings
                ]
            )

        # === Step 2: Multi-robot coordination safety check ===
        robot_id = params.get("robot_id")
        if robot_id and self.coordination_verifier:
            logger.debug(f"Checking multi-robot coordination safety")
            coord_result = self.coordination_verifier.verify_multi_robot_safety(
                robot_id, op_def.category, params, self.world_state
            )
            details["coordination_check"] = coord_result.to_dict()

            if not coord_result.safe:
                issue_msgs = [
                    f"{i.issue_type}: {i.description}" for i in coord_result.issues
                ]
                logger.error(f"Coordination safety check failed: {issue_msgs}")
                return {
                    "safe": False,
                    "error": f"Multi-robot coordination issue: {'; '.join(issue_msgs)}",
                    "warnings": warnings,
                    "details": details,
                }

            # Collect coordination warnings
            if coord_result.warnings:
                warnings.extend(
                    [
                        f"Coordination - {w.issue_type}: {w.description}"
                        for w in coord_result.warnings
                    ]
                )

        # === All checks passed ===
        logger.debug(f"Unified verification passed for {op_def.name}")
        return {"safe": True, "error": None, "warnings": warnings, "details": details}

    def _auto_capture_outputs(self, operation_name: str, result: Dict[str, Any]):
        """
        Automatically capture operation outputs based on ParameterFlow definitions.

        This enables automatic chaining: detect_object output → move_to_coordinate input.

        Args:
            operation_name: Name of the operation that just completed
            result: Operation result containing output values
        """
        if not result:
            return

        # Get operation definition to access relationships
        op_def = self.registry.get_operation_by_name(operation_name)
        if not op_def or not op_def.relationships:
            return

        # Check if operation has parameter flows
        param_flows = op_def.relationships.parameter_flows
        if not param_flows:
            return

        # Capture each output defined in parameter flows
        for flow in param_flows:
            output_key = flow.source_output_key
            if output_key in result:
                # Store with a namespaced variable name: {operation}_{key}
                var_name = f"{operation_name}_{output_key}"
                self._variables[var_name] = result[output_key]
                logger.debug(
                    f"Auto-captured {output_key}={result[output_key]} to ${var_name} (from {operation_name})"
                )

                # Also store in operation-level dict for easier access
                op_result_var = f"{operation_name}_result"
                if op_result_var not in self._variables:
                    self._variables[op_result_var] = {}
                self._variables[op_result_var][output_key] = result[output_key]

    def _auto_inject_parameters(
        self, operation_name: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Automatically inject parameters from previous operations based on ParameterFlow definitions.

        This enables automatic chaining: previous operation output → current operation input.

        Args:
            operation_name: Name of the operation about to execute
            params: Current parameters (may be incomplete)

        Returns:
            Parameters with auto-injected values from previous operations
        """
        # Get operation definition to access relationships
        op_def = self.registry.get_operation_by_name(operation_name)
        if not op_def or not op_def.relationships:
            return params

        # Check if operation has parameter flows (inputs from other operations)
        param_flows = op_def.relationships.parameter_flows
        if not param_flows:
            return params

        enhanced_params = dict(params)

        # Inject parameters from previous operations
        for flow in param_flows:
            # Check if this flow targets the current operation
            if (
                flow.target_operation != op_def.operation_id
                and flow.target_operation != operation_name
            ):
                continue

            # Check if parameter is already provided
            target_param = flow.target_input_param
            if target_param in enhanced_params:
                logger.debug(
                    f"Parameter {target_param} already provided, skipping auto-injection"
                )
                continue

            # Try to get value from captured variables
            source_var_name = f"{flow.source_operation}_{flow.source_output_key}"
            if source_var_name in self._variables:
                var_value = self._variables[source_var_name]

                # Special handling for object_id parameter with detection result
                if target_param == "object_id" and isinstance(var_value, dict):
                    # Extract object identifier from detection result
                    # Detection results have 'color' field (e.g., "blue_cube")
                    if "color" in var_value:
                        enhanced_params[target_param] = var_value["color"]
                        logger.info(
                            f"Auto-injected {target_param}='{var_value['color']}' (extracted from detection result ${source_var_name})"
                        )
                    else:
                        logger.warning(f"Detection result missing 'color' field, cannot auto-inject {target_param}")
                else:
                    enhanced_params[target_param] = var_value
                    logger.info(
                        f"Auto-injected {target_param}={var_value} from ${source_var_name}"
                    )
            else:
                logger.debug(
                    f"No captured value for {source_var_name}, cannot auto-inject {target_param}"
                )

        return enhanced_params

    def _resolve_variables(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve variable references in parameters.

        Variables are referenced with $ prefix (e.g., $target).
        Supports dotted notation (e.g., $target.x) and arithmetic expressions (e.g., $target.z + 0.05).

        Args:
            params: Parameters that may contain variable references

        Returns:
            Parameters with variables resolved
        """
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and "$" in value:
                # Handle expressions with arithmetic (e.g., "$target.z + 0.05")
                if any(op in value for op in ["+", "-", "*", "/"]):
                    resolved_value = self._resolve_expression(value)
                    if resolved_value is not None:
                        resolved[key] = resolved_value
                    else:
                        logger.warning(f"Could not resolve expression: {value}")
                        resolved[key] = value
                # Handle dotted notation (e.g., "$target.x")
                elif "." in value and value.startswith("$"):
                    resolved_value = self._resolve_dotted_variable(value)
                    if resolved_value is not None:
                        resolved[key] = resolved_value
                    else:
                        logger.warning(f"Variable {value} not found")
                        resolved[key] = value
                # Handle simple variable reference (e.g., "$target")
                elif value.startswith("$"):
                    var_name = value[1:]  # Remove $ prefix

                    if var_name in self._variables:
                        var_value = self._variables[var_name]

                        # Special handling for position/coordinate parameters
                        if key in ["x", "y", "z"] and isinstance(var_value, dict):
                            # Extract coordinate from detection result
                            resolved[key] = var_value.get(key, 0.0)
                        elif key == "position" and isinstance(var_value, dict):
                            # Expand position variable to x, y, z
                            resolved["x"] = var_value.get("x", 0.0)
                            resolved["y"] = var_value.get("y", 0.0)
                            resolved["z"] = var_value.get("z", 0.0)
                        # Special handling for object_id parameter with detection result
                        elif key == "object_id" and isinstance(var_value, dict):
                            # Extract object identifier from detection result
                            # Detection results have 'color' field (e.g., "blue_cube")
                            if "color" in var_value:
                                resolved[key] = var_value["color"]
                                logger.info(f"Extracted object_id='{var_value['color']}' from detection result")
                            else:
                                logger.warning(f"Detection result missing 'color' field, cannot extract object_id")
                                resolved[key] = value
                        else:
                            resolved[key] = var_value
                    else:
                        logger.warning(f"Variable ${var_name} not found")
                        resolved[key] = value
                else:
                    resolved[key] = value
            else:
                resolved[key] = value

        return resolved

    def _resolve_dotted_variable(self, var_ref: str) -> Optional[Any]:
        """
        Resolve dotted variable notation (e.g., $target.x).

        Args:
            var_ref: Variable reference string (e.g., "$target.x")

        Returns:
            Resolved value or None if not found
        """
        if not var_ref.startswith("$"):
            return None

        # Remove $ prefix and split by dots
        parts = var_ref[1:].split(".")

        # Start with base variable
        value = self._variables.get(parts[0])
        if value is None:
            return None

        # Navigate through dotted path
        for part in parts[1:]:
            if isinstance(value, dict):
                value = value.get(part)
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                return None

        return value

    def _resolve_expression(self, expr: str) -> Optional[float]:
        """
        Resolve arithmetic expressions containing variables (e.g., "$target.z + 0.05").

        Args:
            expr: Expression string containing variables and arithmetic

        Returns:
            Evaluated result or None if evaluation fails
        """
        import re

        # Find all variable references (e.g., $target.x, $target.z)
        var_pattern = r"\$[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*"
        variables = re.findall(var_pattern, expr)

        # Replace each variable with its value
        resolved_expr = expr
        for var_ref in variables:
            value = self._resolve_dotted_variable(var_ref)
            if value is None:
                logger.warning(f"Could not resolve {var_ref} in expression: {expr}")
                return None
            resolved_expr = resolved_expr.replace(var_ref, str(value))

        # Safely evaluate the expression
        try:
            # Only allow safe mathematical operations
            import ast
            import operator

            # Parse the expression
            node = ast.parse(resolved_expr, mode="eval")

            # Define allowed operators
            safe_operators = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.USub: operator.neg,
            }

            def eval_node(node):
                if isinstance(node, ast.Expression):
                    return eval_node(node.body)
                elif isinstance(node, ast.Constant):
                    # ast.Constant handles all literal values (numbers, strings, etc.) since Python 3.8
                    return node.value
                elif isinstance(node, ast.BinOp):
                    left = eval_node(node.left)
                    right = eval_node(node.right)
                    op = safe_operators.get(type(node.op))
                    if op is None:
                        raise ValueError(f"Unsupported operator: {type(node.op)}")
                    return op(left, right)
                elif isinstance(node, ast.UnaryOp):
                    operand = eval_node(node.operand)
                    op = safe_operators.get(type(node.op))
                    if op is None:
                        raise ValueError(f"Unsupported operator: {type(node.op)}")
                    return op(operand)
                else:
                    raise ValueError(f"Unsupported node type: {type(node)}")

            result = eval_node(node)
            # Ensure we return a float, handle various numeric types
            if isinstance(result, (int, float)):
                return float(result)
            else:
                raise ValueError(f"Expression did not evaluate to a number: {result}")

        except Exception as e:
            logger.error(f"Error evaluating expression '{expr}': {e}")
            return None

    def get_variable(self, name: str) -> Optional[Any]:
        """Get a stored variable value."""
        return self._variables.get(name)

    def set_variable(self, name: str, value: Any):
        """Set a variable value manually."""
        self._variables[name] = value


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
        timeout_per_command: Optional[float] = None,
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

    def wait_for_completion(
        self, timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
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
