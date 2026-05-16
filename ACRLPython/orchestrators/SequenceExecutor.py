#!/usr/bin/env python3
"""Executes parsed command sequences, waiting for each op to complete before proceeding."""

from typing import Dict, Any, List, Optional, Callable
import time
import logging
import threading
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
    TimeoutError as FuturesTimeoutError,
)

# Lazy imports to avoid circular dependency
# Handle both direct execution and package import
try:
    from ..operations.Base import OperationCategory
    from ..operations.Verification import OperationVerifier
    from ..operations.CoordinationVerifier import CoordinationVerifier
    from ..core.Imports import get_world_state
    from ..config.Servers import REFLEXION_ENABLED, REFLEXION_MAX_RETRIES
except ImportError:
    from operations.Base import OperationCategory
    from operations.Verification import OperationVerifier
    from operations.CoordinationVerifier import CoordinationVerifier
    from core.Imports import get_world_state
    from config.Servers import REFLEXION_ENABLED, REFLEXION_MAX_RETRIES

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from ..core.LoggingSetup import _safe_log, _make_handler_safe
except ImportError:
    from core.LoggingSetup import _safe_log, _make_handler_safe

# Apply safe patching to all existing handlers (including pytest's handlers)
for _handler in logger.handlers[:]:
    _make_handler_safe(_handler)
for _handler in logging.root.handlers[:]:
    _make_handler_safe(_handler)


def _extract_waypoint_from_verification(
    verification_result: dict,
) -> "tuple[float, float, float] | None":
    """Parse the first WAYPOINT:x,y,z suggestion from a CoordinationVerifier result."""
    details = verification_result.get("details", {})
    coord_check = (
        details.get("coordination_check", {}) if isinstance(details, dict) else {}
    )
    issues = coord_check.get("issues", []) if isinstance(coord_check, dict) else []
    for issue in issues:
        for suggestion in issue.get("resolution_suggestions", []):
            if isinstance(suggestion, str) and suggestion.startswith("WAYPOINT:"):
                try:
                    parts = suggestion[len("WAYPOINT:") :].split(",")
                    if len(parts) == 3:
                        return (float(parts[0]), float(parts[1]), float(parts[2]))
                except ValueError:
                    pass
    return None


class SequenceExecutor:
    """Executes command sequences with completion tracking, variable passing, and Reflexion retries."""

    # Class-level atomic counter for request IDs (shared across all instances)
    _request_id_counter = 0
    _request_id_lock = threading.Lock()

    class _MetricsTracker:
        """Welford online-average tracker for per-executor operation metrics."""

        def __init__(self):
            self._lock = threading.Lock()
            self._ops_executed: int = 0
            self._ops_succeeded: int = 0
            self._ops_failed: int = 0
            self._avg_duration_ms: float = 0.0

        def record(self, success: bool, duration_ms: float):
            with self._lock:
                self._ops_executed += 1
                if success:
                    self._ops_succeeded += 1
                else:
                    self._ops_failed += 1
                self._avg_duration_ms += (
                    duration_ms - self._avg_duration_ms
                ) / self._ops_executed

        def snapshot(self) -> Dict[str, Any]:
            with self._lock:
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

        def reset(self):
            with self._lock:
                self._ops_executed = 0
                self._ops_succeeded = 0
                self._ops_failed = 0
                self._avg_duration_ms = 0.0

    def __init__(
        self,
        default_timeout: float = 120.0,  # Increased from 90s; handoff grasps can take 40-60s under load
        check_completion: bool = True,
        enable_verification: bool = True,
    ):
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

        # Operation metrics — delegated to nested _MetricsTracker
        self._metrics = self._MetricsTracker()

        if enable_verification:
            self.verifier = OperationVerifier()
            self.coordination_verifier = CoordinationVerifier()
            self.world_state = get_world_state()
        else:
            self.verifier = None
            self.coordination_verifier = None
            self.world_state = None

        self.outcome_tracker: Optional[Any] = None

    def _get_command_broadcaster(self):
        from core.Imports import get_command_broadcaster

        return get_command_broadcaster()

    @classmethod
    def _generate_request_id(cls) -> int:
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
        """Execute commands sequentially or in parallel groups; returns full result dict."""
        self._abort_flag = False
        self._current_sequence_id = sequence_id or f"seq_{int(time.time() * 1000)}"
        timeout = timeout_per_command or self.default_timeout

        # Clear variables for new sequence
        self._variables = {}

        start_time = time.time()
        results = []
        completed = 0
        reflexion_recoveries = 0

        logger.info(
            f"Starting sequence {self._current_sequence_id} with {len(commands)} commands"
        )

        # Check if commands use parallel_group
        has_parallel_groups = any("parallel_group" in cmd for cmd in commands)

        if has_parallel_groups:
            # Execute with parallel group support
            logger.info("Parallel execution mode enabled")
            group_results, group_completed = self._execute_parallel_groups(
                commands, timeout
            )
            results = group_results
            completed = group_completed
        else:
            # Sequential execution
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

                # Auto-inject parameters from previous operations based on ParameterFlow definitions
                params = self._auto_inject_parameters(operation, params)

                # Warn about unresolved $ references before resolution
                for _k, _v in params.items():
                    if isinstance(_v, str) and _v.startswith("$"):
                        _var_name = _v.lstrip("$").split(".")[0]
                        if _var_name not in self._variables:
                            logger.warning(
                                "Pre-exec: param '%s' references $%s which is not yet captured",
                                _k,
                                _var_name,
                            )

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
                        self._capture_result_to_var(capture_var, cmd_result["result"])

                    # Automatically capture outputs based on ParameterFlow definitions
                    self._auto_capture_outputs(operation, cmd_result.get("result", {}))
                else:
                    self._notify_progress(i, len(commands), operation, "failed")
                    _safe_log(
                        logger.error,
                        f"Command {i + 1} failed: {cmd_result.get('error')}",
                    )

                    # Reflexion retry: re-parse the original command with error context.
                    # Only applies to NAVIGATION and MANIPULATION ops — re-planning a
                    # status check or sync primitive with the LLM makes no sense and
                    # would block for LLM_REQUEST_TIMEOUT seconds if the LLM is unavailable.
                    reflexion_succeeded = False
                    original_text = cmd.get("_original_text", "")
                    _reflexion_eligible_categories = {
                        OperationCategory.NAVIGATION,
                        OperationCategory.MANIPULATION,
                        OperationCategory.PERCEPTION,
                    }
                    try:
                        from ..core.Imports import get_global_registry
                    except ImportError:
                        from core.Imports import get_global_registry
                    _registry = get_global_registry()
                    _op_def = (
                        _registry.get_operation_by_name(operation)
                        if _registry
                        else None
                    )
                    _op_category = _op_def.category if _op_def else None
                    _reflexion_allowed = _op_category in _reflexion_eligible_categories
                    if (
                        original_text
                        and REFLEXION_ENABLED
                        and REFLEXION_MAX_RETRIES > 0
                        and _reflexion_allowed
                    ):
                        error_msg = cmd_result.get("error", "Unknown error")
                        recovery = cmd_result.get("recovery_suggestions", [])
                        hint = f"Previous attempt failed: {error_msg}."
                        if recovery:
                            hint += " Suggestions: " + "; ".join(recovery)

                        try:
                            from ..orchestrators.CommandParser import get_command_parser
                        except ImportError:
                            from orchestrators.CommandParser import get_command_parser

                        parser = get_command_parser()
                        robot_id = params.get("robot_id", "Robot1")

                        for retry_n in range(1, REFLEXION_MAX_RETRIES + 1):
                            logger.info(
                                f"Reflexion retry {retry_n}/{REFLEXION_MAX_RETRIES} "
                                f"for command {i + 1}: {operation}"
                            )
                            retry_parse = parser.parse_with_hint(
                                original_text,
                                robot_id=robot_id,
                                hint=hint,
                                use_motion_layer=False,
                            )
                            if (
                                not retry_parse["success"]
                                or not retry_parse["commands"]
                            ):
                                logger.warning(
                                    f"Reflexion retry {retry_n} parse failed"
                                )
                                continue

                            # Re-resolve variables for the retried command(s)
                            retry_cmd = retry_parse["commands"][0]
                            retry_op = retry_cmd.get("operation", operation)
                            retry_params = self._resolve_variables(
                                retry_cmd.get("params", params)
                            )
                            retry_params = self._auto_inject_parameters(
                                retry_op, retry_params
                            )

                            retry_start = time.time()
                            retry_result = self._execute_single_command(
                                retry_op, retry_params, timeout
                            )
                            retry_duration = (time.time() - retry_start) * 1000

                            if retry_result["success"]:
                                logger.info(
                                    f"Reflexion retry {retry_n} succeeded for command {i + 1}"
                                )
                                # Overwrite the failed result entry with the successful one
                                results[-1] = {
                                    "index": i,
                                    "operation": retry_op,
                                    "success": True,
                                    "result": retry_result.get("result"),
                                    "error": None,
                                    "error_code": None,
                                    "duration_ms": retry_duration,
                                }
                                completed += 1
                                self._notify_progress(
                                    i, len(commands), retry_op, "completed"
                                )
                                if cmd.get("capture_var") and retry_result.get(
                                    "result"
                                ):
                                    self._capture_result_to_var(
                                        cmd["capture_var"], retry_result["result"]
                                    )
                                self._auto_capture_outputs(
                                    retry_op, retry_result.get("result", {})
                                )
                                reflexion_succeeded = True
                                reflexion_recoveries += 1
                                break
                            else:
                                error_msg = retry_result.get("error", "Unknown error")
                                recovery = retry_result.get("recovery_suggestions", [])
                                hint = f"Retry {retry_n} failed: {error_msg}."
                                if recovery:
                                    hint += " Suggestions: " + "; ".join(recovery)
                                logger.warning(
                                    f"Reflexion retry {retry_n} failed: {error_msg}"
                                )

                    if not reflexion_succeeded:
                        # Stop sequence on exhausted retries
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
            "reflexion_recoveries": reflexion_recoveries,
        }

        _safe_log(
            logger.info,
            f"Sequence {self._current_sequence_id} finished: "
            f"{completed}/{len(commands)} commands in {total_duration:.0f}ms",
        )

        # Broadcast outcome to WorldState for peer-robot awareness
        try:
            ws = get_world_state()
            if ws is not None:
                final_states = {}
                for r in results:
                    if r is None:
                        continue
                    obj_state = (r.get("result") or {}).get("object_state")
                    if obj_state and isinstance(obj_state, dict):
                        obj_id = obj_state.get("object_id")
                        if obj_id:
                            final_states[obj_id] = obj_state
                # Derive robot_id from first command's params
                _broadcast_robot_id = "unknown"
                if commands:
                    _broadcast_robot_id = (
                        commands[0].get("params", {}).get("robot_id", "unknown")
                    )
                ws.broadcast_task_outcome(
                    robot_id=_broadcast_robot_id,
                    task_id=self._current_sequence_id or "unknown",
                    success=success,
                    duration_ms=total_duration,
                    final_object_states=final_states,
                )
        except Exception:
            pass  # Broadcasting is best-effort; never fail a sequence over it

        return result

    def _execute_parallel_groups(
        self, commands: List[Dict[str, Any]], timeout: float
    ) -> tuple[List[Optional[Dict[str, Any]]], int]:
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

            # Launch all futures for this group via ThreadPoolExecutor.
            # Using as_completed with an overall timeout so that unhandled
            # exceptions inside execute_command_thread propagate via
            # future.result() instead of silently disappearing.
            #
            # NOTE: We manage the pool explicitly (not via context manager) so
            # that shutdown(wait=False) is used on timeout/abort.  The context
            # manager form always calls shutdown(wait=True) on __exit__, which
            # would block here for the full operation timeout even after
            # as_completed gives up — the root cause of the 30-second hang on
            # graceful shutdown.
            pool = ThreadPoolExecutor(max_workers=len(group_commands))
            try:
                future_to_idx = {
                    pool.submit(execute_command_thread, idx, cmd): idx
                    for idx, cmd in group_commands
                }
                try:
                    for future in as_completed(future_to_idx, timeout=timeout + 5.0):
                        idx = future_to_idx[future]
                        try:
                            future.result()  # re-raises any exception from the worker
                        except Exception as exc:
                            logger.error(
                                f"[Group {group_num}] Command {idx} raised an exception: {exc}"
                            )
                            # Write an error result so the group-result loop sees it
                            cmd = dict(group_commands)[idx]
                            with result_lock:
                                thread_results[idx] = (
                                    {
                                        "index": idx,
                                        "operation": cmd.get("operation", ""),
                                        "success": False,
                                        "result": None,
                                        "error": str(exc),
                                        "duration_ms": 0,
                                        "parallel_group": group_num,
                                    },
                                    {"success": False, "error": str(exc)},
                                    cmd.get("capture_var"),
                                )
                except FuturesTimeoutError:
                    # One or more futures did not finish within timeout+5s.
                    # Return immediately — do not wait for stray threads.
                    logger.warning(
                        f"[Group {group_num}] as_completed timed out; some commands may not have finished"
                    )
            finally:
                # shutdown(wait=False) returns immediately; any still-running
                # threads will complete in the background without blocking us.
                pool.shutdown(wait=False)

            # Process results from this group
            group_success = True
            for _, (idx, cmd) in enumerate(group_commands):
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
                            self._capture_result_to_var(
                                capture_var, cmd_result["result"]
                            )

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
                        f"[Group {group_num}] Command {idx} did not complete (thread timeout): "
                        f"operation={cmd.get('operation', '?')} params={cmd.get('params', {})}"
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
        self._metrics.record(success, duration_ms)

    def get_metrics(self) -> Dict[str, Any]:
        return self._metrics.snapshot()

    def reset_metrics(self):
        """Reset all operation metrics counters to zero."""
        self._metrics.reset()

    def _execute_single_command(
        self,
        operation: str,
        params: Dict[str, Any],
        timeout: float,
        _replan_depth: int = 0,
    ) -> Dict[str, Any]:
        request_id = self._generate_request_id()
        _cmd_start = time.time()

        # If completion checking is enabled, create queue before sending command
        if self.check_completion:
            self._get_command_broadcaster().create_completion_queue(request_id)
            logger.debug(f"Created completion queue for request_id {request_id}")

        _result: Dict[str, Any] = {
            "success": False,
            "result": None,
            "error": "internal",
        }
        try:
            # Resolve common LLM name abbreviations to registered names.
            _OP_ALIASES = {
                "return_to_start": "return_to_start_position",
            }
            operation = _OP_ALIASES.get(operation, operation)

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
                    # Attempt waypoint replan if CoordinationVerifier embedded one (max 1 replan)
                    waypoint = _extract_waypoint_from_verification(verification_result)
                    if waypoint is not None and _replan_depth == 0:
                        logger.info(
                            f"Proximity pre-check blocked {operation} — replanning via safe waypoint {waypoint}"
                        )
                        robot_id = params.get("robot_id", "")
                        wp_result = self.registry.execute_operation_by_name(
                            "move_to_coordinate",
                            robot_id=robot_id,
                            x=waypoint[0],
                            y=waypoint[1],
                            z=waypoint[2],
                            request_id=self._generate_request_id(),
                        )
                        if wp_result.success:
                            # Waypoint dispatched — retry original command (no further replanning)
                            return self._execute_single_command(
                                operation, params, timeout, _replan_depth=1
                            )
                        logger.warning(
                            "Waypoint move failed — aborting original command"
                        )
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
                        if impl is None:
                            raise RuntimeError(
                                f"Operation '{op_def.name}' has no implementation"
                            )
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
            # (no Unity command with outer request_id was sent).
            if op_result.result and op_result.result.get("status") in (
                "ros_executed",
                "ros_command_sent",
                "ros_executed_with_grasp_planning",
                "vgn_ros_executed",
                "handoff_received",
            ):
                logger.debug(
                    f"Skipping completion wait for ROS-executed operation: {operation}"
                )
                _result = {"success": True, "result": op_result.result, "error": None}
                return _result

            # Skip completion waiting for operations that execute in Python only
            op_def = self.registry.get_operation_by_name(operation)
            if op_def and op_def.category in (
                OperationCategory.PERCEPTION,
                OperationCategory.SYNC,
            ):
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
            _duration_ms = (time.time() - _cmd_start) * 1000
            self._record_metric(
                success=_result.get("success", False),
                duration_ms=_duration_ms,
            )

    def _wait_for_completion(
        self, operation: str, request_id: int, timeout: float
    ) -> bool:
        """Poll with adaptive backoff (50ms → 500ms) for Unity's completion signal."""
        start_time = time.time()

        # Adaptive polling parameters
        min_poll_interval = 0.05  # Start at 50ms
        max_poll_interval = 0.5  # Max 500ms
        poll_increase_rate = 1.1  # Increase by 10% each iteration
        current_poll_interval = min_poll_interval

        # Queue was already created in _execute_single_command

        while time.time() - start_time < timeout:
            if self._abort_flag:
                return False

            # Detect mid-execution proximity freeze reported by Unity
            if self.world_state is not None:
                for robot_id, robot_state in list(
                    self.world_state._robot_states.items()
                ):
                    if getattr(robot_state, "proximity_frozen", False):
                        logger.warning(
                            f"Robot {robot_id} frozen by proximity mid-execution of {operation}"
                        )
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
        self._progress_callbacks.append(callback)

    def _notify_progress(self, index: int, total: int, operation: str, status: str):
        for callback in self._progress_callbacks:
            try:
                callback(index, total, operation, status)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    def _verify_operation_safety(
        self, op_def, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Single pre-execution check combining preconditions and multi-robot coordination."""
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

        # === Step 1.5: Knowledge Graph spatial feasibility check ===
        robot_id_for_kg = params.get("robot_id")
        if robot_id_for_kg:
            kg_result = self._check_spatial_feasibility(op_def, params, robot_id_for_kg)
            if not kg_result["safe"]:
                return {
                    "safe": False,
                    "error": f"Spatial feasibility: {kg_result['warning']}",
                    "warnings": warnings,
                    "details": details,
                }
            if kg_result.get("warning"):
                warnings.append(f"KG spatial - {kg_result['warning']}")

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

    def _check_spatial_feasibility(
        self, op_def, params: Dict[str, Any], robot_id: str
    ) -> Dict[str, Any]:
        """KG path-block and reachability check; returns safe=True on any exception."""
        try:
            from config.KnowledgeGraph import KNOWLEDGE_GRAPH_ENABLED

            if not KNOWLEDGE_GRAPH_ENABLED:
                return {"safe": True}

            from core.Imports import get_graph_query_engine

            qe = get_graph_query_engine()
            if qe is None:
                return {"safe": True}

            op_name = op_def.name

            # --- Move operations: path-blocked check ---
            MOVE_OPS = {
                "move_to_coordinate",
                "move_from_a_to_b",
                "move_relative_to_object",
                "move_between_objects",
            }
            if op_name in MOVE_OPS:
                target = None
                if "position" in params:
                    pos = params["position"]
                    if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                        target = tuple(pos[:3])
                    elif isinstance(pos, dict):
                        target = (
                            float(pos.get("x", 0)),
                            float(pos.get("y", 0)),
                            float(pos.get("z", 0)),
                        )
                elif all(k in params for k in ("x", "y", "z")):
                    target = (
                        float(params["x"]),
                        float(params["y"]),
                        float(params["z"]),
                    )

                if target is not None and qe.is_path_blocked(robot_id, target):
                    return {
                        "safe": False,
                        "warning": f"Path to {target} appears blocked for {robot_id}",
                    }

            # --- Grasp/stabilize operations: reachability check ---
            GRASP_OPS = {"grasp_object", "grip_object", "stabilize_object"}
            if op_name in GRASP_OPS:
                object_id = params.get("object_id")
                if object_id:
                    reachable_robots = qe.find_reachable_robots(object_id)
                    # Only block when the list is populated AND robot is not in it.
                    # Empty list = KG not yet populated = don't block.
                    if reachable_robots and robot_id not in reachable_robots:
                        return {
                            "safe": False,
                            "warning": (
                                f"{robot_id} is not in the reachable set for '{object_id}' "
                                f"(reachable: {reachable_robots})"
                            ),
                        }

            return {"safe": True}

        except Exception as e:
            logger.debug(f"KG spatial feasibility check skipped: {e}")
            return {"safe": True, "warning": f"KG check skipped: {e}"}

    def _capture_result_to_var(self, capture_var: str, result: Dict[str, Any]) -> None:
        """Store result under capture_var; flattens field-detection center dict for easy $var.x access."""
        if isinstance(result.get("center"), dict):
            # Field-detection result: expose center coordinates at the top level
            # so $field_g.x / $field_g resolve correctly via existing coord logic.
            self._variables[capture_var] = result["center"]
            self._variables[f"{capture_var}_result"] = result
            logger.debug(
                f"Captured field center to ${capture_var} and full result to ${capture_var}_result"
            )
        else:
            self._variables[capture_var] = result
            logger.debug(f"Captured result to ${capture_var}")

    def _auto_capture_outputs(self, operation_name: str, result: Dict[str, Any]):
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

            # Resolve dotted keys (e.g. "center.x" → result["center"]["x"])
            if "." in output_key:
                parts = output_key.split(".", 1)
                parent_val = result.get(parts[0])
                value = (
                    parent_val.get(parts[1]) if isinstance(parent_val, dict) else None
                )
            else:
                value = result.get(output_key)

            if value is None:
                continue

            # Store with a namespaced variable name: {operation}_{key}
            # Dots are replaced with underscores so the name stays a valid key.
            safe_key = output_key.replace(".", "_")
            var_name = f"{operation_name}_{safe_key}"
            self._variables[var_name] = value
            logger.debug(
                f"Auto-captured {output_key}={value} to ${var_name} (from {operation_name})"
            )

            # Also store in operation-level dict for easier access
            op_result_var = f"{operation_name}_result"
            if op_result_var not in self._variables:
                self._variables[op_result_var] = {}
            self._variables[op_result_var][safe_key] = value

    def _auto_inject_parameters(
        self, operation_name: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Get operation definition to access relationships
        op_def = self.registry.get_operation_by_name(operation_name)
        if not op_def:
            return params
        relationships = getattr(op_def, "relationships", None)
        if not relationships:
            return params

        # Check if operation has parameter flows (inputs from other operations)
        param_flows = relationships.parameter_flows
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
            # Dots in source_output_key are stored as underscores (see _auto_capture_outputs)
            safe_key = flow.source_output_key.replace(".", "_")
            source_var_name = f"{flow.source_operation}_{safe_key}"
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
                        logger.warning(
                            f"Detection result missing 'color' field, cannot auto-inject {target_param}"
                        )
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

    def _resolve_single_value(self, key: str, value: Any) -> Any:
        """Resolve one $var reference (arithmetic, dotted, or simple); returns original string on failure."""
        # Handle expressions with arithmetic (e.g., "$target.z + 0.05")
        if any(op in value for op in ["+", "-", "*", "/"]):
            resolved_value = self._resolve_expression(value)
            if resolved_value is not None:
                return resolved_value
            logger.warning(f"Could not resolve expression: {value}")
            return value

        # Handle dotted notation (e.g., "$target.x") — resolves to a scalar directly
        if "." in value and value.startswith("$"):
            resolved_value = self._resolve_dotted_variable(value)
            if resolved_value is not None:
                return resolved_value

            # Dotted resolution failed — try known fallback patterns.
            parts = value[1:].split(".")  # strip "$", split by dots
            base_var = parts[0]
            base_val = self._variables.get(base_var)

            # Pattern: "$field.center.x" where detect_field already stored the center
            # dict directly under the capture variable (i.e. $field == {"x":…,"y":…,"z":…}).
            # The LLM may still emit ".center." — strip that level and retry.
            if (
                len(parts) == 3
                and parts[1] == "center"
                and parts[2] in ("x", "y", "z")
                and isinstance(base_val, dict)
                and parts[2] in base_val
            ):
                recovered = base_val[parts[2]]
                logger.warning(
                    f"Variable {value} not found — recovered as ${base_var}.{parts[2]}={recovered}"
                )
                return recovered

            # Pattern: "$target.id" / "$target.name" used instead of "$target.color"
            # for object_id parameters in grasp operations.
            if (
                key.startswith("object_id")
                and isinstance(base_val, dict)
                and "color" in base_val
            ):
                logger.warning(
                    f"Variable {value} not found for object_id — "
                    f"falling back to ${base_var}.color='{base_val['color']}'"
                )
                return base_val["color"]

            logger.warning(f"Variable {value} not found")
            return value

        # Handle simple variable reference (e.g., "$target")
        if value.startswith("$"):
            var_name = value[1:]
            if var_name in self._variables:
                var_value = self._variables[var_name]

                # Special handling for position/coordinate parameters
                if key in ["x", "y", "z"] and isinstance(var_value, dict):
                    return var_value.get(key, 0.0)
                if key == "position" and isinstance(var_value, dict):
                    # Caller handles the dict → x/y/z expansion in _resolve_variables
                    return var_value
                # Special handling for object_id / object_id1 / object_id2
                if key.startswith("object_id") and isinstance(var_value, dict):
                    if "color" in var_value:
                        logger.info(
                            f"Extracted object_id='{var_value['color']}' from detection result"
                        )
                        return var_value["color"]
                    logger.warning(
                        "Detection result missing 'color' field, cannot extract object_id"
                    )
                    return value
                return var_value

            logger.warning(f"Variable ${var_name} not found")
            return value

        return value

    def _resolve_variables(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve all $var references in a params dict, including dotted, arithmetic, and list elements."""
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and "$" in value:
                result = self._resolve_single_value(key, value)
                # Special case: simple $var that resolved to a position dict
                if key == "position" and isinstance(result, dict):
                    resolved["x"] = result.get("x", 0.0)
                    resolved["y"] = result.get("y", 0.0)
                    resolved["z"] = result.get("z", 0.0)
                else:
                    resolved[key] = result
            elif isinstance(value, (list, tuple)):
                # Resolve any $ references inside list/tuple elements (e.g., object_ref)
                resolved_list = []
                for element in value:
                    if isinstance(element, str) and "$" in element:
                        # Use a neutral key so coordinate/object_id special-casing
                        # doesn't fire — dotted refs like "$p.x" already yield scalars.
                        resolved_list.append(self._resolve_single_value("", element))
                    else:
                        resolved_list.append(element)
                resolved[key] = (
                    type(value)(resolved_list)
                    if isinstance(value, tuple)
                    else resolved_list
                )
            else:
                resolved[key] = value

        return resolved

    def _resolve_dotted_variable(self, var_ref: str) -> Optional[Any]:
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

    def negotiate_if_needed(
        self,
        command_text: str,
        robot_id: str = "Robot1",
    ) -> Optional[List[Dict[str, Any]]]:
        """Run NegotiationHub if the command needs multi-robot planning; returns None to fall back."""
        try:
            from core.Imports import get_negotiation_hub

            hub = get_negotiation_hub()
            if hub is None:
                return None

            if not hub.needs_negotiation(command_text, robot_id):
                return None

            logger.info(f"Negotiation triggered for: {command_text[:80]}")
            handoff_ctx = self._get_handoff_context(command_text, robot_id)
            result = hub.negotiate(command_text, spatial_context=handoff_ctx)

            if result.success and result.commands:
                logger.info(
                    f"Negotiation succeeded: {len(result.commands)} commands, "
                    f"{result.rounds_taken} rounds, {result.duration_s:.1f}s"
                )
                return result.commands

            logger.warning(
                f"Negotiation failed ({result.state.value}): {result.reasoning}. "
                f"Falling back to normal parsing."
            )
            return None

        except ImportError:
            logger.debug("NegotiationHub not available, skipping negotiation")
            return None
        except Exception as e:
            logger.error(f"Negotiation error: {e}. Falling back to normal parsing.")
            return None

    def _get_handoff_context(
        self, command_text: str, robot_id: str
    ) -> Optional[Dict[str, Any]]:
        HANDOFF_KEYWORDS = {"hand", "pass", "give", "transfer", "handoff"}
        try:
            if not any(kw in command_text.lower() for kw in HANDOFF_KEYWORDS):
                return None

            from config.KnowledgeGraph import KNOWLEDGE_GRAPH_ENABLED

            if not KNOWLEDGE_GRAPH_ENABLED:
                return None

            from core.Imports import get_graph_query_engine

            qe = get_graph_query_engine()
            if qe is None:
                return None

            # Get all object node IDs from the KG graph
            from knowledge_graph._singleton import get_knowledge_graph

            kg = get_knowledge_graph()
            all_objects = kg.get_all_nodes(node_type="object")

            # Naive match: first object whose ID appears in the command text
            matched_object = None
            for obj_id in all_objects:
                if obj_id.lower() in command_text.lower():
                    matched_object = obj_id
                    break

            if not matched_object:
                return None

            # Determine other robot (simple two-robot assumption)
            other_robot = "Robot2" if robot_id == "Robot1" else "Robot1"

            candidates = qe.get_handoff_candidates(
                robot_id, other_robot, matched_object
            )

            return {
                "handoff_candidates": candidates,
                "handoff_object": matched_object,
            }

        except Exception as e:
            logger.debug(f"KG handoff context unavailable: {e}")
            return None

    def get_variable(self, name: str) -> Optional[Any]:
        return self._variables.get(name)

    def set_variable(self, name: str, value: Any):
        self._variables[name] = value


class AsyncSequenceExecutor:
    """Runs SequenceExecutor in a background thread."""

    def __init__(self, executor: Optional[SequenceExecutor] = None):
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
        return self._current_thread is not None and self._current_thread.is_alive()

    def get_result(self) -> Optional[Dict[str, Any]]:
        return self._result

    def wait_for_completion(
        self, timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        if self._current_thread is not None:
            self._current_thread.join(timeout=timeout)
        return self._result

    def abort(self):
        self.executor.abort()

    def add_completion_callback(self, callback: Callable):
        self._completion_callbacks.append(callback)

    def _notify_completion(self):
        for callback in self._completion_callbacks:
            try:
                callback(self._result)
            except Exception as e:
                logger.error(f"Completion callback error: {e}")


# Singleton instances
_executor_instance: Optional[SequenceExecutor] = None
_async_executor_instance: Optional[AsyncSequenceExecutor] = None


def get_sequence_executor() -> SequenceExecutor:
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = SequenceExecutor()
    return _executor_instance


def get_async_executor() -> AsyncSequenceExecutor:
    global _async_executor_instance
    if _async_executor_instance is None:
        _async_executor_instance = AsyncSequenceExecutor()
    return _async_executor_instance
