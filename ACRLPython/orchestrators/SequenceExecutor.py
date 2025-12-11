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
from operations.Base import OperationCategory
from operations.Verification import OperationVerifier
from operations.CoordinationVerifier import CoordinationVerifier
from operations.WorldState import get_world_state
from servers.CommandServer import get_command_broadcaster

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SequenceExecutor:
    """
    Executes command sequences with completion tracking and error handling.

    Each command is executed and the executor waits for completion before
    proceeding to the next command in the sequence.

    Supports variable passing between operations:
        detect_object -> $target
        move_to_coordinate with position=$target
    """

    def __init__(
        self,
        default_timeout: float = 60.0,
        check_completion: bool = True,
        enable_verification: bool = True
    ):
        """
        Initialize the SequenceExecutor.

        Args:
            default_timeout: Default timeout in seconds for each operation (60s for movements)
            check_completion: Whether to check for operation completion via StatusServer
            enable_verification: Whether to enable formal verification (preconditions/postconditions)
        """
        self.registry = get_global_registry()
        self.default_timeout = default_timeout
        self.check_completion = check_completion
        self.enable_verification = enable_verification
        self._abort_flag = False
        self._current_sequence_id: Optional[str] = None
        self._progress_callbacks: List[Callable] = []
        self._variables: Dict[str, Any] = {}  # Variable storage for passing results between operations

        # Initialize verification components
        if enable_verification:
            self.verifier = OperationVerifier()
            self.coordination_verifier = CoordinationVerifier()
            self.world_state = get_world_state()
        else:
            self.verifier = None
            self.coordination_verifier = None
            self.world_state = None

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

        # Clear variables for new sequence
        self._variables = {}

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
                "duration_ms": cmd_duration
            }
            results.append(result_entry)

            if cmd_result["success"]:
                completed += 1
                self._notify_progress(i, len(commands), operation, "completed")
                logger.info(f"Command {i + 1} completed successfully in {cmd_duration:.0f}ms")

                # Capture result to variable if specified (manual capture)
                if capture_var and cmd_result.get("result"):
                    self._variables[capture_var] = cmd_result["result"]
                    logger.debug(f"Captured result to ${capture_var}")

                # === PRIORITY 2: Automatic Parameter Flow ===
                # Automatically capture outputs based on ParameterFlow definitions
                self._auto_capture_outputs(operation, cmd_result.get("result", {}))
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
            get_command_broadcaster().create_completion_queue(request_id)
            logger.debug(f"Created completion queue for request_id {request_id}")

        try:
            # Get operation definition for verification
            op_def = self.registry.get_operation_by_name(operation)
            if op_def is None:
                return {
                    "success": False,
                    "result": None,
                    "error": f"Operation '{operation}' not found in registry"
                }

            # === PRIORITY 3: Unified Verification ===
            if self.enable_verification and self.verifier:
                verification_result = self._verify_operation_safety(op_def, params)

                if not verification_result["safe"]:
                    return {
                        "success": False,
                        "result": None,
                        "error": verification_result["error"],
                        "verification_details": verification_result["details"]
                    }

                # Log any warnings
                if verification_result["warnings"]:
                    for warning in verification_result["warnings"]:
                        logger.warning(f"Verification warning: {warning}")

            # Add request_id to params so operation includes it in command to Unity
            params_with_request_id = {**params, "request_id": request_id}

            # Execute the operation
            op_result = self.registry.execute_operation_by_name(operation, **params_with_request_id)

            # === PHASE 3: Postcondition Verification ===
            if self.enable_verification and self.verifier and op_result.success:
                logger.debug(f"Verifying postconditions for {operation}")
                post_result = self.verifier.verify_postconditions(
                    op_def, op_result, params, self.world_state
                )

                # Postcondition failures are warnings, not blockers
                if not post_result.success:
                    violation_msgs = [f"{v.predicate}: {v.reason}" for v in post_result.violations]
                    logger.warning(f"Postcondition verification failed: {violation_msgs}")

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

            # Skip completion waiting for perception operations (they complete immediately)
            op_def = self.registry.get_operation_by_name(operation)
            if op_def and op_def.category == OperationCategory.PERCEPTION:
                logger.debug(f"Skipping completion wait for perception operation: {operation}")
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
                get_command_broadcaster().remove_completion_queue(request_id)
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
                response = get_command_broadcaster().get_completion(request_id, timeout=poll_interval)

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

    def _verify_operation_safety(self, op_def, params: Dict[str, Any]) -> Dict[str, Any]:
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

        # === Step 1: Verify operation preconditions ===
        logger.debug(f"Running unified verification for {op_def.name}")
        pre_result = self.verifier.verify_preconditions(op_def, params, self.world_state)
        details["precondition_check"] = pre_result.to_dict()

        if not pre_result.execution_allowed:
            violation_msgs = [f"{v.predicate}: {v.reason}" for v in pre_result.violations]
            logger.error(f"Precondition verification failed: {violation_msgs}")
            return {
                "safe": False,
                "error": f"Precondition failed: {'; '.join(violation_msgs)}",
                "warnings": warnings,
                "details": details
            }

        # Collect precondition warnings
        if pre_result.warnings:
            warnings.extend([f"Precondition - {w.predicate}: {w.reason}" for w in pre_result.warnings])

        # === Step 2: Multi-robot coordination safety check ===
        robot_id = params.get("robot_id")
        if robot_id and self.coordination_verifier:
            logger.debug(f"Checking multi-robot coordination safety")
            coord_result = self.coordination_verifier.verify_multi_robot_safety(
                robot_id, op_def.category, params, self.world_state
            )
            details["coordination_check"] = coord_result.to_dict()

            if not coord_result.safe:
                issue_msgs = [f"{i.issue_type}: {i.description}" for i in coord_result.issues]
                logger.error(f"Coordination safety check failed: {issue_msgs}")
                return {
                    "safe": False,
                    "error": f"Multi-robot coordination issue: {'; '.join(issue_msgs)}",
                    "warnings": warnings,
                    "details": details
                }

            # Collect coordination warnings
            if coord_result.warnings:
                warnings.extend([f"Coordination - {w.issue_type}: {w.description}" for w in coord_result.warnings])

        # === All checks passed ===
        logger.debug(f"Unified verification passed for {op_def.name}")
        return {
            "safe": True,
            "error": None,
            "warnings": warnings,
            "details": details
        }

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
                logger.debug(f"Auto-captured {output_key}={result[output_key]} to ${var_name} (from {operation_name})")

                # Also store in operation-level dict for easier access
                op_result_var = f"{operation_name}_result"
                if op_result_var not in self._variables:
                    self._variables[op_result_var] = {}
                self._variables[op_result_var][output_key] = result[output_key]

    def _auto_inject_parameters(self, operation_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
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
            if flow.target_operation != op_def.operation_id and flow.target_operation != operation_name:
                continue

            # Check if parameter is already provided
            target_param = flow.target_input_param
            if target_param in enhanced_params:
                logger.debug(f"Parameter {target_param} already provided, skipping auto-injection")
                continue

            # Try to get value from captured variables
            source_var_name = f"{flow.source_operation}_{flow.source_output_key}"
            if source_var_name in self._variables:
                enhanced_params[target_param] = self._variables[source_var_name]
                logger.info(f"Auto-injected {target_param}={self._variables[source_var_name]} from ${source_var_name}")
            else:
                logger.debug(f"No captured value for {source_var_name}, cannot auto-inject {target_param}")

        return enhanced_params

    def _resolve_variables(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve variable references in parameters.

        Variables are referenced with $ prefix (e.g., $target).
        Special handling for position variables from detect_object results.

        Args:
            params: Parameters that may contain variable references

        Returns:
            Parameters with variables resolved
        """
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith('$'):
                var_name = value[1:]  # Remove $ prefix

                if var_name in self._variables:
                    var_value = self._variables[var_name]

                    # Special handling for position/coordinate parameters
                    if key in ['x', 'y', 'z'] and isinstance(var_value, dict):
                        # Extract coordinate from detection result
                        resolved[key] = var_value.get(key, 0.0)
                    elif key == 'position' and isinstance(var_value, dict):
                        # Expand position variable to x, y, z
                        resolved['x'] = var_value.get('x', 0.0)
                        resolved['y'] = var_value.get('y', 0.0)
                        resolved['z'] = var_value.get('z', 0.0)
                    else:
                        resolved[key] = var_value
                else:
                    logger.warning(f"Variable ${var_name} not found")
                    resolved[key] = value
            else:
                resolved[key] = value

        return resolved

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
