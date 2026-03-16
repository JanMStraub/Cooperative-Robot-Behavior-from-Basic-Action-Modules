#!/usr/bin/env python3
"""
TEMPLATE: Robot Operation Implementation
==========================================

This template shows how to implement a new robot operation that can be:
1. Registered in the operation registry
2. Retrieved via RAG semantic search
3. Executed by LLM-driven robot control

Copy this file to create new operations. Replace all [PLACEHOLDER] sections.

RECENT ENHANCEMENTS (December 2025):
- OperationResult dataclass with helper methods (success_result, error_result)
- Enhanced OperationRelationship with reasons and parameter flows
- Automatic parameter validation via OperationParameter.validate()
- Support for request_id in Protocol V2 for request/response correlation
- Backward compatible dict-like access for OperationResult
"""

from typing import Dict, Any
import time

# Import from centralized lazy import system (prevents circular dependencies)
try:
    from ..core.Imports import get_command_broadcaster
except ImportError:
    from core.Imports import get_command_broadcaster

# Handle both direct execution and package import
try:
    from .Base import (
        BasicOperation,
        OperationCategory,
        OperationComplexity,
        OperationParameter,
        OperationResult,
        ParameterFlow,
        OperationRelationship,
    )
except ImportError:
    from operations.Base import (
        BasicOperation,
        OperationCategory,
        OperationComplexity,
        OperationParameter,
        OperationResult,
        ParameterFlow,
        OperationRelationship,
    )

# Configure logging
from core.LoggingSetup import get_logger
logger = get_logger(__name__)


# ============================================================================
# STEP 1: Implement the operation function
# ============================================================================


def [operation_name](
    # Required parameters
    robot_id: str,
    # [Add your required parameters here]

    # Optional parameters with defaults
    # [Add your optional parameters here]

    request_id: int = 0,
) -> OperationResult:
    """
    [Brief one-line description of what this operation does]

    [Detailed description explaining:
    - What the operation does
    - How it works
    - What happens in Unity
    - Any important safety considerations
    ]

    Args:
        robot_id: ID of the robot to control (e.g., "Robot1", "AR4_Robot")
        [parameter_name]: [Description], range: [min, max]
        [Add descriptions for all parameters]

    Returns:
        OperationResult with the following structure:
        - success (bool): True if command was sent successfully
        - result (dict or None): Result data if successful
        - error (dict or None): Error information if failed

        Success result structure:
        {
            "robot_id": str,
            [Add your result fields here]
            "status": str,
            "timestamp": float
        }

        Error structure:
        {
            "code": str,                    # Error code (e.g., "INVALID_PARAMETER")
            "message": str,                 # Human-readable error message
            "recovery_suggestions": list    # List of suggested actions
        }

        Note: Returns OperationResult dataclass, but supports dict-like access
        for backward compatibility (result["success"], result["result"], etc.)

    Example:
        >>> # [Example 1]
        >>> result = [operation_name]("Robot1", ...)
        >>> if result["success"]:
        ...     print(f"Operation completed: {result['result']}")

        >>> # [Example 2]
        >>> result = [operation_name]("Robot1", ...)
    """
    try:
        # ==================================================================
        # VALIDATION: Validate all parameters before execution
        # ==================================================================

        # Validate robot_id
        if not robot_id or not isinstance(robot_id, str):
            return OperationResult.error_result(
                "INVALID_ROBOT_ID",
                f"Robot ID must be a non-empty string, got: {robot_id}",
                [
                    "Provide a valid robot ID (e.g., 'Robot1', 'AR4_Robot')",
                    "Check RobotManager in Unity for available robot IDs",
                ],
            )

        # [Add validation for each parameter]
        # Example parameter validation using OperationResult.error_result():
        # if not (min_value <= parameter <= max_value):
        #     return OperationResult.error_result(
        #         "INVALID_[PARAMETER]",
        #         f"[Parameter] {parameter} out of range [min, max]",
        #         [
        #             "Adjust [parameter] to be within range [min, max]",
        #         ],
        #     )

        # ==================================================================
        # EXECUTION: Perform the operation logic
        # ==================================================================

        # [Add your operation logic here]
        # This could be:
        # - Constructing a command to send to Unity
        # - Performing calculations
        # - Querying state
        # - etc.

        # Example: Construct command for Unity
        command = {
            "command_type": "[operation_name]",
            "robot_id": robot_id,
            "parameters": {
                # [Add your parameters here]
            },
            "timestamp": time.time(),
            "request_id": request_id,
        }

        # Send to Unity via CommandBroadcaster
        logger.info(f"Sending [operation_name] command to {robot_id}")

        success = get_command_broadcaster().send_command(command, request_id)

        if not success:
            return OperationResult.error_result(
                "COMMUNICATION_FAILED",
                "Failed to send command to Unity - no clients connected",
                [
                    "Ensure Unity is running with UnifiedPythonReceiver active",
                    "Verify CommandServer is running (port 5010)",
                    "Check Unity console for connection errors",
                ],
            )

        # ==================================================================
        # SUCCESS: Return success result
        # ==================================================================

        logger.info(f"Successfully sent [operation_name] command to {robot_id}")

        return OperationResult.success_result(
            {
                "robot_id": robot_id,
                # [Add your result data here]
                "status": "command_sent",
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        # ==================================================================
        # ERROR HANDLING: Catch unexpected errors
        # ==================================================================

        logger.error(f"Unexpected error in [operation_name]: {e}", exc_info=True)
        return OperationResult.error_result(
            "UNEXPECTED_ERROR",
            f"Unexpected error occurred: {str(e)}",
            [
                "Check logs for detailed error information",
                "Verify all parameters are correct types",
                "Retry the operation",
                "Report bug if error persists",
            ],
        )


# ============================================================================
# STEP 2: Create the BasicOperation definition for RAG system
# ============================================================================


def create_[operation_name]_operation() -> BasicOperation:
    """
    Create the BasicOperation definition for [operation_name].

    This provides rich metadata for RAG retrieval and LLM task planning.
    """
    return BasicOperation(
        # =================================================================
        # IDENTITY: Unique identifiers and categorization
        # =================================================================

        operation_id="[category]_[name]_[number]",  # e.g., "motion_rotate_001"
        name="[operation_name]",
        category=OperationCategory.[CATEGORY],  # PERCEPTION, NAVIGATION, MANIPULATION, STATE_CHECK, COORDINATION
        complexity=OperationComplexity.[COMPLEXITY],  # ATOMIC, BASIC, INTERMEDIATE, COMPLEX

        # =================================================================
        # DESCRIPTIONS: Natural language for RAG retrieval
        # =================================================================

        description="[One-line description of what this operation does]",

        long_description="""
            [Detailed multi-paragraph description explaining:
            - What the operation does
            - How it works internally
            - What happens in Unity
            - Any important considerations
            - Performance characteristics
            ]
        """,

        usage_examples=[
            "[Example 1: Common use case with sample code]",
            "[Example 2: Another typical scenario]",
            "[Example 3: Edge case or advanced usage]",
        ],

        # =================================================================
        # PARAMETERS: Define all operation parameters
        # =================================================================

        parameters=[
            # Required parameter example
            OperationParameter(
                name="robot_id",
                type="str",
                description="ID of the robot to control (e.g., 'Robot1', 'AR4_Robot')",
                required=True,
            ),
            # [Add your required parameters]

            # Optional parameter example with range validation
            # OperationParameter(
            #     name="speed",
            #     type="float",
            #     description="Movement speed multiplier",
            #     required=False,
            #     default=1.0,
            #     valid_range=(0.1, 2.0),  # Auto-validates parameter is in this range
            # ),

            # Optional parameter with enum-like validation
            # OperationParameter(
            #     name="approach_type",
            #     type="str",
            #     description="Approach strategy for grasping",
            #     required=False,
            #     default="top",
            #     valid_values=["top", "front", "side"],  # Auto-validates value is in this list
            # ),

            # [Add your optional parameters]

            # Note: Parameters with valid_range or valid_values will be automatically
            # validated by BasicOperation.execute() before calling your implementation.
            # You can also manually validate using OperationParameter.validate(value).
        ],

        # =================================================================
        # CONDITIONS: Pre and post conditions
        # =================================================================

        preconditions=[
            "[Condition that must be true before operation can run]",
            "[Another precondition]",
            # Examples:
            # "Robot arm is initialized and calibrated",
            # "Unity is running with LLMResultsReceiver active",
        ],

        postconditions=[
            "[State that will be true after operation completes]",
            "[Another postcondition]",
            # Examples:
            # "Command has been sent to Unity via TCP",
            # "Robot end effector will be at target position",
        ],

        # =================================================================
        # PERFORMANCE METRICS: For LLM decision making
        # =================================================================

        average_duration_ms=[typical_duration],  # Milliseconds
        success_rate=[0.0 to 1.0],  # e.g., 0.95 for 95% success rate

        failure_modes=[
            "[Common failure mode 1]",
            "[Common failure mode 2]",
            # Examples:
            # "Communication failed - Unity not connected",
            # "Parameter validation failed",
        ],

        # =================================================================
        # RELATIONSHIPS: How this relates to other operations
        # =================================================================

        relationships=OperationRelationship(
            operation_id="[category]_[name]_[number]",

            # Required operations with explanations
            required_operations=["status_check_robot_001"],
            required_reasons={
                "status_check_robot_001": "Verify robot is ready before executing operation",
            },

            # Commonly paired operations with reasons
            commonly_paired_with=["perception_detect_001", "motion_move_001"],
            pairing_reasons={
                "perception_detect_001": "Detect objects before acting on them",
                "motion_move_001": "Position robot after detection",
            },

            # Mutually exclusive operations with reasons
            mutually_exclusive_with=[],
            exclusion_reasons={},

            # Parameter flows for automatic chaining
            parameter_flows=[
                # Example: Pass detection coordinates to movement
                # ParameterFlow(
                #     source_operation="perception_detect_001",
                #     source_output_key="x",
                #     target_operation="motion_move_001",
                #     target_input_param="x",
                #     description="X coordinate of detected object"
                # ),
            ],

            # Temporal ordering hints
            typical_before=["manipulation_grip_001"],  # Operations usually after this one
            typical_after=["perception_detect_001"],   # Operations usually before this one

            # Multi-robot coordination (optional)
            coordination_requirements={
                # Example: "sync_required": True, "min_robots": 2
            },
        ),

        # =================================================================
        # IMPLEMENTATION: Link to the actual function
        # =================================================================

        implementation=[operation_name],
    )


# ============================================================================
# STEP 3: Create the operation instance for export
# ============================================================================

[OPERATION_CONSTANT] = create_[operation_name]_operation()


# ============================================================================
# STEP 4: Register in Registry.py
# ============================================================================
# After creating this file:
# 1. Import your operation in Registry.py:
#    from .YourOperationFile import YOUR_OPERATION_CONSTANT
#
# 2. Add to operations list in _initialize_operations():
#    operations = [
#        MOVE_TO_COORDINATE_OPERATION,
#        YOUR_OPERATION_CONSTANT,  # <-- Add here
#        ...
#    ]
# ============================================================================


# ============================================================================
# ALTERNATIVE PATTERNS
# ============================================================================
#
# Pattern 1: Command-based operations (shown above)
# - Send commands to Unity via CommandServer
# - Unity executes and optionally sends results back
# - Examples: move_to_coordinate, control_gripper
#
# Pattern 2: Data-retrieval operations (perception)
# - Retrieve data from ImageStorage or other Python-side storage
# - Process locally and return results directly
# - Examples: detect_objects, analyze_scene
# - Implementation pattern:
#
#   def detect_operation(robot_id: str, camera_id: str = "main") -> OperationResult:
#       try:
#           # Get data from storage
#           image_storage = ImageStorage.get_instance()
#           image = image_storage.get_camera_image(camera_id)
#
#           if image is None:
#               return OperationResult.error_result(
#                   "NO_IMAGE",
#                   f"No image available from camera '{camera_id}'",
#                   ["Ensure Unity is sending images", "Check StreamingServer is running"]
#               )
#
#           # Process locally
#           from vision.ObjectDetector import CubeDetector
#           detector = CubeDetector()
#           result = detector.detect_objects(image, camera_id=camera_id)
#
#           # Return results directly
#           return OperationResult.success_result({
#               "camera_id": camera_id,
#               "detections": [det.to_dict() for det in result.detections],
#               "count": len(result.detections),
#               "timestamp": time.time(),
#           })
#       except Exception as e:
#           return OperationResult.error_result(
#               "DETECTION_ERROR", f"Detection failed: {str(e)}",
#               ["Check logs", "Verify image format"]
#           )
#
# Pattern 3: Synchronization operations (multi-robot coordination)
# - Use shared state or signaling mechanisms
# - Examples: signal, wait_for_signal, wait
# - See SyncOperations.py for reference implementation
#
# ============================================================================


# ============================================================================
# ADVANCED USAGE: Using BasicOperation.execute()
# ============================================================================
#
# Once your operation is registered, you can execute it in two ways:
#
# Method 1: Direct function call (common in orchestrators)
#   result = move_to_coordinate(robot_id="Robot1", x=0.3, y=0.15, z=0.1)
#
# Method 2: Via BasicOperation.execute() (automatic parameter validation)
#   from operations.Registry import get_operation_registry
#   registry = get_operation_registry()
#   operation = registry.get_operation("motion_move_to_coord_001")
#   result = operation.execute(robot_id="Robot1", x=0.3, y=0.15, z=0.1)
#
# Method 2 provides:
# - Automatic parameter validation using OperationParameter.validate()
# - Consistent error handling and result formatting
# - Type checking and range validation
# - Better error messages for invalid parameters
#
# OperationResult supports dict-like access for backward compatibility:
#   if result["success"]:  # Works
#   if result.success:     # Also works
#   data = result["result"]  # Works
#   data = result.result     # Also works
#
# ============================================================================
