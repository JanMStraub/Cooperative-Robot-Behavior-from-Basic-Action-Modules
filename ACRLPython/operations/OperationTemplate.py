"""
TEMPLATE: Robot Operation Implementation
==========================================

This template shows how to implement a new robot operation that can be:
1. Registered in the operation registry
2. Retrieved via RAG semantic search
3. Executed by LLM-driven robot control

Copy this file to create new operations. Replace all [PLACEHOLDER] sections.
"""

from typing import Dict, Any
import time
import logging
from servers.CommandServer import get_command_broadcaster
from .Base import (
    BasicOperation,
    OperationCategory,
    OperationComplexity,
    OperationParameter,
    OperationResult,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# STEP 1: Implement the operation function
# ============================================================================


def [operation_name](
    # Required parameters
    robot_id: str,
    # [Add your required parameters here]

    # Optional parameters with defaults
    # [Add your optional parameters here]
) -> Dict[str, Any]:
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
        Dict with the following structure:
        {
            "success": bool,           # True if command was sent successfully
            "result": dict or None,    # Result data if successful
            "error": dict or None      # Error information if failed
        }

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
            return {
                "success": False,
                "result": None,
                "error": {
                    "code": "INVALID_ROBOT_ID",
                    "message": f"Robot ID must be a non-empty string, got: {robot_id}",
                    "recovery_suggestions": [
                        "Provide a valid robot ID (e.g., 'Robot1', 'AR4_Robot')",
                        "Check RobotManager in Unity for available robot IDs",
                    ],
                },
            }

        # [Add validation for each parameter]
        # Example parameter validation:
        # if not (min_value <= parameter <= max_value):
        #     return {
        #         "success": False,
        #         "result": None,
        #         "error": {
        #             "code": "INVALID_[PARAMETER]",
        #             "message": f"[Parameter] {parameter} out of range [min, max]",
        #             "recovery_suggestions": [
        #                 "Adjust [parameter] to be within range [min, max]",
        #             ],
        #         },
        #     }

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
        }

        # Send to Unity via CommandBroadcaster
        logger.info(f"Sending [operation_name] command to {robot_id}")

        success = get_command_broadcaster().send_command(command)

        if not success:
            return {
                "success": False,
                "result": None,
                "error": {
                    "code": "COMMUNICATION_FAILED",
                    "message": "Failed to send command to Unity - no clients connected",
                    "recovery_suggestions": [
                        "Ensure Unity is running with LLMResultsReceiver active",
                        "Verify ResultsServer is running (port 5010)",
                        "Check Unity console for connection errors",
                    ],
                },
            }

        # ==================================================================
        # SUCCESS: Return success result
        # ==================================================================

        logger.info(f"Successfully sent [operation_name] command to {robot_id}")

        return {
            "success": True,
            "result": {
                "robot_id": robot_id,
                # [Add your result data here]
                "status": "command_sent",
                "timestamp": time.time(),
            },
            "error": None,
        }

    except Exception as e:
        # ==================================================================
        # ERROR HANDLING: Catch unexpected errors
        # ==================================================================

        logger.error(f"Unexpected error in [operation_name]: {e}", exc_info=True)
        return {
            "success": False,
            "result": None,
            "error": {
                "code": "UNEXPECTED_ERROR",
                "message": f"Unexpected error occurred: {str(e)}",
                "recovery_suggestions": [
                    "Check logs for detailed error information",
                    "Verify all parameters are correct types",
                    "Retry the operation",
                    "Report bug if error persists",
                ],
            },
        }


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

            # Optional parameter example with range
            # OperationParameter(
            #     name="speed",
            #     type="float",
            #     description="Movement speed multiplier",
            #     required=False,
            #     default=1.0,
            #     valid_range=(0.1, 2.0),
            # ),
            # [Add your optional parameters]
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

        required_operations=[
            # List operation IDs that must be executed first
            # Example: ["perception_detect_001"] if you need to detect before acting
        ],

        commonly_paired_with=[
            # List operation IDs often used together
            # Example: ["motion_move_001", "manipulation_grip_001"]
        ],

        mutually_exclusive_with=[
            # List operation IDs that can't run simultaneously
            # Example: ["motion_rotate_001"]
        ],

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
