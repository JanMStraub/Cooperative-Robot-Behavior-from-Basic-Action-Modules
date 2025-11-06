# Robot Operations System

A comprehensive framework for defining, registering, and executing robot control operations with rich metadata for RAG-enabled LLM control.

## Overview

This operations system provides:

- **Structured Operation Definitions**: Each operation includes parameters, preconditions, postconditions, and failure modes
- **RAG-Optimized Metadata**: Rich natural language descriptions for semantic search and LLM retrieval
- **Parameter Validation**: Automatic validation with detailed error messages
- **Operation Registry**: Central catalog of all available operations
- **Execution Framework**: Standardized execution with consistent return types
- **Unity Integration**: TCP communication with Unity robot controllers

## Architecture

```
operations/
├── base.py              # Base classes (BasicOperation, OperationParameter, etc.)
├── move_operations.py   # Movement operations (move_to_coordinate)
├── registry.py          # Operation registry and global instance
├── example_usage.py     # Example script showing usage
└── README.md            # This file
```

### Communication Flow

```
Python Operation → ResultsBroadcaster (port 5006)
                 ↓
           ResultsServer (TCP)
                 ↓
         Unity LLMResultsReceiver
                 ↓
         PythonCommandHandler
                 ↓
         RobotController.SetTarget()
```

## Quick Start

### 1. Basic Usage

```python
from LLMCommunication.operations import get_global_registry

# Get the global registry
registry = get_global_registry()

# Execute an operation
result = registry.execute_operation_by_name(
    "move_to_coordinate",
    robot_id="Robot1",
    x=0.3,
    y=0.15,
    z=0.1,
    speed=1.0
)

if result.success:
    print(f"Success! {result.result}")
else:
    print(f"Error: {result.error['message']}")
    print(f"Suggestions: {result.error['recovery_suggestions']}")
```

### 2. Direct Function Call

```python
from LLMCommunication.operations import move_to_coordinate

# Call the function directly
result = move_to_coordinate(
    robot_id="Robot1",
    x=0.3,
    y=0.15,
    z=0.1
)
```

### 3. List Available Operations

```python
from LLMCommunication.operations import get_global_registry, OperationCategory

registry = get_global_registry()

# Get all operations
all_ops = registry.get_all_operations()
print(f"Total operations: {len(all_ops)}")

# Get operations by category
nav_ops = registry.get_operations_by_category(OperationCategory.NAVIGATION)
for op in nav_ops:
    print(f"- {op.name}: {op.description}")
```

## Prerequisites

### Unity Side

1. **LLMResultsReceiver**: Must be active in the scene (listens on port 5006)
2. **PythonCommandHandler**: Must be attached to a GameObject (processes commands)
3. **RobotManager**: Must have registered robots with valid IDs
4. **RobotController**: Each robot must have a RobotController component

### Python Side

1. **ResultsServer**: Must be running (port 5006)

```bash
# Start the ResultsServer with analyzer
python -m LLMCommunication.orchestrators.RunAnalyzer

# Or start just the ResultsServer
python -m LLMCommunication.servers.ResultsServer
```

## Available Operations

### move_to_coordinate

Move the robot's end effector to a specific 3D coordinate.

**Parameters:**
- `robot_id` (str, required): ID of the robot (e.g., "Robot1")
- `x` (float, required): X coordinate in meters, range: [-0.5, 0.5]
- `y` (float, required): Y coordinate in meters, range: [-0.5, 0.5]
- `z` (float, required): Z coordinate in meters, range: [0.0, 0.6]
- `speed` (float, optional): Speed multiplier, default: 1.0, range: [0.1, 2.0]
- `approach_offset` (float, optional): Stop distance before target in meters, default: 0.0, range: [0.0, 0.1]

**Example:**
```python
result = registry.execute_operation_by_name(
    "move_to_coordinate",
    robot_id="Robot1",
    x=0.3,
    y=0.15,
    z=0.1,
    speed=0.5,
    approach_offset=0.05  # Stop 5cm before target
)
```

**Returns:**
```python
{
    "success": True,
    "result": {
        "robot_id": "Robot1",
        "target_position": {"x": 0.3, "y": 0.15, "z": 0.15},  # z includes offset
        "original_target": {"x": 0.3, "y": 0.15, "z": 0.1},
        "speed": 0.5,
        "approach_offset": 0.05,
        "status": "command_sent",
        "timestamp": 1730896543.123
    },
    "error": None
}
```

## Creating New Operations

### Step 1: Implement the Function

```python
# In a new file, e.g., grip_operations.py

from typing import Dict, Any
import logging
from ..servers.ResultsServer import ResultsBroadcaster

logger = logging.getLogger(__name__)


def grip_object(robot_id: str, force: float = 0.5) -> Dict[str, Any]:
    """
    Close the gripper to grasp an object.

    Args:
        robot_id: ID of the robot
        force: Grip force (0.1 to 1.0)

    Returns:
        Result dict with success, result, error keys
    """
    # Validate parameters
    if not (0.1 <= force <= 1.0):
        return {
            "success": False,
            "result": None,
            "error": {
                "code": "INVALID_FORCE",
                "message": f"Force {force} out of range [0.1, 1.0]",
                "recovery_suggestions": ["Use force between 0.1 and 1.0"]
            }
        }

    # Send command to Unity
    command = {
        "command_type": "grip_object",
        "robot_id": robot_id,
        "parameters": {"force": force}
    }

    success = ResultsBroadcaster.send_result(command)

    if not success:
        return {
            "success": False,
            "result": None,
            "error": {
                "code": "COMMUNICATION_FAILED",
                "message": "Failed to send command to Unity",
                "recovery_suggestions": ["Check Unity connection"]
            }
        }

    return {
        "success": True,
        "result": {"robot_id": robot_id, "force": force, "status": "command_sent"},
        "error": None
    }
```

### Step 2: Create BasicOperation Definition

```python
from .base import (
    BasicOperation,
    OperationCategory,
    OperationComplexity,
    OperationParameter
)


def create_grip_object_operation() -> BasicOperation:
    """Create the BasicOperation definition for grip_object"""
    return BasicOperation(
        operation_id="manip_grip_object_001",
        name="grip_object",
        category=OperationCategory.MANIPULATION,
        complexity=OperationComplexity.BASIC,

        description="Close the gripper to grasp an object with appropriate force",

        long_description="""
This operation closes the robot's gripper fingers to securely grasp an object.
The gripper will close with the specified force level.
        """,

        usage_examples=[
            "After moving to object: grip_object(robot_id='Robot1', force=0.5)",
            "Gentle grip: grip_object(robot_id='Robot1', force=0.2)"
        ],

        parameters=[
            OperationParameter(
                name="robot_id",
                type="str",
                description="ID of the robot",
                required=True
            ),
            OperationParameter(
                name="force",
                type="float",
                description="Grip force (0.1=gentle, 1.0=maximum)",
                required=False,
                default=0.5,
                valid_range=(0.1, 1.0)
            )
        ],

        preconditions=[
            "Gripper is open",
            "Robot is at object location"
        ],

        postconditions=[
            "Gripper is closed",
            "Object is grasped securely"
        ],

        average_duration_ms=800.0,
        success_rate=0.88,

        failure_modes=[
            "No object between gripper fingers",
            "Object too large or too small"
        ],

        required_operations=["move_to_coordinate"],
        commonly_paired_with=["detect_object", "lift_object"],
        mutually_exclusive_with=["release_object"],

        implementation=grip_object  # Link to implementation
    )


# Create operation instance
GRIP_OBJECT_OPERATION = create_grip_object_operation()
```

### Step 3: Register the Operation

```python
# In registry.py, add to _initialize_operations():

from .grip_operations import GRIP_OBJECT_OPERATION

def _initialize_operations(self):
    operations = [
        MOVE_TO_COORDINATE_OPERATION,
        GRIP_OBJECT_OPERATION,  # Add new operation
    ]
    # ...
```

### Step 4: Export in __init__.py

```python
# In __init__.py

from .grip_operations import (
    grip_object,
    GRIP_OBJECT_OPERATION,
    create_grip_object_operation
)

__all__ = [
    # ... existing exports ...
    'grip_object',
    'GRIP_OBJECT_OPERATION',
    'create_grip_object_operation'
]
```

## RAG System Integration

### Export Operations for RAG

```python
from LLMCommunication.operations import get_global_registry

registry = get_global_registry()
registry.export_for_rag("./rag_documents")
```

This creates:
- Individual `.txt` files for each operation (optimized for semantic search)
- `operations_index.json` with operation metadata

### RAG Document Format

Each operation is exported as a structured text document:

```
OPERATION: move_to_coordinate (ID: motion_move_to_coord_001)
Category: navigation | Complexity: basic

DESCRIPTION:
[Long description of what the operation does...]

WHEN TO USE THIS OPERATION:
[Short description for quick matching...]

USAGE EXAMPLES:
- Example 1...
- Example 2...

PARAMETERS:
- robot_id (str): Description [Required]
- x (float): Description [Required]
...

PRECONDITIONS (must be true before execution):
- Condition 1
- Condition 2

POSTCONDITIONS (will be true after execution):
- Result 1
- Result 2

PERFORMANCE METRICS:
- Average Duration: 1200ms
- Success Rate: 96%

KNOWN FAILURE MODES:
- Failure 1
- Failure 2

RELATED OPERATIONS:
- Required: []
- Commonly paired with: [detect_object, grip_object]
- Mutually exclusive: [rotate_gripper]
```

### Using in RAG Pipeline

1. **Ingest** the exported documents into your vector database (e.g., ChromaDB, Pinecone)
2. **Query** with natural language: "How do I move the robot to a position?"
3. **Retrieve** relevant operations
4. **Extract** parameters and call the operation

```python
# Pseudocode for RAG-LLM pipeline
query = "Move the robot to pick up the detected object"
relevant_ops = rag_search(query)  # Returns ["move_to_coordinate", ...]

# LLM generates parameters based on context
params = llm_extract_parameters(query, relevant_ops, detected_object_pos)

# Execute the operation
result = registry.execute_operation_by_name("move_to_coordinate", **params)
```

## Error Handling

All operations return a standardized result structure:

```python
{
    "success": bool,
    "result": dict or None,    # If successful
    "error": {                 # If failed
        "code": str,                    # Error code (e.g., "INVALID_PARAMETER")
        "message": str,                 # Human-readable message
        "recovery_suggestions": list    # Suggested fixes
    }
}
```

**Common Error Codes:**
- `INVALID_PARAMETER`: Parameter validation failed
- `COMMUNICATION_FAILED`: Unity connection lost
- `OPERATION_NOT_FOUND`: Operation ID/name not in registry
- `NOT_IMPLEMENTED`: Operation has no implementation
- `EXECUTION_ERROR`: Exception during execution
- `UNEXPECTED_ERROR`: Unhandled exception

## Testing

### Run Example Script

```bash
cd /Users/jan/Code/MS/ACRLPython
python -m LLMCommunication.operations.example_usage
```

### Manual Testing

```python
from LLMCommunication.operations import get_global_registry

registry = get_global_registry()

# Test parameter validation
result = registry.execute_operation_by_name(
    "move_to_coordinate",
    robot_id="Robot1",
    x=10.0,  # Invalid - out of range
    y=0.0,
    z=0.1
)

assert not result.success
assert result.error["code"] == "INVALID_X_COORDINATE"
print(result.error["recovery_suggestions"])
```

## Troubleshooting

### "COMMUNICATION_FAILED" Error

**Cause**: Unity not connected to ResultsServer

**Solutions:**
1. Ensure Unity is running with LLMResultsReceiver active
2. Start ResultsServer: `python -m LLMCommunication.orchestrators.RunAnalyzer`
3. Check Unity console for connection errors
4. Verify port 5006 is not blocked

### "OPERATION_NOT_FOUND" Error

**Cause**: Operation not registered in registry

**Solutions:**
1. Check operation name spelling
2. Verify operation was added to registry._initialize_operations()
3. List available operations: `registry.get_all_operations()`

### "Robot ID not found" Error

**Cause**: Robot ID doesn't exist in Unity's RobotManager

**Solutions:**
1. Check robot GameObject name in Unity
2. Verify RobotManager.RobotInstances contains the robot
3. Use correct robot ID (e.g., "Robot1", "AR4_Robot")

## Future Enhancements

- [ ] Add synchronous execution mode (wait for completion)
- [ ] Add operation chaining/sequencing
- [ ] Add gripper operations (grip, release)
- [ ] Add perception operations (detect_object, get_robot_state)
- [ ] Add composite operations (pick_and_place)
- [ ] Add operation history/logging
- [ ] Add performance metrics tracking
- [ ] Add operation dependencies validation

## API Reference

See docstrings in source files for detailed API documentation:
- `base.py`: Core classes
- `move_operations.py`: Movement operations
- `registry.py`: Registry and execution
