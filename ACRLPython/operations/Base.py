#!/usr/bin/env python3
"""
Basic Operations Foundation
============================

This module provides the base classes and data structures for defining robot
operations that can be executed through the Unity robot control system.

Operations are structured with rich metadata for RAG retrieval and LLM consumption.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import json


class OperationComplexity(Enum):
    """Complexity levels for operations"""

    ATOMIC = "atomic"  # Single action, cannot be decomposed
    BASIC = "basic"  # Simple coordinated action
    INTERMEDIATE = "intermediate"  # Multi-step sequence
    COMPLEX = "complex"  # Task-level operation


class OperationCategory(Enum):
    """Functional categories for operations"""

    PERCEPTION = "perception"  # Vision and sensing operations
    NAVIGATION = "navigation"  # Movement and positioning
    MANIPULATION = "manipulation"  # Grasping and object manipulation
    STATE_CHECK = "state_check"  # Status and verification
    COORDINATION = "coordination"  # Multi-robot coordination (deprecated)
    SYNC = "sync"  # Synchronization primitives for multi-robot tasks


@dataclass
class OperationParameter:
    """
    Definition of an operation parameter.

    Attributes:
        name: Parameter name (used in function calls)
        type: Python type as string (e.g., "float", "str", "bool")
        description: Human-readable description for LLM
        required: Whether parameter is mandatory
        default: Default value if not required
        valid_range: Optional tuple of (min, max) for numeric parameters
        valid_values: Optional list of valid values for enum-like parameters
    """

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    valid_range: Optional[tuple] = None
    valid_values: Optional[List[Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "required": self.required,
            "default": self.default,
            "valid_range": self.valid_range,
            "valid_values": self.valid_values,
        }

    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """
        Validate a parameter value.

        Returns:
            (is_valid, error_message)
        """
        # Check required
        if self.required and value is None:
            return False, f"Parameter '{self.name}' is required"

        # Check valid values (enum-like validation)
        # Normalize the string "None" to Python None (LLMs sometimes emit the
        # string literal instead of JSON null when the prompt uses repr()).
        if value == "None":
            value = None
        if self.valid_values and value is not None:
            if value not in self.valid_values:
                valid_str = ", ".join([f"'{v}'" for v in self.valid_values])
                return (
                    False,
                    f"Parameter '{self.name}' value '{value}' not in valid values: {valid_str}",
                )

        # Check range for numeric types
        if self.valid_range and value is not None:
            if isinstance(value, (int, float)):
                min_val, max_val = self.valid_range
                if not (min_val <= value <= max_val):
                    return (
                        False,
                        f"Parameter '{self.name}' value {value} out of range [{min_val}, {max_val}]",
                    )

        return True, None


@dataclass
class OperationResult:
    """
    Standardized result structure for all operations.

    Attributes:
        success: True if operation completed successfully
        result: Operation-specific result data (if successful)
        error: Error information (if failed)
    """

    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {"success": self.success, "result": self.result, "error": self.error}

    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access for backward compatibility"""
        if key == "success":
            return self.success
        elif key == "result":
            return self.result
        elif key == "error":
            return self.error
        else:
            raise KeyError(f"Invalid key: {key}")

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for backward compatibility"""
        return key in ("success", "result", "error")

    @staticmethod
    def success_result(result_data: Dict[str, Any]) -> "OperationResult":
        """Create a success result"""
        return OperationResult(success=True, result=result_data, error=None)

    @staticmethod
    def error_result(
        error_code: str, message: str, recovery_suggestions: List[str]
    ) -> "OperationResult":
        """Create an error result"""
        return OperationResult(
            success=False,
            result=None,
            error={
                "code": error_code,
                "message": message,
                "recovery_suggestions": recovery_suggestions,
            },
        )


@dataclass
class ParameterFlow:
    """
    Defines how data flows from one operation's output to another's input.

    This enables automatic parameter chaining, where the output of one operation
    (e.g., detected object coordinates) can be used as input to another operation (e.g., move_to_coordinate).

    Attributes:
        source_operation: Operation ID that produces the output
        source_output_key: Key in the source operation's result dict
        target_operation: Operation ID that consumes the input
        target_input_param: Parameter name in the target operation
        description: Human-readable explanation of the data flow
        transform: Optional transformation function name (e.g., "meters_to_mm")

    Example:
        ParameterFlow(
            source_operation="detect_object_stereo",
            source_output_key="x",
            target_operation="move_to_coordinate",
            target_input_param="x",
            description="Object X coordinate for robot positioning"
        )
    """

    source_operation: str
    source_output_key: str
    target_operation: str
    target_input_param: str
    description: str
    transform: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "source_operation": self.source_operation,
            "source_output_key": self.source_output_key,
            "target_operation": self.target_operation,
            "target_input_param": self.target_input_param,
            "description": self.description,
            "transform": self.transform,
        }


@dataclass
class OperationRelationship:
    """
    Rich relationship metadata for an operation.

    Extends basic relationship lists with explanations, parameter flows,
    and temporal ordering hints to help LLMs understand operation sequences.

    Attributes:
        operation_id: The operation this relationship describes
        required_operations: Operations that must exist/complete first
        required_reasons: Why each required operation is needed (op_id -> reason)
        commonly_paired_with: Operations often used together
        pairing_reasons: Why operations are paired (op_id -> reason)
        mutually_exclusive_with: Operations that conflict with this one
        exclusion_reasons: Why operations are exclusive (op_id -> reason)
        parameter_flows: Data connections from/to other operations
        typical_before: Operations typically executed before this one
        typical_after: Operations typically executed after this one
        coordination_requirements: Multi-robot coordination constraints

    Example:
        OperationRelationship(
            operation_id="detect_object_stereo",
            commonly_paired_with=["move_to_coordinate", "control_gripper"],
            pairing_reasons={
                "move_to_coordinate": "Move robot to detected object position",
                "control_gripper": "Grasp object after positioning"
            },
            typical_before=["move_to_coordinate"],
            parameter_flows=[
                ParameterFlow("detect_object_stereo", "x", "move_to_coordinate", "x", "X coordinate"),
                ParameterFlow("detect_object_stereo", "y", "move_to_coordinate", "y", "Y coordinate"),
                ParameterFlow("detect_object_stereo", "z", "move_to_coordinate", "z", "Z coordinate")
            ]
        )
    """

    operation_id: str
    required_operations: List[str] = field(default_factory=list)
    required_reasons: Dict[str, str] = field(default_factory=dict)
    commonly_paired_with: List[str] = field(default_factory=list)
    pairing_reasons: Dict[str, str] = field(default_factory=dict)
    mutually_exclusive_with: List[str] = field(default_factory=list)
    exclusion_reasons: Dict[str, str] = field(default_factory=dict)
    parameter_flows: List[ParameterFlow] = field(default_factory=list)
    typical_before: List[str] = field(default_factory=list)
    typical_after: List[str] = field(default_factory=list)
    coordination_requirements: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "operation_id": self.operation_id,
            "required_operations": self.required_operations,
            "required_reasons": self.required_reasons,
            "commonly_paired_with": self.commonly_paired_with,
            "pairing_reasons": self.pairing_reasons,
            "mutually_exclusive_with": self.mutually_exclusive_with,
            "exclusion_reasons": self.exclusion_reasons,
            "parameter_flows": [pf.to_dict() for pf in self.parameter_flows],
            "typical_before": self.typical_before,
            "typical_after": self.typical_after,
            "coordination_requirements": self.coordination_requirements,
        }


@dataclass
class BasicOperation:
    """
    Complete definition of a basic operation that can be retrieved by RAG.

    This class combines:
    - Rich natural language descriptions for LLM understanding
    - Technical specifications (parameters, pre/postconditions)
    - Performance metadata for decision making
    - Relationship information for task planning
    - Executable implementation function
    """

    # Identity
    operation_id: str
    name: str
    category: OperationCategory
    complexity: OperationComplexity

    # Natural language descriptions for RAG retrieval
    description: str
    long_description: str
    usage_examples: List[str]

    # Technical specifications
    parameters: List[OperationParameter]
    preconditions: List[str]
    postconditions: List[str]

    # Metadata for LLM decision making
    average_duration_ms: float
    success_rate: float
    failure_modes: List[str]

    # Rich relationship metadata
    relationships: Optional[OperationRelationship] = None

    # Flat relationship lists (legacy / convenience aliases for test construction)
    required_operations: List[str] = field(default_factory=list)
    commonly_paired_with: List[str] = field(default_factory=list)
    mutually_exclusive_with: List[str] = field(default_factory=list)

    # The actual implementation function
    implementation: Optional[Callable] = None

    def execute(self, **kwargs) -> OperationResult:
        """
        Execute the operation with given parameters.

        Args:
            **kwargs: Operation parameters

        Returns:
            OperationResult with success status and data
        """
        if self.implementation is None:
            return OperationResult.error_result(
                error_code="NOT_IMPLEMENTED",
                message=f"Operation '{self.name}' has no implementation",
                recovery_suggestions=["Contact developer to implement this operation"],
            )

        # Validate parameters
        validation_error = self.validate_parameters(kwargs)
        if validation_error:
            return validation_error

        # Execute implementation
        try:
            result = self.implementation(**kwargs)

            # Convert dict result to OperationResult if needed
            if isinstance(result, dict):
                return OperationResult(
                    success=result.get("success", False),
                    result=result.get("result"),
                    error=result.get("error"),
                )
            elif isinstance(result, OperationResult):
                return result
            else:
                return OperationResult.error_result(
                    error_code="INVALID_RETURN_TYPE",
                    message=f"Implementation returned invalid type: {type(result)}",
                    recovery_suggestions=[
                        "Implementation must return OperationResult or dict"
                    ],
                )
        except Exception as e:
            return OperationResult.error_result(
                error_code="EXECUTION_ERROR",
                message=f"Error executing operation: {str(e)}",
                recovery_suggestions=[
                    "Check logs for details",
                    "Verify parameters are correct",
                ],
            )

    def validate_parameters(self, kwargs: Dict[str, Any]) -> Optional[OperationResult]:
        """
        Validate operation parameters.

        Returns:
            OperationResult with error if validation fails, None if valid
        """
        for param in self.parameters:
            value = kwargs.get(param.name)
            is_valid, error_msg = param.validate(value)

            if not is_valid:
                return OperationResult.error_result(
                    error_code="INVALID_PARAMETER",
                    message=error_msg or f"Invalid parameter '{param.name}'",
                    recovery_suggestions=[
                        f"Check parameter '{param.name}' specification",
                        f"Expected type: {param.type}",
                        (
                            f"Valid range: {param.valid_range}"
                            if param.valid_range
                            else ""
                        ),
                    ],
                )

        return None

    def to_rag_document(self) -> str:
        """
        Convert operation to a rich text document optimized for RAG retrieval.

        Includes rich relationship metadata and parameter flow information
        to help LLMs understand operation sequencing and data dependencies.
        """
        # Build basic document sections
        doc = f"""
                OPERATION: {self.name} (ID: {self.operation_id})
                Category: {self.category.value} | Complexity: {self.complexity.value}

                DESCRIPTION:
                {self.long_description}

                WHEN TO USE THIS OPERATION:
                {self.description}

                USAGE EXAMPLES:
                {chr(10).join(f"- {example}" for example in self.usage_examples)}

                PARAMETERS:
                {chr(10).join(f"- {p.name} ({p.type}): {p.description}" +
                            (f" [Required: {p.required}, Default: {p.default}]" if not p.required else " [Required]")
                            for p in self.parameters)}

                PRECONDITIONS (must be true before execution):
                {chr(10).join(f"- {pre}" for pre in self.preconditions)}

                POSTCONDITIONS (will be true after execution):
                {chr(10).join(f"- {post}" for post in self.postconditions)}

                PERFORMANCE METRICS:
                - Average Duration: {self.average_duration_ms}ms
                - Success Rate: {self.success_rate * 100}%

                KNOWN FAILURE MODES:
                {chr(10).join(f"- {mode}" for mode in self.failure_modes)}
              """

        # Add enhanced relationship information if available
        if self.relationships:
            doc += "\n\n                OPERATION RELATIONSHIPS:\n"

            # Required operations with reasons
            if self.relationships.required_operations:
                doc += "\n                Required Operations (must be available/complete first):\n"
                for op_id in self.relationships.required_operations:
                    reason = self.relationships.required_reasons.get(
                        op_id, "Dependency required"
                    )
                    doc += f"                - {op_id}: {reason}\n"

            # Commonly paired operations with reasons
            if self.relationships.commonly_paired_with:
                doc += "\n                Commonly Paired With (frequently used together):\n"
                for op_id in self.relationships.commonly_paired_with:
                    reason = self.relationships.pairing_reasons.get(
                        op_id, "Often used in workflows"
                    )
                    doc += f"                - {op_id}: {reason}\n"

            # Mutually exclusive operations with reasons
            if self.relationships.mutually_exclusive_with:
                doc += "\n                Mutually Exclusive With (cannot use simultaneously):\n"
                for op_id in self.relationships.mutually_exclusive_with:
                    reason = self.relationships.exclusion_reasons.get(
                        op_id, "Conflicts with this operation"
                    )
                    doc += f"                - {op_id}: {reason}\n"

            # Parameter flows
            if self.relationships.parameter_flows:
                doc += "\n                Parameter Flows (data connections to other operations):\n"
                for pf in self.relationships.parameter_flows:
                    doc += f"                - Output '{pf.source_output_key}' → {pf.target_operation}.{pf.target_input_param}: {pf.description}\n"

            # Temporal ordering hints
            if self.relationships.typical_before:
                doc += f"\n                Typical Sequence: Usually executed BEFORE {', '.join(self.relationships.typical_before)}\n"

            if self.relationships.typical_after:
                doc += f"                Typical Sequence: Usually executed AFTER {', '.join(self.relationships.typical_after)}\n"

            # Coordination requirements for multi-robot operations
            if self.relationships.coordination_requirements:
                doc += "\n                Multi-Robot Coordination Requirements:\n"
                for key, value in self.relationships.coordination_requirements.items():
                    doc += f"                - {key}: {value}\n"

        return doc

    def to_json(self) -> str:
        """Convert to JSON for structured storage."""
        data: Dict[str, Any] = {
            "operation_id": self.operation_id,
            "name": self.name,
            "category": self.category.value,
            "complexity": self.complexity.value,
            "description": self.description,
            "long_description": self.long_description,
            "usage_examples": self.usage_examples,
            "parameters": [p.to_dict() for p in self.parameters],
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "average_duration_ms": self.average_duration_ms,
            "success_rate": self.success_rate,
            "failure_modes": self.failure_modes,
            "relationships": (
                self.relationships.to_dict() if self.relationships else None
            ),
        }

        return json.dumps(data, indent=2)
