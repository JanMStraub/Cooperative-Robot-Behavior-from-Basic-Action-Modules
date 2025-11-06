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
    COORDINATION = "coordination"  # Multi-robot coordination


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
    """

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    valid_range: Optional[tuple] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "required": self.required,
            "default": self.default,
            "valid_range": self.valid_range,
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

    # Relationships to other operations
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
        """
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

                RELATED OPERATIONS:
                - Required operations: {', '.join(self.required_operations) if self.required_operations else 'None'}
                - Commonly paired with: {', '.join(self.commonly_paired_with) if self.commonly_paired_with else 'None'}
                - Mutually exclusive with: {', '.join(self.mutually_exclusive_with) if self.mutually_exclusive_with else 'None'}
              """
        return doc

    def to_json(self) -> str:
        """Convert to JSON for structured storage"""
        return json.dumps(
            {
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
                "required_operations": self.required_operations,
                "commonly_paired_with": self.commonly_paired_with,
                "mutually_exclusive_with": self.mutually_exclusive_with,
            },
            indent=2,
        )
