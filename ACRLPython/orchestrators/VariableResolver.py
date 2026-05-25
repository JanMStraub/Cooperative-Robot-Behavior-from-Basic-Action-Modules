"""Variable resolution and capture for SequenceExecutor."""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class VariableResolver:
    """Resolves $var references and manages variable capture from operation results."""

    def __init__(self, variables: Dict[str, Any], registry=None):
        # Shared dict — SequenceExecutor and VariableResolver both see the same object
        self._variables = variables
        self._registry = registry

    def capture_result_to_var(self, capture_var: str, result: Dict[str, Any]) -> None:
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

    def auto_capture_outputs(self, operation_name: str, result: Dict[str, Any]):
        if not result:
            return
        if self._registry is None:
            return

        # Get operation definition to access relationships
        op_def = self._registry.get_operation_by_name(operation_name)
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

    def auto_inject_parameters(
        self, operation_name: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self._registry is None:
            return params
        # Get operation definition to access relationships
        op_def = self._registry.get_operation_by_name(operation_name)
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
            # Dots in source_output_key are stored as underscores (see auto_capture_outputs)
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

    def resolve_single_value(self, key: str, value: Any) -> Any:
        """Resolve one $var reference (arithmetic, dotted, or simple); returns original string on failure."""
        # Handle expressions with arithmetic (e.g., "$target.z + 0.05")
        if any(op in value for op in ["+", "-", "*", "/"]):
            resolved_value = self.resolve_expression(value)
            if resolved_value is not None:
                return resolved_value
            logger.warning(f"Could not resolve expression: {value}")
            return value

        # Handle dotted notation (e.g., "$target.x") — resolves to a scalar directly
        if "." in value and value.startswith("$"):
            resolved_value = self.resolve_dotted_variable(value)
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
                    # Caller handles the dict → x/y/z expansion in resolve_variables
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

    def resolve_variables(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve all $var references in a params dict, including dotted, arithmetic, and list elements."""
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and "$" in value:
                result = self.resolve_single_value(key, value)
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
                        resolved_list.append(self.resolve_single_value("", element))
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

    def resolve_dotted_variable(self, var_ref: str) -> Optional[Any]:
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

    def resolve_expression(self, expr: str) -> Optional[float]:
        import re

        # Find all variable references (e.g., $target.x, $target.z)
        var_pattern = r"\$[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*"
        variables = re.findall(var_pattern, expr)

        # Replace each variable with its value
        resolved_expr = expr
        for var_ref in variables:
            value = self.resolve_dotted_variable(var_ref)
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
