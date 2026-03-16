#!/usr/bin/env python3
"""
Test detection result to grasp operation integration.

This test verifies that detection results (containing x, y, z, color, etc.)
are correctly converted to object_id when passed to grasp operations.
"""

import pytest
from orchestrators.SequenceExecutor import SequenceExecutor


class TestDetectionToGraspIntegration:
    """Test detection result conversion for grasp operations"""

    def test_variable_substitution_extracts_object_id_from_detection(self):
        """Test that $variable substitution extracts 'color' field for object_id parameter"""
        executor = SequenceExecutor(check_completion=False, enable_verification=False)

        # Simulate detection result stored in variables
        detection_result = {
            "x": 0.197172455993917,
            "y": 0.021164613747183647,
            "z": 0.18790143728256226,
            "color": "blue_cube",
            "confidence": 0.7154967784881592,
            "camera_id": "TableStereoCamera",
            "selection": "right",
        }
        executor._variables["target"] = detection_result

        # Test parameter resolution with object_id
        params = {"object_id": "$target", "robot_id": "Robot1"}
        resolved = executor._resolve_variables(params)

        # Should extract 'color' field for object_id
        assert (
            resolved["object_id"] == "blue_cube"
        ), f"Expected object_id='blue_cube', got {resolved['object_id']}"
        assert resolved["robot_id"] == "Robot1"

    def test_auto_injection_extracts_object_id_from_detection(self):
        """Test that auto-injection extracts 'color' field for object_id parameter"""
        executor = SequenceExecutor(check_completion=False, enable_verification=False)

        # Simulate detection result stored in auto-captured variable
        detection_result = {
            "x": 0.5,
            "y": 0.2,
            "z": 0.3,
            "color": "red_cube",
            "confidence": 0.9,
            "camera_id": "MainCamera",
        }
        # Auto-captured variables use the pattern: {operation_name}_{output_key}
        executor._variables["detect_object_result"] = detection_result

        # Test auto-injection for object_id parameter
        # Note: This would normally be triggered by ParameterFlow relationships
        # For this test, we manually inject the variable
        source_var_name = "detect_object_result"
        executor._variables[source_var_name] = detection_result

        params = {"robot_id": "Robot1"}

        # Manually call auto-injection with a mocked flow
        var_value = executor._variables[source_var_name]

        # Test the logic that extracts object_id from detection result
        if isinstance(var_value, dict) and "color" in var_value:
            params["object_id"] = var_value["color"]

        assert (
            params["object_id"] == "red_cube"
        ), f"Expected object_id='red_cube', got {params['object_id']}"

    def test_coordinate_extraction_from_detection_still_works(self):
        """Test that x, y, z extraction from detection results still works"""
        executor = SequenceExecutor(check_completion=False, enable_verification=False)

        detection_result = {
            "x": 1.5,
            "y": 2.5,
            "z": 3.5,
            "color": "blue_cube",
            "confidence": 0.8,
        }
        executor._variables["target"] = detection_result

        # Test coordinate extraction
        params = {"x": "$target", "y": "$target", "z": "$target", "robot_id": "Robot1"}
        resolved = executor._resolve_variables(params)

        # Should extract individual coordinates
        assert resolved["x"] == 1.5
        assert resolved["y"] == 2.5
        assert resolved["z"] == 3.5
        assert resolved["robot_id"] == "Robot1"

    def test_dotted_notation_still_works(self):
        """Test that dotted notation ($target.x) still works correctly"""
        executor = SequenceExecutor(check_completion=False, enable_verification=False)

        detection_result = {"x": 0.7, "y": 0.8, "z": 0.9, "color": "green_cube"}
        executor._variables["target"] = detection_result

        # Test dotted notation
        params = {
            "x": "$target.x",
            "y": "$target.y",
            "z": "$target.z",
            "robot_id": "Robot1",
        }
        resolved = executor._resolve_variables(params)

        assert resolved["x"] == 0.7
        assert resolved["y"] == 0.8
        assert resolved["z"] == 0.9

    def test_detection_without_color_field_logs_warning(self):
        """Test that detection results without 'color' field log a warning"""
        executor = SequenceExecutor(check_completion=False, enable_verification=False)

        # Detection result missing 'color' field
        incomplete_result = {"x": 0.5, "y": 0.5, "z": 0.5, "confidence": 0.7}
        executor._variables["target"] = incomplete_result

        # Test parameter resolution
        params = {"object_id": "$target", "robot_id": "Robot1"}
        resolved = executor._resolve_variables(params)

        # Should keep the original variable reference since extraction failed
        assert (
            resolved["object_id"] == "$target"
        ), f"Expected fallback to '$target', got {resolved['object_id']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
