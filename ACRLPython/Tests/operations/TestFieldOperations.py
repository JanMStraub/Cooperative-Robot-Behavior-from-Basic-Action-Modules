#!/usr/bin/env python3
"""
Unit tests for FieldOperations.py

Tests the field detection operations including:
- detect_field: Labeled field detection (A-I)
- get_field_center: Field coordinate extraction
- detect_all_fields: Multi-field detection
- YOLO integration and confidence scoring
- Field boundary validation
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from operations.FieldOperations import (
    detect_field,
    get_field_center,
    detect_all_fields,
    DETECT_FIELD_OPERATION,
    GET_FIELD_CENTER_OPERATION,
    DETECT_ALL_FIELDS_OPERATION,
)
from operations.Base import OperationResult


# ============================================================================
# Helper Mocks
# ============================================================================


@pytest.fixture
def mock_yolo_detector():
    """Create a mock YOLO detector."""
    detector = Mock()
    return detector


@pytest.fixture
def mock_detection_result():
    """Create a mock detection result from YOLO."""
    detection = Mock()
    detection.class_name = "fielda"
    detection.color = "fielda"
    detection.bbox = {"x": 100, "y": 100, "width": 50, "height": 50}
    detection.confidence = 0.85
    detection.world_position = (0.3, 0.0, 0.1)  # Tuple format

    results = Mock()
    results.detections = [detection]
    return results


@pytest.fixture
def mock_image_storage_with_stereo():
    """Create a mock image storage with stereo images."""
    storage = Mock()

    # Create sample stereo images
    left_image = np.zeros((480, 640, 3), dtype=np.uint8)
    right_image = np.zeros((480, 640, 3), dtype=np.uint8)

    storage.get_latest_stereo_image = Mock(
        return_value=(left_image, right_image, "", 0.0, {"camera_params": {}})
    )
    return storage


# ============================================================================
# Test Class: detect_field - Field Detection
# ============================================================================


class TestDetectField:
    """Test field detection operation."""

    def test_detect_field_success(
        self, mock_image_storage_with_stereo, mock_detection_result, monkeypatch
    ):
        """Test successful field detection."""
        # Patch UnifiedImageStorage
        monkeypatch.setattr(
            "operations.FieldOperations.get_unified_image_storage",
            lambda: mock_image_storage_with_stereo,
        )

        # Mock YOLODetector
        with patch("vision.YOLODetector.YOLODetector") as mock_yolo_class:
            mock_detector = Mock()
            mock_detector.detect_objects_stereo = Mock(return_value=mock_detection_result)
            mock_yolo_class.return_value = mock_detector

            result = detect_field("Robot1", "A")

            assert result.success is True
            assert result.result is not None
            assert result.result["field_label"] == "A"
            assert "center" in result.result
            assert result.result["center"] == {"x": 0.3, "y": 0.0, "z": 0.1}
            assert result.result["confidence"] == 0.85
            assert "bounds" in result.result

    def test_detect_field_lowercase_input(
        self, mock_image_storage_with_stereo, mock_detection_result, monkeypatch
    ):
        """Test field detection with lowercase input."""
        monkeypatch.setattr(
            "operations.FieldOperations.get_unified_image_storage",
            lambda: mock_image_storage_with_stereo,
        )

        with patch("vision.YOLODetector.YOLODetector") as mock_yolo_class:
            mock_detector = Mock()
            mock_detector.detect_objects_stereo = Mock(return_value=mock_detection_result)
            mock_yolo_class.return_value = mock_detector

            result = detect_field("Robot1", "a")  # Lowercase

            assert result.success is True
            assert result.result is not None
            assert result.result["field_label"] == "A"  # Should be uppercase in result

    def test_detect_field_all_labels(
        self, mock_image_storage_with_stereo, monkeypatch
    ):
        """Test detection of all field labels A-I."""
        monkeypatch.setattr(
            "operations.FieldOperations.get_unified_image_storage",
            lambda: mock_image_storage_with_stereo,
        )

        for letter in "ABCDEFGHI":
            # Create detection result for this letter
            detection = Mock()
            detection.class_name = f"field{letter.lower()}"
            detection.color = f"field{letter.lower()}"
            detection.bbox = {"x": 100, "y": 100, "width": 50, "height": 50}
            detection.confidence = 0.8
            detection.world_position = (0.2, 0.0, 0.1)

            results = Mock()
            results.detections = [detection]

            with patch("vision.YOLODetector.YOLODetector") as mock_yolo_class:
                mock_detector = Mock()
                mock_detector.detect_objects_stereo = Mock(return_value=results)
                mock_yolo_class.return_value = mock_detector

                result = detect_field("Robot1", letter)

                assert result.success is True
                assert result.result is not None
                assert result.result["field_label"] == letter

    def test_detect_field_invalid_robot_id(self):
        """Test field detection with invalid robot ID."""
        result = detect_field("", "A")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_ROBOT_ID"

    def test_detect_field_invalid_label_too_long(self):
        """Test field detection with invalid label (too long)."""
        result = detect_field("Robot1", "AB")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_FIELD_LABEL"

    def test_detect_field_invalid_label_not_letter(self):
        """Test field detection with invalid label (not a letter)."""
        result = detect_field("Robot1", "1")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_FIELD_LABEL"

    def test_detect_field_no_stereo_images(self, monkeypatch):
        """Test field detection when stereo images unavailable."""
        storage = Mock()
        storage.get_latest_stereo_image = Mock(return_value=None)

        monkeypatch.setattr(
            "operations.FieldOperations.get_unified_image_storage", lambda: storage
        )

        result = detect_field("Robot1", "A")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "NO_STEREO_IMAGES"

    def test_detect_field_incomplete_stereo_pair(self, monkeypatch):
        """Test field detection when stereo pair is incomplete."""
        storage = Mock()
        storage.get_latest_stereo_image = Mock(
            return_value=(np.zeros((480, 640, 3), dtype=np.uint8), None, "", 0.0, {})
        )

        monkeypatch.setattr(
            "operations.FieldOperations.get_unified_image_storage", lambda: storage
        )

        result = detect_field("Robot1", "A")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INCOMPLETE_STEREO_PAIR"

    def test_detect_field_not_detected(
        self, mock_image_storage_with_stereo, monkeypatch
    ):
        """Test field detection when field not found."""
        monkeypatch.setattr(
            "operations.FieldOperations.get_unified_image_storage",
            lambda: mock_image_storage_with_stereo,
        )

        # Empty detection results
        empty_results = Mock()
        empty_results.detections = []

        with patch("vision.YOLODetector.YOLODetector") as mock_yolo_class:
            mock_detector = Mock()
            mock_detector.detect_objects_stereo = Mock(return_value=empty_results)
            mock_yolo_class.return_value = mock_detector

            result = detect_field("Robot1", "D")

            assert result.success is False
            assert result.error is not None
            assert result.error["code"] == "FIELD_NOT_DETECTED"

    def test_detect_field_with_confidence_threshold(
        self, mock_image_storage_with_stereo, mock_detection_result, monkeypatch
    ):
        """Test field detection with custom confidence threshold."""
        monkeypatch.setattr(
            "operations.FieldOperations.get_unified_image_storage",
            lambda: mock_image_storage_with_stereo,
        )

        with patch("vision.YOLODetector.YOLODetector") as mock_yolo_class:
            mock_detector = Mock()
            mock_detector.detect_objects_stereo = Mock(return_value=mock_detection_result)
            mock_yolo_class.return_value = mock_detector

            result = detect_field("Robot1", "A", confidence_threshold=0.7)

            assert result.success is True
            # Verify YOLO was called with correct confidence threshold
            assert mock_detector.conf_threshold == 0.7

    def test_detect_field_no_3d_coordinates(
        self, mock_image_storage_with_stereo, monkeypatch
    ):
        """Test field detection when 3D coordinates missing."""
        monkeypatch.setattr(
            "operations.FieldOperations.get_unified_image_storage",
            lambda: mock_image_storage_with_stereo,
        )

        # Detection without world position
        detection = Mock()
        detection.class_name = "fielda"
        detection.color = "fielda"
        detection.bbox = {"x": 100, "y": 100, "width": 50, "height": 50}
        detection.confidence = 0.85
        detection.world_position = None  # No 3D position

        results = Mock()
        results.detections = [detection]

        with patch("vision.YOLODetector.YOLODetector") as mock_yolo_class:
            mock_detector = Mock()
            mock_detector.detect_objects_stereo = Mock(return_value=results)
            mock_yolo_class.return_value = mock_detector

            result = detect_field("Robot1", "A")

            assert result.success is False
            assert result.error is not None
            assert result.error["code"] == "NO_3D_COORDINATES"

    def test_detect_field_world_position_as_dict(
        self, mock_image_storage_with_stereo, monkeypatch
    ):
        """Test field detection when world_position is dict instead of tuple."""
        monkeypatch.setattr(
            "operations.FieldOperations.get_unified_image_storage",
            lambda: mock_image_storage_with_stereo,
        )

        # Detection with world position as dict
        detection = Mock()
        detection.class_name = "fieldb"
        detection.color = "fieldb"
        detection.bbox = {"x": 100, "y": 100, "width": 50, "height": 50}
        detection.confidence = 0.85
        detection.world_position = {"x": 0.4, "y": 0.1, "z": 0.0}  # Dict format

        results = Mock()
        results.detections = [detection]

        with patch("vision.YOLODetector.YOLODetector") as mock_yolo_class:
            mock_detector = Mock()
            mock_detector.detect_objects_stereo = Mock(return_value=results)
            mock_yolo_class.return_value = mock_detector

            result = detect_field("Robot1", "B")

            assert result.success is True
            assert result.result is not None
            assert result.result["center"] == {"x": 0.4, "y": 0.1, "z": 0.0}


# ============================================================================
# Test Class: get_field_center - Field Center Extraction
# ============================================================================


class TestGetFieldCenter:
    """Test field center coordinate extraction operation."""

    def test_get_field_center_success(
        self, mock_image_storage_with_stereo, mock_detection_result, monkeypatch
    ):
        """Test successful field center extraction."""
        monkeypatch.setattr(
            "operations.FieldOperations.get_unified_image_storage",
            lambda: mock_image_storage_with_stereo,
        )

        with patch("vision.YOLODetector.YOLODetector") as mock_yolo_class:
            mock_detector = Mock()
            mock_detector.detect_objects_stereo = Mock(return_value=mock_detection_result)
            mock_yolo_class.return_value = mock_detector

            result = get_field_center("Robot1", "E")

            assert result.success is True
            assert result.result is not None
            assert result.result["field_label"] == "E"
            assert "center" in result.result
            assert result.result["center"]["x"] == 0.3
            assert result.result["center"]["y"] == 0.0
            assert result.result["center"]["z"] == 0.1

    def test_get_field_center_forwards_error(self, monkeypatch):
        """Test that get_field_center forwards errors from detect_field."""
        storage = Mock()
        storage.get_latest_stereo_image = Mock(return_value=None)

        monkeypatch.setattr(
            "operations.FieldOperations.get_unified_image_storage", lambda: storage
        )

        result = get_field_center("Robot1", "C")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "NO_STEREO_IMAGES"


# ============================================================================
# Test Class: detect_all_fields - Multi-Field Detection
# ============================================================================


class TestDetectAllFields:
    """Test detection of all visible fields."""

    def test_detect_all_fields_success(
        self, mock_image_storage_with_stereo, monkeypatch
    ):
        """Test successful detection of multiple fields."""
        monkeypatch.setattr(
            "operations.FieldOperations.get_unified_image_storage",
            lambda: mock_image_storage_with_stereo,
        )

        # Create multiple field detections
        detection_a = Mock()
        detection_a.class_name = "fielda"
        detection_a.color = "fielda"
        detection_a.bbox = {"x": 100, "y": 100, "width": 50, "height": 50}
        detection_a.confidence = 0.85
        detection_a.world_position = (0.2, 0.0, 0.1)

        detection_d = Mock()
        detection_d.class_name = "fieldd"
        detection_d.color = "fieldd"
        detection_d.bbox = {"x": 200, "y": 100, "width": 50, "height": 50}
        detection_d.confidence = 0.90
        detection_d.world_position = (0.3, 0.1, 0.1)

        results = Mock()
        results.detections = [detection_a, detection_d]

        with patch("vision.YOLODetector.YOLODetector") as mock_yolo_class:
            mock_detector = Mock()
            mock_detector.detect_objects_stereo = Mock(return_value=results)
            mock_yolo_class.return_value = mock_detector

            result = detect_all_fields("Robot1")

            assert result.success is True
            assert result.result is not None
            assert result.result["count"] == 2
            assert len(result.result["fields"]) == 2

            # Check field labels
            labels = [f["label"] for f in result.result["fields"]]
            assert "A" in labels
            assert "D" in labels

    def test_detect_all_fields_no_fields_detected(
        self, mock_image_storage_with_stereo, monkeypatch
    ):
        """Test detection when no fields are visible."""
        monkeypatch.setattr(
            "operations.FieldOperations.get_unified_image_storage",
            lambda: mock_image_storage_with_stereo,
        )

        # Empty detection results
        empty_results = Mock()
        empty_results.detections = []

        with patch("vision.YOLODetector.YOLODetector") as mock_yolo_class:
            mock_detector = Mock()
            mock_detector.detect_objects_stereo = Mock(return_value=empty_results)
            mock_yolo_class.return_value = mock_detector

            result = detect_all_fields("Robot1")

            assert result.success is True
            assert result.result is not None
            assert result.result["count"] == 0
            assert result.result["fields"] == []

    def test_detect_all_fields_invalid_robot_id(self):
        """Test all fields detection with invalid robot ID."""
        result = detect_all_fields("")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_ROBOT_ID"

    def test_detect_all_fields_filters_correct_classes(
        self, mock_image_storage_with_stereo, monkeypatch
    ):
        """Test that detect_all_fields filters for field classes."""
        monkeypatch.setattr(
            "operations.FieldOperations.get_unified_image_storage",
            lambda: mock_image_storage_with_stereo,
        )

        with patch("vision.YOLODetector.YOLODetector") as mock_yolo_class:
            mock_detector = Mock()
            mock_detector.detect_objects_stereo = Mock(return_value=Mock(detections=[]))
            mock_yolo_class.return_value = mock_detector

            result = detect_all_fields("Robot1")

            # Verify YOLO was called with field class filter
            call_kwargs = mock_detector.detect_objects_stereo.call_args[1]
            filter_classes = call_kwargs["filter_classes"]

            # Should filter for fielda through fieldi
            expected_classes = [f"field{chr(ord('a') + i)}" for i in range(9)]
            assert filter_classes == expected_classes

    def test_detect_all_fields_with_confidence_threshold(
        self, mock_image_storage_with_stereo, monkeypatch
    ):
        """Test all fields detection with custom confidence threshold."""
        monkeypatch.setattr(
            "operations.FieldOperations.get_unified_image_storage",
            lambda: mock_image_storage_with_stereo,
        )

        with patch("vision.YOLODetector.YOLODetector") as mock_yolo_class:
            mock_detector = Mock()
            mock_detector.detect_objects_stereo = Mock(return_value=Mock(detections=[]))
            mock_yolo_class.return_value = mock_detector

            result = detect_all_fields("Robot1", confidence_threshold=0.8)

            # Verify YOLO was called with correct confidence threshold
            assert mock_detector.conf_threshold == 0.8


# ============================================================================
# Test Class: Operation Definitions
# ============================================================================


class TestFieldOperationDefinitions:
    """Test BasicOperation definitions for field operations."""

    def test_detect_field_operation_definition(self):
        """Test DETECT_FIELD_OPERATION is properly defined."""
        assert DETECT_FIELD_OPERATION is not None
        assert DETECT_FIELD_OPERATION.name == "detect_field"
        assert DETECT_FIELD_OPERATION.operation_id == "perception_detect_field_004"

    def test_get_field_center_operation_definition(self):
        """Test GET_FIELD_CENTER_OPERATION is properly defined."""
        assert GET_FIELD_CENTER_OPERATION is not None
        assert GET_FIELD_CENTER_OPERATION.name == "get_field_center"
        assert GET_FIELD_CENTER_OPERATION.operation_id == "perception_get_field_center_005"

    def test_detect_all_fields_operation_definition(self):
        """Test DETECT_ALL_FIELDS_OPERATION is properly defined."""
        assert DETECT_ALL_FIELDS_OPERATION is not None
        assert DETECT_ALL_FIELDS_OPERATION.name == "detect_all_fields"
        assert DETECT_ALL_FIELDS_OPERATION.operation_id == "perception_detect_all_fields_006"

    def test_all_operations_have_metadata(self):
        """Test all operations have required metadata."""
        operations = [
            DETECT_FIELD_OPERATION,
            GET_FIELD_CENTER_OPERATION,
            DETECT_ALL_FIELDS_OPERATION,
        ]

        for op in operations:
            assert op.description is not None
            assert op.parameters is not None
            assert len(op.parameters) >= 1
            assert op.implementation is not None
            assert op.preconditions is not None
            assert op.postconditions is not None


# ============================================================================
# Test Class: Edge Cases
# ============================================================================


class TestFieldOperationsEdgeCases:
    """Test edge cases for field operations."""

    def test_detect_field_with_whitespace_in_label(
        self, mock_image_storage_with_stereo, mock_detection_result, monkeypatch
    ):
        """Test field detection with whitespace in label."""
        monkeypatch.setattr(
            "operations.FieldOperations.get_unified_image_storage",
            lambda: mock_image_storage_with_stereo,
        )

        with patch("vision.YOLODetector.YOLODetector") as mock_yolo_class:
            mock_detector = Mock()
            mock_detector.detect_objects_stereo = Mock(return_value=mock_detection_result)
            mock_yolo_class.return_value = mock_detector

            # Whitespace should be stripped
            result = detect_field("Robot1", " A ")

            assert result.success is True
            assert result.result is not None
            assert result.result["field_label"] == "A"

    def test_detect_all_fields_all_nine_fields(
        self, mock_image_storage_with_stereo, monkeypatch
    ):
        """Test detection of all 9 fields simultaneously."""
        monkeypatch.setattr(
            "operations.FieldOperations.get_unified_image_storage",
            lambda: mock_image_storage_with_stereo,
        )

        # Create 9 field detections
        detections = []
        for i, letter in enumerate("ABCDEFGHI"):
            detection = Mock()
            detection.class_name = f"field{letter.lower()}"
            detection.color = f"field{letter.lower()}"
            detection.bbox = {"x": i * 50, "y": 100, "width": 50, "height": 50}
            detection.confidence = 0.85
            detection.world_position = (0.1 + i * 0.05, 0.0, 0.1)
            detections.append(detection)

        results = Mock()
        results.detections = detections

        with patch("vision.YOLODetector.YOLODetector") as mock_yolo_class:
            mock_detector = Mock()
            mock_detector.detect_objects_stereo = Mock(return_value=results)
            mock_yolo_class.return_value = mock_detector

            result = detect_all_fields("Robot1")

            assert result.success is True
            assert result.result is not None
            assert result.result["count"] == 9
            labels = [f["label"] for f in result.result["fields"]]
            assert len(set(labels)) == 9  # All unique

    def test_detect_field_case_insensitive(
        self, mock_image_storage_with_stereo, mock_detection_result, monkeypatch
    ):
        """Test field detection is case-insensitive."""
        monkeypatch.setattr(
            "operations.FieldOperations.get_unified_image_storage",
            lambda: mock_image_storage_with_stereo,
        )

        with patch("vision.YOLODetector.YOLODetector") as mock_yolo_class:
            mock_detector = Mock()
            mock_detector.detect_objects_stereo = Mock(return_value=mock_detection_result)
            mock_yolo_class.return_value = mock_detector

            # Both should work
            result_upper = detect_field("Robot1", "A")
            result_lower = detect_field("Robot1", "a")

            assert result_upper.success is True
            assert result_lower.success is True
            assert result_upper.result is not None
            assert result_lower.result is not None
            assert result_upper.result["field_label"] == "A"
            assert result_lower.result["field_label"] == "A"
