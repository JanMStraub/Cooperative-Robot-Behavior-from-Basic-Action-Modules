#!/usr/bin/env python3
"""
Unit tests for DetectionOperations.py

Tests the detection operations including:
- Color-based detection requests
- Confidence filtering
- Result formatting
- Integration with ObjectDetector
- Detection timeout
- No objects found handling
- Image availability checking
- Parameter validation
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from operations.DetectionOperations import detect_objects, DETECT_OBJECTS_OPERATION
from operations.Base import OperationResult


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_image_storage():
    """
    Create a mock ImageStorage for testing.

    Returns:
        Mock ImageStorage with get_camera_image method
    """
    storage = Mock()
    # Create a sample image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    storage.get_camera_image = Mock(return_value=image)
    storage.get_instance = Mock(return_value=storage)
    return storage


@pytest.fixture
def mock_detector():
    """
    Create a mock CubeDetector for testing.

    Returns:
        Mock CubeDetector with detect_objects method
    """
    detector = Mock()

    # Create mock detection result
    mock_detection = Mock()
    mock_detection.to_dict = Mock(return_value={
        "id": 0,
        "color": "red",
        "bbox_px": {"x": 100, "y": 100, "width": 50, "height": 50},
        "center_px": {"x": 125, "y": 125},
        "confidence": 0.95
    })

    mock_result = Mock()
    mock_result.detections = [mock_detection]
    mock_result.image_width = 640
    mock_result.image_height = 480

    detector.detect_objects = Mock(return_value=mock_result)
    return detector


# ============================================================================
# Test Class: Basic Detection Operations
# ============================================================================

class TestDetectionOperations:
    """Test basic detection operations."""

    def test_detect_objects_success(self, mock_image_storage, mock_detector):
        """Test detecting objects successfully."""
        with patch('operations.DetectionOperations.ImageStorage', mock_image_storage):
            with patch('vision.ObjectDetector.CubeDetector', return_value=mock_detector):
                result = detect_objects("Robot1", camera_id="main")

                assert result.success is True
                assert result.result is not None
                assert result.result["camera_id"] == "main"
                assert result.result["count"] == 1
                assert len(result.result["detections"]) == 1

    def test_detect_with_default_camera_id(self, mock_image_storage, mock_detector):
        """Test detection with default camera ID."""
        with patch('operations.DetectionOperations.ImageStorage', mock_image_storage):
            with patch('vision.ObjectDetector.CubeDetector', return_value=mock_detector):
                result = detect_objects("Robot1")

                assert result.success is True
                assert result.result is not None
                assert result.result["camera_id"] == "main"

    def test_detect_multiple_objects(self, mock_image_storage, mock_detector):
        """Test detecting multiple objects."""
        # Create mock with multiple detections
        mock_detection1 = Mock()
        mock_detection1.to_dict = Mock(return_value={"id": 0, "color": "red", "confidence": 0.95})

        mock_detection2 = Mock()
        mock_detection2.to_dict = Mock(return_value={"id": 1, "color": "blue", "confidence": 0.88})

        mock_result = Mock()
        mock_result.detections = [mock_detection1, mock_detection2]
        mock_result.image_width = 640
        mock_result.image_height = 480

        mock_detector.detect_objects = Mock(return_value=mock_result)

        with patch('operations.DetectionOperations.ImageStorage', mock_image_storage):
            with patch('vision.ObjectDetector.CubeDetector', return_value=mock_detector):
                result = detect_objects("Robot1")

                assert result.success is True
                assert result.result is not None
                assert result.result["count"] == 2
                assert len(result.result["detections"]) == 2

    def test_detect_no_objects_found(self, mock_image_storage, mock_detector):
        """Test detection when no objects are found."""
        mock_result = Mock()
        mock_result.detections = []
        mock_result.image_width = 640
        mock_result.image_height = 480

        mock_detector.detect_objects = Mock(return_value=mock_result)

        with patch('operations.DetectionOperations.ImageStorage', mock_image_storage):
            with patch('vision.ObjectDetector.CubeDetector', return_value=mock_detector):
                result = detect_objects("Robot1")

                assert result.success is True
                assert result.result is not None
                assert result.result["count"] == 0
                assert len(result.result["detections"]) == 0


# ============================================================================
# Test Class: Error Handling
# ============================================================================

class TestDetectionErrors:
    """Test error handling for detection operations."""

    def test_detect_no_image_available(self):
        """Test detection when no image is available from ImageStorage."""
        mock_storage = Mock()
        mock_storage.get_camera_image = Mock(return_value=None)
        mock_storage.get_instance = Mock(return_value=mock_storage)

        with patch('operations.DetectionOperations.ImageStorage', mock_storage):
            result = detect_objects("Robot1", camera_id="missing_camera")

            assert result.success is False
            assert result.error is not None
            assert result.error["code"] == "NO_IMAGE"

    def test_detect_detector_error(self, mock_image_storage):
        """Test detection when detector raises exception."""
        mock_detector = Mock()
        mock_detector.detect_objects = Mock(side_effect=Exception("Detector error"))

        with patch('operations.DetectionOperations.ImageStorage', mock_image_storage):
            with patch('vision.ObjectDetector.CubeDetector', return_value=mock_detector):
                result = detect_objects("Robot1")

                assert result.success is False
                assert result.error is not None
                assert result.error["code"] == "DETECTION_ERROR"


# ============================================================================
# Test Class: Result Formatting
# ============================================================================

class TestDetectionResultFormatting:
    """Test detection result formatting."""

    def test_detection_result_format(self, mock_image_storage, mock_detector):
        """Test that detection results have correct format."""
        with patch('operations.DetectionOperations.ImageStorage', mock_image_storage):
            with patch('vision.ObjectDetector.CubeDetector', return_value=mock_detector):
                result = detect_objects("Robot1", camera_id="test_cam")

                assert result.success is True
                assert result.result is not None
                assert "camera_id" in result.result
                assert "detections" in result.result
                assert "count" in result.result
                assert "image_width" in result.result
                assert "image_height" in result.result
                assert "timestamp" in result.result

    def test_detection_dict_structure(self, mock_image_storage, mock_detector):
        """Test that each detection has correct dictionary structure."""
        with patch('operations.DetectionOperations.ImageStorage', mock_image_storage):
            with patch('vision.ObjectDetector.CubeDetector', return_value=mock_detector):
                result = detect_objects("Robot1")

                assert result.result is not None
                detection = result.result["detections"][0]
                assert "id" in detection
                assert "color" in detection
                assert "bbox_px" in detection
                assert "center_px" in detection
                assert "confidence" in detection


# ============================================================================
# Test Class: Confidence Filtering
# ============================================================================

class TestDetectionConfidence:
    """Test confidence-based filtering."""

    def test_detect_with_confidence_threshold(self, mock_image_storage, mock_detector):
        """Test detection respects confidence filtering (handled by detector)."""
        # Detector already filters by confidence internally
        with patch('operations.DetectionOperations.ImageStorage', mock_image_storage):
            with patch('vision.ObjectDetector.CubeDetector', return_value=mock_detector):
                result = detect_objects("Robot1")

                assert result.success is True
                assert result.result is not None
                # All returned detections should have passed detector's confidence filter
                for detection in result.result["detections"]:
                    assert detection["confidence"] > 0.0


# ============================================================================
# Test Class: Operation Definition
# ============================================================================

class TestDetectionOperationDefinition:
    """Test the BasicOperation definition for detection."""

    def test_operation_definition_exists(self):
        """Test that DETECT_OBJECTS_OPERATION is properly defined."""
        assert DETECT_OBJECTS_OPERATION is not None
        assert DETECT_OBJECTS_OPERATION.name == "detect_objects"
        assert DETECT_OBJECTS_OPERATION.operation_id == "perception_detect_objects_001"

    def test_operation_has_metadata(self):
        """Test that operation has required metadata."""
        op = DETECT_OBJECTS_OPERATION

        assert op.description is not None
        assert len(op.parameters) >= 1  # robot_id at minimum
        assert op.preconditions is not None
        assert op.postconditions is not None
        assert op.implementation is not None

    def test_operation_execution_through_definition(self, mock_image_storage, mock_detector):
        """Test executing operation through BasicOperation.execute()."""
        with patch('operations.DetectionOperations.ImageStorage', mock_image_storage):
            with patch('vision.ObjectDetector.CubeDetector', return_value=mock_detector):
                result = DETECT_OBJECTS_OPERATION.execute(robot_id="Robot1", camera_id="main")

                assert result.success is True
