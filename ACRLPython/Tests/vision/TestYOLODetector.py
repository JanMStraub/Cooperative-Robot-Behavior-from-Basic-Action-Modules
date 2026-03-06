#!/usr/bin/env python3
"""
Unit tests for YOLODetector.py

Tests YOLO-based cube detection
"""

import pytest
import numpy as np
from unittest.mock import patch, Mock, MagicMock

# Import YOLO detector
try:
    from vision.YOLODetector import YOLODetector, YOLO_AVAILABLE
    from vision.DetectionDataModels import DetectionObject, DetectionResult
except ImportError:
    pytest.skip("YOLODetector not available", allow_module_level=True)


class TestYOLODetectorInitialization:
    """Test YOLODetector initialization"""

    @pytest.mark.skipif(not YOLO_AVAILABLE, reason="YOLO not available")
    @patch("vision.YOLODetector.YOLO")
    def test_yolo_detector_init_with_default_model(self, mock_yolo_class):
        """Test YOLODetector initialization with default model"""
        mock_model = Mock()
        mock_model.names = None
        mock_yolo_class.return_value = mock_model

        detector = YOLODetector()

        # Should load default YOLOv8n model
        mock_yolo_class.assert_called_once()
        assert detector.model is not None
        assert detector.conf_threshold == 0.5
        assert detector.iou_threshold == 0.45

    @pytest.mark.skipif(not YOLO_AVAILABLE, reason="YOLO not available")
    @patch("vision.YOLODetector.YOLO")
    def test_yolo_detector_init_with_custom_model(self, mock_yolo_class):
        """Test YOLODetector initialization with custom model path"""
        mock_model = Mock()
        mock_model.names = None
        mock_yolo_class.return_value = mock_model

        custom_model_path = "models/cube_detector.pt"
        detector = YOLODetector(model_path=custom_model_path)

        # Should load custom model with task parameter
        mock_yolo_class.assert_called_once_with(custom_model_path, task="detect")
        assert detector.model_path.name == "cube_detector.pt"

    @pytest.mark.skipif(not YOLO_AVAILABLE, reason="YOLO not available")
    @patch("vision.YOLODetector.YOLO")
    def test_yolo_detector_class_mapping(self, mock_yolo_class):
        """Test custom class mapping"""
        mock_model = Mock()
        mock_model.names = None
        mock_yolo_class.return_value = mock_model

        custom_mapping = {
            0: "red",
            1: "blue",
            2: "green",
        }

        detector = YOLODetector(class_mapping=custom_mapping)

        assert detector.class_mapping == custom_mapping


class TestYOLODetection:
    """Test YOLO detection functionality"""

    # Helper to simulate PyTorch tensor .cpu().numpy() chain
    def _create_mock_tensor(self, data):
        mock_tensor = Mock()
        mock_tensor.cpu.return_value.numpy.return_value = data
        return mock_tensor

    @pytest.mark.skipif(not YOLO_AVAILABLE, reason="YOLO not available")
    @patch("vision.YOLODetector.YOLO")
    def test_detect_objects_empty_image(self, mock_yolo_class):
        """Test detection with empty image"""
        mock_model = Mock()
        mock_model.names = None
        mock_yolo_class.return_value = mock_model

        detector = YOLODetector()

        # Empty image
        empty_image = np.array([])
        result = detector.detect_objects(empty_image, camera_id="test")

        assert isinstance(result, DetectionResult)
        assert result.camera_id == "test"
        assert len(result.detections) == 0

    @pytest.mark.skipif(not YOLO_AVAILABLE, reason="YOLO not available")
    @patch("vision.YOLODetector.YOLO")
    def test_detect_objects_with_detections(self, mock_yolo_class):
        """Test detection with mock YOLO results"""
        # Setup mock YOLO model
        mock_model = Mock()
        mock_model.names = None
        mock_yolo_class.return_value = mock_model

        # Create mock YOLO detection result
        mock_boxes = Mock()
        # Use mock objects that support .cpu().numpy() for all fields
        mock_boxes.xyxy = [self._create_mock_tensor(np.array([100, 200, 150, 260]))]
        mock_boxes.cls = [self._create_mock_tensor(np.array(0))]  # class ID 0 (red)
        mock_boxes.conf = [self._create_mock_tensor(np.array(0.85))]  # confidence
        mock_boxes.__len__ = Mock(return_value=1)

        mock_result = Mock()
        mock_result.boxes = mock_boxes

        mock_model.predict.return_value = [mock_result]

        detector = YOLODetector()

        # Test image (640x480 RGB)
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect_objects(test_image, camera_id="test")

        # Verify result
        assert isinstance(result, DetectionResult)
        assert result.camera_id == "test"
        assert result.image_width == 640
        assert result.image_height == 480
        assert len(result.detections) == 1

        # Verify detection details
        det = result.detections[0]
        assert isinstance(det, DetectionObject)
        assert det.color == "red_cube"
        assert det.bbox_x == 100
        assert det.bbox_y == 200
        assert det.bbox_w == 50  # x2 - x1
        assert det.bbox_h == 60  # y2 - y1
        assert det.confidence == 0.85

    @pytest.mark.skipif(not YOLO_AVAILABLE, reason="YOLO not available")
    @patch("vision.YOLODetector.YOLO")
    def test_detect_objects_filters_by_area(self, mock_yolo_class):
        """Test that detections are filtered by area"""
        # Setup mock YOLO model
        mock_model = Mock()
        mock_model.names = None
        mock_yolo_class.return_value = mock_model

        # Create mock YOLO detection with very small area (should be filtered)
        mock_boxes = Mock()
        # Use mock objects that support .cpu().numpy() for all fields
        mock_boxes.xyxy = [
            self._create_mock_tensor(np.array([100, 200, 102, 202]))  # 2x2 pixels
        ]
        mock_boxes.cls = [self._create_mock_tensor(np.array(0))]
        mock_boxes.conf = [self._create_mock_tensor(np.array(0.85))]
        mock_boxes.__len__ = Mock(return_value=1)

        mock_result = Mock()
        mock_result.boxes = mock_boxes

        mock_model.predict.return_value = [mock_result]

        detector = YOLODetector()

        # Test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect_objects(test_image, camera_id="test")

        # Should filter out small detection
        assert len(result.detections) == 0

    @pytest.mark.skipif(not YOLO_AVAILABLE, reason="YOLO not available")
    @patch("vision.YOLODetector.YOLO")
    def test_detect_objects_multiple_detections(self, mock_yolo_class):
        """Test detection with multiple cubes"""
        # Setup mock YOLO model
        mock_model = Mock()
        mock_model.names = None
        mock_yolo_class.return_value = mock_model

        # Create mock YOLO detections (red and blue cubes)
        mock_boxes = Mock()
        # Use mock objects that support .cpu().numpy() for all fields
        mock_boxes.xyxy = [
            self._create_mock_tensor(np.array([100, 200, 150, 260])),  # Red cube
            self._create_mock_tensor(np.array([300, 150, 360, 220])),  # Blue cube
        ]
        mock_boxes.cls = [
            self._create_mock_tensor(np.array(0)),  # class ID 0 (red_cube)
            self._create_mock_tensor(np.array(1))   # class ID 1 (blue_cube)
        ]
        mock_boxes.conf = [
            self._create_mock_tensor(np.array(0.85)),  # confidence
            self._create_mock_tensor(np.array(0.92))   # confidence
        ]
        mock_boxes.__len__ = Mock(return_value=2)

        mock_result = Mock()
        mock_result.boxes = mock_boxes

        mock_model.predict.return_value = [mock_result]

        detector = YOLODetector()

        # Test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect_objects(test_image, camera_id="test")

        # Verify results
        assert len(result.detections) == 2

        # Check first detection (red_cube)
        det1 = result.detections[0]
        assert det1.color == "red_cube"
        assert det1.confidence == 0.85

        # Check second detection (blue_cube)
        det2 = result.detections[1]
        assert det2.color == "blue_cube"
        assert det2.confidence == 0.92


class TestYOLOIntegration:
    """Test YOLO integration with ObjectDetector"""

    @pytest.mark.skipif(not YOLO_AVAILABLE, reason="YOLO not available")
    @patch("vision.ObjectDetector.YOLO_AVAILABLE", True)
    @patch("vision.ObjectDetector.YOLODetector")
    @patch("vision.ObjectDetector.USE_YOLO", True)
    def test_cube_detector_uses_yolo_when_enabled(self, mock_yolo_detector_class):
        """Test that CubeDetector uses YOLO when enabled in config"""
        from vision.ObjectDetector import CubeDetector

        # Setup mock
        mock_yolo_instance = Mock()
        mock_yolo_detector_class.return_value = mock_yolo_instance

        # Create CubeDetector
        detector = CubeDetector()

        # Verify YOLO was initialized
        assert detector.use_yolo is True
        mock_yolo_detector_class.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])




class TestYOLODetectorSegmentation:
    """Test mask population when YOLO task=segment."""

    @pytest.mark.skipif(not YOLO_AVAILABLE, reason="YOLO not available")
    @patch("vision.YOLODetector.YOLO")
    def test_mask_populated_when_segment_result(self, mock_yolo_class):
        """DetectionObject.mask is populated when results[0].masks is present."""
        import numpy as np

        mock_model = MagicMock()
        mock_model.names = {0: "red_cube"}
        mock_yolo_class.return_value = mock_model

        fake_mask = np.ones((480, 640), dtype=np.uint8)

        mock_box = MagicMock()
        mock_box.__len__ = MagicMock(return_value=1)
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].cpu.return_value.numpy.return_value = [10, 10, 100, 100]
        mock_box.cls = [MagicMock()]
        mock_box.cls[0].cpu.return_value.numpy.return_value = 0
        mock_box.conf = [MagicMock()]
        mock_box.conf[0].cpu.return_value.numpy.return_value = 0.9

        mock_masks = MagicMock()
        mock_masks.data = MagicMock()
        mock_masks.data.cpu.return_value.numpy.return_value = fake_mask[np.newaxis, :]

        mock_result = MagicMock()
        mock_result.boxes = mock_box
        mock_result.masks = mock_masks
        mock_model.predict.return_value = [mock_result]

        detector = YOLODetector(model_path="fake.pt", task="segment")
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_objects(image, camera_id="test")

        assert len(result.detections) == 1
        det = result.detections[0]
        assert det.mask is not None
        assert det.mask.shape == (480, 640)

    @pytest.mark.skipif(not YOLO_AVAILABLE, reason="YOLO not available")
    @patch("vision.YOLODetector.YOLO")
    def test_mask_is_none_when_no_masks(self, mock_yolo_class):
        """DetectionObject.mask is None when results[0].masks is None (detect mode)."""
        import numpy as np

        mock_model = MagicMock()
        mock_model.names = {0: "red_cube"}
        mock_yolo_class.return_value = mock_model

        mock_box = MagicMock()
        mock_box.__len__ = MagicMock(return_value=1)
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].cpu.return_value.numpy.return_value = [10, 10, 100, 100]
        mock_box.cls = [MagicMock()]
        mock_box.cls[0].cpu.return_value.numpy.return_value = 0
        mock_box.conf = [MagicMock()]
        mock_box.conf[0].cpu.return_value.numpy.return_value = 0.9

        mock_result = MagicMock()
        mock_result.boxes = mock_box
        mock_result.masks = None

        mock_model.predict.return_value = [mock_result]

        detector = YOLODetector(model_path="fake.pt", task="detect")
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_objects(image, camera_id="test")

        assert len(result.detections) == 1
        assert result.detections[0].mask is None
