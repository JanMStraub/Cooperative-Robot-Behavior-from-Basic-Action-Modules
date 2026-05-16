#!/usr/bin/env python3
"""Unit tests for FieldOperations"""

from unittest.mock import MagicMock, patch

import pytest

from operations.FieldOperations import detect_field, detect_all_fields

# Shared helpers


def _make_detection(color: str, world_position=(0.1, 0.2, 0.3), confidence=0.85):
    """Create a mock YOLO DetectionObject."""
    d = MagicMock()
    d.color = color
    d.world_position = world_position
    d.confidence = confidence
    d.bbox_x = 10
    d.bbox_y = 20
    d.bbox_w = 50
    d.bbox_h = 50
    return d


def _make_detections_result(detections):
    r = MagicMock()
    r.detections = detections
    return r


def _stereo_data(left=MagicMock(), right=MagicMock(), metadata=None):
    return (left, right, None, None, metadata or {})


def _stereo_params(metadata=None):
    params = MagicMock()
    params.camera_config = MagicMock()
    params.camera_position = (0, 0, 0)
    params.camera_rotation = (0, 0, 0, 1)
    return params


# detect_field


class TestDetectField:
    def _patch_all(self, detections_result, stereo=None):
        """Context manager that patches all external deps for detect_field."""
        image_storage = MagicMock()
        image_storage.get_latest_stereo_image.return_value = (
            stereo if stereo is not None else _stereo_data()
        )

        detector = MagicMock()
        detector.detect_objects_stereo.return_value = detections_result

        world_state = MagicMock()

        patches = [
            patch("core.Imports.get_unified_image_storage", return_value=image_storage),
            patch(
                "operations.FieldOperations.get_unified_image_storage",
                return_value=image_storage,
            ),
            patch("vision.YOLODetector.YOLODetector", return_value=detector),
            patch(
                (
                    "operations.FieldOperations.YOLODetector"
                    if False
                    else "vision.YOLODetector.YOLODetector"
                ),
                return_value=detector,
            ),
            patch(
                "operations.StereoUtils.camera_config_from_metadata",
                return_value=_stereo_params(),
            ),
            patch("config.KnowledgeGraph.KNOWLEDGE_GRAPH_ENABLED", False),
            patch("core.Imports.get_world_state", return_value=world_state),
        ]
        return patches, detector, world_state

    def _run(self, robot_id="Robot1", field_label="D", detections=None, stereo=None):
        """Run detect_field with all deps mocked."""
        if detections is None:
            detections = [_make_detection("field_d")]

        det_result = _make_detections_result(detections)
        image_storage = MagicMock()
        image_storage.get_latest_stereo_image.return_value = (
            stereo if stereo is not None else _stereo_data()
        )
        detector_inst = MagicMock()
        detector_inst.detect_objects_stereo.return_value = det_result
        world_state = MagicMock()

        # camera_config_from_metadata and YOLODetector are local imports inside the
        # function body, so patch at the source module, not on FieldOperations.
        with (
            patch(
                "operations.FieldOperations.get_unified_image_storage",
                return_value=image_storage,
            ),
            patch(
                "operations.StereoUtils.camera_config_from_metadata",
                return_value=_stereo_params(),
            ),
            patch("vision.YOLODetector.YOLODetector", return_value=detector_inst),
            patch("config.Vision.YOLO_MODEL_PATH", "/fake/model.onnx"),
            patch("core.Imports.get_world_state", return_value=world_state),
        ):
            result = detect_field(robot_id, field_label)

        return result, world_state, detector_inst

    def test_happy_path_returns_field_label(self):
        result, _, _ = self._run(
            field_label="D", detections=[_make_detection("field_d")]
        )
        assert result.success is True
        assert result.result is not None
        assert result.result["field_label"] == "D"

    def test_happy_path_center_dict(self):
        det = _make_detection("field_g", world_position=(0.5, 0.1, 0.8))
        result, _, _ = self._run(field_label="G", detections=[det])
        assert result.success is True
        assert result.result is not None
        center = result.result["center"]
        assert center["x"] == pytest.approx(0.5)
        assert center["y"] == pytest.approx(0.1)
        assert center["z"] == pytest.approx(0.8)

    def test_world_position_as_dict(self):
        det = _make_detection("field_a", world_position={"x": 1.0, "y": 2.0, "z": 3.0})
        result, _, _ = self._run(field_label="A", detections=[det])
        assert result.success is True
        assert result.result is not None
        assert result.result["center"]["x"] == pytest.approx(1.0)

    def test_world_state_updated(self):
        det = _make_detection("field_d", world_position=(0.1, 0.2, 0.3))
        _, ws, _ = self._run(field_label="D", detections=[det])
        ws.update_object_position.assert_called_once()
        call_kwargs = ws.update_object_position.call_args.kwargs
        assert call_kwargs["object_id"] == "field_d"
        assert call_kwargs["object_type"] == "field"

    def test_invalid_robot_id_empty(self):
        result = detect_field("", "D")
        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_ROBOT_ID"

    def test_invalid_robot_id_none(self):
        result = detect_field(None, "D")  # type: ignore[arg-type]
        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_ROBOT_ID"

    def test_invalid_field_label_digit(self):
        result = detect_field("Robot1", "1")
        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_FIELD_LABEL"

    def test_invalid_field_label_multichar(self):
        result = detect_field("Robot1", "AB")
        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_FIELD_LABEL"

    def test_no_stereo_images(self):
        image_storage = MagicMock()
        image_storage.get_latest_stereo_image.return_value = None
        with patch(
            "operations.FieldOperations.get_unified_image_storage",
            return_value=image_storage,
        ):
            result = detect_field("Robot1", "D")
        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "NO_STEREO_IMAGES"

    def test_incomplete_stereo_pair(self):
        image_storage = MagicMock()
        image_storage.get_latest_stereo_image.return_value = (
            None,
            None,
            None,
            None,
            {},
        )
        with patch(
            "operations.FieldOperations.get_unified_image_storage",
            return_value=image_storage,
        ):
            result = detect_field("Robot1", "D")
        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INCOMPLETE_STEREO_PAIR"

    def test_no_detections(self):
        result, _, _ = self._run(field_label="D", detections=[])
        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "FIELD_NOT_DETECTED"

    def test_no_3d_coordinates(self):
        det = _make_detection("field_d", world_position=None)
        result, _, _ = self._run(field_label="D", detections=[det])
        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "NO_3D_COORDINATES"

    def test_confidence_in_result(self):
        det = _make_detection(
            "field_b", world_position=(0.1, 0.2, 0.3), confidence=0.92
        )
        result, _, _ = self._run(field_label="B", detections=[det])
        assert result.result is not None
        assert result.result["confidence"] == pytest.approx(0.92)

    def test_field_label_case_insensitive(self):
        det = _make_detection("field_g")
        result, _, _ = self._run(field_label="g", detections=[det])
        assert result.success is True
        assert result.result is not None
        assert result.result["field_label"] == "G"

    def test_yolo_class_constructed_correctly(self):
        det = _make_detection("field_e", world_position=(0, 0, 0))
        _, _, detector = self._run(field_label="E", detections=[det])
        call_kwargs = detector.detect_objects_stereo.call_args.kwargs
        assert call_kwargs["filter_classes"] == ["field_e"]


# detect_all_fields


class TestDetectAllFields:
    def _run(self, detections=None, stereo="ok"):
        image_storage = MagicMock()
        if stereo == "ok":
            image_storage.get_latest_stereo_image.return_value = _stereo_data()
        elif stereo is None:
            image_storage.get_latest_stereo_image.return_value = None
        else:
            image_storage.get_latest_stereo_image.return_value = stereo

        det_result = _make_detections_result(detections or [])
        detector_inst = MagicMock()
        detector_inst.detect_objects_stereo.return_value = det_result

        with (
            patch(
                "operations.FieldOperations.get_unified_image_storage",
                return_value=image_storage,
            ),
            patch(
                "operations.StereoUtils.camera_config_from_metadata",
                return_value=_stereo_params(),
            ),
            patch("vision.YOLODetector.YOLODetector", return_value=detector_inst),
            patch("config.Vision.YOLO_MODEL_PATH", "/fake/model.onnx"),
        ):
            result = detect_all_fields("Robot1")

        return result, detector_inst

    def test_empty_detections_success(self):
        result, _ = self._run(detections=[])
        assert result.success is True
        assert result.result is not None
        assert result.result["fields"] == []
        assert result.result["count"] == 0

    def test_multiple_fields_detected(self):
        dets = [
            _make_detection("field_a", world_position=(0.1, 0.0, 0.1)),
            _make_detection("field_d", world_position=(0.2, 0.0, 0.2)),
            _make_detection("field_g", world_position=(0.3, 0.0, 0.3)),
        ]
        result, _ = self._run(detections=dets)
        assert result.success is True
        assert result.result is not None
        assert result.result["count"] == 3
        labels = [f["label"] for f in result.result["fields"]]
        assert set(labels) == {"A", "D", "G"}

    def test_skips_non_field_detections(self):
        dets = [
            _make_detection("blue_cube", world_position=(0.1, 0, 0.1)),
            _make_detection("field_b", world_position=(0.2, 0, 0.2)),
        ]
        result, _ = self._run(detections=dets)
        assert result.result is not None
        assert result.result["count"] == 1
        assert result.result["fields"][0]["label"] == "B"

    def test_field_classes_filter_passed(self):
        _, detector = self._run(detections=[])
        call_kwargs = detector.detect_objects_stereo.call_args.kwargs
        filter_classes = call_kwargs["filter_classes"]
        assert len(filter_classes) == 9
        assert "field_a" in filter_classes
        assert "field_i" in filter_classes

    def test_no_stereo_images(self):
        result, _ = self._run(stereo=None)  # type: ignore[arg-type]
        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "NO_STEREO_IMAGES"

    def test_incomplete_stereo_pair(self):
        result, _ = self._run(stereo=(None, None, None, None, {}))  # type: ignore[arg-type]
        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INCOMPLETE_STEREO_PAIR"

    def test_invalid_robot_id(self):
        # robot_id validation fires before any external dep is called
        result = detect_all_fields("")
        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_ROBOT_ID"

    def test_center_dict_in_result(self):
        det = _make_detection("field_c", world_position=(1.0, 2.0, 3.0))
        result, _ = self._run(detections=[det])
        assert result.result is not None
        field = result.result["fields"][0]
        assert field["center"]["x"] == pytest.approx(1.0)
        assert field["center"]["z"] == pytest.approx(3.0)

    def test_world_position_as_dict(self):
        det = _make_detection("field_f", world_position={"x": 0.5, "y": 0.0, "z": 0.7})
        result, _ = self._run(detections=[det])
        assert result.success is True
        assert result.result is not None
        assert result.result["fields"][0]["center"]["x"] == pytest.approx(0.5)
