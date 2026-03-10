"""
Unit tests for GraspNetClient
==============================

Tests the HTTP client that talks to the Contact-GraspNet FastAPI inference
service.  All network I/O is mocked via ``unittest.mock.patch`` on
``urllib.request.urlopen`` so no live service is required.

Coverage:
- is_available: healthy service, unreachable service, TTL cache
- predict_grasps: success, empty-after-mask, service returns success=false,
  network error, optional colors/mask handling, cache refresh on success
"""

import base64
import json
import time
import io
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from operations.GraspNetClient import GraspNetClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_health_response(status: int = 200):
    """Return a mock urllib response for the /health endpoint."""
    resp = MagicMock()
    resp.status = status
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _make_predict_response(grasps: list, success: bool = True, inference_ms: float = 42.0):
    """Return a mock urllib response for the /predict_grasps endpoint."""
    payload = {
        "success": success,
        "grasps": grasps,
        "count": len(grasps),
        "inference_time_ms": inference_ms,
    }
    raw = json.dumps(payload).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = raw
    resp.status = 200
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _sample_points(n: int = 100) -> np.ndarray:
    """Return a minimal float32 (N, 3) point array."""
    return np.random.rand(n, 3).astype(np.float32)


def _sample_grasp() -> dict:
    """Return a minimal grasp dict matching the service response schema."""
    return {
        "position": [0.1, 0.2, 0.5],
        "rotation": [0.0, 0.0, 0.0, 1.0],
        "score": 0.85,
        "width": 0.08,
        "approach_direction": [0.0, -1.0, 0.0],
    }


# ---------------------------------------------------------------------------
# is_available tests
# ---------------------------------------------------------------------------


class TestIsAvailable:
    """Tests for GraspNetClient.is_available()."""

    def _fresh_client(self) -> GraspNetClient:
        """Return a client with an expired health cache."""
        c = GraspNetClient()
        c._last_health_check = 0.0  # force cache miss
        return c

    def test_returns_true_when_service_healthy(self):
        """HTTP 200 on /health → True."""
        client = self._fresh_client()
        resp = _make_health_response(200)
        with patch("urllib.request.urlopen", return_value=resp):
            assert client.is_available() is True

    def test_returns_false_when_service_unreachable(self):
        """Network error → False (no exception raised to caller)."""
        client = self._fresh_client()
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            assert client.is_available() is False

    def test_returns_false_when_non_200_status(self):
        """HTTP 503 → False."""
        client = self._fresh_client()
        resp = _make_health_response(503)
        with patch("urllib.request.urlopen", return_value=resp):
            assert client.is_available() is False

    def test_ttl_cache_avoids_second_request(self):
        """Within TTL, is_available() uses cached result without hitting network."""
        client = self._fresh_client()
        resp = _make_health_response(200)
        with patch("urllib.request.urlopen", return_value=resp) as mock_urlopen:
            client.is_available()   # fills cache
            client.is_available()   # should use cache
            assert mock_urlopen.call_count == 1

    def test_cache_expires_after_ttl(self):
        """After TTL seconds, is_available() makes a new network request."""
        client = self._fresh_client()
        client._health_cache_ttl = 0.01  # very short TTL
        resp = _make_health_response(200)
        with patch("urllib.request.urlopen", return_value=resp) as mock_urlopen:
            client.is_available()
            time.sleep(0.05)       # let TTL expire
            client.is_available()
            assert mock_urlopen.call_count == 2

    def test_failed_check_stores_false_in_cache(self):
        """After a failed check, cached result is False."""
        client = self._fresh_client()
        with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
            client.is_available()
        assert client._last_health_result is False

    def test_healthy_check_stores_true_in_cache(self):
        """After a successful check, cached result is True."""
        client = self._fresh_client()
        resp = _make_health_response(200)
        with patch("urllib.request.urlopen", return_value=resp):
            client.is_available()
        assert client._last_health_result is True


# ---------------------------------------------------------------------------
# predict_grasps tests
# ---------------------------------------------------------------------------


class TestPredictGrasps:
    """Tests for GraspNetClient.predict_grasps()."""

    def _client(self) -> GraspNetClient:
        return GraspNetClient()

    def test_returns_grasp_list_on_success(self):
        """Valid response with grasps → list of dicts."""
        client = self._client()
        grasps = [_sample_grasp(), _sample_grasp()]
        resp = _make_predict_response(grasps)
        with patch("urllib.request.urlopen", return_value=resp):
            result = client.predict_grasps(_sample_points())
        assert result is not None
        assert len(result) == 2
        assert result[0]["score"] == pytest.approx(0.85)

    def test_returns_none_when_success_false(self):
        """Service returns success=false → None."""
        client = self._client()
        resp = _make_predict_response([], success=False)
        with patch("urllib.request.urlopen", return_value=resp):
            result = client.predict_grasps(_sample_points())
        assert result is None

    def test_returns_none_on_network_error(self):
        """Network error → None (no exception propagated)."""
        client = self._client()
        with patch("urllib.request.urlopen", side_effect=OSError("reset")):
            result = client.predict_grasps(_sample_points())
        assert result is None

    def test_returns_none_for_empty_point_cloud_after_mask(self):
        """Segmentation mask that keeps at least one True but leaves zero points after
        numpy boolean index should not happen — but an all-True mask on an empty array
        produces an empty result.  More practically: mask.any() is False → mask branch
        is skipped entirely and full pts is sent.  When pts itself is empty, return None."""
        client = self._client()
        pts = np.zeros((0, 3), dtype=np.float32)   # empty array
        mask = np.zeros(0, dtype=bool)
        with patch("urllib.request.urlopen") as mock_urlopen:
            result = client.predict_grasps(pts, segmentation_mask=mask)
        assert result is None
        mock_urlopen.assert_not_called()

    def test_segmentation_mask_filters_points(self):
        """Points outside the mask are excluded from the payload.

        The client also un-negates X before sending (LH→RH frame conversion for
        the GraspNet server), so the sent X values are -X of the input.
        """
        client = self._client()
        pts = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
        mask = np.array([True, False, True])
        resp = _make_predict_response([_sample_grasp()])

        captured_payload = {}

        def fake_urlopen(req, timeout=None):
            captured_payload.update(json.loads(req.data))
            return resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client.predict_grasps(pts, segmentation_mask=mask)

        # Decode base64 points and reshape using points_shape
        shape = captured_payload["points_shape"]
        sent_pts = np.frombuffer(
            base64.b64decode(captured_payload["points"]), dtype=np.float32
        ).reshape(shape)
        assert len(sent_pts) == 2                          # mask filtered one out
        # X is negated (LH Unity → RH OpenCV) before sending
        assert sent_pts[0].tolist() == pytest.approx([-1, 0, 0])
        assert sent_pts[1].tolist() == pytest.approx([-3, 0, 0])

    def test_colors_included_in_payload_when_provided(self):
        """Colors array is serialized into the JSON payload."""
        client = self._client()
        pts = _sample_points(10)
        clr = np.zeros((10, 3), dtype=np.uint8)
        resp = _make_predict_response([_sample_grasp()])

        captured_payload = {}

        def fake_urlopen(req, timeout=None):
            captured_payload.update(json.loads(req.data))
            return resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client.predict_grasps(pts, colors=clr)

        assert "colors" in captured_payload

    def test_top_k_default_used_when_not_specified(self):
        """top_k in payload matches _top_k_default when caller passes None."""
        client = self._client()
        client._top_k_default = 15
        resp = _make_predict_response([])

        captured_payload = {}

        def fake_urlopen(req, timeout=None):
            captured_payload.update(json.loads(req.data))
            return resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client.predict_grasps(_sample_points())

        assert captured_payload["top_k"] == 15

    def test_top_k_override_respected(self):
        """Explicit top_k parameter overrides the default."""
        client = self._client()
        resp = _make_predict_response([])

        captured_payload = {}

        def fake_urlopen(req, timeout=None):
            captured_payload.update(json.loads(req.data))
            return resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client.predict_grasps(_sample_points(), top_k=5)

        assert captured_payload["top_k"] == 5

    def test_successful_predict_refreshes_health_cache(self):
        """A successful predict_grasps marks the service as healthy in cache."""
        client = self._client()
        client._last_health_result = False
        client._last_health_check = 0.0
        resp = _make_predict_response([_sample_grasp()])
        with patch("urllib.request.urlopen", return_value=resp):
            client.predict_grasps(_sample_points())
        assert client._last_health_result is True

    def test_network_error_invalidates_health_cache(self):
        """A failed predict_grasps resets the health cache so next is_available re-checks."""
        client = self._client()
        client._last_health_result = True
        client._last_health_check = time.time()
        with patch("urllib.request.urlopen", side_effect=OSError("dropped")):
            client.predict_grasps(_sample_points())
        assert client._last_health_result is False
        assert client._last_health_check == pytest.approx(0.0)

    def test_request_sent_to_correct_url(self):
        """POST is made to <base_url>/predict_grasps."""
        client = self._client()
        client._base_url = "http://10.0.0.1:8766"
        resp = _make_predict_response([])

        captured_url = {}

        def fake_urlopen(req, timeout=None):
            captured_url["url"] = req.full_url
            return resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client.predict_grasps(_sample_points())

        assert captured_url["url"] == "http://10.0.0.1:8766/predict_grasps"

    def test_returns_empty_list_when_service_returns_no_grasps(self):
        """Service success=true but grasps=[] → empty list (not None)."""
        client = self._client()
        resp = _make_predict_response([])
        with patch("urllib.request.urlopen", return_value=resp):
            result = client.predict_grasps(_sample_points())
        # Empty list is falsy; service returned success=true so result is []
        assert result == []

    def test_x_axis_negated_before_sending(self):
        """Points are sent with X negated to convert Unity LH frame to RH OpenCV frame.

        StereoReconstruction bakes a -X flip into its Q-matrix so all incoming
        points have X already negated (Unity LH convention).  The client must
        un-negate X before sending so Contact-GraspNet receives the correct
        right-handed geometry (X-right, Y-down, Z-forward).
        """
        client = self._client()
        # Input: Unity LH frame, X negated (as produced by StereoReconstruction)
        pts = np.array([[-1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]], dtype=np.float32)
        resp = _make_predict_response([_sample_grasp()])

        captured_payload = {}

        def fake_urlopen(req, timeout=None):
            captured_payload.update(json.loads(req.data))
            return resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client.predict_grasps(pts)

        # Decode base64 points and reshape using points_shape
        shape = captured_payload["points_shape"]
        sent_pts = np.frombuffer(
            base64.b64decode(captured_payload["points"]), dtype=np.float32
        ).reshape(shape)
        # X should be un-negated (flipped back to RH positive)
        assert sent_pts[0].tolist() == pytest.approx([1.0, 2.0, 3.0])
        assert sent_pts[1].tolist() == pytest.approx([4.0, 5.0, 6.0])

    def test_x_negation_does_not_mutate_input_array(self):
        """The coordinate flip must not modify the caller's original numpy array."""
        client = self._client()
        pts = np.array([[-1.0, 2.0, 3.0]], dtype=np.float32)
        original_x = pts[0, 0]
        resp = _make_predict_response([])
        with patch("urllib.request.urlopen", return_value=resp):
            client.predict_grasps(pts)
        assert pts[0, 0] == pytest.approx(original_x)  # unchanged

    def test_points_encoded_as_base64_string(self):
        """Payload 'points' field is a base64 string, not a list of lists."""
        client = self._client()
        resp = _make_predict_response([])

        captured_payload = {}

        def fake_urlopen(req, timeout=None):
            captured_payload.update(json.loads(req.data))
            return resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client.predict_grasps(_sample_points())

        assert isinstance(captured_payload["points"], str)
        assert "points_shape" in captured_payload
        # Must be valid base64 — decode should not raise
        decoded = base64.b64decode(captured_payload["points"])
        assert len(decoded) > 0

    def test_points_shape_matches_actual_points(self):
        """points_shape in payload matches the shape of the decoded points array."""
        client = self._client()
        n_pts = 50
        pts = _sample_points(n_pts)
        resp = _make_predict_response([])

        captured_payload = {}

        def fake_urlopen(req, timeout=None):
            captured_payload.update(json.loads(req.data))
            return resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client.predict_grasps(pts)

        shape = captured_payload["points_shape"]
        decoded = np.frombuffer(
            base64.b64decode(captured_payload["points"]), dtype=np.float32
        ).reshape(shape)
        assert decoded.shape == (n_pts, 3)
