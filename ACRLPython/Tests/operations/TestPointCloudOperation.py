#!/usr/bin/env python3
"""
Unit tests for PointCloudOperations.py

Tests:
- Successful point cloud generation from mocked stereo images
- Stale image rejection
- Incomplete stereo pair handling
- Uniform random downsampling
- Integration marker for tests that need live Unity (skipped by default)
"""

import time
import pytest
import numpy as np
from unittest.mock import Mock, patch

from operations.PointCloudOperations import generate_point_cloud, GENERATE_POINT_CLOUD_OPERATION
from operations.Base import OperationResult


def _load_backend_client():
    """Load BackendClient using an absolute file path to avoid package import issues.

    Test subdirectories have no __init__.py, so package-style imports like
    'from tests.integration.helpers.backend_client import ...' fail.  We use
    importlib to load the module directly from its file path instead.
    """
    import importlib.util
    import os
    helper_path = os.path.join(
        os.path.dirname(__file__), "..", "integration", "helpers", "backend_client.py"
    )
    spec = importlib.util.spec_from_file_location("backend_client", helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BackendClient


def _backend_available() -> bool:
    """Return True when the SequenceServer (port 5013) is reachable.

    The SequenceServer is only active once the backend process has started
    and Unity has connected.  Probing the port is sufficient to decide
    whether integration tests should run.
    """
    try:
        import socket as _socket
        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        sock.settimeout(2.0)
        result = sock.connect_ex(("localhost", 5013))
        sock.close()
        return result == 0
    except Exception:
        return False


_STEREO_AVAILABLE = _backend_available()
_STEREO_SKIP = "Backend not reachable on port 5013 — run Unity and start the backend servers"


# ============================================================================
# Helpers
# ============================================================================


def _make_stereo_images(width: int = 64, height: int = 48) -> tuple:
    """Return a minimal stereo pair (left, right) with simple chessboard texture."""
    board = np.zeros((height, width, 3), dtype=np.uint8)
    # Chessboard pattern so SGBM can find disparity
    for r in range(height):
        for c in range(width):
            if (r // 4 + c // 4) % 2 == 0:
                board[r, c] = [200, 200, 200]
    return board.copy(), board.copy()


def _make_storage_mock(
    left,
    right,
    timestamp=None,
    metadata=None,
):
    """Return a mock UnifiedImageStorage with get_latest_stereo_image configured."""
    if timestamp is None:
        timestamp = time.time()
    if metadata is None:
        metadata = {
            "fov": 60.0,
            "baseline": 0.1,
            "camera_position": [0.0, 1.0, 0.5],
            "camera_rotation": [0.0, 0.0, 0.0, 1.0],
        }
    storage = Mock()
    storage.get_latest_stereo_image.return_value = (left, right, "", timestamp, metadata)
    return storage


# ============================================================================
# Tests
# ============================================================================


class TestGeneratePointCloud:
    """Unit tests for generate_point_cloud — all IO is mocked."""

    def _patch_storage(self, monkeypatch, storage):
        """Patch get_unified_image_storage to return the given mock storage."""
        monkeypatch.setattr(
            "operations.PointCloudOperations.get_unified_image_storage",
            Mock(return_value=storage),
            raising=False,
        )

    def _patch_reconstruct(self, monkeypatch, fake_pc):
        """Patch stereo_reconstruct_stream to return a synthetic point cloud."""
        monkeypatch.setattr(
            "operations.PointCloudOperations.stereo_reconstruct_stream",
            Mock(return_value=fake_pc),
            raising=False,
        )

    def test_success_returns_expected_keys(self, monkeypatch):
        """A valid stereo pair produces a result with all required keys."""
        left, right = _make_stereo_images()
        storage = _make_storage_mock(left, right)
        self._patch_storage(monkeypatch, storage)

        # Provide a minimal stereo_reconstruct_stream that returns synthetic data
        n_pts = 200
        fake_pc = {
            "points": np.random.rand(n_pts, 3).astype(np.float32),
            "colors": np.random.randint(0, 255, (n_pts, 3), dtype=np.uint8),
        }
        self._patch_reconstruct(monkeypatch, fake_pc)

        result = generate_point_cloud("Robot1")

        assert result.success, f"Expected success but got error: {result.error}"
        r = result.result
        assert "points" in r
        assert "colors" in r
        assert "point_count" in r
        assert "camera_position" in r
        assert "camera_rotation" in r
        assert "fov" in r
        assert "baseline" in r
        assert "timestamp" in r
        assert r["point_count"] == n_pts

    def test_stale_image_rejected(self, monkeypatch):
        """Images older than max_age_seconds return STALE_IMAGE error."""
        left, right = _make_stereo_images()
        stale_ts = time.time() - 60.0  # 60 seconds old
        storage = _make_storage_mock(left, right, timestamp=stale_ts)
        self._patch_storage(monkeypatch, storage)

        result = generate_point_cloud("Robot1", max_age_seconds=2.0)

        assert not result.success
        assert result.error["code"] == "STALE_IMAGE"

    def test_no_stereo_images_available(self, monkeypatch):
        """When storage returns None, operation returns NO_STEREO_IMAGES error."""
        storage = Mock()
        storage.get_latest_stereo_image.return_value = None
        self._patch_storage(monkeypatch, storage)

        result = generate_point_cloud("Robot1")

        assert not result.success
        assert result.error["code"] == "NO_STEREO_IMAGES"

    def test_incomplete_stereo_pair(self, monkeypatch):
        """When right image is None, operation returns INCOMPLETE_STEREO_PAIR."""
        left, _ = _make_stereo_images()
        storage = _make_storage_mock(left, None)
        self._patch_storage(monkeypatch, storage)

        result = generate_point_cloud("Robot1")

        assert not result.success
        assert result.error["code"] == "INCOMPLETE_STEREO_PAIR"

    def test_downsample_applied_when_above_max_points(self, monkeypatch):
        """When raw cloud exceeds max_points, result is capped at max_points."""
        left, right = _make_stereo_images()
        storage = _make_storage_mock(left, right)
        self._patch_storage(monkeypatch, storage)

        n_raw = 10_000
        fake_pc = {
            "points": np.random.rand(n_raw, 3).astype(np.float32),
            "colors": np.random.randint(0, 255, (n_raw, 3), dtype=np.uint8),
        }
        self._patch_reconstruct(monkeypatch, fake_pc)

        max_pts = 500
        result = generate_point_cloud("Robot1", max_points=max_pts)

        assert result.success
        assert result.result["point_count"] == max_pts
        assert len(result.result["points"]) == max_pts

    def test_nan_points_filtered(self, monkeypatch):
        """NaN/Inf points from SGBM are removed before downsampling."""
        left, right = _make_stereo_images()
        storage = _make_storage_mock(left, right)
        self._patch_storage(monkeypatch, storage)

        pts = np.array([[1.0, 2.0, 3.0], [float("nan"), 0.0, 0.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        clr = np.zeros((3, 3), dtype=np.uint8)
        fake_pc = {"points": pts, "colors": clr}
        self._patch_reconstruct(monkeypatch, fake_pc)

        result = generate_point_cloud("Robot1")

        assert result.success
        # Only the 2 valid points should remain
        assert result.result["point_count"] == 2

    def test_camera_position_in_result(self, monkeypatch):
        """camera_position metadata is propagated correctly from stereo metadata."""
        left, right = _make_stereo_images()
        expected_pos = [1.5, 0.8, 2.3]
        metadata = {
            "fov": 60.0,
            "baseline": 0.1,
            "camera_position": expected_pos,
            "camera_rotation": [0.0, 0.0, 0.0, 1.0],
        }
        storage = _make_storage_mock(left, right, metadata=metadata)
        self._patch_storage(monkeypatch, storage)

        n_pts = 10
        fake_pc = {
            "points": np.random.rand(n_pts, 3).astype(np.float32),
            "colors": np.zeros((n_pts, 3), dtype=np.uint8),
        }
        self._patch_reconstruct(monkeypatch, fake_pc)

        result = generate_point_cloud("Robot1")

        assert result.success
        assert result.result["camera_position"] == pytest.approx(expected_pos)

    def test_operation_definition_has_implementation(self):
        """Registry operation definition must carry an implementation callable."""
        assert GENERATE_POINT_CLOUD_OPERATION.implementation is not None
        assert callable(GENERATE_POINT_CLOUD_OPERATION.implementation)


# ============================================================================
# Integration tests (require live Unity + stereo camera — skipped by default)
# ============================================================================


@pytest.mark.integration
@pytest.mark.requires_unity
@pytest.mark.skipif(not _STEREO_AVAILABLE, reason=_STEREO_SKIP)
class TestGeneratePointCloudIntegration:
    """Integration tests that need a live Unity session with stereo cameras.

    These tests send commands to the running backend over TCP (port 5013),
    which ensures the correct process's UnifiedImageStorage is used.
    The backend populates its storage from Unity's stereo camera stream;
    we trigger a detect_object_stereo command to warm it up before the
    point cloud tests run.
    """

    @pytest.fixture(autouse=True)
    def warmup_stereo_storage(self):
        """Trigger a stereo detection on the backend to populate UnifiedImageStorage.

        generate_point_cloud reads from the backend server's UnifiedImageStorage,
        so we send a detect_object_stereo command via the SequenceServer to ensure
        at least one stereo frame has been captured before the test runs.
        """
        try:
            BackendClient = _load_backend_client()
            with BackendClient(timeout=15.0) as client:
                client.send_command(
                    command="detect object stereo for Robot1",
                    robot_id="Robot1",
                    camera_id="TableStereoCamera",
                    request_id=9901,
                )
        except Exception:
            pass  # If warmup fails the test itself will surface the real error

    def _run_point_cloud(self, request_id: int) -> dict:
        """Capture a fresh stereo frame then immediately generate a point cloud.

        Sends both commands as a two-step sequence so the images are always
        within max_age_seconds when generate_point_cloud runs.
        """
        BackendClient = _load_backend_client()
        # Step 1: capture a fresh stereo frame into the backend's storage.
        with BackendClient(timeout=15.0) as client:
            client.send_command(
                command="detect object stereo for Robot1",
                robot_id="Robot1",
                camera_id="TableStereoCamera",
                request_id=request_id,
            )
        # Step 2: generate the point cloud immediately after (images are fresh).
        with BackendClient(timeout=30.0) as client:
            return client.send_command(
                command="generate point cloud for Robot1",
                robot_id="Robot1",
                camera_id="TableStereoCamera",
                request_id=request_id + 1,
            )

    def test_live_point_cloud_has_valid_points(self):
        """Point count > 0 and all points are finite."""
        result_dict = self._run_point_cloud(request_id=9902)
        assert result_dict.get("success"), result_dict.get("error")
        cmd_results = result_dict.get("results", [{}])
        r = cmd_results[0].get("result", {}) if cmd_results else {}
        assert r.get("point_count", 0) > 0
        pts = np.array(r["points"])
        assert np.isfinite(pts).all(), "Point cloud contains non-finite values"

    def test_live_camera_position_non_zero(self):
        """camera_position from Unity metadata should not be the origin."""
        result_dict = self._run_point_cloud(request_id=9904)
        assert result_dict.get("success"), result_dict.get("error")
        cmd_results = result_dict.get("results", [{}])
        r = cmd_results[0].get("result", {}) if cmd_results else {}
        cam_pos = r.get("camera_position", [0, 0, 0])
        assert any(abs(v) > 1e-3 for v in cam_pos), (
            "camera_position is all zeros — check Unity stereo camera metadata"
        )
