"""
Integration tests for _grasp_via_graspnet orchestration
=========================================================

Tests the full GraspNet fast-path inside ``grasp_object`` by mocking every
external collaborator so no live Unity, GPU server, or stereo cameras are
required.  The goal is to verify that the orchestration logic correctly
chains the components and falls back to the geometric pipeline when any
step fails.

Coverage:
- Happy path: point cloud → detection → GraspNet → transform → broadcaster
- Fallback when GraspNet service is unavailable
- Fallback when point cloud generation fails
- Fallback when GraspNet returns no candidates
- Fallback when frame transform produces no valid poses
- Broadcaster unavailable → error result (not fallback)
- Broadcaster send failure → error result
- Pre-grasp position computed along approach direction
- Correct number of candidates forwarded to Unity
- grasp_object routes to GraspNet path when GRASPNET_ENABLED=true
- grasp_object falls back to TCP path when GraspNet returns None
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from operations.GraspOperations import _grasp_via_graspnet, grasp_object
from operations.Base import OperationResult


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------


def _pc_success(n_pts: int = 200):
    """Return a successful OperationResult wrapping a minimal point cloud."""
    pts = np.random.rand(n_pts, 3).astype(np.float32).tolist()
    clr = np.zeros((n_pts, 3), dtype=np.uint8).tolist()
    return OperationResult.success_result({
        "points": pts,
        "colors": clr,
        "point_count": n_pts,
        "camera_position": [0.0, 1.0, 0.5],
        "camera_rotation": [0.0, 0.0, 0.0, 1.0],
        "fov": 60.0,
        "baseline": 0.1,
        "timestamp": 1234567890.0,
    })


def _pc_failure():
    return OperationResult.error_result("STALE_IMAGE", "Image too old", [])


def _sample_grasps(n: int = 3):
    """Return minimal grasp dicts as returned by the GraspNet service."""
    return [
        {
            "position": [0.1 * i, 0.2, 0.5],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "score": 0.9 - 0.1 * i,
            "width": 0.08,
            "approach_direction": [0.0, -1.0, 0.0],
        }
        for i in range(n)
    ]


def _world_grasps(n: int = 3):
    """Return world-frame grasps as produced by GraspFrameTransform."""
    return [
        {
            "position": [0.2 * i, 0.3, 0.6],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "score": 0.9 - 0.1 * i,
            "width": 0.08,
            "approach_direction": [0.0, -1.0, 0.0],
        }
        for i in range(n)
    ]


_UNSET = object()  # sentinel: distinguishes "caller omitted" from explicit None


class _GraspNetPatch:
    """Context manager that patches all collaborators of _grasp_via_graspnet."""

    def __init__(
        self,
        graspnet_available: bool = True,
        pc_result=_UNSET,
        detect_result=None,
        raw_grasps=_UNSET,
        world_grasps_result=_UNSET,
        broadcaster_available: bool = True,
        send_command_return: bool = True,
    ):
        self.graspnet_available = graspnet_available
        self.pc_result = _pc_success() if pc_result is _UNSET else pc_result
        self.detect_result = detect_result
        self.raw_grasps = _sample_grasps() if raw_grasps is _UNSET else raw_grasps
        self.world_grasps_result = _world_grasps() if world_grasps_result is _UNSET else world_grasps_result
        self.broadcaster_available = broadcaster_available
        self.send_command_return = send_command_return

        self._patches = []
        self.mock_client = None
        self.mock_broadcaster = None

    def __enter__(self):
        # GraspNetClient and its lazy-import siblings live in their own modules;
        # _grasp_via_graspnet imports them locally so we must patch the source
        # module, not the GraspOperations namespace.

        # Mock GraspNetClient class in its own module
        self.mock_client = MagicMock()
        self.mock_client.is_available.return_value = self.graspnet_available
        self.mock_client.predict_grasps.return_value = self.raw_grasps

        client_cls = patch(
            "operations.GraspNetClient.GraspNetClient",
            return_value=self.mock_client,
        )
        self._patches.append(client_cls.start())

        # Mock generate_point_cloud in its source module
        pc_patch = patch(
            "operations.PointCloudOperations.generate_point_cloud",
            return_value=self.pc_result,
        )
        self._patches.append(pc_patch.start())

        # Mock detect_objects (optional detection, non-fatal if not provided)
        if self.detect_result is not None:
            det_patch = patch(
                "operations.DetectionOperations.detect_objects",
                return_value=self.detect_result,
            )
            self._patches.append(det_patch.start())

        # Mock transform in its source module
        transform_patch = patch(
            "operations.GraspFrameTransform.transform_graspnet_poses_to_unity",
            return_value=self.world_grasps_result,
        )
        self._patches.append(transform_patch.start())

        # Mock CommandBroadcaster — this IS a module-level name in GraspOperations
        self.mock_broadcaster = MagicMock()
        self.mock_broadcaster.send_command.return_value = self.send_command_return

        bc_patch = patch(
            "operations.GraspOperations._get_command_broadcaster",
            return_value=self.mock_broadcaster if self.broadcaster_available else None,
        )
        self._patches.append(bc_patch.start())

        return self

    def __exit__(self, *args):
        import unittest.mock as _um
        _um.patch.stopall()


# ---------------------------------------------------------------------------
# _grasp_via_graspnet happy path
# ---------------------------------------------------------------------------


class TestGraspViaGraspnetHappyPath:
    """Tests for the successful execution path."""

    def test_returns_success_result(self):
        """Happy path returns a successful OperationResult."""
        with _GraspNetPatch() as ctx:
            result = _grasp_via_graspnet(
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach="top",
                use_advanced_planning=True,
                pre_grasp_distance=0.0,
                enable_retreat=True,
                retreat_distance=0.0,
                request_id=1,
            )
        assert result is not None
        assert result.success is True

    def test_result_contains_graspnet_candidate_count(self):
        """Result dict includes how many GraspNet candidates were forwarded."""
        n = 5
        with _GraspNetPatch(raw_grasps=_sample_grasps(n), world_grasps_result=_world_grasps(n)) as ctx:
            result = _grasp_via_graspnet(
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach="auto",
                use_advanced_planning=True,
                pre_grasp_distance=0.0,
                enable_retreat=True,
                retreat_distance=0.0,
                request_id=2,
            )
        assert result.result["graspnet_candidates"] == n

    def test_command_sent_with_precomputed_candidates(self):
        """grasp_object command payload contains precomputed_candidates list."""
        n = 3
        with _GraspNetPatch(raw_grasps=_sample_grasps(n), world_grasps_result=_world_grasps(n)) as ctx:
            _grasp_via_graspnet(
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach="top",
                use_advanced_planning=True,
                pre_grasp_distance=0.0,
                enable_retreat=True,
                retreat_distance=0.0,
                request_id=3,
            )
            call_args = ctx.mock_broadcaster.send_command.call_args[0][0]
            assert "precomputed_candidates" in call_args["parameters"]
            assert len(call_args["parameters"]["precomputed_candidates"]) == n

    def test_pre_grasp_position_offset_along_approach(self):
        """Pre-grasp position is grasp position + approach_direction * hover."""
        world = [{
            "position": [0.0, 0.0, 0.5],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "score": 0.9,
            "width": 0.08,
            "approach_direction": [0.0, 1.0, 0.0],  # +Y
        }]
        hover = 0.1
        with _GraspNetPatch(raw_grasps=_sample_grasps(1), world_grasps_result=world) as ctx:
            _grasp_via_graspnet(
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach="auto",
                use_advanced_planning=True,
                pre_grasp_distance=hover,
                enable_retreat=True,
                retreat_distance=0.0,
                request_id=4,
            )
            call_args = ctx.mock_broadcaster.send_command.call_args[0][0]
            cand = call_args["parameters"]["precomputed_candidates"][0]
            # pre_grasp = grasp + approach * hover
            assert cand["pre_grasp_position"]["x"] == pytest.approx(0.0)
            assert cand["pre_grasp_position"]["y"] == pytest.approx(0.1)
            assert cand["pre_grasp_position"]["z"] == pytest.approx(0.5)

    def test_robot_id_and_object_id_in_command(self):
        """Command payload carries correct robot_id and object_id."""
        with _GraspNetPatch() as ctx:
            _grasp_via_graspnet(
                robot_id="RobotA",
                object_id="Box_99",
                preferred_approach="side",
                use_advanced_planning=False,
                pre_grasp_distance=0.0,
                enable_retreat=False,
                retreat_distance=0.0,
                request_id=5,
            )
            cmd = ctx.mock_broadcaster.send_command.call_args[0][0]
            assert cmd["robot_id"] == "RobotA"
            assert cmd["parameters"]["object_id"] == "Box_99"

    def test_request_id_forwarded_to_command(self):
        """request_id is forwarded into the Unity command."""
        with _GraspNetPatch() as ctx:
            _grasp_via_graspnet(
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach="auto",
                use_advanced_planning=True,
                pre_grasp_distance=0.0,
                enable_retreat=True,
                retreat_distance=0.0,
                request_id=42,
            )
            cmd = ctx.mock_broadcaster.send_command.call_args[0][0]
            assert cmd["request_id"] == 42


# ---------------------------------------------------------------------------
# Fallback paths (return None → caller uses geometric pipeline)
# ---------------------------------------------------------------------------


class TestGraspViaGraspnetFallback:
    """Tests for scenarios where _grasp_via_graspnet returns None."""

    def test_returns_none_when_graspnet_unavailable(self):
        """Service health check fails → None (no exception)."""
        with _GraspNetPatch(graspnet_available=False):
            result = _grasp_via_graspnet(
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach="auto",
                use_advanced_planning=True,
                pre_grasp_distance=0.0,
                enable_retreat=True,
                retreat_distance=0.0,
                request_id=10,
            )
        assert result is None

    def test_returns_none_when_point_cloud_fails(self):
        """Point cloud generation error → None."""
        with _GraspNetPatch(pc_result=_pc_failure()):
            result = _grasp_via_graspnet(
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach="auto",
                use_advanced_planning=True,
                pre_grasp_distance=0.0,
                enable_retreat=True,
                retreat_distance=0.0,
                request_id=11,
            )
        assert result is None

    def test_returns_none_when_graspnet_returns_no_candidates(self):
        """GraspNet returns empty list → None."""
        with _GraspNetPatch(raw_grasps=[]):
            result = _grasp_via_graspnet(
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach="auto",
                use_advanced_planning=True,
                pre_grasp_distance=0.0,
                enable_retreat=True,
                retreat_distance=0.0,
                request_id=12,
            )
        assert result is None

    def test_returns_none_when_graspnet_returns_none(self):
        """GraspNet returns None (network error) → None."""
        with _GraspNetPatch(raw_grasps=None):
            result = _grasp_via_graspnet(
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach="auto",
                use_advanced_planning=True,
                pre_grasp_distance=0.0,
                enable_retreat=True,
                retreat_distance=0.0,
                request_id=13,
            )
        assert result is None

    def test_returns_none_when_frame_transform_produces_no_poses(self):
        """Empty world_grasps after transform → None."""
        with _GraspNetPatch(world_grasps_result=[]):
            result = _grasp_via_graspnet(
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach="auto",
                use_advanced_planning=True,
                pre_grasp_distance=0.0,
                enable_retreat=True,
                retreat_distance=0.0,
                request_id=14,
            )
        assert result is None


# ---------------------------------------------------------------------------
# Error results (not fallback — definitive failures)
# ---------------------------------------------------------------------------


class TestGraspViaGraspnetErrors:
    """Tests for scenarios where _grasp_via_graspnet returns an error OperationResult."""

    def test_error_when_broadcaster_unavailable(self):
        """Missing broadcaster → error result (not None)."""
        with _GraspNetPatch(broadcaster_available=False):
            result = _grasp_via_graspnet(
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach="auto",
                use_advanced_planning=True,
                pre_grasp_distance=0.0,
                enable_retreat=True,
                retreat_distance=0.0,
                request_id=20,
            )
        assert result is not None
        assert result.success is False
        assert result.error["code"] == "COMMUNICATION_ERROR"

    def test_error_when_send_command_fails(self):
        """Broadcaster.send_command returns False → error result."""
        with _GraspNetPatch(send_command_return=False):
            result = _grasp_via_graspnet(
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach="auto",
                use_advanced_planning=True,
                pre_grasp_distance=0.0,
                enable_retreat=True,
                retreat_distance=0.0,
                request_id=21,
            )
        assert result is not None
        assert result.success is False
        assert result.error["code"] == "COMMUNICATION_ERROR"


# ---------------------------------------------------------------------------
# grasp_object routing tests
# ---------------------------------------------------------------------------


class TestGraspObjectGraspNetRouting:
    """Tests that grasp_object correctly routes through the GraspNet path."""

    def _base_patches(self, graspnet_enabled: bool = True):
        """Return a dict of patches for grasp_object's outer layer."""
        return {
            "config.Servers.GRASPNET_ENABLED": graspnet_enabled,
            "config.ROS.ROS_ENABLED": False,
        }

    def test_uses_graspnet_path_when_enabled_and_available(self):
        """When GRASPNET_ENABLED and service is up, _grasp_via_graspnet is called."""
        graspnet_result = OperationResult.success_result({
            "command_sent": True,
            "robot_id": "Robot1",
            "object_id": "Cube_01",
            "request_id": 0,
            "graspnet_candidates": 3,
        })
        with patch("config.Servers.GRASPNET_ENABLED", True), \
             patch("config.ROS.ROS_ENABLED", False), \
             patch("operations.GraspOperations._grasp_via_graspnet", return_value=graspnet_result) as mock_gvg, \
             patch("operations.GraspOperations._get_command_broadcaster"):
            result = grasp_object(robot_id="Robot1", object_id="Cube_01")
        mock_gvg.assert_called_once()
        assert result.success is True
        assert result.result.get("graspnet_candidates") == 3

    def test_falls_back_to_tcp_when_graspnet_returns_none(self):
        """When _grasp_via_graspnet returns None, geometric TCP path is used."""
        broadcaster = MagicMock()
        broadcaster.send_command.return_value = True
        with patch("config.Servers.GRASPNET_ENABLED", True), \
             patch("config.ROS.ROS_ENABLED", False), \
             patch("operations.GraspOperations._grasp_via_graspnet", return_value=None), \
             patch("operations.GraspOperations._get_command_broadcaster", return_value=broadcaster):
            result = grasp_object(robot_id="Robot1", object_id="Cube_01")
        # TCP path sends the command
        assert result.success is True
        broadcaster.send_command.assert_called_once()
        cmd = broadcaster.send_command.call_args[0][0]
        # TCP path does NOT include precomputed_candidates
        assert "precomputed_candidates" not in cmd.get("parameters", {})

    def test_skips_graspnet_when_disabled(self):
        """When GRASPNET_ENABLED=False, _grasp_via_graspnet is never called."""
        broadcaster = MagicMock()
        broadcaster.send_command.return_value = True
        with patch("config.Servers.GRASPNET_ENABLED", False), \
             patch("config.ROS.ROS_ENABLED", False), \
             patch("operations.GraspOperations._grasp_via_graspnet") as mock_gvg, \
             patch("operations.GraspOperations._get_command_broadcaster", return_value=broadcaster):
            grasp_object(robot_id="Robot1", object_id="Cube_01")
        mock_gvg.assert_not_called()
