"""
Integration tests for _grasp_via_vgn orchestration
=====================================================

Tests the full VGN fast-path inside ``grasp_object`` by mocking every
external collaborator so no live Unity, model checkpoint, or stereo cameras are
required.  The goal is to verify that the orchestration logic correctly
chains the components and falls back to the geometric pipeline when any
step fails.

Coverage:
- Happy path: point cloud → detection → VGN → transform → broadcaster
- Fallback when VGN model is unavailable
- Fallback when point cloud generation fails
- Fallback when VGN returns no candidates
- Fallback when frame transform produces no valid poses
- Broadcaster unavailable → error result (not fallback)
- Broadcaster send failure → error result
- Pre-grasp position computed along approach direction
- Correct number of candidates forwarded to Unity
- grasp_object routes to VGN path when VGN_ENABLED=true
- grasp_object falls back to TCP path when VGN returns None
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from operations.GraspOperations import _grasp_via_vgn, grasp_object
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
    """Return minimal grasp dicts as returned by the VGN pipeline."""
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


class _VGNPatch:
    """Context manager that patches all collaborators of _grasp_via_vgn."""

    def __init__(
        self,
        vgn_available: bool = True,
        pc_result=_UNSET,
        detect_result=None,
        raw_grasps=_UNSET,
        world_grasps_result=_UNSET,
        broadcaster_available: bool = True,
        send_command_return: bool = True,
    ):
        self.vgn_available = vgn_available
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
        # VGNClient mock — patch in its own module
        self.mock_client = MagicMock()
        self.mock_client.is_available.return_value = self.vgn_available
        self.mock_client.predict_grasps.return_value = self.raw_grasps

        client_cls = patch(
            "operations.VGNClient.VGNClient",
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
            "operations.GraspFrameTransform.transform_grasp_poses_to_unity",
            return_value=self.world_grasps_result,
        )
        self._patches.append(transform_patch.start())

        # Mock unified image storage (non-fatal, returns None)
        storage_mock = MagicMock()
        storage_mock.get_latest_image.return_value = None
        storage_patch = patch(
            "core.Imports.get_unified_image_storage",
            return_value=storage_mock,
        )
        self._patches.append(storage_patch.start())

        # Mock CommandBroadcaster
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
# _grasp_via_vgn happy path
# ---------------------------------------------------------------------------


class TestGraspViaVGNHappyPath:
    """Tests for the successful execution path."""

    def test_returns_success_result(self):
        """Happy path returns a successful OperationResult."""
        with _VGNPatch() as ctx:
            result = _grasp_via_vgn(
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

    def test_result_contains_vgn_candidate_count(self):
        """Result dict includes how many VGN candidates were forwarded."""
        n = 5
        with _VGNPatch(raw_grasps=_sample_grasps(n), world_grasps_result=_world_grasps(n)) as ctx:
            result = _grasp_via_vgn(
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach="auto",
                use_advanced_planning=True,
                pre_grasp_distance=0.0,
                enable_retreat=True,
                retreat_distance=0.0,
                request_id=2,
            )
        assert result.result["vgn_candidates"] == n

    def test_command_sent_with_precomputed_candidates(self):
        """grasp_object command payload contains precomputed_candidates list."""
        n = 3
        with _VGNPatch(raw_grasps=_sample_grasps(n), world_grasps_result=_world_grasps(n)) as ctx:
            _grasp_via_vgn(
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
        """Pre-grasp position is grasp position - approach_direction * hover.

        approach_direction points *toward* the object (VGN convention), so the
        pre-grasp hover position must be placed *against* the approach direction
        (i.e. subtract), not along it.  Adding would place the arm on the far
        side of the object and cause a collision.
        """
        world = [{
            "position": [0.0, 0.0, 0.5],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "score": 0.9,
            "width": 0.08,
            "approach_direction": [0.0, 1.0, 0.0],  # +Y
        }]
        hover = 0.1
        with _VGNPatch(raw_grasps=_sample_grasps(1), world_grasps_result=world) as ctx:
            _grasp_via_vgn(
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
            # pre_grasp = grasp - approach * hover  (step BACK from object)
            assert cand["pre_grasp_position"]["x"] == pytest.approx(0.0)
            assert cand["pre_grasp_position"]["y"] == pytest.approx(-0.1)
            assert cand["pre_grasp_position"]["z"] == pytest.approx(0.5)

    def test_robot_id_and_object_id_in_command(self):
        """Command payload carries correct robot_id and object_id."""
        with _VGNPatch() as ctx:
            _grasp_via_vgn(
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
        with _VGNPatch() as ctx:
            _grasp_via_vgn(
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


class TestGraspViaVGNFallback:
    """Tests for scenarios where _grasp_via_vgn returns None."""

    def test_returns_none_when_vgn_unavailable(self):
        """Model not available → None (no exception)."""
        with _VGNPatch(vgn_available=False):
            result = _grasp_via_vgn(
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
        with _VGNPatch(pc_result=_pc_failure()):
            result = _grasp_via_vgn(
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

    def test_returns_none_when_vgn_returns_no_candidates(self):
        """VGN returns empty list → None."""
        with _VGNPatch(raw_grasps=[]):
            result = _grasp_via_vgn(
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

    def test_returns_none_when_vgn_returns_none(self):
        """VGN returns None (e.g. too few points) → None."""
        with _VGNPatch(raw_grasps=None):
            result = _grasp_via_vgn(
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
        with _VGNPatch(world_grasps_result=[]):
            result = _grasp_via_vgn(
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


class TestGraspViaVGNErrors:
    """Tests for scenarios where _grasp_via_vgn returns an error OperationResult."""

    def test_error_when_broadcaster_unavailable(self):
        """Missing broadcaster → error result (not None)."""
        with _VGNPatch(broadcaster_available=False):
            result = _grasp_via_vgn(
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
        with _VGNPatch(send_command_return=False):
            result = _grasp_via_vgn(
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


class TestGraspObjectVGNRouting:
    """Tests that grasp_object correctly routes through the VGN path."""

    def test_uses_vgn_path_when_enabled_and_available(self):
        """When VGN_ENABLED and model is available, _grasp_via_vgn is called."""
        vgn_result = OperationResult.success_result({
            "command_sent": True,
            "robot_id": "Robot1",
            "object_id": "Cube_01",
            "request_id": 0,
            "vgn_candidates": 3,
        })
        with patch("config.Servers.VGN_ENABLED", True), \
             patch("config.ROS.ROS_ENABLED", False), \
             patch("operations.GraspOperations._grasp_via_vgn", return_value=vgn_result) as mock_vgn, \
             patch("operations.GraspOperations._get_command_broadcaster"):
            result = grasp_object(robot_id="Robot1", object_id="Cube_01")
        mock_vgn.assert_called_once()
        assert result.success is True
        assert result.result.get("vgn_candidates") == 3

    def test_falls_back_to_tcp_when_vgn_returns_none(self):
        """When _grasp_via_vgn returns None, geometric TCP path is used."""
        broadcaster = MagicMock()
        broadcaster.send_command.return_value = True
        with patch("config.Servers.VGN_ENABLED", True), \
             patch("config.ROS.ROS_ENABLED", False), \
             patch("operations.GraspOperations._grasp_via_vgn", return_value=None), \
             patch("operations.GraspOperations._get_command_broadcaster", return_value=broadcaster):
            result = grasp_object(robot_id="Robot1", object_id="Cube_01")
        # TCP path sends the command
        assert result.success is True
        broadcaster.send_command.assert_called_once()
        cmd = broadcaster.send_command.call_args[0][0]
        # TCP path does NOT include precomputed_candidates
        assert "precomputed_candidates" not in cmd.get("parameters", {})

    def test_skips_vgn_when_disabled(self):
        """When VGN_ENABLED=False, _grasp_via_vgn is never called."""
        broadcaster = MagicMock()
        broadcaster.send_command.return_value = True
        with patch("config.Servers.VGN_ENABLED", False), \
             patch("config.ROS.ROS_ENABLED", False), \
             patch("operations.GraspOperations._grasp_via_vgn") as mock_vgn, \
             patch("operations.GraspOperations._get_command_broadcaster", return_value=broadcaster):
            grasp_object(robot_id="Robot1", object_id="Cube_01")
        mock_vgn.assert_not_called()


# ---------------------------------------------------------------------------
# _grasp_via_vgn_with_ros helpers and tests
# ---------------------------------------------------------------------------


class _VGNROSPatch:
    """Context manager that patches all collaborators of _grasp_via_vgn_with_ros.

    Mirrors _VGNPatch but additionally mocks the ROS bridge and the
    follow-target helper so no live MoveIt or Unity connection is required.
    """

    def __init__(
        self,
        vgn_available: bool = True,
        pc_result=_UNSET,
        raw_grasps=_UNSET,
        world_grasps_result=_UNSET,
        pre_grasp_success: bool = True,
        descent_success: bool = True,
        gripper_success: bool = True,
    ):
        self.vgn_available = vgn_available
        self.pc_result = _pc_success() if pc_result is _UNSET else pc_result
        self.raw_grasps = _sample_grasps() if raw_grasps is _UNSET else raw_grasps
        self.world_grasps_result = _world_grasps() if world_grasps_result is _UNSET else world_grasps_result
        self.pre_grasp_success = pre_grasp_success
        self.descent_success = descent_success
        self.gripper_success = gripper_success

        self._patches = []
        self.mock_client = None
        self.mock_bridge = None

    def __enter__(self):
        # VGNClient mock
        self.mock_client = MagicMock()
        self.mock_client.is_available.return_value = self.vgn_available
        self.mock_client.predict_grasps.return_value = self.raw_grasps

        client_cls = patch(
            "operations.VGNClient.VGNClient",
            return_value=self.mock_client,
        )
        self._patches.append(client_cls.start())

        # generate_point_cloud mock
        pc_patch = patch(
            "operations.PointCloudOperations.generate_point_cloud",
            return_value=self.pc_result,
        )
        self._patches.append(pc_patch.start())

        # Frame transform mock
        transform_patch = patch(
            "operations.GraspFrameTransform.transform_grasp_poses_to_unity",
            return_value=self.world_grasps_result,
        )
        self._patches.append(transform_patch.start())

        # Image storage mock (non-fatal)
        storage_mock = MagicMock()
        storage_mock.get_latest_image.return_value = None
        storage_patch = patch(
            "core.Imports.get_unified_image_storage",
            return_value=storage_mock,
        )
        self._patches.append(storage_patch.start())

        # ROSBridge mock
        self.mock_bridge = MagicMock()
        self.mock_bridge.plan_and_execute.return_value = {"success": self.pre_grasp_success}
        self.mock_bridge.plan_cartesian_descent.return_value = {"success": self.descent_success}
        bridge_patch = patch(
            "ros2.ROSBridge.ROSBridge.get_instance",
            return_value=self.mock_bridge,
        )
        self._patches.append(bridge_patch.start())

        # Follow-target + gripper helper mock
        follow_patch = patch(
            "operations.GraspOperations._execute_grasp_with_follow_target",
            return_value=self.gripper_success,
        )
        self._patches.append(follow_patch.start())

        return self

    def __exit__(self, *args):
        import unittest.mock as _um
        _um.patch.stopall()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestGraspViaVGNWithROSHappyPath:
    """Tests for the successful execution path of _grasp_via_vgn_with_ros."""

    def test_returns_success_result(self):
        """Full pipeline succeeds → OperationResult with success=True."""
        with _VGNROSPatch() as ctx:
            from operations.GraspOperations import _grasp_via_vgn_with_ros
            result = _grasp_via_vgn_with_ros(
                bridge=ctx.mock_bridge,
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach="top",
                pre_grasp_distance=0.0,
                request_id=1,
                world_state=None,
            )
        assert result is not None
        assert result.success is True

    def test_status_is_vgn_ros_executed(self):
        """Result status key equals 'vgn_ros_executed'."""
        with _VGNROSPatch() as ctx:
            from operations.GraspOperations import _grasp_via_vgn_with_ros
            result = _grasp_via_vgn_with_ros(
                bridge=ctx.mock_bridge,
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach="top",
                pre_grasp_distance=0.0,
                request_id=2,
                world_state=None,
            )
        assert result.result["status"] == "vgn_ros_executed"

    def test_vgn_candidates_count_in_result(self):
        """result['vgn_candidates'] equals the number of world-frame poses."""
        n = 4
        with _VGNROSPatch(raw_grasps=_sample_grasps(n), world_grasps_result=_world_grasps(n)) as ctx:
            from operations.GraspOperations import _grasp_via_vgn_with_ros
            result = _grasp_via_vgn_with_ros(
                bridge=ctx.mock_bridge,
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach="auto",
                pre_grasp_distance=0.0,
                request_id=3,
                world_state=None,
            )
        assert result.result["vgn_candidates"] == n

    def test_plan_and_execute_called_with_vgn_orientation(self):
        """plan_and_execute receives the orientation from the top VGN candidate."""
        world = [{
            "position": [0.1, 0.2, 0.5],
            "rotation": [0.1, 0.2, 0.3, 0.9],
            "score": 0.95,
            "width": 0.08,
            "approach_direction": [0.0, -1.0, 0.0],
        }]
        with _VGNROSPatch(raw_grasps=_sample_grasps(1), world_grasps_result=world) as ctx:
            from operations.GraspOperations import _grasp_via_vgn_with_ros
            _grasp_via_vgn_with_ros(
                bridge=ctx.mock_bridge,
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach="top",
                pre_grasp_distance=0.0,
                request_id=4,
                world_state=None,
            )
        call_kwargs = ctx.mock_bridge.plan_and_execute.call_args.kwargs
        assert call_kwargs["orientation"] == {"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.9}

    def test_cartesian_descent_called_at_grasp_position(self):
        """plan_cartesian_descent is called at grasp position (not pre-grasp)."""
        world = [{
            "position": [0.3, 0.4, 0.5],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "score": 0.9,
            "width": 0.08,
            "approach_direction": [0.0, 1.0, 0.0],
        }]
        with _VGNROSPatch(raw_grasps=_sample_grasps(1), world_grasps_result=world) as ctx:
            from operations.GraspOperations import _grasp_via_vgn_with_ros
            _grasp_via_vgn_with_ros(
                bridge=ctx.mock_bridge,
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach="top",
                pre_grasp_distance=0.1,
                request_id=5,
                world_state=None,
            )
        descent_kwargs = ctx.mock_bridge.plan_cartesian_descent.call_args.kwargs
        assert descent_kwargs["position"] == {"x": 0.3, "y": 0.4, "z": 0.5}

    def test_follow_target_called_after_descent(self):
        """_execute_grasp_with_follow_target is called after Cartesian descent."""
        with _VGNROSPatch() as ctx:
            from operations.GraspOperations import _grasp_via_vgn_with_ros
            import operations.GraspOperations as go_module
            with patch.object(go_module, "_execute_grasp_with_follow_target", return_value=True) as mock_follow:
                _grasp_via_vgn_with_ros(
                    bridge=ctx.mock_bridge,
                    robot_id="Robot1",
                    object_id="Cube_01",
                    preferred_approach="top",
                    pre_grasp_distance=0.0,
                    request_id=6,
                    world_state=None,
                )
            mock_follow.assert_called_once()


# ---------------------------------------------------------------------------
# Fallback paths (return None)
# ---------------------------------------------------------------------------


class TestGraspViaVGNWithROSFallback:
    """Tests for scenarios where _grasp_via_vgn_with_ros returns None."""

    def test_returns_none_when_vgn_unavailable(self):
        """VGN model not available → None."""
        with _VGNROSPatch(vgn_available=False) as ctx:
            from operations.GraspOperations import _grasp_via_vgn_with_ros
            result = _grasp_via_vgn_with_ros(
                bridge=ctx.mock_bridge,
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach="top",
                pre_grasp_distance=0.0,
                request_id=10,
                world_state=None,
            )
        assert result is None

    def test_returns_none_when_point_cloud_fails(self):
        """Point cloud failure → None."""
        with _VGNROSPatch(pc_result=_pc_failure()) as ctx:
            from operations.GraspOperations import _grasp_via_vgn_with_ros
            result = _grasp_via_vgn_with_ros(
                bridge=ctx.mock_bridge,
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach="top",
                pre_grasp_distance=0.0,
                request_id=11,
                world_state=None,
            )
        assert result is None

    def test_returns_none_when_vgn_returns_no_candidates(self):
        """VGN returns empty list → None."""
        with _VGNROSPatch(raw_grasps=[]) as ctx:
            from operations.GraspOperations import _grasp_via_vgn_with_ros
            result = _grasp_via_vgn_with_ros(
                bridge=ctx.mock_bridge,
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach="top",
                pre_grasp_distance=0.0,
                request_id=12,
                world_state=None,
            )
        assert result is None

    def test_returns_none_when_frame_transform_empty(self):
        """Frame transform produces no valid poses → None."""
        with _VGNROSPatch(world_grasps_result=[]) as ctx:
            from operations.GraspOperations import _grasp_via_vgn_with_ros
            result = _grasp_via_vgn_with_ros(
                bridge=ctx.mock_bridge,
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach="top",
                pre_grasp_distance=0.0,
                request_id=13,
                world_state=None,
            )
        assert result is None

    def test_returns_none_when_pre_grasp_move_fails(self):
        """MoveIt pre-grasp planning failure → None (arm has not moved)."""
        with _VGNROSPatch(pre_grasp_success=False) as ctx:
            from operations.GraspOperations import _grasp_via_vgn_with_ros
            result = _grasp_via_vgn_with_ros(
                bridge=ctx.mock_bridge,
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach="top",
                pre_grasp_distance=0.0,
                request_id=14,
                world_state=None,
            )
        assert result is None

    def test_returns_none_when_descent_fails(self):
        """MoveIt Cartesian descent failure → None (arm at pre-grasp, not target)."""
        with _VGNROSPatch(descent_success=False) as ctx:
            from operations.GraspOperations import _grasp_via_vgn_with_ros
            result = _grasp_via_vgn_with_ros(
                bridge=ctx.mock_bridge,
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach="top",
                pre_grasp_distance=0.0,
                request_id=15,
                world_state=None,
            )
        assert result is None


# ---------------------------------------------------------------------------
# Error results (definitive failures after arm has moved)
# ---------------------------------------------------------------------------


class TestGraspViaVGNWithROSErrors:
    """Tests for scenarios that return an error OperationResult (not None)."""

    def test_error_when_gripper_close_fails(self):
        """Arm descended but gripper close failed → GRIPPER_CLOSE_FAILED error result."""
        with _VGNROSPatch(gripper_success=False) as ctx:
            from operations.GraspOperations import _grasp_via_vgn_with_ros
            result = _grasp_via_vgn_with_ros(
                bridge=ctx.mock_bridge,
                robot_id="Robot1",
                object_id="Cube_01",
                preferred_approach="top",
                pre_grasp_distance=0.0,
                request_id=20,
                world_state=None,
            )
        assert result is not None
        assert result.success is False
        assert result.error["code"] == "GRIPPER_CLOSE_FAILED"


# ---------------------------------------------------------------------------
# Routing tests via grasp_object() with both ROS and VGN enabled
# ---------------------------------------------------------------------------


class TestGraspObjectRoutingWithBothEnabled:
    """Tests that grasp_object routes correctly when both VGN_ENABLED and ROS are on."""

    def _make_world_state(self):
        """Return a minimal WorldState mock that satisfies grasp_object's resolution logic."""
        ws = MagicMock()
        ws.get_object_position.return_value = (0.3, 0.1, 0.4)
        ws.get_object_dimensions.return_value = None  # forces position-only ROS path
        ws.get_robot_state.return_value = None
        ws._objects = {"Cube_01": MagicMock()}
        return ws

    def test_vgn_ros_path_attempted_first(self):
        """When both enabled, _grasp_via_vgn_with_ros is called before geometric ROS."""
        vgn_ros_result = OperationResult.success_result({
            "robot_id": "Robot1",
            "object_id": "Cube_01",
            "request_id": 0,
            "vgn_candidates": 3,
            "status": "vgn_ros_executed",
        })
        bridge_mock = MagicMock()
        bridge_mock.is_connected = True
        world_state = self._make_world_state()

        with patch("config.ROS.ROS_ENABLED", True), \
             patch("config.ROS.DEFAULT_CONTROL_MODE", "ros"), \
             patch("config.Servers.VGN_ENABLED", True), \
             patch("ros2.ROSBridge.ROSBridge") as mock_ros_cls, \
             patch("core.Imports.get_world_state", return_value=world_state), \
             patch("operations.GraspOperations._grasp_via_vgn_with_ros",
                   return_value=vgn_ros_result) as mock_vgn_ros:
            mock_ros_cls.get_instance.return_value = bridge_mock
            result = grasp_object(robot_id="Robot1", object_id="Cube_01")

        mock_vgn_ros.assert_called_once()
        assert result.result["status"] == "vgn_ros_executed"

    def test_falls_back_to_geometric_ros_when_vgn_ros_returns_none(self):
        """When _grasp_via_vgn_with_ros returns None, geometric ROS is used."""
        bridge_mock = MagicMock()
        bridge_mock.is_connected = True
        bridge_mock.plan_and_execute.return_value = {"success": True}
        bridge_mock.plan_cartesian_descent.return_value = {"success": True}
        bridge_mock.control_gripper.return_value = {"success": True}
        world_state = self._make_world_state()

        with patch("config.ROS.ROS_ENABLED", True), \
             patch("config.ROS.DEFAULT_CONTROL_MODE", "ros"), \
             patch("config.Servers.VGN_ENABLED", True), \
             patch("ros2.ROSBridge.ROSBridge") as mock_ros_cls, \
             patch("core.Imports.get_world_state", return_value=world_state), \
             patch("operations.GraspOperations._grasp_via_vgn_with_ros",
                   return_value=None), \
             patch("operations.GraspOperations._grasp_via_ros_position_only",
                   return_value=(OperationResult.success_result({"status": "ros_executed"}), False)) as mock_geo:
            mock_ros_cls.get_instance.return_value = bridge_mock
            result = grasp_object(robot_id="Robot1", object_id="Cube_01")

        mock_geo.assert_called_once()
        assert result.result["status"] == "ros_executed"

    def test_vgn_unity_path_when_ros_disabled(self):
        """When ROS is off, _grasp_via_vgn (not _with_ros) is called."""
        vgn_result = OperationResult.success_result({
            "command_sent": True,
            "robot_id": "Robot1",
            "object_id": "Cube_01",
            "request_id": 0,
            "vgn_candidates": 2,
        })
        with patch("config.ROS.ROS_ENABLED", False), \
             patch("config.Servers.VGN_ENABLED", True), \
             patch("operations.GraspOperations._grasp_via_vgn",
                   return_value=vgn_result) as mock_vgn, \
             patch("operations.GraspOperations._grasp_via_vgn_with_ros") as mock_vgn_ros:
            result = grasp_object(robot_id="Robot1", object_id="Cube_01")

        mock_vgn.assert_called_once()
        mock_vgn_ros.assert_not_called()
        assert result.success is True
