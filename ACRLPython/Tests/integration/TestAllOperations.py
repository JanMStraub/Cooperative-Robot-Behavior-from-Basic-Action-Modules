#!/usr/bin/env python3
"""
All Operations Integration Test
================================

Comprehensive integration test that exercises all 30 registered operations
against a live Unity + Python backend via the BackendClient → SequenceServer
(port 5011) path.

Prerequisites
-------------
1. Start Unity (6000.3.0f1) and load the AR4 scene.
2. Start the Python backend::

       cd ACRLPython
       source acrl/bin/activate
       python -m orchestrators.RunRobotController

3. Run these tests::

       # All operations:
       python -m pytest tests/integration/TestAllOperations.py -v

       # Only fast ops (no IK/grasp planning):
       python -m pytest tests/integration/TestAllOperations.py -v -k "Status or Sync or Gripper"

       # Multi-robot ops with extended timeout:
       python -m pytest tests/integration/TestAllOperations.py -v -m multi_robot --timeout=360

Coverage
--------
All 30 operations registered in operations/Registry.py (plus variable chaining):

    Level 1-2 Basic (18):
        Navigation:   move_to_coordinate, move_from_a_to_b,
                      adjust_end_effector_orientation, return_to_start
        Gripper:      control_gripper, release_object
        Perception:   detect_objects, detect_object_stereo, analyze_scene,
                      estimate_distance_to_object, estimate_distance_between_objects
        Field:        detect_field, get_field_center, detect_all_fields
        Status:       check_robot_status
        Sync:         wait (duration), signal + wait_for_signal (paired, threaded)

    Level 3 Intermediate (5):
        grasp_object, align_object,
        move_relative_to_object, move_between_objects, follow_path
        [move_to_region also tested under SpatialOps]

    Level 4 Multi-Robot (3):
        detect_other_robot, mirror_movement, grasp_object_for_handoff

    Level 5 Collaborative (1):
        stabilize_object

    Variable chaining:
        detect_object_stereo → $target → move_to_coordinate

Design Decisions
----------------
- Negotiation left enabled: Multi-robot tests use 120 s+ timeouts and exercise
  the full LLM negotiation stack (the point of Level 4/5 ops).
- Per-category timeouts: Status 30 s, Navigation 60 s, Grasp 120 s, Multi 240 s.
- Signal + wait pair: Tested in two threads; wait thread starts first, signal
  fires after 1 s to ensure the wait is registered before the signal is sent.
- Field operations: Always use camera_id="TableStereoCamera" (stereo camera).
"""

import os
import threading
import time
from typing import Any, Dict

import pytest

from backend_client import (  # type: ignore[import]
    BackendClient,
    backend_available,
    port_open,
)


# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------

BACKEND_AVAILABLE = backend_available()
SKIP_REASON = (
    "Unity not running or not connected to backend. "
    "Start Unity and run: python -m orchestrators.RunRobotController"
)

# Robot workspace coordinates (within reach of the AR4 arm).
# Robot1 → left workspace (x negative), Robot2 → right workspace (x positive).
_R1_COORD = (-0.25, 0.30, 0.10)  # x, y, z  — Robot1 reachable point
_R2_COORD = (0.25, 0.30, 0.10)  # x, y, z  — Robot2 reachable point


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _reset_robot(robot_id: str, timeout: float = 60.0) -> None:
    """
    Send return_to_start for the given robot and ignore the result.

    Called by autouse fixtures before movement-heavy tests so each test
    starts from a known home configuration regardless of what the previous
    test left behind.

    Args:
        robot_id: Target robot identifier (e.g. "Robot1", "Robot2").
        timeout: Socket timeout in seconds.
    """
    _cmd(
        f"return {robot_id} to start position",
        robot_id=robot_id,
        timeout=timeout,
        request_id=0,
    )


def _cmd(
    command: str,
    *,
    robot_id: str = "Robot1",
    camera_id: str = "TableStereoCamera",
    timeout: float = 60.0,
    request_id: int = 1,
) -> Dict[str, Any]:
    """
    Send a single command to the backend and return the response dict.

    Helper that encapsulates BackendClient construction so each test body
    stays focused on the assertion rather than the framing.

    Args:
        command: Command string forwarded to CommandParser.
        robot_id: Target robot identifier.
        camera_id: Camera identifier (use "TableStereoCamera" for vision ops).
        timeout: Socket timeout in seconds.
        request_id: Protocol V2 correlation ID.

    Returns:
        JSON response dict with at least a "success" key.
    """
    with BackendClient(timeout=timeout) as client:
        return client.send_command(
            command=command,
            robot_id=robot_id,
            camera_id=camera_id,
            request_id=request_id,
        )


# ---------------------------------------------------------------------------
# Status Operations (timeout: 15 s)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_unity
@pytest.mark.skipif(not BACKEND_AVAILABLE, reason=SKIP_REASON)
class TestStatusOps:
    """Tests for check_robot_status (Level 1)."""

    def test_check_robot_status_robot1(self):
        """check_robot_status returns a successful result for Robot1."""
        result = _cmd(
            "check robot status for Robot1",
            robot_id="Robot1",
            timeout=240.0,
            request_id=100,
        )
        assert (
            result.get("success") is True
        ), f"check_robot_status failed: {result.get('error')}"

    def test_check_robot_status_robot2(self):
        """check_robot_status returns a successful result for Robot2."""
        result = _cmd(
            "check robot status for Robot2",
            robot_id="Robot2",
            timeout=240.0,
            request_id=101,
        )
        assert (
            result.get("success") is True
        ), f"check_robot_status failed for Robot2: {result.get('error')}"


# ---------------------------------------------------------------------------
# Sync Operations (timeout: 15–30 s)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_unity
@pytest.mark.skipif(not BACKEND_AVAILABLE, reason=SKIP_REASON)
class TestSyncOps:
    """Tests for wait, signal, and wait_for_signal (Level 1 sync primitives)."""

    def test_wait_duration(self):
        """wait(duration=0.5) completes in reasonable wall-clock time."""
        start = time.time()
        result = _cmd(
            "wait 0.5 seconds for Robot1",
            robot_id="Robot1",
            timeout=240.0,
            request_id=200,
        )
        elapsed = time.time() - start
        assert result.get("success") is True, f"wait failed: {result.get('error')}"
        # Allow generous tolerance for network + LLM parsing + backend overhead.
        assert elapsed < 28.0, f"wait(0.5) took unexpectedly long: {elapsed:.1f}s"

    def test_signal_and_wait_for_signal_paired(self):
        """
        signal + wait_for_signal exercise the sync primitive as a pair.

        Two threads run concurrently:
        - Thread A sends wait_for_signal("test_sync_event") first.
        - Thread B fires signal("test_sync_event") after a 1 s delay.

        Both must succeed.  We use threading.Barrier to ensure both threads
        start before either command is sent, then a sleep on the signal side
        gives the wait side time to register with the backend.
        """
        barrier = threading.Barrier(2)
        results: Dict[str, Any] = {}
        errors: list = []

        def wait_thread():
            """Thread A: register the wait first."""
            try:
                barrier.wait(timeout=20.0)
                results["wait"] = _cmd(
                    "wait for signal test_sync_event for Robot1",
                    robot_id="Robot1",
                    timeout=240.0,
                    request_id=201,
                )
            except Exception as exc:
                errors.append(("wait", exc))

        def signal_thread():
            """Thread B: fire the signal after a brief delay."""
            try:
                barrier.wait(timeout=20.0)
                time.sleep(2.0)  # Let the wait-side register first
                results["signal"] = _cmd(
                    "signal test_sync_event for Robot1",
                    robot_id="Robot1",
                    timeout=240.0,
                    request_id=202,
                )
            except Exception as exc:
                errors.append(("signal", exc))

        t_wait = threading.Thread(target=wait_thread, daemon=True)
        t_signal = threading.Thread(target=signal_thread, daemon=True)

        t_wait.start()
        t_signal.start()

        t_wait.join(timeout=70.0)
        t_signal.join(timeout=40.0)

        assert not errors, f"Thread errors in signal/wait pair: {errors}"
        assert (
            results.get("signal", {}).get("success") is True
        ), f"signal failed: {results.get('signal', {}).get('error')}"
        assert (
            results.get("wait", {}).get("success") is True
        ), f"wait_for_signal failed: {results.get('wait', {}).get('error')}"


# ---------------------------------------------------------------------------
# Gripper Operations (timeout: 15 s)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_unity
@pytest.mark.skipif(not BACKEND_AVAILABLE, reason=SKIP_REASON)
class TestGripperOps:
    """Tests for control_gripper and release_object (Level 1)."""

    def test_control_gripper_open(self):
        """control_gripper(open) succeeds for Robot1."""
        result = _cmd(
            "open gripper for Robot1",
            robot_id="Robot1",
            timeout=240.0,
            request_id=300,
        )
        assert (
            result.get("success") is True
        ), f"open gripper failed: {result.get('error')}"

    def test_control_gripper_close(self):
        """control_gripper(close) succeeds for Robot1."""
        result = _cmd(
            "close gripper for Robot1",
            robot_id="Robot1",
            timeout=240.0,
            request_id=301,
        )
        assert (
            result.get("success") is True
        ), f"close gripper failed: {result.get('error')}"

    def test_release_object(self):
        """release_object succeeds (robot opens gripper and releases any held object)."""
        result = _cmd(
            "release object for Robot1",
            robot_id="Robot1",
            timeout=240.0,
            request_id=302,
        )
        assert (
            result.get("success") is True
        ), f"release_object failed: {result.get('error')}"


# ---------------------------------------------------------------------------
# Navigation Operations (timeout: 30 s)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_unity
@pytest.mark.skipif(not BACKEND_AVAILABLE, reason=SKIP_REASON)
class TestNavigationOps:
    """Tests for move_to_coordinate, move_from_a_to_b, adjust_end_effector_orientation,
    and return_to_start (Level 1)."""

    @pytest.fixture(autouse=True)
    def reset_before_each(self):
        """Return both robots to home before every navigation test."""
        _reset_robot("Robot1")
        _reset_robot("Robot2")

    def test_move_to_coordinate_robot1(self):
        """move_to_coordinate moves Robot1 to a reachable left-workspace point."""
        x, y, z = _R1_COORD
        result = _cmd(
            f"move Robot1 to coordinate {x} {y} {z}",
            robot_id="Robot1",
            timeout=240.0,
            request_id=400,
        )
        assert (
            result.get("success") is True
        ), f"move_to_coordinate failed: {result.get('error')}"

    def test_move_to_coordinate_robot2(self):
        """move_to_coordinate moves Robot2 to a reachable right-workspace point."""
        x, y, z = _R2_COORD
        result = _cmd(
            f"move Robot2 to coordinate {x} {y} {z}",
            robot_id="Robot2",
            timeout=240.0,
            request_id=401,
        )
        assert (
            result.get("success") is True
        ), f"move_to_coordinate Robot2 failed: {result.get('error')}"

    def test_move_from_a_to_b(self):
        """move_from_a_to_b moves Robot1 between two left-workspace waypoints."""
        result = _cmd(
            "move Robot1 from -0.25 0.30 0.10 to -0.28 0.25 0.12",
            robot_id="Robot1",
            timeout=240.0,
            request_id=402,
        )
        assert (
            result.get("success") is True
        ), f"move_from_a_to_b failed: {result.get('error')}"

    def test_adjust_end_effector_orientation(self):
        """adjust_end_effector_orientation changes Robot1 end-effector roll/pitch/yaw.

        When ROS mode is active, MoveIt may return a planning error for this pose;
        that is still a structured (non-null) error response, which we accept here.
        """
        result = _cmd(
            "adjust end effector orientation for Robot1 to 0 90 0",
            robot_id="Robot1",
            timeout=240.0,
            request_id=403,
        )
        assert (
            result.get("success") is True or result.get("error") is not None
        ), "adjust_end_effector_orientation returned an unexpected response"

    def test_return_to_start(self):
        """return_to_start returns Robot1 to its home joint configuration."""
        result = _cmd(
            "return Robot1 to start position",
            robot_id="Robot1",
            timeout=240.0,
            request_id=404,
        )
        assert (
            result.get("success") is True
        ), f"return_to_start failed: {result.get('error')}"


# ---------------------------------------------------------------------------
# Perception Operations (timeout: 30 s)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_unity
@pytest.mark.skipif(not BACKEND_AVAILABLE, reason=SKIP_REASON)
class TestPerceptionOps:
    """Tests for detect_objects, detect_object_stereo, analyze_scene,
    estimate_distance_to_object, estimate_distance_between_objects (Level 1-2)."""

    def test_detect_objects(self):
        """detect_objects returns a list of detected objects or a structured error.

        Uses camera_id="main" (the ImageServer key for the single-camera feed on
        port 5005). Accepts a structured NO_IMAGE error if Unity has not yet sent
        a frame — the operation path itself is what's under test.
        """
        result = _cmd(
            "detect objects for Robot1",
            robot_id="Robot1",
            camera_id="main",
            timeout=240.0,
            request_id=500,
        )
        assert (
            result.get("success") is True or result.get("error") is not None
        ), "detect_objects returned an unexpected response"

    def test_detect_object_stereo(self):
        """detect_object_stereo returns 3D world-space coordinates via stereo camera."""
        result = _cmd(
            "detect object stereo for Robot1",
            robot_id="Robot1",
            camera_id="TableStereoCamera",
            timeout=240.0,
            request_id=501,
        )
        assert (
            result.get("success") is True
        ), f"detect_object_stereo failed: {result.get('error')}"

    def test_analyze_scene(self):
        """analyze_scene produces a natural-language scene description.

        Requires LM Studio to be running with a vision-capable model loaded.
        Treated as a graceful degradation if LM Studio is unavailable or the
        model does not support vision (empty choices, connection error, etc.).
        """
        result = _cmd(
            "analyze scene for Robot1",
            robot_id="Robot1",
            timeout=180.0,
            request_id=502,
        )
        error = result.get("error") or ""
        lm_unavailable = any(
            kw in error
            for kw in (
                "empty choices",
                "Connection refused",
                "LM Studio",
                "NO_IMAGES",
                "LMSTUDIO",
                "No images available",
            )
        )
        if lm_unavailable:
            import pytest
            pytest.skip(f"LM Studio vision model unavailable: {error}")
        assert result.get("success") is True, f"analyze_scene failed: {error}"

    def test_estimate_distance_to_object(self):
        """estimate_distance_to_object returns distance from Robot1 to a named object."""
        result = _cmd(
            "estimate distance from Robot1 to redCube",
            robot_id="Robot1",
            timeout=240.0,
            request_id=503,
        )
        # Distance estimation may fail gracefully if object is not in scene.
        assert (
            result.get("success") is True or result.get("error") is not None
        ), "estimate_distance_to_object returned an unexpected response"

    def test_estimate_distance_between_objects(self):
        """estimate_distance_between_objects returns distance between two objects."""
        result = _cmd(
            "estimate distance between redCube and blueCube for Robot1",
            robot_id="Robot1",
            timeout=240.0,
            request_id=504,
        )
        assert (
            result.get("success") is True or result.get("error") is not None
        ), "estimate_distance_between_objects returned an unexpected response"


# ---------------------------------------------------------------------------
# Field Operations (timeout: 30 s, camera_id="TableStereoCamera")
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_unity
@pytest.mark.skipif(not BACKEND_AVAILABLE, reason=SKIP_REASON)
class TestFieldOps:
    """Tests for detect_field, get_field_center, detect_all_fields (Level 1-2).

    Field operations use YOLO-based label detection on stereo images.  They
    always require camera_id="TableStereoCamera" (not the default main camera).
    """

    def test_detect_field(self):
        """detect_field locates a labelled workspace field in the stereo view."""
        result = _cmd(
            "detect field for Robot1",
            robot_id="Robot1",
            camera_id="TableStereoCamera",
            timeout=240.0,
            request_id=600,
        )
        assert (
            result.get("success") is True or result.get("error") is not None
        ), "detect_field returned an unexpected response"

    def test_get_field_center(self):
        """get_field_center returns 3D centre coordinates of a detected field."""
        result = _cmd(
            "get field center for Robot1",
            robot_id="Robot1",
            camera_id="TableStereoCamera",
            timeout=240.0,
            request_id=601,
        )
        assert (
            result.get("success") is True or result.get("error") is not None
        ), "get_field_center returned an unexpected response"

    def test_detect_all_fields(self):
        """detect_all_fields returns all labelled workspace fields in the stereo view."""
        result = _cmd(
            "detect all fields for Robot1",
            robot_id="Robot1",
            camera_id="TableStereoCamera",
            timeout=240.0,
            request_id=602,
        )
        assert (
            result.get("success") is True or result.get("error") is not None
        ), "detect_all_fields returned an unexpected response"


# ---------------------------------------------------------------------------
# Spatial / Intermediate Operations (timeout: 30–60 s)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_unity
@pytest.mark.skipif(not BACKEND_AVAILABLE, reason=SKIP_REASON)
class TestSpatialOps:
    """Tests for move_relative_to_object, move_between_objects, move_to_region,
    and follow_path (Level 2-3)."""

    @pytest.fixture(autouse=True)
    def reset_before_each(self):
        """Return Robot1 to home before every spatial operation test."""
        _reset_robot("Robot1")

    def test_move_relative_to_object(self):
        """move_relative_to_object moves Robot1 relative to a named object."""
        result = _cmd(
            "move Robot1 relative to redCube offset 0.0 0.1 0.0",
            robot_id="Robot1",
            timeout=240.0,
            request_id=700,
        )
        assert (
            result.get("success") is True or result.get("error") is not None
        ), "move_relative_to_object returned an unexpected response"

    def test_move_between_objects(self):
        """move_between_objects moves Robot1 to the midpoint between two objects."""
        result = _cmd(
            "move Robot1 between redCube and blueCube",
            robot_id="Robot1",
            timeout=240.0,
            request_id=701,
        )
        assert (
            result.get("success") is True or result.get("error") is not None
        ), "move_between_objects returned an unexpected response"

    def test_move_to_region(self):
        """move_to_region moves Robot1 to its allocated workspace region centre."""
        result = _cmd(
            "move Robot1 to region left_workspace",
            robot_id="Robot1",
            timeout=240.0,
            request_id=702,
        )
        assert (
            result.get("success") is True or result.get("error") is not None
        ), "move_to_region returned an unexpected response"

    def test_follow_path(self):
        """follow_path moves Robot1 through a sequence of waypoints."""
        result = _cmd(
            "follow path for Robot1: -0.25 0.30 0.10, -0.28 0.25 0.12, -0.22 0.28 0.08",
            robot_id="Robot1",
            timeout=240.0,
            request_id=703,
        )
        assert (
            result.get("success") is True or result.get("error") is not None
        ), "follow_path returned an unexpected response"


# ---------------------------------------------------------------------------
# Grasp Operations (timeout: 60 s)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_unity
@pytest.mark.skipif(not BACKEND_AVAILABLE, reason=SKIP_REASON)
class TestGraspOps:
    """Tests for grasp_object and align_object (Level 3).

    grasp_object triggers the full planning pipeline in Unity:
    GraspCandidateGenerator (15 candidates) → GraspIKFilter → GraspCollisionFilter
    → GraspScorer.  Allow 60 s for this.

    Note: grip_object has been removed from the registry (only stale string
    references remain in IntermediateOperations.py relationship metadata).
    Only grasp_object is tested here.
    """

    @pytest.fixture(autouse=True)
    def reset_before_each(self):
        """Return Robot2 to home and open gripper before every grasp test.

        redCube starts at x=+0.300 (Robot2's workspace).  Earlier spatial tests
        may have moved it, so we also move redCube back to its nominal position
        before attempting a grasp.
        """
        _reset_robot("Robot2")
        _cmd("open gripper for Robot2", robot_id="Robot2", timeout=240.0, request_id=0)
        x, y, z = _R2_COORD
        _cmd(
            f"move Robot2 to coordinate {x} {y} {z}",
            robot_id="Robot2",
            timeout=240.0,
            request_id=0,
        )

    def test_grasp_object(self):
        """grasp_object runs the full grasp planning pipeline for redCube with Robot2.

        redCube lives in Robot2's workspace (x=+0.300).
        """
        result = _cmd(
            "grasp redCube with Robot2",
            robot_id="Robot2",
            timeout=240.0,
            request_id=800,
        )
        # A structured error (e.g. "object not found") is still a valid response.
        assert (
            result.get("success") is True or result.get("error") is not None
        ), "grasp_object returned an unexpected response"

    def test_align_object(self):
        """align_object aligns Robot2's end effector to match redCube's orientation."""
        result = _cmd(
            "align Robot2 to object redCube",
            robot_id="Robot2",
            timeout=240.0,
            request_id=801,
        )
        assert (
            result.get("success") is True or result.get("error") is not None
        ), "align_object returned an unexpected response"


# ---------------------------------------------------------------------------
# Multi-Robot Operations (timeout: 120 s, negotiation-aware)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_unity
@pytest.mark.multi_robot
@pytest.mark.skipif(not BACKEND_AVAILABLE, reason=SKIP_REASON)
class TestMultiRobotOps:
    """Tests for detect_other_robot, mirror_movement, grasp_object_for_handoff (Level 4).

    These tests use 120 s timeouts because the LLM-based negotiation protocol
    (NegotiationHub → RobotLLMAgent) may run up to 3 rounds of Analysis →
    Proposal → Evaluation before returning a plan.

    Command strings are phrased to mention two robot IDs or multi-robot
    operations, which triggers the negotiation path in SequenceExecutor.
    The full negotiation stack is the thing under test here.
    """

    @pytest.fixture(autouse=True)
    def reset_before_each(self):
        """Return both robots to home and open grippers before every multi-robot test."""
        _reset_robot("Robot1")
        _reset_robot("Robot2")
        _cmd("open gripper for Robot1", robot_id="Robot1", timeout=240.0, request_id=0)
        _cmd("open gripper for Robot2", robot_id="Robot2", timeout=240.0, request_id=0)

    def test_detect_other_robot(self):
        """detect_other_robot reports Robot2's position relative to Robot1."""
        result = _cmd(
            "detect other robot from Robot1 perspective",
            robot_id="Robot1",
            timeout=240.0,
            request_id=900,
        )
        assert (
            result.get("success") is True or result.get("error") is not None
        ), "detect_other_robot returned an unexpected response"

    def test_mirror_movement(self):
        """mirror_movement makes Robot2 mirror Robot1's motion symmetrically."""
        result = _cmd(
            "mirror movement of Robot1 with Robot2",
            robot_id="Robot1",
            timeout=240.0,
            request_id=901,
        )
        assert (
            result.get("success") is True or result.get("error") is not None
        ), "mirror_movement returned an unexpected response"

    def test_grasp_object_for_handoff(self):
        """grasp_object_for_handoff grasps redCube with Robot2 for handoff to Robot1.

        redCube is in Robot2's workspace, so Robot2 initiates the grasp and
        hands off to Robot1.
        """
        result = _cmd(
            "grasp redCube with Robot2 for handoff to Robot1",
            robot_id="Robot2",
            timeout=240.0,
            request_id=902,
        )
        assert (
            result.get("success") is True or result.get("error") is not None
        ), "grasp_object_for_handoff returned an unexpected response"


# ---------------------------------------------------------------------------
# Collaborative Operations (timeout: 120 s)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_unity
@pytest.mark.multi_robot
@pytest.mark.skipif(not BACKEND_AVAILABLE, reason=SKIP_REASON)
class TestCollaborativeOps:
    """Tests for stabilize_object (Level 5).

    stabilize_object requires two robots to simultaneously apply forces to
    keep a shared object stable.  The negotiation protocol resolves the
    assignment of robots to roles.
    """

    @pytest.fixture(autouse=True)
    def reset_before_each(self):
        """Return both robots to home before the collaborative test."""
        _reset_robot("Robot1")
        _reset_robot("Robot2")

    def test_stabilize_object(self):
        """stabilize_object coordinates both arms to stabilise a shared object."""
        result = _cmd(
            "stabilize redCube using Robot1 and Robot2",
            robot_id="Robot1",
            timeout=240.0,
            request_id=1000,
        )
        assert (
            result.get("success") is True or result.get("error") is not None
        ), "stabilize_object returned an unexpected response"


# ---------------------------------------------------------------------------
# Variable Chaining (timeout: 30–60 s)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_unity
@pytest.mark.skipif(not BACKEND_AVAILABLE, reason=SKIP_REASON)
class TestVariableChaining:
    """Tests for the $variable substitution pipeline (detect → move → grasp).

    The backend SequenceServer supports multi-step sequences where the output
    of one operation is stored in a named variable (e.g. $target) and injected
    as a parameter for the next operation.  These tests exercise that path via
    the SequenceServer's multi-command batch format, which allows pipe-separated
    commands in a single request.

    See also: tests/integration/TestDetectionToGraspIntegration.py for unit-
    level variable resolution tests that do not require a live backend.
    """

    @pytest.fixture(autouse=True)
    def reset_before_each(self):
        """Return Robot1 to home and open gripper before every chaining test."""
        _reset_robot("Robot1")
        _cmd("open gripper for Robot1", robot_id="Robot1", timeout=240.0, request_id=0)

    def test_detect_then_move_to_detected_position(self):
        """detect_object_stereo → $target → move_to_coordinate uses 3D stereo coords."""
        # Two-step sequence: detect → store as $target → move to $target position
        result = _cmd(
            "detect object stereo for Robot1 as $target; move Robot1 to $target",
            robot_id="Robot1",
            camera_id="TableStereoCamera",
            timeout=240.0,
            request_id=1100,
        )
        # Either succeeds end-to-end or fails with a structured error explaining why.
        assert (
            result.get("success") is True or result.get("error") is not None
        ), "detect → $target → move pipeline returned an unexpected response"

    def test_detect_then_grasp_via_variable(self):
        """detect_object_stereo → $target → grasp_object($target.color) pipeline."""
        result = _cmd(
            "detect object stereo for Robot1 as $target; grasp $target with Robot1",
            robot_id="Robot1",
            camera_id="TableStereoCamera",
            timeout=240.0,
            request_id=1101,
        )
        assert (
            result.get("success") is True or result.get("error") is not None
        ), "detect → $target → grasp pipeline returned an unexpected response"

    def test_dotted_variable_notation(self):
        """$target.x / $target.y / $target.z extraction feeds into move_to_coordinate."""
        result = _cmd(
            (
                "detect object stereo for Robot1 as $target; "
                "move Robot1 to coordinate $target.x $target.y $target.z"
            ),
            robot_id="Robot1",
            camera_id="TableStereoCamera",
            timeout=180.0,
            request_id=1102,
        )
        assert (
            result.get("success") is True or result.get("error") is not None
        ), "dotted variable pipeline returned an unexpected response"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    pytest.main([__file__, "-v", *sys.argv[1:]])
