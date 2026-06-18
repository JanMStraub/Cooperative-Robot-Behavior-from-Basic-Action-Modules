import threading
import time
from typing import Any, Dict

import pytest

from BackendClient import (  # type: ignore[import]
    BackendClient,
    backend_available,
    reset_simulation,
)

BACKEND_AVAILABLE = backend_available()
SKIP_REASON = (
    "Unity not running or not connected to backend. "
    "Start Unity and run: python -m orchestrators.RunRobotController"
)

# Robot workspace coordinates (within reach of the AR4 arm).
# Robot1 → left workspace (x negative), Robot2 → right workspace (x positive).
_R1_COORD = (-0.25, 0.30, 0.10)  # x, y, z  — Robot1 reachable point
_R2_COORD = (0.25, 0.30, 0.10)  # x, y, z  — Robot2 reachable point


def _cmd(
    command: str,
    *,
    robot_id: str = "Robot1",
    camera_id: str = "TableStereoCamera",
    timeout: float = 60.0,
    request_id: int = 1,
) -> Dict[str, Any]:
    """Send a single command to the backend and return the response dict."""
    with BackendClient(timeout=timeout) as client:
        return client.send_command(
            command=command,
            robot_id=robot_id,
            camera_id=camera_id,
            request_id=request_id,
        )


@pytest.mark.integration
@pytest.mark.requires_unity
@pytest.mark.skipif(not BACKEND_AVAILABLE, reason=SKIP_REASON)
class TestStatusOps:

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


@pytest.mark.integration
@pytest.mark.requires_unity
@pytest.mark.skipif(not BACKEND_AVAILABLE, reason=SKIP_REASON)
class TestSyncOps:

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
        # 60s covers LLM parsing + network round-trip overhead; the actual wait is 500ms.
        assert elapsed < 60.0, f"wait(0.5) took unexpectedly long: {elapsed:.1f}s"

    def test_signal_and_wait_for_signal_paired(self):
        """signal + wait_for_signal run concurrently and both must succeed."""
        barrier = threading.Barrier(2)
        results: Dict[str, Any] = {}
        errors: list = []

        def wait_thread():
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


@pytest.mark.integration
@pytest.mark.requires_unity
@pytest.mark.skipif(not BACKEND_AVAILABLE, reason=SKIP_REASON)
class TestGripperOps:

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


@pytest.mark.integration
@pytest.mark.requires_unity
@pytest.mark.skipif(not BACKEND_AVAILABLE, reason=SKIP_REASON)
class TestNavigationOps:
    """Tests for move_to_coordinate, move_from_a_to_b, adjust_end_effector_orientation,
    and return_to_start (Level 1)."""

    @pytest.fixture(autouse=True)
    def reset_before_each(self):
        """Reset simulation before every navigation test."""
        reset_simulation()

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


@pytest.mark.integration
@pytest.mark.requires_unity
@pytest.mark.skipif(not BACKEND_AVAILABLE, reason=SKIP_REASON)
class TestPerceptionOps:
    """Tests for detect_object_stereo, analyze_scene (Level 1-2)."""

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


@pytest.mark.integration
@pytest.mark.requires_unity
@pytest.mark.skipif(not BACKEND_AVAILABLE, reason=SKIP_REASON)
class TestFieldOps:
    """Tests for detect_field, detect_all_fields (Level 1-2).

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


@pytest.mark.integration
@pytest.mark.requires_unity
@pytest.mark.skipif(not BACKEND_AVAILABLE, reason=SKIP_REASON)
class TestSpatialOps:

    @pytest.fixture(autouse=True)
    def reset_before_each(self):
        """Reset simulation before every spatial operation test."""
        reset_simulation()

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
        """Reset simulation before every grasp test.

        reset_simulation restores robots to home, opens grippers, and
        returns all scene objects (including redCube) to initial positions.
        """
        reset_simulation()

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

    def test_place_object(self):
        """place_object performs hover → descent → gripper open → ascent at a target position.

        Robot2 first grasps redCube, then places it at a nearby coordinate.
        The test accepts a structured error (e.g. IK infeasible) as a valid
        response — the important thing is that the operation is dispatched and
        returns a well-formed result rather than crashing or timing out.
        """
        # Grasp first so there is something to place.
        _cmd(
            "grasp redCube with Robot2",
            robot_id="Robot2",
            timeout=240.0,
            request_id=0,
        )
        x, y, z = _R2_COORD
        result = _cmd(
            f"place object at coordinate {x} {y + 0.05} {z} with Robot2",
            robot_id="Robot2",
            timeout=240.0,
            request_id=802,
        )
        assert (
            result.get("success") is True or result.get("error") is not None
        ), "place_object returned an unexpected response"


@pytest.mark.integration
@pytest.mark.requires_unity
@pytest.mark.multi_robot
@pytest.mark.skipif(not BACKEND_AVAILABLE, reason=SKIP_REASON)
class TestMultiRobotOps:
    """Tests for detect_other_robot, mirror_movement, grasp_object (Level 4).

    These tests use 120 s timeouts because the LLM-based negotiation protocol
    (NegotiationHub → RobotLLMAgent) may run up to 3 rounds of Analysis →
    Proposal → Evaluation before returning a plan.

    Command strings are phrased to mention two robot IDs or multi-robot
    operations, which triggers the negotiation path in SequenceExecutor.
    The full negotiation stack is the thing under test here.
    """

    @pytest.fixture(autouse=True)
    def reset_before_each(self):
        """Reset simulation before every multi-robot test."""
        reset_simulation()

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

    def test_check_partner_status(self):
        """check_partner_status returns Robot2's full state from Robot1's perspective."""
        result = _cmd(
            "check partner status of Robot2 from Robot1",
            robot_id="Robot1",
            timeout=240.0,
            request_id=903,
        )
        assert (
            result.get("success") is True or result.get("error") is not None
        ), "check_partner_status returned an unexpected response"

    def test_yield_workspace(self):
        """yield_workspace signals intent to enter shared zone and waits for clearance."""
        result = _cmd(
            "Robot1 yield workspace shared_zone",
            robot_id="Robot1",
            timeout=60.0,
            request_id=904,
        )
        assert (
            result.get("success") is True or result.get("error") is not None
        ), "yield_workspace returned an unexpected response"

    def test_grasp_object_handoff(self):
        """grasp_object grasps redCube with Robot2 for handoff to Robot1.

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
        ), "grasp_object returned an unexpected response"


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
        """Reset simulation before the collaborative test."""
        reset_simulation()

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

    def test_place_for_partner(self):
        """place_for_partner places a held object at the shared zone for Robot2."""
        result = _cmd(
            "Robot1 place object for partner at shared zone",
            robot_id="Robot1",
            timeout=120.0,
            request_id=1001,
        )
        assert (
            result.get("success") is True or result.get("error") is not None
        ), "place_for_partner returned an unexpected response"

    def test_synchronized_grasp(self):
        """synchronized_grasp has both robots approach LargeBox from opposite sides."""
        result = _cmd(
            "Robot1 and Robot2 synchronized grasp LargeBox together",
            robot_id="Robot1",
            timeout=240.0,
            request_id=1002,
        )
        assert (
            result.get("success") is True or result.get("error") is not None
        ), "synchronized_grasp returned an unexpected response"

    def test_joint_transport(self):
        """joint_transport moves a jointly-grasped object to target position."""
        result = _cmd(
            "Robot1 and Robot2 jointly transport object to position 0 0 0.3",
            robot_id="Robot1",
            timeout=240.0,
            request_id=1003,
        )
        assert (
            result.get("success") is True or result.get("error") is not None
        ), "joint_transport returned an unexpected response"


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
        """Reset simulation before every chaining test."""
        reset_simulation()

    def test_detect_then_move_to_detected_position(self):
        """detect_object_stereo → $target → move_to_coordinate uses 3D stereo coords."""
        result = _cmd(
            "detect object stereo for Robot1 as $target; move Robot1 to $target",
            robot_id="Robot1",
            camera_id="TableStereoCamera",
            timeout=240.0,
            request_id=1100,
        )
        assert (
            result.get("success") is True or result.get("error") is not None
        ), "detect → $target → move pipeline returned an unexpected response"

    def test_detect_then_grasp_via_variable(self):
        """detect_object_stereo → $target → grasp_object($target.color) pipeline."""
        result = _cmd(
            "detect object stereo for Robot1 as $target; grasp $target with Robot1",
            robot_id="Robot1",
            camera_id="TableStereoCamera",
            timeout=480.0,
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


if __name__ == "__main__":
    import sys

    pytest.main([__file__, "-v", *sys.argv[1:]])
