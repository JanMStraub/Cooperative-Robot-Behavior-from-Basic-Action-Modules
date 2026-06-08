import time
import pytest

from ros2.ROSBridge import ROSBridge

# Configure logging
from core.LoggingSetup import get_logger

logger = get_logger(__name__)


def _ros_bridge_available() -> bool:
    """Check whether the ROS bridge is reachable."""
    bridge = ROSBridge.get_instance()
    return bridge.connect(timeout=3.0)


pytestmark = pytest.mark.skipif(
    not _ros_bridge_available(),
    reason="ROS bridge unavailable (Docker not running on port 5020)",
)


def test_dual_robot_connection():
    """Test connection to ROS bridge and verify multi-robot support."""
    bridge = ROSBridge.get_instance()

    if not bridge.connect(timeout=10.0):
        pytest.fail(
            "Failed to connect to ROS bridge on port 5020. "
            "Make sure Docker services are running: cd ros_unity_integration && ./start_ros_endpoint.sh"
        )

    logger.info("Successfully connected to ROS bridge")

    # Verify ping works
    if not bridge.ping():
        pytest.fail("Ping failed - bridge not responsive")

    logger.info("Bridge is responsive")


def test_robot1_motion():
    bridge = ROSBridge.get_instance()

    # Unity world coords; Robot1 at (-0.475, 0, 0), rotation 0°
    # local = (-0.2 - (-0.475), 0.15, 0) = (0.275, 0.15, 0)
    target_position = {"x": -0.2, "y": 0.15, "z": 0.0}

    logger.info(f"Requesting motion for Robot1 to {target_position}")
    result = bridge.plan_and_execute(
        position=target_position, robot_id="Robot1", planning_time=10.0
    )

    if result and result.get("success"):
        logger.info(f"Robot1 motion successful!")
        logger.info(f"  Planning time: {result.get('planning_time', 0):.2f}s")
        logger.info(f"  Trajectory points: {result.get('trajectory_points', 0)}")
        logger.info(f"  Status: {result.get('status')}")
    else:
        error = result.get("error", "Unknown error") if result else "No response"
        pytest.fail(f"Robot1 motion failed: {error}")


def test_robot2_motion():
    bridge = ROSBridge.get_instance()

    # Unity world coords; Robot2 at (0.475, 0, 0), rotation 180°
    # local: translate (-0.275), rotate 180° → (0.275, 0.15, 0)
    target_position = {"x": 0.2, "y": 0.15, "z": 0.0}

    logger.info(f"Requesting motion for Robot2 to {target_position}")
    result = bridge.plan_and_execute(
        position=target_position, robot_id="Robot2", planning_time=10.0
    )

    if result and result.get("success"):
        logger.info(f"Robot2 motion successful!")
        logger.info(f"  Planning time: {result.get('planning_time', 0):.2f}s")
        logger.info(f"  Trajectory points: {result.get('trajectory_points', 0)}")
        logger.info(f"  Status: {result.get('status')}")
    else:
        error = result.get("error", "Unknown error") if result else "No response"
        pytest.fail(f"Robot2 motion failed: {error}")


def test_simultaneous_motion():
    bridge = ROSBridge.get_instance()

    robot1_target = {"x": -0.1, "y": 0.15, "z": 0.0}
    robot2_target = {"x": 0.1, "y": 0.15, "z": 0.0}

    result1 = bridge.plan_and_execute(
        position=robot1_target, robot_id="Robot1", planning_time=10.0
    )

    time.sleep(0.5)

    result2 = bridge.plan_and_execute(
        position=robot2_target, robot_id="Robot2", planning_time=10.0
    )

    success1 = result1 and result1.get("success")
    success2 = result2 and result2.get("success")

    if success1 and success2:
        logger.info("Both robots executed successfully!")
        if result1:
            logger.info(f"  Robot1 planning: {result1.get('planning_time', 0):.2f}s")
        if result2:
            logger.info(f"  Robot2 planning: {result2.get('planning_time', 0):.2f}s")
    else:
        errors = []
        if not success1:
            errors.append(
                f"Robot1: {result1.get('error', 'Unknown') if result1 else 'No response'}"
            )
        if not success2:
            errors.append(
                f"Robot2: {result2.get('error', 'Unknown') if result2 else 'No response'}"
            )
        pytest.fail(f"Simultaneous motion failed - {'; '.join(errors)}")


def test_gripper_control():
    logger.info("=" * 60)
    logger.info("Test 5: Dual-Robot Gripper Control")
    logger.info("=" * 60)

    bridge = ROSBridge.get_instance()

    # Close both grippers
    logger.info("Robot1: Closing gripper")
    result1 = bridge.control_gripper(position=0.0, robot_id="Robot1")

    time.sleep(1.0)

    logger.info("Robot2: Closing gripper")
    result2 = bridge.control_gripper(position=0.0, robot_id="Robot2")

    time.sleep(3.0)

    # Test Robot1 gripper
    logger.info("Robot1: Opening gripper")
    result3 = bridge.control_gripper(position=0.014, robot_id="Robot1")

    time.sleep(1.0)

    # Test Robot2 gripper
    logger.info("Robot2: Opening gripper")
    result4 = bridge.control_gripper(position=0.014, robot_id="Robot2")

    success = all(
        [
            result1 and result1.get("success"),
            result2 and result2.get("success"),
            result3 and result3.get("success"),
            result4 and result4.get("success"),
        ]
    )

    if success:
        logger.info("All gripper commands successful!")
    else:
        pytest.fail("Some gripper commands failed")


def test_get_joint_states():
    logger.info("=" * 60)
    logger.info("Test 6: Get Joint States")
    logger.info("=" * 60)

    bridge = ROSBridge.get_instance()

    # Get Robot1 joint states
    logger.info("Requesting Robot1 joint states")
    result1 = bridge.get_current_pose(robot_id="Robot1")

    if result1 and result1.get("success"):
        logger.info(f"Robot1 joint states received:")
        logger.info(f"  Joint names: {result1.get('joint_names')}")
        logger.info(
            f"  Joint positions: {[f'{p:.3f}' for p in result1.get('joint_positions', [])]}"
        )
    else:
        logger.warning(
            f"Robot1 joint states not available: {result1.get('error') if result1 else 'No response'}"
        )

    # Get Robot2 joint states
    logger.info("Requesting Robot2 joint states")
    result2 = bridge.get_current_pose(robot_id="Robot2")

    if result2 and result2.get("success"):
        logger.info(f"Robot2 joint states received:")
        logger.info(f"  Joint names: {result2.get('joint_names')}")
        logger.info(
            f"  Joint positions: {[f'{p:.3f}' for p in result2.get('joint_positions', [])]}"
        )
    else:
        logger.warning(
            f"Robot2 joint states not available: {result2.get('error') if result2 else 'No response'}"
        )

    # Success if at least one robot reports joint states
    success = (result1 and result1.get("success")) or (
        result2 and result2.get("success")
    )

    if success:
        logger.info("Joint state retrieval successful")
    else:
        pytest.fail("No joint states available - Unity may not be running")
