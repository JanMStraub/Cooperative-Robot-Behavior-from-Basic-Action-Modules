#!/usr/bin/env python3
"""
Test Dual-Robot ROS Control
============================

Tests the multi-robot MoveIt setup with separate instances for Robot1 and Robot2.

Requirements:
- Docker ROS services running: cd ros_unity_integration && ./start_ros_endpoint.sh
- Unity simulation running with both Robot1 and Robot2
- Both robots publishing joint states to namespaced topics

Test flow:
1. Connect to ROS bridge
2. Send commands to Robot1 (should route to moveit_robot1 container)
3. Send commands to Robot2 (should route to moveit_robot2 container)
4. Verify both robots move independently
"""

import sys
import os
import time
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
    logger.info("=" * 60)
    logger.info("Test 1: Verify ROS Bridge Connection")
    logger.info("=" * 60)

    # Use singleton instance to ensure connection persists across tests
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
    """Test motion planning for Robot1."""
    logger.info("=" * 60)
    logger.info("Test 2: Robot1 Motion Planning")
    logger.info("=" * 60)

    bridge = ROSBridge.get_instance()

    # Test position for Robot1 (Unity world coordinates - will be auto-transformed)
    # Send: world (-0.2, 0.15, 0.0)
    # Robot1 at world (-0.475, 0, 0), rotation 0°
    # Transform: local = (-0.2 - (-0.475), 0.15, 0) = (0.275, 0.15, 0)
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
    """Test motion planning for Robot2."""
    logger.info("=" * 60)
    logger.info("Test 3: Robot2 Motion Planning")
    logger.info("=" * 60)

    bridge = ROSBridge.get_instance()

    # Test position for Robot2 (Unity world coordinates - will be auto-transformed)
    # Send: world (0.2, 0.15, 0)
    # Robot2 at world (0.475, 0, 0), rotation 180°
    # Transform: translate (0.2 - 0.475 = -0.275), rotate 180° -> (0.275, 0.15, 0)
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
    """Test simultaneous motion planning for both robots."""
    logger.info("=" * 60)
    logger.info("Test 4: Simultaneous Dual-Robot Motion")
    logger.info("=" * 60)

    bridge = ROSBridge.get_instance()

    # Target positions for both robots
    robot1_target = {"x": -0.1, "y": 0.15, "z": 0.0}
    robot2_target = {"x": 0.1, "y": 0.15, "z": 0.0}

    logger.info("Sending simultaneous commands to both robots")

    # Send Robot1 command
    logger.info(f"Robot1 -> {robot1_target}")
    result1 = bridge.plan_and_execute(
        position=robot1_target, robot_id="Robot1", planning_time=10.0
    )

    # Small delay to avoid overwhelming the system
    time.sleep(0.5)

    # Send Robot2 command
    logger.info(f"Robot2 -> {robot2_target}")
    result2 = bridge.plan_and_execute(
        position=robot2_target, robot_id="Robot2", planning_time=10.0
    )

    # Check results
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
    """Test gripper control for both robots."""
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
    """Test getting joint states for both robots."""
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


def run_all_tests():
    """Run all dual-robot tests."""
    logger.info("\n" + "=" * 60)
    logger.info("DUAL-ROBOT ROS CONTROL TEST SUITE")
    logger.info("=" * 60 + "\n")

    results = {}

    # Test 1: Connection
    results["connection"] = test_dual_robot_connection()
    if not results["connection"]:
        logger.error("\nConnection test failed - aborting remaining tests")
        return results

    time.sleep(2)

    # Test 2: Robot1 motion
    results["robot1_motion"] = test_robot1_motion()
    time.sleep(3)

    # Test 3: Robot2 motion
    results["robot2_motion"] = test_robot2_motion()
    time.sleep(3)

    # Test 4: Simultaneous motion
    results["simultaneous"] = test_simultaneous_motion()
    time.sleep(3)

    # Test 5: Gripper control
    results["gripper"] = test_gripper_control()
    time.sleep(2)

    # Test 6: Joint states
    results["joint_states"] = test_get_joint_states()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    for test_name, success in results.items():
        status = "PASS" if success else "FAIL"
        logger.info(f"  {test_name:20s}: {status}")

    total = len(results)
    passed = sum(results.values())
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    logger.info("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    try:
        results = run_all_tests()

        # Exit with appropriate code
        all_passed = all(results.values())
        sys.exit(0 if all_passed else 1)

    except KeyboardInterrupt:
        logger.info("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
