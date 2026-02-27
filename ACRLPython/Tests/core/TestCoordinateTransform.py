"""
Test Coordinate Transformation
================================

Verifies that world-to-local coordinate transformation works correctly
for dual-robot ROS integration.

The real _transform_world_to_local performs three steps:
  1. Translate: Unity world → robot-centered Unity coordinates
  2. Rotate: Apply robot's Y-axis rotation (for Robot2's 180° flip)
  3. Axis conversion: Unity (Y-up, left-handed) → ROS (Z-up, right-handed)
     Unity (X, Y, Z) → ROS (Z, -X, Y)
"""

import sys
import os
import math
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from utils.CoordinateTransforms import world_to_robot_frame_np, robot_to_world_frame_np


class MockROSMotionServer:
    """Mirrors the full _transform_world_to_local logic from ROSMotionClient.py."""

    ROBOT_BASE_TRANSFORMS = {
        "Robot1": {"position": (-0.475, 0.0, 0.0), "y_rotation": 0.0},
        "Robot2": {"position": (0.475, 0.0, 0.0), "y_rotation": 180.0},
    }

    def _transform_world_to_local(self, world_position: dict, robot_id: str) -> dict:
        """Transform Unity world coordinates to ROS base_link coordinates.

        Step 1: Translate (Unity world → robot-centered Unity)
        Step 2: Rotate around Y axis (for Robot2's 180° flip)
        Step 3: Axis convert Unity (Y-up) → ROS (Z-up)
                Unity (X, Y, Z) → ROS (Z, -X, Y)
        """
        if robot_id not in self.ROBOT_BASE_TRANSFORMS:
            return world_position

        transform = self.ROBOT_BASE_TRANSFORMS[robot_id]
        base_x, base_y, base_z = transform["position"]
        y_rotation_deg = transform["y_rotation"]

        # Step 1: Translate
        unity_local_x = world_position.get("x", 0.0) - base_x
        unity_local_y = world_position.get("y", 0.0) - base_y
        unity_local_z = world_position.get("z", 0.0) - base_z

        # Step 2: Rotate around Y-axis
        y_rotation_rad = math.radians(y_rotation_deg)
        cos_theta = math.cos(y_rotation_rad)
        sin_theta = math.sin(y_rotation_rad)

        rotated_x = cos_theta * unity_local_x + sin_theta * unity_local_z
        rotated_y = unity_local_y
        rotated_z = -sin_theta * unity_local_x + cos_theta * unity_local_z

        # Step 3: Unity (X, Y, Z) → ROS (Z, -X, Y)
        ros_x = rotated_z
        ros_y = -rotated_x
        ros_z = rotated_y

        return {"x": ros_x, "y": ros_y, "z": ros_z}


def test_robot1_axis_conversion():
    """Robot1 (no rotation): verify full translate + axis-convert pipeline."""
    server = MockROSMotionServer()

    # Object 0.3m in front of Robot1 (Unity +Z), at table height 0.05m (Unity +Y)
    world_pos = {"x": -0.475, "y": 0.05, "z": 0.3}
    ros_pos = server._transform_world_to_local(world_pos, "Robot1")

    print("Robot1 axis conversion test:")
    print(f"  Unity world: {world_pos}")
    print(f"  ROS base_link: {ros_pos}")

    # After translate: (0.0, 0.05, 0.3) in Unity local
    # No rotation (0°)
    # Axis convert: ros_x=Z=0.3, ros_y=-X=0.0, ros_z=Y=0.05
    assert abs(ros_pos["x"] - 0.3) < 0.001, f"ROS X (Unity Z) wrong: {ros_pos['x']}"
    assert abs(ros_pos["y"] - 0.0) < 0.001, f"ROS Y (-Unity X) wrong: {ros_pos['y']}"
    assert abs(ros_pos["z"] - 0.05) < 0.001, f"ROS Z (Unity Y) wrong: {ros_pos['z']}"

    print("  ✅ Robot1 axis conversion correct")


def test_robot2_axis_conversion():
    """Robot2 (180° rotation): verify translate + rotate + axis-convert pipeline."""
    server = MockROSMotionServer()

    # Object symmetric to Robot1 case: 0.3m in front of Robot2 (Unity -Z from world),
    # i.e., at Unity world z = -0.3, at table height 0.05m
    world_pos = {"x": 0.475, "y": 0.05, "z": -0.3}
    ros_pos = server._transform_world_to_local(world_pos, "Robot2")

    print("\nRobot2 axis conversion test:")
    print(f"  Unity world: {world_pos}")
    print(f"  ROS base_link: {ros_pos}")

    # After translate: (0.0, 0.05, -0.3) in Unity local
    # Rotate 180°: rotated_x = cos(180)*0 + sin(180)*(-0.3) = 0,
    #              rotated_z = -sin(180)*0 + cos(180)*(-0.3) = 0.3
    # Axis convert: ros_x = rotated_z = 0.3, ros_y = -rotated_x = 0, ros_z = 0.05
    assert abs(ros_pos["x"] - 0.3) < 0.001, f"ROS X wrong: {ros_pos['x']}"
    assert abs(ros_pos["y"] - 0.0) < 0.001, f"ROS Y wrong: {ros_pos['y']}"
    assert abs(ros_pos["z"] - 0.05) < 0.001, f"ROS Z (height) wrong: {ros_pos['z']}"

    print("  ✅ Robot2 axis conversion correct")


def test_symmetric_world_positions_give_same_ros_coords():
    """Objects equidistant from their robot's front should give equal ROS coords."""
    server = MockROSMotionServer()

    # Robot1 at (-0.475, 0, 0) facing +Z: object 0.25m in front, 0.1m up
    world_left = {"x": -0.475, "y": 0.1, "z": 0.25}
    # Robot2 at (0.475, 0, 0) facing -Z: object 0.25m in front (world -Z), 0.1m up
    world_right = {"x": 0.475, "y": 0.1, "z": -0.25}

    ros_left = server._transform_world_to_local(world_left, "Robot1")
    ros_right = server._transform_world_to_local(world_right, "Robot2")

    print("\nSymmetric world positions test:")
    print(f"  Robot1 world {world_left} → ROS {ros_left}")
    print(f"  Robot2 world {world_right} → ROS {ros_right}")

    # Both should be 0.25m in front (ROS X), 0 lateral (ROS Y), 0.1m up (ROS Z)
    assert abs(ros_left["x"] - 0.25) < 0.001, f"Robot1 ROS X: {ros_left['x']}"
    assert abs(ros_left["y"] - 0.0) < 0.001, f"Robot1 ROS Y: {ros_left['y']}"
    assert abs(ros_left["z"] - 0.1) < 0.001, f"Robot1 ROS Z: {ros_left['z']}"

    assert abs(ros_right["x"] - 0.25) < 0.001, f"Robot2 ROS X: {ros_right['x']}"
    assert abs(ros_right["y"] - 0.0) < 0.001, f"Robot2 ROS Y: {ros_right['y']}"
    assert abs(ros_right["z"] - 0.1) < 0.001, f"Robot2 ROS Z: {ros_right['z']}"

    print("  ✅ Symmetric positions give equal ROS coordinates")


def test_lateral_offset_axis_conversion():
    """Verify Unity X (lateral) maps to ROS -Y (left/right in ROS frame)."""
    server = MockROSMotionServer()

    # Object 0.1m to the right of Robot1 (Unity +X) and 0.3m forward
    world_pos = {"x": -0.475 + 0.1, "y": 0.05, "z": 0.3}
    ros_pos = server._transform_world_to_local(world_pos, "Robot1")

    # Unity local: (0.1, 0.05, 0.3)
    # Axis convert: ros_x=0.3, ros_y=-0.1, ros_z=0.05
    assert abs(ros_pos["x"] - 0.3) < 0.001, f"ROS X: {ros_pos['x']}"
    assert abs(ros_pos["y"] - (-0.1)) < 0.001, f"ROS Y (expect -0.1 for right): {ros_pos['y']}"
    assert abs(ros_pos["z"] - 0.05) < 0.001, f"ROS Z: {ros_pos['z']}"

    print("\nLateral offset test:")
    print(f"  0.1m right of Robot1 → ROS Y = {ros_pos['y']:.3f} (expect -0.1)")
    print("  ✅ Unity X → ROS -Y correct")


# ---------------------------------------------------------------------------
# Direct tests of world_to_robot_frame_np / robot_to_world_frame_np
# These exercise the actual CoordinateTransforms module (not the mock).
# ---------------------------------------------------------------------------

def test_np_robot1_axis_conversion():
    """NumPy wrapper: Robot1 world pos → ROS frame matches mock behavior."""
    world_pos = np.array([-0.475, 0.05, 0.3])
    ros_pos = world_to_robot_frame_np(world_pos, "Robot1")

    # Same expected values as test_robot1_axis_conversion
    assert abs(ros_pos[0] - 0.3) < 0.001, f"ROS X wrong: {ros_pos[0]}"
    assert abs(ros_pos[1] - 0.0) < 0.001, f"ROS Y wrong: {ros_pos[1]}"
    assert abs(ros_pos[2] - 0.05) < 0.001, f"ROS Z wrong: {ros_pos[2]}"

    print("\nNumPy Robot1 axis conversion:")
    print(f"  world_pos={world_pos} → ros_pos={ros_pos}")
    print("  ✅ matches mock")


def test_np_robot2_axis_conversion():
    """NumPy wrapper: Robot2 world pos → ROS frame matches mock behavior."""
    world_pos = np.array([0.475, 0.05, -0.3])
    ros_pos = world_to_robot_frame_np(world_pos, "Robot2")

    assert abs(ros_pos[0] - 0.3) < 0.001, f"ROS X wrong: {ros_pos[0]}"
    assert abs(ros_pos[1] - 0.0) < 0.001, f"ROS Y wrong: {ros_pos[1]}"
    assert abs(ros_pos[2] - 0.05) < 0.001, f"ROS Z wrong: {ros_pos[2]}"

    print("\nNumPy Robot2 axis conversion:")
    print(f"  world_pos={world_pos} → ros_pos={ros_pos}")
    print("  ✅ matches mock")


def test_np_lateral_offset():
    """NumPy wrapper: lateral Unity X offset maps to ROS -Y."""
    world_pos = np.array([-0.475 + 0.1, 0.05, 0.3])
    ros_pos = world_to_robot_frame_np(world_pos, "Robot1")

    assert abs(ros_pos[0] - 0.3) < 0.001, f"ROS X: {ros_pos[0]}"
    assert abs(ros_pos[1] - (-0.1)) < 0.001, f"ROS Y (expect -0.1): {ros_pos[1]}"
    assert abs(ros_pos[2] - 0.05) < 0.001, f"ROS Z: {ros_pos[2]}"

    print("\nNumPy lateral offset test:")
    print(f"  0.1m right of Robot1 → ROS Y={ros_pos[1]:.3f} (expect -0.1)")
    print("  ✅ correct")


def test_np_roundtrip_robot1():
    """world → robot → world roundtrip for Robot1 should recover original position."""
    original = np.array([-0.3, 0.15, 0.25])
    ros = world_to_robot_frame_np(original, "Robot1")
    recovered = robot_to_world_frame_np(ros, "Robot1")
    assert np.allclose(recovered, original, atol=1e-10), (
        f"Roundtrip Robot1 failed: {original} → {ros} → {recovered}"
    )
    print("\nNumPy roundtrip Robot1:")
    print(f"  {original} → ros {ros} → {recovered}")
    print("  ✅ roundtrip exact")


def test_np_roundtrip_robot2():
    """world → robot → world roundtrip for Robot2 should recover original position."""
    original = np.array([0.6, 0.1, -0.2])
    ros = world_to_robot_frame_np(original, "Robot2")
    recovered = robot_to_world_frame_np(ros, "Robot2")
    assert np.allclose(recovered, original, atol=1e-10), (
        f"Roundtrip Robot2 failed: {original} → {ros} → {recovered}"
    )
    print("\nNumPy roundtrip Robot2:")
    print(f"  {original} → ros {ros} → {recovered}")
    print("  ✅ roundtrip exact")


def test_np_matches_dict_robot1():
    """NumPy result should be numerically identical to dict-based world_to_robot_frame."""
    server = MockROSMotionServer()
    world_dict = {"x": -0.2, "y": 0.08, "z": 0.35}
    world_np = np.array([world_dict["x"], world_dict["y"], world_dict["z"]])

    ros_dict = server._transform_world_to_local(world_dict, "Robot1")
    ros_np = world_to_robot_frame_np(world_np, "Robot1")

    assert abs(ros_np[0] - ros_dict["x"]) < 1e-10, f"X mismatch: {ros_np[0]} vs {ros_dict['x']}"
    assert abs(ros_np[1] - ros_dict["y"]) < 1e-10, f"Y mismatch: {ros_np[1]} vs {ros_dict['y']}"
    assert abs(ros_np[2] - ros_dict["z"]) < 1e-10, f"Z mismatch: {ros_np[2]} vs {ros_dict['z']}"

    print("\nNumPy vs dict equivalence (Robot1):")
    print(f"  dict: {ros_dict}, np: {ros_np}")
    print("  ✅ identical")


def test_np_matches_dict_robot2():
    """NumPy result should be numerically identical to dict-based world_to_robot_frame for Robot2."""
    server = MockROSMotionServer()
    world_dict = {"x": 0.6, "y": 0.12, "z": -0.4}
    world_np = np.array([world_dict["x"], world_dict["y"], world_dict["z"]])

    ros_dict = server._transform_world_to_local(world_dict, "Robot2")
    ros_np = world_to_robot_frame_np(world_np, "Robot2")

    assert abs(ros_np[0] - ros_dict["x"]) < 1e-10, f"X mismatch: {ros_np[0]} vs {ros_dict['x']}"
    assert abs(ros_np[1] - ros_dict["y"]) < 1e-10, f"Y mismatch: {ros_np[1]} vs {ros_dict['y']}"
    assert abs(ros_np[2] - ros_dict["z"]) < 1e-10, f"Z mismatch: {ros_np[2]} vs {ros_dict['z']}"

    print("\nNumPy vs dict equivalence (Robot2):")
    print(f"  dict: {ros_dict}, np: {ros_np}")
    print("  ✅ identical")


if __name__ == "__main__":
    try:
        test_robot1_axis_conversion()
        test_robot2_axis_conversion()
        test_symmetric_world_positions_give_same_ros_coords()
        test_lateral_offset_axis_conversion()

        test_np_robot1_axis_conversion()
        test_np_robot2_axis_conversion()
        test_np_lateral_offset()
        test_np_roundtrip_robot1()
        test_np_roundtrip_robot2()
        test_np_matches_dict_robot1()
        test_np_matches_dict_robot2()

        print("\n" + "=" * 60)
        print("ALL COORDINATE TRANSFORMATION TESTS PASSED")
        print("=" * 60)
        sys.exit(0)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
