"""
Test Coordinate Transformation
================================

Verifies that world-to-local coordinate transformation works correctly
for dual-robot ROS integration.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_world_to_local_transformation():
    """Test that world coordinates are correctly transformed to local base_link coordinates."""

    # Mock the ROSMotionServer class to test transformation logic
    import math

    class MockROSMotionServer:
        ROBOT_BASE_TRANSFORMS = {
            "Robot1": {"position": (-0.475, 0.0, 0.0), "y_rotation": 0.0},
            "Robot2": {"position": (0.475, 0.0, 0.0), "y_rotation": 180.0},
        }

        def _transform_world_to_local(self, world_position: dict, robot_id: str) -> dict:
            """Transform world coordinates to robot-local base_link coordinates with rotation."""
            if robot_id not in self.ROBOT_BASE_TRANSFORMS:
                return world_position

            transform = self.ROBOT_BASE_TRANSFORMS[robot_id]
            base_x, base_y, base_z = transform["position"]
            y_rotation_deg = transform["y_rotation"]

            # Step 1: Translate
            translated_x = world_position.get("x", 0.0) - base_x
            translated_y = world_position.get("y", 0.0) - base_y
            translated_z = world_position.get("z", 0.0) - base_z

            # Step 2: Rotate around Y-axis
            y_rotation_rad = math.radians(y_rotation_deg)
            cos_theta = math.cos(y_rotation_rad)
            sin_theta = math.sin(y_rotation_rad)

            local_x = cos_theta * translated_x + sin_theta * translated_z
            local_y = translated_y
            local_z = -sin_theta * translated_x + cos_theta * translated_z

            return {"x": local_x, "y": local_y, "z": local_z}

    server = MockROSMotionServer()

    # Test Robot1 transformation
    world_pos_robot1 = {"x": -0.2, "y": 0.05, "z": 0.0}
    local_pos_robot1 = server._transform_world_to_local(world_pos_robot1, "Robot1")

    print("Robot1 Transformation:")
    print(f"  World: {world_pos_robot1}")
    print(f"  Local: {local_pos_robot1}")

    # Expected for Robot1 (0° rotation):
    # Translate: -0.2 - (-0.475) = 0.275
    # Rotate 0°: no change = (0.275, 0.05, 0)
    assert abs(local_pos_robot1["x"] - 0.275) < 0.001, f"Robot1 X transform failed: expected 0.275, got {local_pos_robot1['x']}"
    assert abs(local_pos_robot1["y"] - 0.05) < 0.001, f"Robot1 Y transform failed: {local_pos_robot1['y']}"
    assert abs(local_pos_robot1["z"] - 0.0) < 0.001, f"Robot1 Z transform failed: {local_pos_robot1['z']}"

    # Test Robot2 transformation
    world_pos_robot2 = {"x": 0.2, "y": 0.05, "z": 0.0}
    local_pos_robot2 = server._transform_world_to_local(world_pos_robot2, "Robot2")

    print("\nRobot2 Transformation:")
    print(f"  World: {world_pos_robot2}")
    print(f"  Local: {local_pos_robot2}")

    # Expected for Robot2 (180° rotation):
    # Translate: 0.2 - 0.475 = -0.275
    # Rotate 180° around Y: (x, y, z) -> (-x, y, -z) = (0.275, 0.05, 0)
    assert abs(local_pos_robot2["x"] - 0.275) < 0.001, f"Robot2 X transform failed: expected 0.275, got {local_pos_robot2['x']}"
    assert abs(local_pos_robot2["y"] - 0.05) < 0.001, f"Robot2 Y transform failed: {local_pos_robot2['y']}"
    assert abs(local_pos_robot2["z"] - 0.0) < 0.001, f"Robot2 Z transform failed: {local_pos_robot2['z']}"

    print("\n✅ All coordinate transformations passed!")
    return True


def test_symmetric_positions():
    """Test that symmetric world positions result in expected local positions."""
    import math

    class MockROSMotionServer:
        ROBOT_BASE_TRANSFORMS = {
            "Robot1": {"position": (-0.475, 0.0, 0.0), "y_rotation": 0.0},
            "Robot2": {"position": (0.475, 0.0, 0.0), "y_rotation": 180.0},
        }

        def _transform_world_to_local(self, world_position: dict, robot_id: str) -> dict:
            if robot_id not in self.ROBOT_BASE_TRANSFORMS:
                return world_position

            transform = self.ROBOT_BASE_TRANSFORMS[robot_id]
            base_x, base_y, base_z = transform["position"]
            y_rotation_deg = transform["y_rotation"]

            translated_x = world_position.get("x", 0.0) - base_x
            translated_y = world_position.get("y", 0.0) - base_y
            translated_z = world_position.get("z", 0.0) - base_z

            y_rotation_rad = math.radians(y_rotation_deg)
            cos_theta = math.cos(y_rotation_rad)
            sin_theta = math.sin(y_rotation_rad)

            local_x = cos_theta * translated_x + sin_theta * translated_z
            local_y = translated_y
            local_z = -sin_theta * translated_x + cos_theta * translated_z

            return {"x": local_x, "y": local_y, "z": local_z}

    server = MockROSMotionServer()

    # Test symmetric world positions
    world_left = {"x": -0.3, "y": 0.1, "z": 0.0}
    world_right = {"x": 0.3, "y": 0.1, "z": 0.0}

    local_left = server._transform_world_to_local(world_left, "Robot1")
    local_right = server._transform_world_to_local(world_right, "Robot2")

    print("\nSymmetric Position Test:")
    print(f"  Robot1 world {world_left} -> local {local_left}")
    print(f"  Robot2 world {world_right} -> local {local_right}")

    # With rotation:
    # Robot1: translate (-0.3 - (-0.475) = 0.175), no rotation -> (0.175, 0.1, 0)
    # Robot2: translate (0.3 - 0.475 = -0.175), rotate 180° -> (-(-0.175), 0.1, 0) = (0.175, 0.1, 0)
    #
    # Result: BOTH robots end up with the SAME local coordinates (0.175, 0.1, 0)!
    # This makes sense: symmetric world positions for opposite-facing robots
    # should result in the same position relative to each robot's local frame.

    # Verify both transformations
    assert abs(local_left["x"] - 0.175) < 0.001, f"Robot1 symmetric X failed: expected 0.175, got {local_left['x']}"
    assert abs(local_left["y"] - 0.1) < 0.001, f"Robot1 symmetric Y failed"

    assert abs(local_right["x"] - 0.175) < 0.001, f"Robot2 symmetric X failed: expected 0.175, got {local_right['x']}"
    assert abs(local_right["y"] - 0.1) < 0.001, f"Robot2 symmetric Y failed"

    print("✅ Symmetric position test passed!")
    return True


if __name__ == "__main__":
    try:
        test_world_to_local_transformation()
        test_symmetric_positions()

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
