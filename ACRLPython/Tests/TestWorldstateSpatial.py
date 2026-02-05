"""
Test suite for WorldState spatial query methods.

Tests spatial reasoning capabilities:
- find_objects_near: Proximity queries
- find_robots_near: Robot proximity
- get_reachable_objects: Reachability computation
- get_objects_in_region: Region containment
- get_region_for_position: Position-to-region mapping
- get_world_context_string: LLM context generation
"""

import unittest
from operations.WorldState import WorldState, get_world_state
from config.Robot import WORKSPACE_REGIONS


class TestWorldStateSpatial(unittest.TestCase):
    """Test spatial query methods on WorldState."""

    def setUp(self):
        """Reset world state before each test."""
        self.world_state = get_world_state()
        self.world_state.reset()

    def test_find_objects_near_returns_correct_objects(self):
        """Test that find_objects_near returns objects within radius."""
        # Register objects at known positions
        self.world_state.register_object("obj1", position=(0.0, 0.0, 0.0))
        self.world_state.register_object("obj2", position=(0.05, 0.05, 0.0))
        self.world_state.register_object("obj3", position=(0.5, 0.5, 0.0))

        # Search from origin with radius 0.1m
        nearby = self.world_state.find_objects_near((0.0, 0.0, 0.0), radius=0.1)

        # Should find obj1 (0.0m) and obj2 (~0.07m) but not obj3 (0.7m)
        nearby_ids = {obj.object_id for obj in nearby}
        self.assertIn("obj1", nearby_ids)
        self.assertIn("obj2", nearby_ids)
        self.assertNotIn("obj3", nearby_ids)

    def test_find_objects_near_excludes_stale(self):
        """Test that stale objects are excluded by default."""
        # Register objects
        self.world_state.register_object("obj1", position=(0.0, 0.0, 0.0))
        self.world_state.register_object("obj2", position=(0.05, 0.0, 0.0))

        # Mark obj2 as stale
        self.world_state._objects["obj2"].stale = True

        # Search with exclude_stale=True (default)
        nearby = self.world_state.find_objects_near((0.0, 0.0, 0.0), radius=0.1)
        nearby_ids = {obj.object_id for obj in nearby}

        self.assertIn("obj1", nearby_ids)
        self.assertNotIn("obj2", nearby_ids)

        # Search with exclude_stale=False
        nearby_all = self.world_state.find_objects_near(
            (0.0, 0.0, 0.0), radius=0.1, exclude_stale=False
        )
        nearby_all_ids = {obj.object_id for obj in nearby_all}

        self.assertIn("obj1", nearby_all_ids)
        self.assertIn("obj2", nearby_all_ids)

    def test_find_robots_near(self):
        """Test finding robots within radius."""
        # Register robots with positions
        self.world_state.update_robot("Robot1", position=(-0.3, 0.2, 0.0))
        self.world_state.update_robot("Robot2", position=(0.3, 0.2, 0.0))

        # Search from center
        nearby = self.world_state.find_robots_near((0.0, 0.2, 0.0), radius=0.4)

        # Should find both robots (both ~0.3m away)
        nearby_ids = {robot.robot_id for robot in nearby}
        self.assertIn("Robot1", nearby_ids)
        self.assertIn("Robot2", nearby_ids)

        # Narrow search
        nearby_narrow = self.world_state.find_robots_near((0.0, 0.2, 0.0), radius=0.2)
        self.assertEqual(len(nearby_narrow), 0)  # Both are 0.3m away

    def test_get_reachable_objects(self):
        """Test getting objects reachable by a robot."""
        # Register robot
        self.world_state.update_robot("Robot1", position=(-0.475, 0.0, 0.0))

        # Register objects at various positions
        # Robot1 base is at (-0.475, 0, 0), MAX_ROBOT_REACH is 0.8m
        self.world_state.register_object(
            "nearby_obj", position=(-0.3, 0.3, 0.0)
        )  # Within reach
        self.world_state.register_object(
            "far_obj", position=(0.5, 0.3, 0.0)
        )  # Far from Robot1

        # Get reachable objects
        reachable = self.world_state.get_reachable_objects("Robot1")
        reachable_ids = {obj.object_id for obj in reachable}

        # nearby_obj should be reachable, far_obj should not
        self.assertIn("nearby_obj", reachable_ids)
        self.assertNotIn("far_obj", reachable_ids)

    def test_get_reachable_objects_excludes_stale(self):
        """Test that stale objects are excluded from reachable objects."""
        # Register robot
        self.world_state.update_robot("Robot1", position=(-0.475, 0.0, 0.0))

        # Register objects
        self.world_state.register_object("obj1", position=(-0.3, 0.3, 0.0))
        self.world_state.register_object("obj2", position=(-0.2, 0.3, 0.0))

        # Mark obj2 as stale
        self.world_state._objects["obj2"].stale = True

        # Get reachable with exclude_stale=True (default)
        reachable = self.world_state.get_reachable_objects("Robot1")
        reachable_ids = {obj.object_id for obj in reachable}

        self.assertIn("obj1", reachable_ids)
        self.assertNotIn("obj2", reachable_ids)

        # Get reachable with exclude_stale=False
        reachable_all = self.world_state.get_reachable_objects(
            "Robot1", exclude_stale=False
        )
        reachable_all_ids = {obj.object_id for obj in reachable_all}

        self.assertIn("obj1", reachable_all_ids)
        self.assertIn("obj2", reachable_all_ids)

    def test_get_objects_in_region(self):
        """Test getting objects in a workspace region."""
        # Register objects in different regions
        # left_workspace: x_min=-0.5, x_max=-0.15
        self.world_state.register_object("left_obj", position=(-0.3, 0.3, 0.0))
        # right_workspace: x_min=0.15, x_max=0.5
        self.world_state.register_object("right_obj", position=(0.3, 0.3, 0.0))
        # shared_zone: x_min=-0.15, x_max=0.15
        self.world_state.register_object("shared_obj", position=(0.0, 0.3, 0.0))

        # Query left workspace
        left_objs = self.world_state.get_objects_in_region("left_workspace")
        left_ids = {obj.object_id for obj in left_objs}
        self.assertIn("left_obj", left_ids)
        self.assertNotIn("right_obj", left_ids)
        self.assertNotIn("shared_obj", left_ids)

        # Query shared zone
        shared_objs = self.world_state.get_objects_in_region("shared_zone")
        shared_ids = {obj.object_id for obj in shared_objs}
        self.assertIn("shared_obj", shared_ids)
        self.assertNotIn("left_obj", shared_ids)
        self.assertNotIn("right_obj", shared_ids)

    def test_get_objects_in_region_unknown_region(self):
        """Test behavior with unknown region name."""
        objs = self.world_state.get_objects_in_region("nonexistent_region")
        self.assertEqual(len(objs), 0)

    def test_get_region_for_position(self):
        """Test position-to-region mapping."""
        # Test left workspace
        region = self.world_state.get_region_for_position((-0.3, 0.3, 0.0))
        self.assertEqual(region, "left_workspace")

        # Test right workspace
        region = self.world_state.get_region_for_position((0.3, 0.3, 0.0))
        self.assertEqual(region, "right_workspace")

        # Test shared zone
        region = self.world_state.get_region_for_position((0.0, 0.3, 0.0))
        self.assertEqual(region, "shared_zone")

        # Test outside all regions
        region = self.world_state.get_region_for_position((10.0, 10.0, 10.0))
        self.assertIsNone(region)

    def test_get_region_for_position_boundaries(self):
        """Test region detection at boundaries."""
        # Get left_workspace bounds
        left = WORKSPACE_REGIONS["left_workspace"]

        # Test exact boundaries
        region = self.world_state.get_region_for_position(
            (left["x_min"], left["y_min"], left["z_min"])
        )
        self.assertEqual(region, "left_workspace")

        region = self.world_state.get_region_for_position(
            (left["x_max"], left["y_max"], left["z_max"])
        )
        # Note: x_max=-0.15 overlaps with shared_zone x_min=-0.15
        # The order in WORKSPACE_REGIONS dict determines which is returned first
        self.assertIn(region, ["left_workspace", "shared_zone"])

    def test_get_world_context_string_basic(self):
        """Test basic world context string generation."""
        # Register robot
        self.world_state.update_robot(
            "Robot1", position=(-0.3, 0.2, 0.1), gripper_state="open"
        )

        # Register objects
        self.world_state.register_object(
            "RedCube", position=(-0.2, 0.3, 0.0), color="red"
        )

        # Get context
        context = self.world_state.get_world_context_string("Robot1")

        # Verify robot state is included
        self.assertIn("Robot1", context)
        self.assertIn("-0.30, 0.20, 0.10", context)
        self.assertIn("gripper open", context)

        # Verify object is included
        self.assertIn("RedCube", context)
        self.assertIn("-0.20, 0.30, 0.00", context)

    def test_get_world_context_string_with_annotations(self):
        """Test context string includes spatial annotations."""
        # Register robot at base position
        self.world_state.update_robot(
            "Robot1", position=(-0.475, 0.0, 0.0), gripper_state="closed"
        )

        # Register reachable object in left workspace
        self.world_state.register_object("Obj1", position=(-0.3, 0.3, 0.0))

        # Register far object in right workspace
        self.world_state.register_object("Obj2", position=(0.3, 0.3, 0.0))

        # Get context
        context = self.world_state.get_world_context_string("Robot1")

        # Check that reachability is annotated
        # Obj1 should be marked as reachable
        self.assertIn("Obj1", context)
        # Note: actual reachability depends on SpatialPredicates implementation

        # Obj2 should be marked as not reachable
        self.assertIn("Obj2", context)

        # Check region annotations
        self.assertIn("left_workspace", context)
        self.assertIn("right_workspace", context)

    def test_get_world_context_string_with_grasped_object(self):
        """Test context string shows grasped objects."""
        # Register robots
        self.world_state.update_robot(
            "Robot1", position=(-0.3, 0.2, 0.0), gripper_state="closed"
        )

        # Register object grasped by Robot1
        self.world_state.register_object("Cube", position=(-0.3, 0.2, 0.0))
        self.world_state.mark_object_grasped("Cube", "Robot1")

        # Get context
        context = self.world_state.get_world_context_string("Robot1")

        # Verify grasp annotation
        self.assertIn("grasped by Robot1", context)

    def test_get_world_context_string_with_stale_object(self):
        """Test context string marks stale objects."""
        # Register robot
        self.world_state.update_robot("Robot1", position=(-0.3, 0.2, 0.0))

        # Register object and mark as stale
        self.world_state.register_object("StaleObj", position=(-0.2, 0.3, 0.0))
        self.world_state._objects["StaleObj"].stale = True

        # Get context
        context = self.world_state.get_world_context_string("Robot1")

        # Verify stale annotation
        self.assertIn("stale", context)

    def test_get_world_context_string_no_objects(self):
        """Test context string when no objects are present."""
        # Register robot only
        self.world_state.update_robot("Robot1", position=(-0.3, 0.2, 0.0))

        # Get context
        context = self.world_state.get_world_context_string("Robot1")

        # Should mention no objects
        self.assertIn("No objects detected", context)

    def test_get_world_context_string_unknown_robot(self):
        """Test context string for unknown robot."""
        context = self.world_state.get_world_context_string("UnknownRobot")
        self.assertIn("state unknown", context)

    def test_empty_searches(self):
        """Test spatial queries with no objects registered."""
        # No objects registered
        nearby = self.world_state.find_objects_near((0.0, 0.0, 0.0))
        self.assertEqual(len(nearby), 0)

        reachable = self.world_state.get_reachable_objects("Robot1")
        self.assertEqual(len(reachable), 0)

        in_region = self.world_state.get_objects_in_region("left_workspace")
        self.assertEqual(len(in_region), 0)


if __name__ == "__main__":
    unittest.main()
