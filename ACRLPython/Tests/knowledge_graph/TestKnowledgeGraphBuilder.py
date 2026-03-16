#!/usr/bin/env python3
"""
Test suite for KnowledgeGraph Builder and QueryEngine
======================================================

Tests graph building from WorldState and high-level queries.
"""

import unittest
from knowledge_graph import KnowledgeGraph, GraphBuilder, GraphQueryEngine
from operations.WorldState import get_world_state


class TestGraphBuilder(unittest.TestCase):
    """Test GraphBuilder functionality."""

    def setUp(self):
        """Create fresh graph and world state for each test."""
        self.world_state = get_world_state()
        self.world_state.reset()
        self.graph = KnowledgeGraph()
        self.builder = GraphBuilder(self.graph, self.world_state)

    def tearDown(self):
        """Clean up."""
        self.graph.clear()
        self.world_state.reset()

    def test_static_regions_initialized(self):
        """Test that static regions are created at initialization."""
        # Check region nodes exist
        self.assertTrue(self.graph.has_node("left_workspace"))
        self.assertTrue(self.graph.has_node("right_workspace"))
        self.assertTrue(self.graph.has_node("shared_zone"))

        # Check ADJACENT_TO edges
        left_neighbors = self.graph.get_neighbors(
            "left_workspace", edge_type="ADJACENT_TO"
        )
        self.assertIn("shared_zone", left_neighbors)

    def test_update_robot_nodes(self):
        """Test robot node creation and updates."""
        # Add robot to WorldState
        self.world_state.update_robot(
            "Robot1", position=(-0.3, 0.2, 0.1), gripper_state="open"
        )

        # Trigger update
        state_data = {"robots": [{"robot_id": "Robot1"}], "objects": []}
        self.builder.on_state_update(state_data)

        # Verify robot node created
        self.assertTrue(self.graph.has_node("Robot1"))
        robot_node = self.graph.get_node("Robot1")
        self.assertIsNotNone(robot_node)
        assert robot_node is not None  # Type narrowing for Pylance
        self.assertEqual(robot_node["node_type"], "robot")
        self.assertEqual(robot_node["gripper_state"], "open")

    def test_update_object_nodes(self):
        """Test object node creation and updates."""
        # Add object to WorldState
        self.world_state.register_object(
            "RedCube", position=(0.1, 0.3, 0.0), color="red"
        )

        # Trigger update
        state_data = {"robots": [], "objects": [{"object_id": "RedCube"}]}
        self.builder.on_state_update(state_data)

        # Verify object node created
        self.assertTrue(self.graph.has_node("RedCube"))
        obj_node = self.graph.get_node("RedCube")
        self.assertIsNotNone(obj_node)
        assert obj_node is not None  # Type narrowing for Pylance
        self.assertEqual(obj_node["node_type"], "object")
        self.assertEqual(obj_node["color"], "red")

    def test_can_reach_edges_computed(self):
        """Test CAN_REACH edges are computed correctly."""
        # Setup: Robot and reachable object
        self.world_state.update_robot("Robot1", position=(-0.475, 0.0, 0.0))
        self.world_state.register_object("NearbyObj", position=(-0.3, 0.3, 0.0))

        # Trigger update
        state_data = {
            "robots": [{"robot_id": "Robot1"}],
            "objects": [{"object_id": "NearbyObj"}],
        }
        self.builder.on_state_update(state_data)

        # Verify CAN_REACH edge exists
        reachable = self.graph.get_neighbors("Robot1", edge_type="CAN_REACH")
        self.assertIn("NearbyObj", reachable)

    def test_near_edges_computed(self):
        """Test NEAR edges are computed for close objects."""
        # Setup: Two objects very close together
        self.world_state.register_object("Obj1", position=(0.0, 0.3, 0.0))
        self.world_state.register_object("Obj2", position=(0.05, 0.3, 0.0))  # 5cm away

        # Trigger update
        state_data = {
            "robots": [],
            "objects": [{"object_id": "Obj1"}, {"object_id": "Obj2"}],
        }
        self.builder.on_state_update(state_data)

        # Verify NEAR edges exist
        near_obj1 = self.graph.get_neighbors("Obj1", edge_type="NEAR")
        self.assertIn("Obj2", near_obj1)

        near_obj2 = self.graph.get_neighbors("Obj2", edge_type="NEAR")
        self.assertIn("Obj1", near_obj2)

    def test_in_region_edges_computed(self):
        """Test IN_REGION edges are computed correctly."""
        # Robot in left workspace
        self.world_state.update_robot("Robot1", position=(-0.3, 0.3, 0.0))

        # Trigger update
        state_data = {"robots": [{"robot_id": "Robot1"}], "objects": []}
        self.builder.on_state_update(state_data)

        # Verify IN_REGION edge
        in_region = self.graph.get_neighbors("Robot1", edge_type="IN_REGION")
        self.assertIn("left_workspace", in_region)

    def test_grasping_edges_computed(self):
        """Test GRASPING edges when robot grasps object."""
        # Setup: Robot with grasped object
        self.world_state.update_robot("Robot1", position=(-0.3, 0.2, 0.0))
        self.world_state.register_object("GraspedCube", position=(-0.3, 0.2, 0.0))
        self.world_state.mark_object_grasped("GraspedCube", "Robot1")

        # Trigger update
        state_data = {
            "robots": [{"robot_id": "Robot1"}],
            "objects": [{"object_id": "GraspedCube"}],
        }
        self.builder.on_state_update(state_data)

        # Verify GRASPING edge
        grasping = self.graph.get_neighbors("Robot1", edge_type="GRASPING")
        self.assertIn("GraspedCube", grasping)

    def test_allocated_edges_computed(self):
        """Test ALLOCATED edges when region is allocated to robot."""
        # Setup: Allocate region
        self.world_state.update_robot("Robot1", position=(-0.3, 0.2, 0.0))
        self.world_state.allocate_workspace("left_workspace", "Robot1")

        # Trigger update
        state_data = {"robots": [{"robot_id": "Robot1"}], "objects": []}
        self.builder.on_state_update(state_data)

        # Verify ALLOCATED edge
        allocated = self.graph.get_neighbors("left_workspace", edge_type="ALLOCATED")
        self.assertIn("Robot1", allocated)

    def test_stale_objects_removed(self):
        """Test that objects no longer in WorldState are removed from graph."""
        # Add object
        self.world_state.register_object("TempObj", position=(0.1, 0.3, 0.0))

        # Trigger update (object present)
        state_data = {"robots": [], "objects": [{"object_id": "TempObj"}]}
        self.builder.on_state_update(state_data)
        self.assertTrue(self.graph.has_node("TempObj"))

        # Trigger update (object gone)
        state_data = {"robots": [], "objects": []}  # No objects
        self.builder.on_state_update(state_data)

        # Verify object removed
        self.assertFalse(self.graph.has_node("TempObj"))


class TestGraphQueryEngine(unittest.TestCase):
    """Test GraphQueryEngine functionality."""

    def setUp(self):
        """Create graph with test data."""
        self.world_state = get_world_state()
        self.world_state.reset()
        self.graph = KnowledgeGraph()
        self.builder = GraphBuilder(self.graph, self.world_state)
        self.query_engine = GraphQueryEngine(self.graph)

        # Setup test scenario
        self.world_state.update_robot("Robot1", position=(-0.475, 0.0, 0.0))
        self.world_state.update_robot("Robot2", position=(0.475, 0.0, 0.0))
        self.world_state.register_object(
            "RedCube", position=(-0.3, 0.3, 0.0), color="red"
        )
        self.world_state.register_object(
            "BlueCube", position=(0.3, 0.3, 0.0), color="blue"
        )

        # Build graph
        state_data = {
            "robots": [{"robot_id": "Robot1"}, {"robot_id": "Robot2"}],
            "objects": [{"object_id": "RedCube"}, {"object_id": "BlueCube"}],
        }
        self.builder.on_state_update(state_data)

    def tearDown(self):
        """Clean up."""
        self.graph.clear()
        self.world_state.reset()

    def test_find_reachable_robots(self):
        """Test finding which robots can reach an object."""
        # RedCube should be reachable by Robot1 (nearby)
        reachable = self.query_engine.find_reachable_robots("RedCube")

        self.assertIn("Robot1", reachable)
        # Robot2 is far away, may not be able to reach
        # (depends on MAX_ROBOT_REACH)

    def test_find_robots_near(self):
        """Test finding robots near another robot."""
        # Robot1 and Robot2 are ~0.95m apart
        nearby = self.query_engine.find_robots_near("Robot1", max_distance=1.0)

        # Should find Robot2
        self.assertEqual(len(nearby), 1)
        self.assertEqual(nearby[0]["robot_id"], "Robot2")
        self.assertGreater(nearby[0]["distance"], 0.9)

    def test_get_objects_in_reach(self):
        """Test getting all objects reachable by a robot."""
        objects = self.query_engine.get_objects_in_reach("Robot1")

        # Should have RedCube
        obj_ids = [obj["object_id"] for obj in objects]
        self.assertIn("RedCube", obj_ids)

        # Should have metadata
        red_cube = next(obj for obj in objects if obj["object_id"] == "RedCube")
        self.assertIsNotNone(red_cube.get("distance"))
        self.assertEqual(red_cube.get("color"), "red")

    def test_get_handoff_candidates(self):
        """Test finding handoff positions."""
        # Place object in shared zone
        self.world_state.register_object("SharedObj", position=(0.0, 0.3, 0.0))

        # Rebuild graph
        state_data = {
            "robots": [{"robot_id": "Robot1"}, {"robot_id": "Robot2"}],
            "objects": [{"object_id": "SharedObj"}],
        }
        self.builder.on_state_update(state_data)

        # Query handoff candidates
        candidates = self.query_engine.get_handoff_candidates(
            "Robot1", "Robot2", "SharedObj"
        )

        # Should have candidates if both can reach
        # (depends on specific positions and MAX_ROBOT_REACH)
        self.assertIsInstance(candidates, list)

    def test_get_graph_stats(self):
        """Test getting graph statistics."""
        stats = self.query_engine.get_graph_stats()

        self.assertIn("total_nodes", stats)
        self.assertIn("total_edges", stats)
        self.assertIn("node_types", stats)
        self.assertIn("edge_types", stats)

        # Should have robots and objects
        self.assertGreater(stats["node_types"].get("robot", 0), 0)
        self.assertGreater(stats["node_types"].get("object", 0), 0)


if __name__ == "__main__":
    unittest.main()
