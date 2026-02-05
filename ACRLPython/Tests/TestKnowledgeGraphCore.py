"""
Test suite for KnowledgeGraph Core
===================================

Tests the NetworkX-based knowledge graph wrapper:
- Node CRUD operations
- Edge CRUD operations with types
- Thread safety
- GraphML save/load
- Neighbor queries with edge type filtering
"""

import unittest
import os
import tempfile
import threading
import time
from knowledge_graph.Core import KnowledgeGraph
from knowledge_graph.Schema import RobotNode


class TestKnowledgeGraphCore(unittest.TestCase):
    """Test KnowledgeGraph core operations."""

    def setUp(self):
        """Create fresh graph for each test."""
        self.graph = KnowledgeGraph()

    def tearDown(self):
        """Clean up."""
        self.graph.clear()

    def test_add_node(self):
        """Test adding nodes to graph."""
        self.graph.add_node("Robot1", node_type="robot", position=(-0.3, 0.2, 0.1))

        self.assertTrue(self.graph.has_node("Robot1"))
        node_attrs = self.graph.get_node("Robot1")
        self.assertIsNotNone(node_attrs)
        assert node_attrs is not None  # Type narrowing for Pylance
        self.assertEqual(node_attrs["node_type"], "robot")
        self.assertEqual(node_attrs["position"], (-0.3, 0.2, 0.1))

    def test_add_node_with_schema(self):
        """Test adding node using schema dataclass."""
        robot = RobotNode(
            node_id="Robot1",
            position=(-0.3, 0.2, 0.1),
            gripper_state="open"
        )

        self.graph.add_node(robot.node_id, **robot.to_dict())

        node_attrs = self.graph.get_node("Robot1")
        self.assertIsNotNone(node_attrs)
        assert node_attrs is not None  # Type narrowing for Pylance
        self.assertEqual(node_attrs["gripper_state"], "open")

    def test_remove_node(self):
        """Test removing nodes."""
        self.graph.add_node("Robot1", node_type="robot")
        self.assertTrue(self.graph.has_node("Robot1"))

        success = self.graph.remove_node("Robot1")
        self.assertTrue(success)
        self.assertFalse(self.graph.has_node("Robot1"))

    def test_remove_nonexistent_node(self):
        """Test removing node that doesn't exist."""
        success = self.graph.remove_node("NonexistentNode")
        self.assertFalse(success)

    def test_get_node_nonexistent(self):
        """Test getting nonexistent node returns None."""
        node_attrs = self.graph.get_node("NonexistentNode")
        self.assertIsNone(node_attrs)

    def test_add_edge(self):
        """Test adding edges."""
        self.graph.add_node("Robot1", node_type="robot")
        self.graph.add_node("RedCube", node_type="object")

        self.graph.add_edge("Robot1", "RedCube", "CAN_REACH", distance=0.5)

        neighbors = self.graph.get_neighbors("Robot1")
        self.assertIn("RedCube", neighbors)

    def test_add_multiple_edges_same_nodes(self):
        """Test adding multiple edges between same nodes (MultiDiGraph)."""
        self.graph.add_node("Robot1", node_type="robot")
        self.graph.add_node("RedCube", node_type="object")

        self.graph.add_edge("Robot1", "RedCube", "CAN_REACH", distance=0.5)
        self.graph.add_edge("Robot1", "RedCube", "NEAR", distance=0.1)

        # Should support both edges
        neighbors = self.graph.get_neighbors("Robot1")
        self.assertIn("RedCube", neighbors)

    def test_remove_edge(self):
        """Test removing edges."""
        self.graph.add_node("Robot1", node_type="robot")
        self.graph.add_node("RedCube", node_type="object")
        self.graph.add_edge("Robot1", "RedCube", "CAN_REACH")

        count = self.graph.remove_edge("Robot1", "RedCube")
        self.assertEqual(count, 1)

        neighbors = self.graph.get_neighbors("Robot1")
        self.assertNotIn("RedCube", neighbors)

    def test_remove_edge_by_type(self):
        """Test removing specific edge type."""
        self.graph.add_node("Robot1", node_type="robot")
        self.graph.add_node("RedCube", node_type="object")
        self.graph.add_edge("Robot1", "RedCube", "CAN_REACH")
        self.graph.add_edge("Robot1", "RedCube", "NEAR")

        # Remove only CAN_REACH edge
        count = self.graph.remove_edge("Robot1", "RedCube", edge_type="CAN_REACH")
        self.assertEqual(count, 1)

        # NEAR edge should still exist
        neighbors_near = self.graph.get_neighbors("Robot1", edge_type="NEAR")
        self.assertIn("RedCube", neighbors_near)

    def test_get_neighbors_filtered_by_edge_type(self):
        """Test getting neighbors filtered by edge type."""
        self.graph.add_node("Robot1", node_type="robot")
        self.graph.add_node("RedCube", node_type="object")
        self.graph.add_node("BlueCube", node_type="object")

        self.graph.add_edge("Robot1", "RedCube", "CAN_REACH")
        self.graph.add_edge("Robot1", "BlueCube", "GRASPING")

        reachable = self.graph.get_neighbors("Robot1", edge_type="CAN_REACH")
        self.assertIn("RedCube", reachable)
        self.assertNotIn("BlueCube", reachable)

        grasped = self.graph.get_neighbors("Robot1", edge_type="GRASPING")
        self.assertIn("BlueCube", grasped)
        self.assertNotIn("RedCube", grasped)

    def test_get_predecessors(self):
        """Test getting predecessors (incoming edges)."""
        self.graph.add_node("Robot1", node_type="robot")
        self.graph.add_node("RedCube", node_type="object")

        self.graph.add_edge("Robot1", "RedCube", "CAN_REACH")

        predecessors = self.graph.get_predecessors("RedCube")
        self.assertIn("Robot1", predecessors)

    def test_get_all_nodes(self):
        """Test getting all nodes."""
        self.graph.add_node("Robot1", node_type="robot")
        self.graph.add_node("Robot2", node_type="robot")
        self.graph.add_node("RedCube", node_type="object")

        all_nodes = self.graph.get_all_nodes()
        self.assertEqual(len(all_nodes), 3)

        # Filter by type
        robots = self.graph.get_all_nodes(node_type="robot")
        self.assertEqual(len(robots), 2)
        self.assertIn("Robot1", robots)
        self.assertIn("Robot2", robots)

        objects = self.graph.get_all_nodes(node_type="object")
        self.assertEqual(len(objects), 1)
        self.assertIn("RedCube", objects)

    def test_node_and_edge_counts(self):
        """Test getting node and edge counts."""
        self.assertEqual(self.graph.node_count(), 0)
        self.assertEqual(self.graph.edge_count(), 0)

        self.graph.add_node("Robot1", node_type="robot")
        self.graph.add_node("RedCube", node_type="object")
        self.assertEqual(self.graph.node_count(), 2)

        self.graph.add_edge("Robot1", "RedCube", "CAN_REACH")
        self.assertEqual(self.graph.edge_count(), 1)

    def test_clear(self):
        """Test clearing graph."""
        self.graph.add_node("Robot1", node_type="robot")
        self.graph.add_node("RedCube", node_type="object")
        self.graph.add_edge("Robot1", "RedCube", "CAN_REACH")

        self.graph.clear()

        self.assertEqual(self.graph.node_count(), 0)
        self.assertEqual(self.graph.edge_count(), 0)

    def test_get_stats(self):
        """Test getting graph statistics."""
        self.graph.add_node("Robot1", node_type="robot")
        self.graph.add_node("Robot2", node_type="robot")
        self.graph.add_node("RedCube", node_type="object")
        self.graph.add_node("left_workspace", node_type="region")

        stats = self.graph.get_stats()

        self.assertEqual(stats["total_nodes"], 4)
        self.assertEqual(stats["node_types"]["robot"], 2)
        self.assertEqual(stats["node_types"]["object"], 1)
        self.assertEqual(stats["node_types"]["region"], 1)

    def test_save_and_load_graphml(self):
        """Test saving and loading graph to/from GraphML."""
        # Build graph
        self.graph.add_node("Robot1", node_type="robot", position=(-0.3, 0.2, 0.1))
        self.graph.add_node("RedCube", node_type="object", color="red")
        self.graph.add_edge("Robot1", "RedCube", "CAN_REACH", distance=0.5)

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.graphml', delete=False) as f:
            temp_path = f.name

        try:
            self.graph.save_graphml(temp_path)

            # Load into new graph
            new_graph = KnowledgeGraph()
            new_graph.load_graphml(temp_path)

            # Verify nodes and edges
            self.assertEqual(new_graph.node_count(), 2)
            self.assertEqual(new_graph.edge_count(), 1)
            self.assertTrue(new_graph.has_node("Robot1"))
            self.assertTrue(new_graph.has_node("RedCube"))

        finally:
            os.unlink(temp_path)

    def test_thread_safety_concurrent_reads(self):
        """Test thread safety with concurrent read operations."""
        # Populate graph
        for i in range(10):
            self.graph.add_node(f"Node{i}", node_type="test")

        errors = []

        def read_nodes():
            try:
                for _ in range(100):
                    nodes = self.graph.get_all_nodes()
                    self.assertGreaterEqual(len(nodes), 0)
            except Exception as e:
                errors.append(e)

        # Start 10 reader threads
        threads = [threading.Thread(target=read_nodes) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors during concurrent reads: {errors}")

    def test_thread_safety_concurrent_writes(self):
        """Test thread safety with concurrent write operations."""
        errors = []

        def add_nodes(offset):
            try:
                for i in range(10):
                    node_id = f"Node{offset}_{i}"
                    self.graph.add_node(node_id, node_type="test")
            except Exception as e:
                errors.append(e)

        # Start 5 writer threads
        threads = [threading.Thread(target=add_nodes, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors during concurrent writes: {errors}")
        self.assertEqual(self.graph.node_count(), 50)  # 5 threads * 10 nodes each

    def test_thread_safety_mixed_operations(self):
        """Test thread safety with mixed read/write operations."""
        # Pre-populate
        for i in range(10):
            self.graph.add_node(f"Node{i}", node_type="test")

        errors = []

        def writer():
            try:
                for i in range(20):
                    self.graph.add_node(f"WriterNode{threading.get_ident()}_{i}", node_type="test")
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    self.graph.get_all_nodes()
                    self.graph.node_count()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        # Start 5 readers and 2 writers
        threads = []
        threads.extend([threading.Thread(target=reader) for _ in range(5)])
        threads.extend([threading.Thread(target=writer) for _ in range(2)])

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors during mixed operations: {errors}")


if __name__ == "__main__":
    unittest.main()
