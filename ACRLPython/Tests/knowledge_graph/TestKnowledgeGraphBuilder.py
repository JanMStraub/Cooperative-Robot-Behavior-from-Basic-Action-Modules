from knowledge_graph import KnowledgeGraph, GraphBuilder, GraphQueryEngine  # type: ignore[attr-defined]
from operations.WorldState import get_world_state


class TestGraphBuilder:

    def setup_method(self):
        self.world_state = get_world_state()
        self.world_state.reset()
        self.graph = KnowledgeGraph()
        self.builder = GraphBuilder(self.graph, self.world_state)

    def teardown_method(self):
        self.graph.clear()
        self.world_state.reset()

    def test_static_regions_initialized(self):
        assert self.graph.has_node("left_workspace")
        assert self.graph.has_node("right_workspace")
        assert self.graph.has_node("shared_zone")

        left_neighbors = self.graph.get_neighbors(
            "left_workspace", edge_type="ADJACENT_TO"
        )
        assert "shared_zone" in left_neighbors

    def test_update_robot_nodes(self):
        self.world_state.update_robot(
            "Robot1", position=(-0.3, 0.2, 0.1), gripper_state="open"
        )

        state_data = {"robots": [{"robot_id": "Robot1"}], "objects": []}
        self.builder.on_state_update(state_data)

        assert self.graph.has_node("Robot1")
        robot_node = self.graph.get_node("Robot1")
        assert robot_node is not None
        assert robot_node["node_type"] == "robot"
        assert robot_node["gripper_state"] == "open"

    def test_update_object_nodes(self):
        self.world_state.register_object(
            "RedCube", position=(0.1, 0.3, 0.0), color="red"
        )

        state_data = {"robots": [], "objects": [{"object_id": "RedCube"}]}
        self.builder.on_state_update(state_data)

        assert self.graph.has_node("RedCube")
        obj_node = self.graph.get_node("RedCube")
        assert obj_node is not None
        assert obj_node["node_type"] == "object"
        assert obj_node["color"] == "red"

    def test_can_reach_edges_computed(self):
        self.world_state.update_robot("Robot1", position=(-0.475, 0.0, 0.0))
        self.world_state.register_object("NearbyObj", position=(-0.3, 0.3, 0.0))

        state_data = {
            "robots": [{"robot_id": "Robot1"}],
            "objects": [{"object_id": "NearbyObj"}],
        }
        self.builder.on_state_update(state_data)

        reachable = self.graph.get_neighbors("Robot1", edge_type="CAN_REACH")
        assert "NearbyObj" in reachable

    def test_near_edges_computed(self):
        self.world_state.register_object("Obj1", position=(0.0, 0.3, 0.0))
        self.world_state.register_object("Obj2", position=(0.01, 0.3, 0.0))  # 1cm away

        state_data = {
            "robots": [],
            "objects": [{"object_id": "Obj1"}, {"object_id": "Obj2"}],
        }
        self.builder.on_state_update(state_data)

        near_obj1 = self.graph.get_neighbors("Obj1", edge_type="NEAR")
        assert "Obj2" in near_obj1

        near_obj2 = self.graph.get_neighbors("Obj2", edge_type="NEAR")
        assert "Obj1" in near_obj2

    def test_in_region_edges_computed(self):
        self.world_state.update_robot("Robot1", position=(-0.3, 0.3, 0.0))

        state_data = {"robots": [{"robot_id": "Robot1"}], "objects": []}
        self.builder.on_state_update(state_data)

        in_region = self.graph.get_neighbors("Robot1", edge_type="IN_REGION")
        assert "left_workspace" in in_region

    def test_grasping_edges_computed(self):
        self.world_state.update_robot("Robot1", position=(-0.3, 0.2, 0.0))
        self.world_state.register_object("GraspedCube", position=(-0.3, 0.2, 0.0))
        self.world_state.mark_object_grasped("GraspedCube", "Robot1")

        state_data = {
            "robots": [{"robot_id": "Robot1"}],
            "objects": [{"object_id": "GraspedCube"}],
        }
        self.builder.on_state_update(state_data)

        grasping = self.graph.get_neighbors("Robot1", edge_type="GRASPING")
        assert "GraspedCube" in grasping

    def test_allocated_edges_computed(self):
        self.world_state.update_robot("Robot1", position=(-0.3, 0.2, 0.0))
        self.world_state.allocate_workspace("left_workspace", "Robot1")

        state_data = {"robots": [{"robot_id": "Robot1"}], "objects": []}
        self.builder.on_state_update(state_data)

        allocated = self.graph.get_neighbors("left_workspace", edge_type="ALLOCATED")
        assert "Robot1" in allocated

    def test_stale_objects_removed(self):
        self.world_state.register_object("TempObj", position=(0.1, 0.3, 0.0))

        state_data = {"robots": [], "objects": [{"object_id": "TempObj"}]}
        self.builder.on_state_update(state_data)
        assert self.graph.has_node("TempObj")

        state_data = {"robots": [], "objects": []}
        self.builder.on_state_update(state_data)

        assert not self.graph.has_node("TempObj")


class TestGraphQueryEngine:

    def setup_method(self):
        self.world_state = get_world_state()
        self.world_state.reset()
        self.graph = KnowledgeGraph()
        self.builder = GraphBuilder(self.graph, self.world_state)
        self.query_engine = GraphQueryEngine(self.graph)

        self.world_state.update_robot("Robot1", position=(-0.475, 0.0, 0.0))
        self.world_state.update_robot("Robot2", position=(0.475, 0.0, 0.0))
        self.world_state.register_object(
            "RedCube", position=(-0.3, 0.3, 0.0), color="red"
        )
        self.world_state.register_object(
            "BlueCube", position=(0.3, 0.3, 0.0), color="blue"
        )

        state_data = {
            "robots": [{"robot_id": "Robot1"}, {"robot_id": "Robot2"}],
            "objects": [{"object_id": "RedCube"}, {"object_id": "BlueCube"}],
        }
        self.builder.on_state_update(state_data)

    def teardown_method(self):
        self.graph.clear()
        self.world_state.reset()

    def test_find_reachable_robots(self):
        reachable = self.query_engine.find_reachable_robots("RedCube")
        assert "Robot1" in reachable

    def test_find_robots_near(self):
        nearby = self.query_engine.find_robots_near("Robot1", max_distance=1.0)

        assert len(nearby) == 1
        assert nearby[0]["robot_id"] == "Robot2"
        assert nearby[0]["distance"] > 0.9

    def test_get_objects_in_reach(self):
        objects = self.query_engine.get_objects_in_reach("Robot1")

        obj_ids = [obj["object_id"] for obj in objects]
        assert "RedCube" in obj_ids

        red_cube = next(obj for obj in objects if obj["object_id"] == "RedCube")
        assert red_cube.get("distance") is not None
        assert red_cube.get("color") == "red"

    def test_get_handoff_candidates(self):
        self.world_state.register_object("SharedObj", position=(0.0, 0.3, 0.0))

        state_data = {
            "robots": [{"robot_id": "Robot1"}, {"robot_id": "Robot2"}],
            "objects": [{"object_id": "SharedObj"}],
        }
        self.builder.on_state_update(state_data)

        candidates = self.query_engine.get_handoff_candidates(
            "Robot1", "Robot2", "SharedObj"
        )

        assert isinstance(candidates, list)

    def test_get_graph_stats(self):
        stats = self.query_engine.get_graph_stats()

        assert "total_nodes" in stats
        assert "total_edges" in stats
        assert "node_types" in stats
        assert "edge_types" in stats

        assert stats["node_types"].get("robot", 0) > 0
        assert stats["node_types"].get("object", 0) > 0
