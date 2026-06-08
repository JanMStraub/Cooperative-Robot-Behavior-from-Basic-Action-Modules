from operations.WorldState import get_world_state
from config.Robot import WORKSPACE_REGIONS


class TestWorldStateSpatial:

    def setup_method(self):
        self.world_state = get_world_state()
        self.world_state.reset()

    def test_find_objects_near_returns_correct_objects(self):
        self.world_state.register_object("obj1", position=(0.0, 0.0, 0.0))
        self.world_state.register_object("obj2", position=(0.05, 0.05, 0.0))
        self.world_state.register_object("obj3", position=(0.5, 0.5, 0.0))

        nearby = self.world_state.find_objects_near((0.0, 0.0, 0.0), radius=0.1)

        # obj1 is 0.0m away, obj2 is ~0.07m, obj3 is 0.7m
        nearby_ids = {obj.object_id for obj in nearby}
        assert "obj1" in nearby_ids
        assert "obj2" in nearby_ids
        assert "obj3" not in nearby_ids

    def test_find_objects_near_excludes_stale(self):
        self.world_state.register_object("obj1", position=(0.0, 0.0, 0.0))
        self.world_state.register_object("obj2", position=(0.05, 0.0, 0.0))
        self.world_state._objects["obj2"].stale = True

        nearby = self.world_state.find_objects_near((0.0, 0.0, 0.0), radius=0.1)
        nearby_ids = {obj.object_id for obj in nearby}
        assert "obj1" in nearby_ids
        assert "obj2" not in nearby_ids

        nearby_all = self.world_state.find_objects_near(
            (0.0, 0.0, 0.0), radius=0.1, exclude_stale=False
        )
        nearby_all_ids = {obj.object_id for obj in nearby_all}
        assert "obj1" in nearby_all_ids
        assert "obj2" in nearby_all_ids

    def test_find_robots_near(self):
        self.world_state.update_robot("Robot1", position=(-0.3, 0.2, 0.0))
        self.world_state.update_robot("Robot2", position=(0.3, 0.2, 0.0))

        nearby = self.world_state.find_robots_near((0.0, 0.2, 0.0), radius=0.4)
        # both robots are ~0.3m from center
        nearby_ids = {robot.robot_id for robot in nearby}
        assert "Robot1" in nearby_ids
        assert "Robot2" in nearby_ids

        nearby_narrow = self.world_state.find_robots_near((0.0, 0.2, 0.0), radius=0.2)
        assert len(nearby_narrow) == 0

    def test_get_reachable_objects(self):
        # Robot1 base is at (-0.475, 0, 0), MAX_ROBOT_REACH is 0.8m
        self.world_state.update_robot("Robot1", position=(-0.475, 0.0, 0.0))
        self.world_state.register_object("nearby_obj", position=(-0.3, 0.3, 0.0))
        self.world_state.register_object("far_obj", position=(0.5, 0.3, 0.0))

        reachable = self.world_state.get_reachable_objects("Robot1")
        reachable_ids = {obj.object_id for obj in reachable}
        assert "nearby_obj" in reachable_ids
        assert "far_obj" not in reachable_ids

    def test_get_reachable_objects_excludes_stale(self):
        self.world_state.update_robot("Robot1", position=(-0.475, 0.0, 0.0))
        self.world_state.register_object("obj1", position=(-0.3, 0.3, 0.0))
        self.world_state.register_object("obj2", position=(-0.2, 0.3, 0.0))
        self.world_state._objects["obj2"].stale = True

        reachable = self.world_state.get_reachable_objects("Robot1")
        reachable_ids = {obj.object_id for obj in reachable}
        assert "obj1" in reachable_ids
        assert "obj2" not in reachable_ids

        reachable_all = self.world_state.get_reachable_objects(
            "Robot1", exclude_stale=False
        )
        reachable_all_ids = {obj.object_id for obj in reachable_all}
        assert "obj1" in reachable_all_ids
        assert "obj2" in reachable_all_ids

    def test_get_objects_in_region(self):
        # left_workspace: x_min=-0.5, x_max=-0.15
        self.world_state.register_object("left_obj", position=(-0.3, 0.3, 0.0))
        # right_workspace: x_min=0.15, x_max=0.5
        self.world_state.register_object("right_obj", position=(0.3, 0.3, 0.0))
        # shared_zone: x_min=-0.15, x_max=0.15
        self.world_state.register_object("shared_obj", position=(0.0, 0.3, 0.0))

        left_objs = self.world_state.get_objects_in_region("left_workspace")
        left_ids = {obj.object_id for obj in left_objs}
        assert "left_obj" in left_ids
        assert "right_obj" not in left_ids
        assert "shared_obj" not in left_ids

        shared_objs = self.world_state.get_objects_in_region("shared_zone")
        shared_ids = {obj.object_id for obj in shared_objs}
        assert "shared_obj" in shared_ids
        assert "left_obj" not in shared_ids
        assert "right_obj" not in shared_ids

    def test_get_objects_in_region_unknown_region(self):
        objs = self.world_state.get_objects_in_region("nonexistent_region")
        assert len(objs) == 0

    def test_get_region_for_position(self):
        assert (
            self.world_state.get_region_for_position((-0.3, 0.3, 0.0))
            == "left_workspace"
        )
        assert (
            self.world_state.get_region_for_position((0.3, 0.3, 0.0))
            == "right_workspace"
        )
        assert (
            self.world_state.get_region_for_position((0.0, 0.3, 0.0)) == "shared_zone"
        )
        assert self.world_state.get_region_for_position((10.0, 10.0, 10.0)) is None

    def test_get_region_for_position_boundaries(self):
        left = WORKSPACE_REGIONS["left_workspace"]

        region = self.world_state.get_region_for_position(
            (left["x_min"], left["y_min"], left["z_min"])
        )
        assert region == "left_workspace"

        region = self.world_state.get_region_for_position(
            (left["x_max"], left["y_max"], left["z_max"])
        )
        # x_max=-0.15 overlaps with shared_zone x_min=-0.15;
        # WORKSPACE_REGIONS dict order determines which is returned first
        assert region in ["left_workspace", "shared_zone"]

    def test_get_world_context_string_basic(self):
        self.world_state.update_robot(
            "Robot1", position=(-0.3, 0.2, 0.1), gripper_state="open"
        )
        self.world_state.register_object(
            "RedCube", position=(-0.2, 0.3, 0.0), color="red"
        )

        context = self.world_state.get_world_context_string("Robot1")

        assert "Robot1" in context
        assert "-0.30, 0.20, 0.10" in context
        assert "gripper open" in context
        assert "RedCube" in context
        assert "-0.20, 0.30, 0.00" in context

    def test_get_world_context_string_with_annotations(self):
        self.world_state.update_robot(
            "Robot1", position=(-0.475, 0.0, 0.0), gripper_state="closed"
        )
        self.world_state.register_object("Obj1", position=(-0.3, 0.3, 0.0))
        self.world_state.register_object("Obj2", position=(0.3, 0.3, 0.0))

        context = self.world_state.get_world_context_string("Robot1")

        assert "Obj1" in context
        assert "Obj2" in context
        assert "left_workspace" in context
        assert "right_workspace" in context

    def test_get_world_context_string_with_grasped_object(self):
        self.world_state.update_robot(
            "Robot1", position=(-0.3, 0.2, 0.0), gripper_state="closed"
        )
        self.world_state.register_object("Cube", position=(-0.3, 0.2, 0.0))
        self.world_state.mark_object_grasped("Cube", "Robot1")

        context = self.world_state.get_world_context_string("Robot1")
        assert "grasped by Robot1" in context

    def test_get_world_context_string_with_stale_object(self):
        self.world_state.update_robot("Robot1", position=(-0.3, 0.2, 0.0))
        self.world_state.register_object("StaleObj", position=(-0.2, 0.3, 0.0))
        self.world_state._objects["StaleObj"].stale = True

        context = self.world_state.get_world_context_string("Robot1")
        assert "stale" in context

    def test_get_world_context_string_no_objects(self):
        self.world_state.update_robot("Robot1", position=(-0.3, 0.2, 0.0))

        context = self.world_state.get_world_context_string("Robot1")
        assert "No objects detected" in context

    def test_get_world_context_string_unknown_robot(self):
        context = self.world_state.get_world_context_string("UnknownRobot")
        assert "state unknown" in context

    def test_empty_searches(self):
        # No objects registered
        assert len(self.world_state.find_objects_near((0.0, 0.0, 0.0))) == 0
        assert len(self.world_state.get_reachable_objects("Robot1")) == 0
        assert len(self.world_state.get_objects_in_region("left_workspace")) == 0
