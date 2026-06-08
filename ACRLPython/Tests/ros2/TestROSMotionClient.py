import math
from unittest.mock import MagicMock, patch

import pytest

# Minimal stub of ROSMotionServer usable without ROS packages


def _make_server(robot1_base=(-0.475, 0.0, 0.0), robot2_base=(0.475, 0.0, 0.0)):
    with patch.dict(
        "sys.modules",
        {
            "rclpy": MagicMock(),
            "rclpy.node": MagicMock(),
            "rclpy.action": MagicMock(),
            "geometry_msgs": MagicMock(),
            "geometry_msgs.msg": MagicMock(),
            "moveit_msgs": MagicMock(),
            "moveit_msgs.action": MagicMock(),
            "moveit_msgs.msg": MagicMock(),
            "moveit_msgs.srv": MagicMock(),
            "sensor_msgs": MagicMock(),
            "sensor_msgs.msg": MagicMock(),
            "shape_msgs": MagicMock(),
            "shape_msgs.msg": MagicMock(),
            "std_msgs": MagicMock(),
            "std_msgs.msg": MagicMock(),
            "trajectory_msgs": MagicMock(),
            "trajectory_msgs.msg": MagicMock(),
        },
    ):
        from ros2.ROSMotionClient import ROSMotionServer

    server = ROSMotionServer.__new__(ROSMotionServer)
    server.ROBOT_BASE_TRANSFORMS = {
        "Robot1": {"position": robot1_base, "y_rotation": 0.0},
        "Robot2": {"position": robot2_base, "y_rotation": 180.0},
    }
    return server


# _transform_world_to_local


class TestTransformWorldToLocal:
    def setup_method(self):
        self.server = _make_server()

    def test_robot1_identity_at_origin(self):
        result = self.server._transform_world_to_local(
            {"x": 0.0, "y": 0.0, "z": 0.0}, "Robot1"
        )
        # unity_local = (0-(-0.475), 0, 0) = (0.475, 0, 0)
        # no rotation (0°)
        # ROS: x=local_z=0, y=-local_x=-0.475, z=local_y=0
        assert result["x"] == pytest.approx(0.0)
        assert result["y"] == pytest.approx(-0.475)
        assert result["z"] == pytest.approx(0.0)

    def test_robot1_forward_target(self):
        result = self.server._transform_world_to_local(
            {"x": -0.475, "y": 0.0, "z": 0.3}, "Robot1"
        )
        # unity_local = (0, 0, 0.3), y_rot=0 → ros: x=0.3, y=0, z=0
        assert result["x"] == pytest.approx(0.3)
        assert result["y"] == pytest.approx(0.0)
        assert result["z"] == pytest.approx(0.0)

    def test_robot1_height(self):
        result = self.server._transform_world_to_local(
            {"x": -0.475, "y": 0.089, "z": 0.0}, "Robot1"
        )
        assert result["z"] == pytest.approx(0.089)
        assert result["x"] == pytest.approx(0.0)

    def test_robot1_documented_example(self):
        result = self.server._transform_world_to_local(
            {"x": 0.0, "y": 0.089, "z": 0.05}, "Robot1"
        )
        # unity_local = (0.475, 0.089, 0.05), no rotation
        # ros: x=0.05, y=-0.475, z=0.089
        assert result["x"] == pytest.approx(0.05)
        assert result["y"] == pytest.approx(-0.475)
        assert result["z"] == pytest.approx(0.089)

    def test_robot2_180_flip(self):
        # Robot2 base at (0.475, 0, 0). Put target at (0.475, 0, -0.3) — directly "in front"
        result = self.server._transform_world_to_local(
            {"x": 0.475, "y": 0.0, "z": -0.3}, "Robot2"
        )
        # unity_local = (0, 0, -0.3)
        # y_rot=180°: cos=−1, sin=0
        # rotated_x = -1*0 + 0*(-0.3) = 0
        # rotated_z = -0*0 + (-1)*(-0.3) = 0.3  ← now +Z after 180 flip
        # ros: x=0.3, y=0, z=0
        assert result["x"] == pytest.approx(0.3)
        assert result["y"] == pytest.approx(0.0)
        assert result["z"] == pytest.approx(0.0)

    def test_robot2_right_maps_to_left(self):
        result = self.server._transform_world_to_local(
            {"x": 0.475 + 0.1, "y": 0.0, "z": 0.0}, "Robot2"
        )
        # unity_local = (0.1, 0, 0)
        # y_rot=180°: rotated_x = -0.1, rotated_z = 0
        # ros: x=0, y=-(-0.1)=0.1, z=0
        assert result["y"] == pytest.approx(0.1)

    def test_unknown_robot_returns_passthrough(self):
        result = self.server._transform_world_to_local(
            {"x": 1.0, "y": 2.0, "z": 3.0}, "RobotUnknown"
        )
        assert result == {"x": 1.0, "y": 2.0, "z": 3.0}

    def test_missing_keys_default_to_zero(self):
        result = self.server._transform_world_to_local({}, "Robot1")
        # unity_local = (0.475, 0, 0), ros: x=0, y=-0.475, z=0
        assert result["y"] == pytest.approx(-0.475)


# _transform_orientation_to_ros


class TestTransformOrientationToRos:
    def setup_method(self):
        self.server = _make_server()

    def _quat_norm(self, q):
        return math.sqrt(q["x"] ** 2 + q["y"] ** 2 + q["z"] ** 2 + q["w"] ** 2)

    def test_identity_quaternion_robot1(self):
        result = self.server._transform_orientation_to_ros(
            {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}, "Robot1"
        )
        # axis relabel: rx=qz=0, ry=-qx=0, rz=qy=0, rw=qw=1
        # no extra robot2 rotation
        assert result["x"] == pytest.approx(0.0)
        assert result["y"] == pytest.approx(0.0)
        assert result["z"] == pytest.approx(0.0)
        assert result["w"] == pytest.approx(1.0)

    def test_unity_z_rotation_maps_to_ros_y(self):
        # 90° around Unity Z: q = (0, 0, sin(45°), cos(45°))
        s = math.sin(math.pi / 4)
        c = math.cos(math.pi / 4)
        result = self.server._transform_orientation_to_ros(
            {"x": 0.0, "y": 0.0, "z": s, "w": c}, "Robot1"
        )
        # axis relabel: rx=qz=s, ry=-qx=0, rz=qy=0, rw=qw=c
        assert result["x"] == pytest.approx(s)
        assert result["y"] == pytest.approx(0.0)
        assert result["z"] == pytest.approx(0.0)
        assert result["w"] == pytest.approx(c)

    def test_output_is_unit_quaternion_robot1(self):
        q = {"x": 0.1, "y": 0.2, "z": 0.3, "w": math.sqrt(1 - 0.1**2 - 0.2**2 - 0.3**2)}
        result = self.server._transform_orientation_to_ros(q, "Robot1")
        assert self._quat_norm(result) == pytest.approx(1.0, abs=1e-6)

    def test_output_is_unit_quaternion_robot2(self):
        q = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
        result = self.server._transform_orientation_to_ros(q, "Robot2")
        assert self._quat_norm(result) == pytest.approx(1.0, abs=1e-6)

    def test_robot2_identity_rotated_180(self):
        result = self.server._transform_orientation_to_ros(
            {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}, "Robot2"
        )
        # After axis relabel: (0, 0, 0, 1) in ROS
        # Robot2 applies inverse -180° around ROS Z: half=-90°, rz2=(0,0,-1,0)
        # Product rz2 * identity = rz2 itself
        half = math.radians(-180.0 / 2.0)  # = -90°
        expected_z = math.sin(half)
        expected_w = math.cos(half)
        assert result["z"] == pytest.approx(expected_z, abs=1e-6)
        assert result["w"] == pytest.approx(expected_w, abs=1e-6)

    def test_w_preserved_not_negated(self):
        q = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 0.9}
        result = self.server._transform_orientation_to_ros(q, "Robot1")
        assert result["w"] == pytest.approx(0.9)

    def test_missing_keys_default_to_identity(self):
        result = self.server._transform_orientation_to_ros({}, "Robot1")
        # default: qx=0, qy=0, qz=0, qw=1 → identity after relabel
        assert result["w"] == pytest.approx(1.0)
        assert result["x"] == pytest.approx(0.0)


# ROBOT_BASE_TRANSFORMS structure


class TestRobotBaseTransforms:
    def test_robot1_y_rotation_zero(self):
        server = _make_server()
        assert server.ROBOT_BASE_TRANSFORMS["Robot1"]["y_rotation"] == pytest.approx(
            0.0
        )

    def test_robot2_y_rotation_180(self):
        server = _make_server()
        assert server.ROBOT_BASE_TRANSFORMS["Robot2"]["y_rotation"] == pytest.approx(
            180.0
        )

    def test_robot1_base_position(self):
        server = _make_server()
        pos = server.ROBOT_BASE_TRANSFORMS["Robot1"]["position"]
        assert pos[0] == pytest.approx(-0.475)

    def test_robot2_base_position(self):
        server = _make_server()
        pos = server.ROBOT_BASE_TRANSFORMS["Robot2"]["position"]
        assert pos[0] == pytest.approx(0.475)


# Inter-robot collision object publishing


class TestPublishOtherRobotCollision:
    def test_publish_other_robot_collision_adds_4_objects(self):
        import threading
        from unittest.mock import MagicMock, patch

        server = _make_server()
        server._joint_states_lock = threading.Lock()

        js = MagicMock()
        js.position = [0.1, -0.2, 0.3, 0.0, 0.1, 0.0]
        server._current_joint_states = {"Robot2": js}
        server._planning_scene_pubs = {}

        mock_pub = MagicMock()
        mock_node = MagicMock()
        mock_node.create_publisher.return_value = mock_pub
        server._node = mock_node

        with patch("ros2.ROSMotionClient.HAS_ROS", True):
            server._publish_other_robot_collision("Robot1", "Robot2")

        # Publisher was stored for Robot1
        assert "Robot1" in server._planning_scene_pubs
        # publish was called exactly once
        assert mock_pub.publish.call_count == 1
        # The published scene has is_diff=True
        published_scene = mock_pub.publish.call_args[0][0]
        # PlanningScene is a MagicMock; we verify is_diff was assigned True
        assert published_scene.is_diff is True

    def test_publish_other_robot_collision_ids_and_count(self):
        import threading
        from unittest.mock import MagicMock, patch

        server = _make_server()
        server._joint_states_lock = threading.Lock()

        js = MagicMock()
        js.position = [0.1, -0.2, 0.3, 0.0, 0.1, 0.0]
        server._current_joint_states = {"Robot2": js}
        server._planning_scene_pubs = {}

        mock_pub = MagicMock()
        mock_node = MagicMock()
        mock_node.create_publisher.return_value = mock_pub
        server._node = mock_node

        mock_co = MagicMock(side_effect=lambda: MagicMock())
        with patch("ros2.ROSMotionClient.HAS_ROS", True), patch(
            "ros2.ROSMotionClient.CollisionObject", mock_co
        ), patch(
            "ros2.ROSMotionClient.SolidPrimitive",
            MagicMock(side_effect=lambda: MagicMock()),
        ), patch(
            "ros2.ROSMotionClient.PlanningScene", MagicMock()
        ), patch(
            "ros2.ROSMotionClient.Pose", MagicMock(side_effect=lambda: MagicMock())
        ):
            server._publish_other_robot_collision("Robot1", "Robot2")

        # Verify the scene's world.collision_objects has 4 entries with correct IDs
        scene = mock_pub.publish.call_args[0][0]
        objs = scene.world.collision_objects
        assert len(objs) == 4
        expected_ids = [f"Robot2_link_{i}" for i in range(4)]
        for i, obj in enumerate(objs):
            assert obj.id == expected_ids[i]

    def test_no_collision_objects_if_no_joint_state(self):
        import threading
        from unittest.mock import patch

        server = _make_server()
        server._joint_states_lock = threading.Lock()
        server._current_joint_states = {}
        server._planning_scene_pubs = {}

        with patch("ros2.ROSMotionClient.HAS_ROS", True):
            server._publish_other_robot_collision("Robot1", "Robot2")

        # No publisher should have been created, nothing was published
        assert server._planning_scene_pubs == {}

    def test_remove_other_robot_collision_removes_4_objects(self):
        from unittest.mock import MagicMock, patch

        server = _make_server()
        mock_pub = MagicMock()
        server._planning_scene_pubs = {"Robot1": mock_pub}

        mock_co = MagicMock()
        with patch("ros2.ROSMotionClient.HAS_ROS", True), patch(
            "ros2.ROSMotionClient.CollisionObject", mock_co
        ), patch("ros2.ROSMotionClient.PlanningScene", MagicMock()):
            server._remove_other_robot_collision("Robot1", "Robot2")

        assert mock_pub.publish.call_count == 1
        scene = mock_pub.publish.call_args[0][0]
        assert scene.is_diff is True
        objs = scene.world.collision_objects
        assert len(objs) == 4

    def test_inter_robot_collision_disabled_skips_publish(self):
        import threading
        from unittest.mock import MagicMock, patch

        server = _make_server()
        server._joint_states_lock = threading.Lock()
        server._current_joint_states = {}
        server._planning_scene_pubs = {}
        server._publish_other_robot_collision = MagicMock()
        server._move_group_server_ready = {"Robot1": True}

        mock_client = MagicMock()
        server._move_group_clients = {"Robot1": mock_client}

        def fake_wait(robot_id, timeout=5.0):
            return True

        server._wait_for_joint_states = fake_wait

        with patch("ros2.ROSMotionClient.INTER_ROBOT_COLLISION_ENABLED", False):
            try:
                server._call_move_group_plan(MagicMock(), "Robot1")
            except Exception:
                pass

        assert server._publish_other_robot_collision.call_count == 0
