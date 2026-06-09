import socket
import time
import threading
import pytest
from servers.WorldStateServer import WorldStateServer
from core.UnityProtocol import UnityProtocol


@pytest.fixture(autouse=True, scope="class")
def server_setup(request):
    from core.TCPServerBase import ServerConfig

    config = ServerConfig(host="127.0.0.1", port=5914)
    server = WorldStateServer(config=config)
    server.start()
    time.sleep(0.5)
    request.cls.server = server
    yield
    server.stop()
    time.sleep(0.5)


class TestWorldStateServer:
    server: WorldStateServer

    def setup_method(self):
        self.client = None

    def teardown_method(self):
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass

    def _create_client(self) -> socket.socket:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", 5914))
        return client

    def _send_world_state(self, client: socket.socket, state_data: dict) -> None:
        message = UnityProtocol.encode_status_response(state_data, request_id=0)
        client.sendall(message)

    def test_server_starts_and_stops(self):
        assert self.server.is_running()

    def test_receive_world_state_update(self):
        world_state = {
            "type": "world_state_update",
            "robots": [
                {
                    "robot_id": "Robot1",
                    "position": {"x": 1.0, "y": 0.5, "z": 2.0},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    "target_position": {"x": 1.5, "y": 0.5, "z": 2.5},
                    "gripper_state": "open",
                    "is_moving": True,
                    "is_initialized": True,
                    "joint_angles": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                }
            ],
            "objects": [
                {
                    "object_id": "RedCube",
                    "position": {"x": 2.0, "y": 0.1, "z": 3.0},
                    "color": "red",
                    "object_type": "cube",
                    "confidence": 0.95,
                }
            ],
            "timestamp": 123.45,
        }

        self.client = self._create_client()
        self._send_world_state(self.client, world_state)
        time.sleep(0.5)

        latest_state = self.server.get_latest_state()
        assert latest_state is not None
        assert latest_state["type"] == "world_state_update"
        assert len(latest_state["robots"]) == 1
        assert len(latest_state["objects"]) == 1
        assert latest_state["timestamp"] == 123.45

    def test_get_robot_state(self):
        world_state = {
            "type": "world_state_update",
            "robots": [
                {
                    "robot_id": "Robot1",
                    "position": {"x": 1.0, "y": 0.5, "z": 2.0},
                    "gripper_state": "closed",
                    "is_moving": False,
                    "is_initialized": True,
                    "joint_angles": [],
                },
                {
                    "robot_id": "Robot2",
                    "position": {"x": -1.0, "y": 0.5, "z": -2.0},
                    "gripper_state": "open",
                    "is_moving": True,
                    "is_initialized": True,
                    "joint_angles": [],
                },
            ],
            "objects": [],
            "timestamp": 200.0,
        }

        self.client = self._create_client()
        self._send_world_state(self.client, world_state)
        time.sleep(0.5)

        robot1 = self.server.get_robot_state("Robot1")
        assert robot1 is not None
        assert robot1["robot_id"] == "Robot1"
        assert robot1["gripper_state"] == "closed"
        assert not robot1["is_moving"]

        robot2 = self.server.get_robot_state("Robot2")
        assert robot2 is not None
        assert robot2["robot_id"] == "Robot2"
        assert robot2["is_moving"]

        assert self.server.get_robot_state("Robot3") is None

    def test_get_object_state(self):
        world_state = {
            "type": "world_state_update",
            "robots": [],
            "objects": [
                {
                    "object_id": "RedCube",
                    "position": {"x": 1.0, "y": 0.1, "z": 2.0},
                    "color": "red",
                    "object_type": "cube",
                    "confidence": 0.9,
                },
                {
                    "object_id": "BlueSphere",
                    "position": {"x": -1.0, "y": 0.2, "z": -2.0},
                    "color": "blue",
                    "object_type": "sphere",
                    "confidence": 0.85,
                },
            ],
            "timestamp": 300.0,
        }

        self.client = self._create_client()
        self._send_world_state(self.client, world_state)
        time.sleep(0.5)

        red_cube = self.server.get_object_state("RedCube")
        assert red_cube is not None
        assert red_cube["object_id"] == "RedCube"
        assert red_cube["color"] == "red"

        blue_sphere = self.server.get_object_state("BlueSphere")
        assert blue_sphere is not None
        assert blue_sphere["color"] == "blue"

        assert self.server.get_object_state("GreenCube") is None

    def test_get_all_ids(self):
        world_state = {
            "type": "world_state_update",
            "robots": [
                {"robot_id": "Robot1", "is_moving": False},
                {"robot_id": "Robot2", "is_moving": True},
            ],
            "objects": [
                {"object_id": "Cube1", "color": "red"},
                {"object_id": "Cube2", "color": "blue"},
                {"object_id": "Sphere1", "color": "green"},
            ],
            "timestamp": 400.0,
        }

        self.client = self._create_client()
        self._send_world_state(self.client, world_state)
        time.sleep(0.5)

        robot_ids = self.server.get_all_robot_ids()
        assert len(robot_ids) == 2
        assert "Robot1" in robot_ids
        assert "Robot2" in robot_ids

        object_ids = self.server.get_all_object_ids()
        assert len(object_ids) == 3
        assert "Cube1" in object_ids
        assert "Cube2" in object_ids
        assert "Sphere1" in object_ids

    def test_statistics(self):
        stats = self.server.get_statistics()
        initial_count = stats["updates_received"]

        for i in range(3):
            world_state = {
                "type": "world_state_update",
                "robots": [],
                "objects": [],
                "timestamp": float(i),
            }
            self.client = self._create_client()
            self._send_world_state(self.client, world_state)
            time.sleep(0.3)
            self.client.close()

        stats = self.server.get_statistics()
        assert stats["updates_received"] == initial_count + 3
        assert stats["has_state"]
        assert stats["last_update_time"] is not None

    def test_thread_safety(self):
        world_state = {
            "type": "world_state_update",
            "robots": [{"robot_id": "Robot1", "is_moving": False}],
            "objects": [],
            "timestamp": 500.0,
        }
        self.client = self._create_client()
        self._send_world_state(self.client, world_state)
        time.sleep(0.5)

        results = []

        def read_state():
            for _ in range(10):
                state = self.server.get_latest_state()
                results.append(state is not None)
                time.sleep(0.01)

        threads = [threading.Thread(target=read_state) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(results)
