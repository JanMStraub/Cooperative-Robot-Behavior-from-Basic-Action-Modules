import numpy as np
import threading
import time
from unittest.mock import Mock, patch

from servers.ImageServer import UnifiedImageStorage
from servers.CommandServer import CommandBroadcaster
from operations.Registry import get_global_registry
from operations.Base import (
    BasicOperation,
    OperationResult,
    OperationCategory,
    OperationComplexity,
)
from operations.MoveOperations import move_to_coordinate
from operations.WorldState import WorldState


class TestConcurrentImageOperations:

    def test_concurrent_image_writes(self, cleanup_singletons):
        storage = UnifiedImageStorage()
        num_threads = 20
        images_per_thread = 10

        errors = []

        def write_images(thread_id):
            try:
                for i in range(images_per_thread):
                    camera_id = f"cam_t{thread_id}_i{i}"
                    image = np.ones((50, 50, 3), dtype=np.uint8) * thread_id
                    storage.store_single_image(
                        camera_id, image, f"prompt_{thread_id}_{i}"
                    )
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=write_images, args=(i,)) for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        # Images may be evicted due to MAX_STORED_IMAGES limit (20 images)
        camera_ids = storage.get_all_camera_ids()
        expected = min(num_threads * images_per_thread, 20)
        assert len(camera_ids) == expected

    def test_concurrent_read_write_mix(self, cleanup_singletons):
        storage = UnifiedImageStorage()
        camera_id = "shared_camera"

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        storage.store_single_image(camera_id, image, "initial")

        read_count = [0]
        write_count = [0]
        errors = []

        def read_loop():
            try:
                for _ in range(100):
                    img = storage.get_single_image(camera_id)
                    if img is not None:
                        read_count[0] += 1
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def write_loop():
            try:
                for i in range(100):
                    new_img = np.ones((100, 100, 3), dtype=np.uint8) * i
                    storage.store_single_image(camera_id, new_img, f"update_{i}")
                    write_count[0] += 1
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        readers = [threading.Thread(target=read_loop) for _ in range(5)]
        writers = [threading.Thread(target=write_loop) for _ in range(3)]

        for t in readers + writers:
            t.start()
        for t in readers + writers:
            t.join()

        assert len(errors) == 0
        assert read_count[0] > 0
        assert write_count[0] == 300  # 3 writers * 100 writes each

    def test_concurrent_cleanup(self, cleanup_singletons):
        storage = UnifiedImageStorage()

        for i in range(50):
            image = np.zeros((50, 50, 3), dtype=np.uint8)
            storage.store_single_image(f"cam_{i}", image, "")

        errors = []

        def access_images():
            try:
                for _ in range(100):
                    cam_ids = storage.get_all_camera_ids()
                    if cam_ids:
                        storage.get_single_image(cam_ids[0])
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def cleanup_old():
            try:
                time.sleep(0.05)
                storage.cleanup_old_images(max_age_seconds=0.01)
            except Exception as e:
                errors.append(e)

        accessors = [threading.Thread(target=access_images) for _ in range(5)]
        cleaner = threading.Thread(target=cleanup_old)

        for t in accessors:
            t.start()
        cleaner.start()

        for t in accessors:
            t.join()
        cleaner.join()

        assert len(errors) == 0


class TestConcurrentCommandBroadcasting:

    def test_concurrent_command_sends(self):
        broadcaster = CommandBroadcaster()

        mock_server = Mock()
        broadcaster.set_server(mock_server)

        num_threads = 10
        commands_per_thread = 20

        errors = []
        sent_count = [0]

        def send_commands(thread_id):
            try:
                for i in range(commands_per_thread):
                    command = {
                        "command_type": "test",
                        "thread_id": thread_id,
                        "index": i,
                    }
                    success = broadcaster.send_command(command)
                    if success:
                        sent_count[0] += 1
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=send_commands, args=(i,))
            for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert sent_count[0] > 0

    def test_concurrent_completion_queue_operations(self):
        broadcaster = CommandBroadcaster()

        errors = []

        def create_and_use_queue(request_id):
            try:
                broadcaster.create_completion_queue(request_id)
                completion = {"status": "success", "request_id": request_id}
                broadcaster.put_completion(request_id, completion)
                result = broadcaster.get_completion(request_id, timeout=1.0)
                assert result is not None
                assert result["request_id"] == request_id
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=create_and_use_queue, args=(i,)) for i in range(20)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestConcurrentRegistryOperations:

    def test_concurrent_operation_execution(self, clean_registry):
        registry = get_global_registry()

        def test_impl(**kwargs):
            time.sleep(0.01)
            return OperationResult.success_result({"executed": True})

        op = BasicOperation(
            operation_id="test_concurrent_001",
            name="test_concurrent",
            category=OperationCategory.NAVIGATION,
            complexity=OperationComplexity.BASIC,
            description="Test operation for concurrency",
            long_description="A simple test operation for concurrency testing",
            usage_examples=["test_concurrent()"],
            parameters=[],
            preconditions=[],
            postconditions=[],
            average_duration_ms=10.0,
            success_rate=1.0,
            failure_modes=[],
            implementation=test_impl,
        )

        # Register manually if needed (use correct dict name: operations)
        if registry.get_operation_by_name("test_concurrent") is None:
            registry.operations["test_concurrent_001"] = op

        errors = []
        success_count = [0]

        def execute_operation(thread_id):
            try:
                for i in range(10):
                    result = registry.execute_operation_by_name(
                        "test_concurrent", thread_id=thread_id, index=i
                    )
                    if result and result.success:
                        success_count[0] += 1
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=execute_operation, args=(i,)) for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert success_count[0] == 100  # 10 threads * 10 executions

    def test_concurrent_registry_lookups(self, clean_registry):
        registry = get_global_registry()

        errors = []
        lookup_count = [0]

        def lookup_operations():
            try:
                for _ in range(100):
                    op = registry.get_operation_by_name("move_to_coordinate")
                    if op:
                        lookup_count[0] += 1

                    # Look up by ID (correct method name: get_operation)
                    op = registry.get_operation("motion_move_to_coord_001")
                    if op:
                        lookup_count[0] += 1

                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=lookup_operations) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert lookup_count[0] == 2000  # 10 threads * 100 iterations * 2 lookups


class TestConcurrentWorldStateUpdates:

    def test_concurrent_robot_state_updates(self, cleanup_world_state):
        world_state = WorldState()

        errors = []

        def update_robot_state(robot_id, iterations):
            try:
                for i in range(iterations):
                    world_state.update_robot(
                        robot_id=robot_id,
                        position=(float(i) * 0.01, 0.0, 0.1),
                        rotation=(0.0, 0.0, 0.0),
                        joint_angles=[0.0] * 6,
                        is_moving=(i % 2 == 0),
                    )
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=update_robot_state, args=(f"Robot{i}", 50))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        for i in range(5):
            robot_state = world_state.get_robot_state(f"Robot{i}")
            assert robot_state is not None

    def test_concurrent_object_registration(self, cleanup_world_state):
        world_state = WorldState()

        errors = []

        def register_objects(start_id, count):
            try:
                for i in range(count):
                    obj_id = f"obj_{start_id}_{i}"
                    world_state.register_object(
                        object_id=obj_id,
                        object_type="cube",
                        position=(float(i) * 0.1, 0.0, 0.0),
                        graspable=True,
                    )
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_objects, args=(i, 20)) for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        all_objects = world_state.get_all_objects()
        assert len(all_objects) == 200  # 10 threads * 20 objects

    def test_concurrent_read_write_world_state(self, cleanup_world_state):
        world_state = WorldState()

        for i in range(10):
            world_state.update_robot(
                robot_id=f"Robot{i}",
                position=(0.0, 0.0, 0.0),
                rotation=(0.0, 0.0, 0.0),
                joint_angles=[0.0] * 6,
            )

        errors = []
        read_count = [0]
        write_count = [0]

        def read_states():
            try:
                for _ in range(100):
                    for i in range(10):
                        state = world_state.get_robot_state(f"Robot{i}")
                        if state:
                            read_count[0] += 1
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def write_states():
            try:
                for iteration in range(50):
                    for i in range(10):
                        world_state.update_robot(
                            robot_id=f"Robot{i}",
                            position=(float(iteration) * 0.01, 0.0, 0.0),
                            rotation=(0.0, 0.0, 0.0),
                            joint_angles=[0.0] * 6,
                        )
                        write_count[0] += 1
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        readers = [threading.Thread(target=read_states) for _ in range(5)]
        writers = [threading.Thread(target=write_states) for _ in range(3)]

        for t in readers + writers:
            t.start()
        for t in readers + writers:
            t.join()

        assert len(errors) == 0
        assert read_count[0] > 0
        assert write_count[0] == 1500  # 3 writers * 50 iterations * 10 robots


class TestSingletonThreadSafety:

    def test_image_storage_singleton_thread_safe(self):
        UnifiedImageStorage._instance = None

        instances = []
        errors = []

        def get_instance():
            try:
                inst = UnifiedImageStorage()
                instances.append(inst)
                time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_instance) for _ in range(50)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(inst is instances[0] for inst in instances)

    def test_world_state_singleton_thread_safe(self):
        WorldState._instance = None

        instances = []
        errors = []

        def get_instance():
            try:
                inst = WorldState
                instances.append(inst)
                time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_instance) for _ in range(50)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(inst is instances[0] for inst in instances)


class TestRaceConditions:

    def test_command_queue_race_condition(self):
        broadcaster = CommandBroadcaster()

        results = []
        errors = []

        def send_and_track(command_id):
            try:
                command = {"id": command_id}
                success = broadcaster.send_command(command)
                results.append((command_id, success))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=send_and_track, args=(i,)) for i in range(100)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 100

    @patch("servers.CommandServer.get_command_broadcaster")
    def test_operation_execution_race_condition(
        self, mock_broadcaster, cleanup_world_state
    ):
        mock_broadcaster.return_value.send_command = Mock(return_value=True)

        execution_order = []
        lock = threading.Lock()
        errors = []

        def execute_move(robot_id, x, y, z):
            try:
                result = move_to_coordinate(robot_id=robot_id, x=x, y=y, z=z)

                with lock:
                    execution_order.append(robot_id)

                return result
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [
            threading.Thread(target=execute_move, args=(f"Robot{i}", 0.3, 0.0, 0.1))
            for i in range(20)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(execution_order) == 20
