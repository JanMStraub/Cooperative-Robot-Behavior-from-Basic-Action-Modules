import sys
from pathlib import Path

# Add parent directory to path so we can import modules
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

import importlib
import pytest
import numpy as np
import socket
from unittest.mock import Mock, MagicMock


def _reset_singleton(module_path: str, class_name: str) -> None:
    # Silent on import/attribute errors - safe to call before a module is imported.
    try:
        mod = importlib.import_module(module_path)
        getattr(mod, class_name)._instance = None
    except (ImportError, AttributeError):
        pass


# Singletons that are commonly reset across the test suite
_ALL_SINGLETONS = [
    ("servers.ImageStorageCore", "UnifiedImageStorage"),
    ("servers.CommandServer", "CommandBroadcaster"),
    ("operations.WorldState", "WorldState"),
    ("operations.SyncOperations", "EventBus"),
    ("servers.NegotiationHub", "NegotiationHub"),
]


@pytest.fixture
def mock_socket():
    sock = Mock(spec=socket.socket)
    sock.recv = Mock(return_value=b"test_data")
    sock.sendall = Mock(return_value=None)
    sock.close = Mock(return_value=None)
    sock.settimeout = Mock(return_value=None)
    sock.bind = Mock(return_value=None)
    sock.listen = Mock(return_value=None)
    sock.accept = Mock(return_value=(Mock(spec=socket.socket), ("127.0.0.1", 12345)))
    return sock


@pytest.fixture
def sample_image():
    # Create a simple gradient image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, :, 0] = np.linspace(0, 255, 640, dtype=np.uint8)  # Red gradient
    image[:, :, 1] = 128  # Constant green
    image[:, :, 2] = np.linspace(255, 0, 640, dtype=np.uint8)  # Blue gradient
    return image


@pytest.fixture
def sample_red_cube_image():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a red cube with proper BGR values
    # Red in BGR = (0, 0, 255)
    image[200:280, 270:370] = [0, 0, 255]  # BGR for red
    return image


@pytest.fixture
def sample_blue_cube_image():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a blue cube with proper BGR values
    # Blue in BGR = (255, 0, 0)
    image[200:280, 270:370] = [255, 0, 0]  # BGR for blue
    return image


@pytest.fixture
def sample_stereo_pair():
    # Create identical images for simplicity (real stereo would have disparity)
    left = np.zeros((480, 640, 3), dtype=np.uint8)
    left[200:280, 270:370, 2] = 255  # Red cube in left image

    # Right image has same cube slightly shifted (simulating parallax)
    right = np.zeros((480, 640, 3), dtype=np.uint8)
    right[200:280, 250:350, 2] = 255  # Shifted 20px left

    return left, right


@pytest.fixture
def server_config():
    from core.TCPServerBase import ServerConfig

    return ServerConfig(
        host="127.0.0.1",
        port=9999,  # Use non-standard port for testing
        max_connections=2,
        max_client_threads=2,
        socket_timeout=0.1,  # Short timeout for tests
    )


@pytest.fixture
def detection_result_dict():
    return {
        "success": True,
        "camera_id": "test_camera",
        "timestamp": "2025-01-01T12:00:00",
        "image_width": 640,
        "image_height": 480,
        "detections": [
            {
                "id": 0,
                "color": "red",
                "bbox_px": {"x": 270, "y": 200, "width": 100, "height": 80},
                "center_px": {"x": 320, "y": 240},
                "confidence": 0.95,
            }
        ],
    }


@pytest.fixture
def llm_result_dict():
    return {
        "success": True,
        "response": "I see a red cube on the table.",
        "camera_id": "AR4Left",
        "timestamp": "2025-01-01T12:00:00",
        "metadata": {
            "model": "llama-3.2-vision",
            "duration_seconds": 2.5,
            "image_count": 1,
            "camera_ids": ["AR4Left"],
            "prompt": "What do you see?",
        },
    }


@pytest.fixture
def mock_lmstudio_client():
    client = MagicMock()

    # Mock models.list() for connection testing
    client.models.list = Mock(return_value=[])

    # Mock chat.completions.create() for vision API
    mock_choice = MagicMock()
    mock_choice.message.content = "This is a test response from the LLM."

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    client.chat.completions.create = Mock(return_value=mock_response)

    return client


@pytest.fixture
def cleanup_singletons():

    def _cleanup():
        for module_path, class_name in _ALL_SINGLETONS:
            _reset_singleton(module_path, class_name)

    _cleanup()  # Clean before test
    yield
    _cleanup()  # Clean after test


@pytest.fixture
def mock_command_broadcaster():
    broadcaster = Mock()
    broadcaster.send_command = Mock(return_value=True)
    broadcaster.wait_for_result = Mock(
        return_value={"success": True, "result": {"status": "completed"}, "error": None}
    )
    return broadcaster


@pytest.fixture
def mock_unified_image_storage(sample_red_cube_image):
    storage = Mock()
    storage.get_single_image = Mock(return_value=sample_red_cube_image)
    storage.get_stereo_pair = Mock(
        return_value=(sample_red_cube_image, sample_red_cube_image)
    )
    storage.get_latest_stereo_image = Mock(
        return_value=(sample_red_cube_image, sample_red_cube_image)
    )
    return storage


@pytest.fixture
def mock_get_global_registry():
    registry = Mock()
    # Mock common registry methods
    registry.get_operation = Mock(return_value=None)
    registry.get_all_operations = Mock(return_value=[])

    def _get_global_registry():
        return registry

    return _get_global_registry


@pytest.fixture(autouse=False)
def patch_command_broadcaster(monkeypatch, mock_command_broadcaster):
    # Patches at core.Imports level so all operation modules get the same mock.
    # Also disables ROS so operations use the TCP path (mocked broadcaster)
    # instead of trying to connect to a ROS bridge that isn't running in tests.
    # Disable ROS so operations use TCP path (the mocked broadcaster)
    try:
        import config.ROS as ros_config

        monkeypatch.setattr(ros_config, "ROS_ENABLED", False)
    except (ImportError, AttributeError):
        pass

    # Patch at the source: core.Imports.get_command_broadcaster
    try:
        import core.Imports as imports_module

        monkeypatch.setattr(
            imports_module, "get_command_broadcaster", lambda: mock_command_broadcaster
        )
    except (ImportError, AttributeError):
        pass

    # Also patch individual modules for backwards compatibility
    modules_with_broadcaster = [
        "operations.MoveOperations",
        "operations.StatusOperations",
        "operations.GripperOperations",
        "operations.DefaultPositionOperation",
        "operations.CoordinationOperations",
        "operations.CollaborativeOperations",
        "operations.IntermediateOperations",
    ]

    for module_name in modules_with_broadcaster:
        try:
            module = __import__(module_name, fromlist=[""])
            if hasattr(module, "_get_command_broadcaster"):
                monkeypatch.setattr(
                    module, "_get_command_broadcaster", lambda: mock_command_broadcaster
                )
        except (ImportError, AttributeError):
            pass

    yield mock_command_broadcaster


@pytest.fixture(autouse=False)
def patch_unified_image_storage(monkeypatch, mock_unified_image_storage):

    # Create a mock class that returns our mock instance
    def mock_unified_storage_class():
        return mock_unified_image_storage

    # Patch in the operations module where it's imported
    try:
        import operations.DetectionOperations as det_ops

        if hasattr(det_ops, "UnifiedImageStorage"):
            monkeypatch.setattr(
                det_ops, "UnifiedImageStorage", mock_unified_storage_class
            )
    except (ImportError, AttributeError):
        pass

    yield mock_unified_image_storage


@pytest.fixture
def patch_world_state():
    # WorldState is imported inside functions, so we patch at the module level.
    # Usage:
    #   with patch_world_state(mock_world_state_instance):
    #       result = detect_other_robot("Robot1", "Robot2")
    from unittest.mock import patch

    def _create_patch(mock_world_state_instance):
        return patch(
            "operations.WorldState.WorldState", return_value=mock_world_state_instance
        )

    return _create_patch


@pytest.fixture
def patch_yolo_detector():
    # YOLODetector is imported inside functions, so we need the patch() context manager.
    # Usage:
    #   with patch_yolo_detector(mock_detector):
    #       result = detect_field("Robot1", "A")
    from unittest.mock import patch

    def _create_patch(mock_detector_instance):
        return patch(
            "operations.FieldOperations.YOLODetector",
            return_value=mock_detector_instance,
        )

    return _create_patch


@pytest.fixture
def temp_output_dir(tmp_path):
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def mock_lmstudio_embeddings_client():
    client = MagicMock()

    # Mock embeddings.create() for embedding API
    mock_embedding_data = MagicMock()
    mock_embedding_data.embedding = [0.1] * 768  # 768-dim embedding

    mock_response = MagicMock()
    mock_response.data = [mock_embedding_data]

    client.embeddings.create = Mock(return_value=mock_response)

    return client


@pytest.fixture
def sample_operation():
    from operations.Base import OperationCategory, OperationComplexity

    op = Mock()
    op.operation_id = "test_op_001"
    op.name = "test_operation"
    op.category = Mock(value="navigation")
    op.complexity = Mock(value="basic")
    op.description = "A test operation"
    op.average_duration_ms = 1000.0
    op.success_rate = 0.95
    op.to_rag_document = Mock(return_value="Test operation RAG document")

    return op


@pytest.fixture
def mock_operation_registry(sample_operation):
    registry = Mock()
    registry.get_all_operations = Mock(return_value=[sample_operation])
    registry.get_operation = Mock(return_value=sample_operation)

    return registry


@pytest.fixture
def temp_vector_store_path(tmp_path):
    return tmp_path / "test_vector_store.pkl"


@pytest.fixture
def cleanup_rag_singletons():
    yield

    # Reset RAG singleton instances if they exist
    try:
        from operations.Registry import _global_registry

        _global_registry = None
    except:
        pass


@pytest.fixture
def clean_registry():

    # Clean up BEFORE the test runs
    def _cleanup():
        try:
            import operations.Registry as registry_module

            registry_module._global_registry = None
        except:
            pass

    _cleanup()  # Clean before test
    yield
    _cleanup()  # Clean after test


@pytest.fixture
def mock_world_state():
    world_state = Mock()
    world_state.get_robot_position = Mock(return_value=(0.3, 0.0, 0.1))
    world_state.get_robot_status = Mock(
        return_value={
            "is_initialized": True,
            "is_moving": False,
            "gripper_state": "open",
            "position": (0.3, 0.0, 0.1),
        }
    )
    world_state._robot_states = {}
    world_state._objects = {}
    world_state.get_workspace_owner = Mock(return_value=None)
    return world_state


@pytest.fixture
def sample_robot_positions():
    return {
        "Robot1": (-0.3, 0.0, 0.0),
        "Robot2": (0.3, 0.0, 0.0),
    }


@pytest.fixture
def sample_operation_with_conditions():
    from operations.Base import OperationCategory, OperationComplexity

    op = Mock()
    op.name = "test_move"
    op.category = OperationCategory.NAVIGATION
    op.complexity = OperationComplexity.BASIC
    op.preconditions = [
        "target_within_reach(robot_id, x, y, z)",
        "robot_is_initialized(robot_id)",
    ]
    op.postconditions = ["robot_is_stationary(robot_id)"]
    op.parameters = []
    return op


@pytest.fixture(autouse=True)
def cleanup_world_state():
    # autouse=True: runs for every test to prevent state pollution between files.
    _reset_singleton("operations.WorldState", "WorldState")
    yield  # Test runs with clean state
    _reset_singleton("operations.WorldState", "WorldState")


@pytest.fixture
def mock_world_state_with_objects():
    world_state = Mock()

    # Sample objects
    world_state._objects = {
        "cube_01": Mock(position=(0.3, 0.2, 0.1), color="red", grasped_by=None),
        "cube_02": Mock(position=(0.4, 0.3, 0.1), color="blue", grasped_by=None),
    }

    world_state.get_object_position = Mock(
        side_effect=lambda obj_id: (
            world_state._objects[obj_id].position
            if obj_id in world_state._objects
            else None
        )
    )

    return world_state


@pytest.fixture
def mock_world_state_multi_robot():
    from operations.WorldState import RobotState

    world_state = Mock()

    # Robot states
    robot1_state = Mock(spec=RobotState)
    robot1_state.robot_id = "Robot1"
    robot1_state.position = (-0.3, 0.0, 0.1)
    robot1_state.is_moving = False
    robot1_state.target_position = None

    robot2_state = Mock(spec=RobotState)
    robot2_state.robot_id = "Robot2"
    robot2_state.position = (0.3, 0.0, 0.1)
    robot2_state.is_moving = False
    robot2_state.target_position = None

    world_state._robot_states = {"Robot1": robot1_state, "Robot2": robot2_state}

    world_state.get_robot_position = Mock(
        side_effect=lambda rid: (
            world_state._robot_states[rid].position
            if rid in world_state._robot_states
            else None
        )
    )

    world_state.get_workspace_owner = Mock(return_value=None)
    world_state.allocate_workspace = Mock(return_value=True)

    world_state._objects = {
        "test_object": Mock(position=(0.0, 0.0, 0.15), grasped_by=None)
    }

    return world_state


@pytest.fixture
def sample_navigation_params():
    return {"robot_id": "Robot1", "x": 0.3, "y": 0.2, "z": 0.1}


@pytest.fixture
def sample_manipulation_params():
    return {"robot_id": "Robot1", "object_id": "cube_01", "action": "grasp"}


@pytest.fixture
def disable_yolo_detection():
    # HSV tests use synthetic pure-color squares that YOLO (trained on realistic cubes)
    # doesn't detect reliably - disable it so the HSV path runs instead.
    import config.Vision as vision_cfg

    original_use_yolo = vision_cfg.USE_YOLO
    vision_cfg.USE_YOLO = False
    yield
    vision_cfg.USE_YOLO = original_use_yolo


@pytest.fixture
def cleanup_event_bus():
    yield
    try:
        from operations.SyncOperations import EventBus

        bus = EventBus()
        bus.reset()
    except Exception:
        # If EventBus not available or already clean, ignore
        pass


@pytest.fixture
def event_bus(cleanup_event_bus):
    from operations.SyncOperations import EventBus

    bus = EventBus()
    bus.reset()
    return bus


@pytest.fixture
def timing_helper():

    def verify_timing(actual_ms, expected_ms, tolerance_percent=10):
        tolerance = expected_ms * (tolerance_percent / 100.0)
        return abs(actual_ms - expected_ms) <= tolerance

    return verify_timing


@pytest.fixture
def thread_barrier():
    import threading

    def create_barrier(num_threads):
        return threading.Barrier(num_threads)

    return create_barrier


@pytest.fixture
def thread_error_collector():
    import threading

    errors = []
    lock = threading.Lock()

    def add_error(error):
        with lock:
            errors.append(error)

    return errors, add_error


@pytest.fixture
def async_executor():
    import threading

    def execute_async(func, *args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return execute_async


@pytest.fixture
def mock_move(monkeypatch):
    from operations.Base import OperationResult

    def _mock_move(**kwargs):
        return OperationResult.success_result(
            {
                "robot_id": kwargs.get("robot_id"),
                "final_position": (kwargs.get("x"), kwargs.get("y"), kwargs.get("z")),
            }
        )

    mock = Mock(side_effect=_mock_move)
    # Auto-patch into SpatialOperations module
    monkeypatch.setattr("operations.SpatialOperations.move_to_coordinate", mock)
    return mock


@pytest.fixture
def mock_get_ws(monkeypatch):
    mock = Mock()
    # Auto-patch into SpatialOperations module
    monkeypatch.setattr("operations.SpatialOperations.get_world_state", mock)
    return mock
