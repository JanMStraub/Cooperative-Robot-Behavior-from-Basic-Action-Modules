"""
Test AutoRT Task Generator

Tests for LLM-based task generation with JSON parsing and validation.
"""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch
from pydantic import ValidationError
from autort.TaskGenerator import TaskGenerator
from autort.DataModels import SceneDescription, GroundedObject, ProposedTask
from operations.Base import BasicOperation, OperationParameter, OperationCategory, OperationComplexity


@pytest.fixture
def mock_config():
    """Mock AutoRT config"""
    config = Mock()
    config.LM_STUDIO_URL = "http://localhost:1234/v1"
    config.TASK_GENERATION_MODEL = "test-model"
    config.MAX_JSON_RETRIES = 3
    return config


def _make_param(param_name, required=True, param_type="float", valid_values=None, valid_range=None, default=None):
    """Helper to create a properly configured parameter mock."""
    p = Mock()
    p.name = param_name
    p.required = required
    p.type = param_type
    p.valid_values = valid_values  # None or a list
    p.valid_range = valid_range    # None or a tuple
    p.default = default
    return p


@pytest.fixture
def mock_registry():
    """Mock operation registry with sample operations"""
    registry = Mock()

    # Create sample operations with properly configured parameter mocks
    wait_op = Mock(spec=BasicOperation)
    wait_op.name = "wait"
    wait_op.description = "Wait for specified seconds"
    wait_op.parameters = [
        _make_param("seconds", required=True, param_type="float"),
    ]

    move_op = Mock(spec=BasicOperation)
    move_op.name = "move_to_coordinate"
    move_op.description = "Move robot to coordinates"
    move_op.parameters = [
        _make_param("x", required=True, param_type="float"),
        _make_param("y", required=True, param_type="float"),
        _make_param("z", required=True, param_type="float"),
        _make_param("velocity", required=False, param_type="float", default=0.5),
    ]

    gripper_op = Mock(spec=BasicOperation)
    gripper_op.name = "control_gripper"
    gripper_op.description = "Control gripper open/close"
    gripper_op.parameters = [
        _make_param("action", required=True, param_type="str", valid_values=["open", "close"]),
    ]

    registry.get_all_operations = Mock(return_value=[wait_op, move_op, gripper_op])
    registry.get_operation_by_name = Mock(side_effect=lambda name: {
        "wait": wait_op,
        "move_to_coordinate": move_op,
        "control_gripper": gripper_op,
    }.get(name))

    return registry


@pytest.fixture
def sample_scene():
    """Sample scene description for task generation"""
    return SceneDescription(
        timestamp=123456.789,
        objects=[
            GroundedObject(
                object_id="cube_01",
                color="red",
                position=(0.3, 0.2, 0.1),
                confidence=0.95,
                graspable=True,
            ),
            GroundedObject(
                object_id="cube_02",
                color="blue",
                position=(0.4, 0.3, 0.1),
                confidence=0.90,
                graspable=True,
            ),
        ],
        scene_summary="Two cubes on table",
        robot_states={},
    )


@pytest.fixture
def valid_task_json():
    """Valid task JSON response"""
    return json.dumps([
        {
            "task_id": "task_001",
            "description": "Pick up red cube",
            "operations": [
                {
                    "type": "move_to_coordinate",
                    "robot_id": "Robot1",
                    "parameters": {"x": 0.3, "y": 0.2, "z": 0.1}
                },
                {
                    "type": "control_gripper",
                    "robot_id": "Robot1",
                    "parameters": {"action": "close"}
                }
            ],
            "required_robots": ["Robot1"],
            "estimated_complexity": 3,
            "reasoning": "Simple pick task"
        }
    ])


# ============================================================================
# Initialization Tests
# ============================================================================


def test_task_generator_init(mock_config, mock_registry):
    """TaskGenerator initializes with config"""
    with patch('autort.TaskGenerator.get_global_registry', return_value=mock_registry):
        with patch('autort.TaskGenerator.OpenAI'):
            generator = TaskGenerator(mock_config)
            assert generator.config == mock_config
            assert generator.max_retries == 3
            assert generator._operations_summary_cache is None


# ============================================================================
# JSON Parsing Tests
# ============================================================================


def test_parse_llm_response_valid_list(mock_config, mock_registry, valid_task_json):
    """Parse valid JSON array response"""
    with patch('autort.TaskGenerator.get_global_registry', return_value=mock_registry):
        with patch('autort.TaskGenerator.OpenAI'):
            generator = TaskGenerator(mock_config)
            tasks = generator._parse_llm_response(valid_task_json)

            assert len(tasks) == 1
            assert tasks[0].task_id == "task_001"
            assert len(tasks[0].operations) == 2


def test_parse_llm_response_single_dict(mock_config, mock_registry):
    """Parse single dict (non-array) response"""
    single_task = json.dumps({
        "task_id": "task_001",
        "description": "test",
        "operations": [
            {"type": "wait", "robot_id": "Robot1", "parameters": {"seconds": 1}}
        ],
        "required_robots": ["Robot1"],
        "estimated_complexity": 1,
        "reasoning": "test"
    })

    with patch('autort.TaskGenerator.get_global_registry', return_value=mock_registry):
        with patch('autort.TaskGenerator.OpenAI'):
            generator = TaskGenerator(mock_config)
            tasks = generator._parse_llm_response(single_task)

            assert len(tasks) == 1
            assert tasks[0].task_id == "task_001"


def test_parse_llm_response_strips_markdown(mock_config, mock_registry, valid_task_json):
    """Parse JSON with markdown code blocks"""
    markdown_wrapped = f"```json\n{valid_task_json}\n```"

    with patch('autort.TaskGenerator.get_global_registry', return_value=mock_registry):
        with patch('autort.TaskGenerator.OpenAI'):
            generator = TaskGenerator(mock_config)
            tasks = generator._parse_llm_response(markdown_wrapped)

            assert len(tasks) == 1
            assert tasks[0].task_id == "task_001"


def test_parse_llm_response_invalid_json(mock_config, mock_registry):
    """Parse invalid JSON raises JSONDecodeError"""
    invalid_json = '{"task_id": "broken" "missing_comma"}'

    with patch('autort.TaskGenerator.get_global_registry', return_value=mock_registry):
        with patch('autort.TaskGenerator.OpenAI'):
            generator = TaskGenerator(mock_config)

            with pytest.raises(json.JSONDecodeError):
                generator._parse_llm_response(invalid_json)


def test_parse_llm_response_invalid_schema(mock_config, mock_registry):
    """Parse JSON with invalid schema raises ValidationError"""
    invalid_schema = json.dumps([
        {
            "task_id": "task_001",
            "description": "test",
            "operations": [],  # Empty operations not allowed
            "required_robots": ["Robot1"],
            "estimated_complexity": 1,
            "reasoning": "test"
        }
    ])

    with patch('autort.TaskGenerator.get_global_registry', return_value=mock_registry):
        with patch('autort.TaskGenerator.OpenAI'):
            generator = TaskGenerator(mock_config)

            with pytest.raises(ValidationError):
                generator._parse_llm_response(invalid_schema)


# ============================================================================
# Operation Validation Tests
# ============================================================================


def test_validate_operations_valid(mock_config, mock_registry):
    """Validate task with valid operation types"""
    task = ProposedTask(
        task_id="task_001",
        description="test",
        operations=[
            {"type": "wait", "robot_id": "Robot1", "parameters": {"seconds": 1}},
            {"type": "move_to_coordinate", "robot_id": "Robot1", "parameters": {"x": 0.3, "y": 0.2, "z": 0.1}},
        ],
        required_robots=["Robot1"],
        estimated_complexity=2,
        reasoning="test"
    )

    with patch('autort.TaskGenerator.get_global_registry', return_value=mock_registry):
        with patch('autort.TaskGenerator.OpenAI'):
            generator = TaskGenerator(mock_config)
            assert generator._validate_operations(task) is True


def test_validate_operations_invalid(mock_config, mock_registry):
    """Validate task with invalid operation type"""
    task = ProposedTask(
        task_id="task_001",
        description="test",
        operations=[
            {"type": "nonexistent_operation", "robot_id": "Robot1", "parameters": {}},
        ],
        required_robots=["Robot1"],
        estimated_complexity=1,
        reasoning="test"
    )

    with patch('autort.TaskGenerator.get_global_registry', return_value=mock_registry):
        with patch('autort.TaskGenerator.OpenAI'):
            generator = TaskGenerator(mock_config)
            assert generator._validate_operations(task) is False


# ============================================================================
# Operations Summary Tests
# ============================================================================


def test_get_operations_summary_caching(mock_config, mock_registry):
    """Operations summary is cached after first call"""
    with patch('autort.TaskGenerator.get_global_registry', return_value=mock_registry):
        with patch('autort.TaskGenerator.OpenAI'):
            generator = TaskGenerator(mock_config)

            # First call
            summary1 = generator._get_operations_summary()
            assert "wait" in summary1
            assert "move_to_coordinate" in summary1

            # Check cache
            assert generator._operations_summary_cache is not None

            # Second call should use cache
            summary2 = generator._get_operations_summary()
            assert summary1 == summary2

            # Registry should only be called once
            assert mock_registry.get_all_operations.call_count == 1


def test_get_operations_summary_format(mock_config, mock_registry):
    """Operations summary has correct format"""
    with patch('autort.TaskGenerator.get_global_registry', return_value=mock_registry):
        with patch('autort.TaskGenerator.OpenAI'):
            generator = TaskGenerator(mock_config)
            summary = generator._get_operations_summary()

            # Required params include type annotation: - name(param :type) - description
            assert "wait" in summary
            assert "seconds" in summary
            assert "move_to_coordinate" in summary
            assert "Wait for specified seconds" in summary
            assert "velocity" in summary  # Optional param appears in summary


# ============================================================================
# Retry Logic Tests
# ============================================================================


def test_generate_tasks_retry_on_json_error(mock_config, mock_registry, sample_scene, valid_task_json):
    """Retry loop recovers from malformed JSON"""
    mock_client = MagicMock()

    # First attempt: malformed JSON
    mock_choice1 = MagicMock()
    mock_choice1.message.content = '{"task_id": "broken" "missing_comma"}'
    mock_response1 = MagicMock()
    mock_response1.choices = [mock_choice1]

    # Second attempt: valid JSON
    mock_choice2 = MagicMock()
    mock_choice2.message.content = valid_task_json
    mock_response2 = MagicMock()
    mock_response2.choices = [mock_choice2]

    mock_client.chat.completions.create = Mock(side_effect=[mock_response1, mock_response2])

    with patch('autort.TaskGenerator.get_global_registry', return_value=mock_registry):
        with patch('autort.TaskGenerator.OpenAI', return_value=mock_client):
            generator = TaskGenerator(mock_config)
            tasks = generator.generate_tasks(sample_scene, robot_ids=["Robot1"], num_tasks=1)

            assert len(tasks) == 1
            assert tasks[0].task_id == "task_001"
            # Should have made 2 LLM calls (1 failed, 1 success)
            assert mock_client.chat.completions.create.call_count == 2


def test_generate_tasks_fails_after_max_retries(mock_config, mock_registry, sample_scene):
    """Generate tasks fails after max retries"""
    mock_client = MagicMock()

    # All attempts return malformed JSON
    mock_choice = MagicMock()
    mock_choice.message.content = '{"invalid": "json'
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client.chat.completions.create = Mock(return_value=mock_response)

    with patch('autort.TaskGenerator.get_global_registry', return_value=mock_registry):
        with patch('autort.TaskGenerator.OpenAI', return_value=mock_client):
            generator = TaskGenerator(mock_config)
            tasks = generator.generate_tasks(sample_scene, robot_ids=["Robot1"], num_tasks=1)

            # Should return empty list after all retries fail
            assert len(tasks) == 0
            # Should have made max_retries attempts
            assert mock_client.chat.completions.create.call_count == mock_config.MAX_JSON_RETRIES


# ============================================================================
# Task Generation Integration Tests
# ============================================================================


def test_generate_tasks_filters_invalid_operations(mock_config, mock_registry, sample_scene):
    """Generate tasks filters out tasks with invalid operations"""
    mock_client = MagicMock()

    # Return mix of valid and invalid operation types
    mixed_json = json.dumps([
        {
            "task_id": "task_valid",
            "description": "Valid task",
            "operations": [
                {"type": "wait", "robot_id": "Robot1", "parameters": {"seconds": 1}}
            ],
            "required_robots": ["Robot1"],
            "estimated_complexity": 1,
            "reasoning": "test"
        },
        {
            "task_id": "task_invalid",
            "description": "Invalid task",
            "operations": [
                {"type": "nonexistent_op", "robot_id": "Robot1", "parameters": {}}
            ],
            "required_robots": ["Robot1"],
            "estimated_complexity": 1,
            "reasoning": "test"
        }
    ])

    mock_choice = MagicMock()
    mock_choice.message.content = mixed_json
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create = Mock(return_value=mock_response)

    with patch('autort.TaskGenerator.get_global_registry', return_value=mock_registry):
        with patch('autort.TaskGenerator.OpenAI', return_value=mock_client):
            generator = TaskGenerator(mock_config)
            tasks = generator.generate_tasks(sample_scene, robot_ids=["Robot1"], num_tasks=2)

            # Should only return valid task
            assert len(tasks) == 1
            assert tasks[0].task_id == "task_valid"


def test_generate_tasks_collaborative_prompt(mock_config, mock_registry, sample_scene, valid_task_json):
    """Generate tasks includes collaborative patterns in prompt"""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = valid_task_json
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create = Mock(return_value=mock_response)

    with patch('autort.TaskGenerator.get_global_registry', return_value=mock_registry):
        with patch('autort.TaskGenerator.OpenAI', return_value=mock_client):
            generator = TaskGenerator(mock_config)
            tasks = generator.generate_tasks(
                sample_scene,
                robot_ids=["Robot1", "Robot2"],
                num_tasks=1,
                include_collaborative=True
            )

            # Check that prompt included collaborative hints (in user message, index 1)
            call_args = mock_client.chat.completions.create.call_args
            # messages[0] is system, messages[1] is user with the actual task prompt
            user_prompt = call_args[1]['messages'][1]['content']

            assert "MULTI-ROBOT COORDINATION" in user_prompt
            assert "Handoff" in user_prompt
            assert "signal" in user_prompt
            assert "wait_for_signal" in user_prompt
