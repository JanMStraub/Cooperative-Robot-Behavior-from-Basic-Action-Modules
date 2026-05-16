#!/usr/bin/env python3
"""
Tests for context-aware AutoRT task sequencing.

Covers ExecutedTaskContext model, SceneDescription backward compat,
_build_previous_task_section logic, _build_task_prompt injection,
and AutoRTOrchestrator state tracking.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pydantic import ValidationError

# Fixtures


def make_scene(last_task_context=None):
    from autort.DataModels import SceneDescription, GroundedObject

    return SceneDescription(
        timestamp=1.0,
        objects=[
            GroundedObject(
                object_id="red_cube",
                color="red",
                position=(0.2, 0.1, 0.05),
                confidence=0.9,
            )
        ],
        last_task_context=last_task_context,
    )


def make_context(op_types, success=True, result_summary="ok", description="Test task"):
    from autort.DataModels import ExecutedTaskContext

    return ExecutedTaskContext(
        task_id="t001",
        description=description,
        operation_types=op_types,
        success=success,
        result_summary=result_summary,
    )


def make_generator():
    """Build a TaskGenerator with all external deps mocked."""
    with patch("autort.TaskGenerator.get_global_registry") as mreg, patch(
        "autort.TaskGenerator.OpenAI"
    ):
        mreg.return_value.get_all_operations.return_value = []
        gen = _build_generator()
    return gen


def _build_generator():
    from autort.TaskGenerator import TaskGenerator

    cfg = Mock()
    cfg.LM_STUDIO_URL = "http://localhost:1234/v1"
    cfg.TASK_GENERATION_MODEL = "test-model"
    cfg.MAX_JSON_RETRIES = 1
    cfg.TASK_GENERATION_TEMPERATURE = 0.7
    return TaskGenerator(cfg)


# Section 1: ExecutedTaskContext model


class TestExecutedTaskContext:
    def test_valid_success(self):
        from autort.DataModels import ExecutedTaskContext

        ctx = ExecutedTaskContext(
            task_id="t1",
            description="Grasp cube",
            operation_types=["detect_object_stereo", "grasp_object"],
            success=True,
            result_summary="grasped red_cube",
        )
        assert ctx.task_id == "t1"
        assert ctx.success is True
        assert ctx.result_summary == "grasped red_cube"

    def test_result_summary_defaults_empty(self):
        from autort.DataModels import ExecutedTaskContext

        ctx = ExecutedTaskContext(
            task_id="t2",
            description="Move",
            operation_types=["move_to_coordinate"],
            success=True,
        )
        assert ctx.result_summary == ""

    def test_failure(self):
        from autort.DataModels import ExecutedTaskContext

        ctx = ExecutedTaskContext(
            task_id="t3",
            description="Grasp blue",
            operation_types=["grasp_object"],
            success=False,
            result_summary="IK failed",
        )
        assert ctx.success is False
        assert "IK" in ctx.result_summary

    def test_missing_required_field_raises(self):
        from autort.DataModels import ExecutedTaskContext

        with pytest.raises(ValidationError):
            ExecutedTaskContext(
                description="No task_id",
                operation_types=[],
                success=True,
            )


# Section 2: SceneDescription backward compat


class TestSceneDescriptionBackwardCompat:
    def test_no_context_still_valid(self):
        scene = make_scene()
        assert scene.last_task_context is None

    def test_with_context_round_trips(self):
        ctx = make_context(["grasp_object"])
        scene = make_scene(last_task_context=ctx)
        assert scene.last_task_context is not None
        assert scene.last_task_context.task_id == "t001"
        assert scene.last_task_context.operation_types == ["grasp_object"]

    def test_serialise_deserialise(self):
        ctx = make_context(["move_to_coordinate"], success=False, result_summary="err")
        scene = make_scene(last_task_context=ctx)
        data = scene.model_dump()
        from autort.DataModels import SceneDescription

        restored = SceneDescription(**data)
        assert restored.last_task_context.success is False


# Section 3: _build_previous_task_section — success paths


class TestBuildPreviousTaskSectionSuccess:
    @pytest.fixture(autouse=True)
    def gen(self):
        with patch("autort.TaskGenerator.get_global_registry") as mreg, patch(
            "autort.TaskGenerator.OpenAI"
        ):
            mreg.return_value.get_all_operations.return_value = []
            from autort.TaskGenerator import TaskGenerator

            cfg = Mock(
                LM_STUDIO_URL="x",
                TASK_GENERATION_MODEL="m",
                MAX_JSON_RETRIES=1,
                TASK_GENERATION_TEMPERATURE=0.7,
            )
            self.gen = TaskGenerator(cfg)

    def test_grasp_success(self):
        ctx = make_context(
            ["grasp_object"], success=True, result_summary="grasped cube"
        )
        section = self.gen._build_previous_task_section(ctx)
        assert "PREVIOUS TASK CONTEXT" in section
        assert "holding an object" in section
        assert "place_object" in section
        assert "succeeded" in section

    def test_detect_success(self):
        ctx = make_context(["detect_object_stereo"], success=True)
        section = self.gen._build_previous_task_section(ctx)
        assert "detection scan" in section
        assert "grasp_object" in section

    def test_release_success(self):
        ctx = make_context(["release_object"], success=True)
        section = self.gen._build_previous_task_section(ctx)
        assert "gripper is open" in section

    def test_unknown_op_generic(self):
        ctx = make_context(["stabilize_object"], success=True)
        section = self.gen._build_previous_task_section(ctx)
        assert "PREVIOUS TASK CONTEXT" in section
        assert "detect_object_stereo" in section  # generic follow-ups


# Section 4: _build_previous_task_section — failure paths


class TestBuildPreviousTaskSectionFailure:
    @pytest.fixture(autouse=True)
    def gen(self):
        with patch("autort.TaskGenerator.get_global_registry") as mreg, patch(
            "autort.TaskGenerator.OpenAI"
        ):
            mreg.return_value.get_all_operations.return_value = []
            from autort.TaskGenerator import TaskGenerator

            cfg = Mock(
                LM_STUDIO_URL="x",
                TASK_GENERATION_MODEL="m",
                MAX_JSON_RETRIES=1,
                TASK_GENERATION_TEMPERATURE=0.7,
            )
            self.gen = TaskGenerator(cfg)

    def test_grasp_failure_suggests_redetect(self):
        ctx = make_context(["grasp_object"], success=False, result_summary="IK failed")
        section = self.gen._build_previous_task_section(ctx)
        assert "FAILED" in section
        assert "Re-detect" in section
        assert "IK failed" in section

    def test_detect_failure_suggests_viewpoint(self):
        ctx = make_context(
            ["detect_object_stereo"], success=False, result_summary="timeout"
        )
        section = self.gen._build_previous_task_section(ctx)
        assert "viewpoint" in section
        assert "timeout" in section

    def test_other_failure_generic_retry(self):
        ctx = make_context(
            ["move_to_coordinate"], success=False, result_summary="collision"
        )
        section = self.gen._build_previous_task_section(ctx)
        assert "FAILED" in section
        assert "collision" in section

    def test_error_summary_in_output(self):
        ctx = make_context(
            ["grasp_object"], success=False, result_summary="motor overheated"
        )
        section = self.gen._build_previous_task_section(ctx)
        assert "motor overheated" in section


# Section 5: _build_task_prompt integration


class TestBuildTaskPromptIntegration:
    @pytest.fixture(autouse=True)
    def gen(self):
        with patch("autort.TaskGenerator.get_global_registry") as mreg, patch(
            "autort.TaskGenerator.OpenAI"
        ):
            mreg.return_value.get_all_operations.return_value = []
            from autort.TaskGenerator import TaskGenerator

            cfg = Mock(
                LM_STUDIO_URL="x",
                TASK_GENERATION_MODEL="m",
                MAX_JSON_RETRIES=1,
                TASK_GENERATION_TEMPERATURE=0.7,
            )
            self.gen = TaskGenerator(cfg)

    def test_no_context_section_absent(self):
        scene = make_scene()  # no last_task_context
        prompt = self.gen._build_task_prompt(scene, ["Robot1"], 1, False)
        assert "PREVIOUS TASK CONTEXT" not in prompt

    def test_with_context_section_present(self):
        ctx = make_context(["grasp_object"])
        scene = make_scene(last_task_context=ctx)
        prompt = self.gen._build_task_prompt(scene, ["Robot1"], 1, False)
        assert "PREVIOUS TASK CONTEXT" in prompt

    def test_section_appears_before_task_label(self):
        ctx = make_context(["grasp_object"])
        scene = make_scene(last_task_context=ctx)
        prompt = self.gen._build_task_prompt(scene, ["Robot1"], 1, False)
        assert prompt.index("PREVIOUS TASK CONTEXT") < prompt.index("TASK:")

    def test_no_context_prompt_contains_task_label(self):
        scene = make_scene()
        prompt = self.gen._build_task_prompt(scene, ["Robot1"], 1, False)
        assert "TASK:" in prompt


# Section 6: AutoRTOrchestrator state tracking


class TestAutoRTOrchestratorState:
    def _make_orchestrator(self):
        with patch("autort.AutoRTLoop.get_global_registry"), patch(
            "autort.AutoRTLoop.get_world_state"
        ), patch("autort.TaskGenerator.get_global_registry") as mreg, patch(
            "autort.TaskGenerator.OpenAI"
        ), patch(
            "autort.AutoRTLoop.TaskGenerator"
        ), patch(
            "autort.AutoRTLoop.RobotConstitution"
        ), patch(
            "autort.AutoRTLoop.TaskSelector"
        ), patch(
            "orchestrators.SequenceExecutor.SequenceExecutor"
        ):
            mreg.return_value.get_all_operations.return_value = []
            from autort.AutoRTLoop import AutoRTOrchestrator

            with patch("orchestrators.SequenceExecutor.SequenceExecutor"):
                orch = AutoRTOrchestrator.__new__(AutoRTOrchestrator)
                orch._last_task_context = None
                orch._running = False
                return orch

    def test_starts_none(self):
        from autort.AutoRTLoop import AutoRTOrchestrator

        with patch("autort.AutoRTLoop.get_global_registry"), patch(
            "autort.AutoRTLoop.get_world_state"
        ), patch("autort.AutoRTLoop.TaskGenerator"), patch(
            "autort.AutoRTLoop.RobotConstitution"
        ), patch(
            "autort.AutoRTLoop.TaskSelector"
        ), patch(
            "orchestrators.SequenceExecutor.SequenceExecutor"
        ):
            orch = AutoRTOrchestrator()
            assert orch._last_task_context is None

    def test_updates_after_execution(self):
        from autort.DataModels import ExecutedTaskContext, Operation, ProposedTask

        op = Operation(type="grasp_object", robot_id="Robot1", parameters={})
        task = ProposedTask(
            task_id="t1",
            description="Grasp red cube",
            operations=[op],
            required_robots=["Robot1"],
            estimated_complexity=3,
        )
        result = {"success": True, "result": "grasped red_cube"}

        from autort.AutoRTLoop import AutoRTOrchestrator

        with patch("autort.AutoRTLoop.get_global_registry"), patch(
            "autort.AutoRTLoop.get_world_state"
        ), patch("autort.AutoRTLoop.TaskGenerator"), patch(
            "autort.AutoRTLoop.RobotConstitution"
        ), patch(
            "autort.AutoRTLoop.TaskSelector"
        ), patch(
            "orchestrators.SequenceExecutor.SequenceExecutor"
        ):
            orch = AutoRTOrchestrator()

        # Simulate what _run_one_iteration does after execution
        orch._last_task_context = ExecutedTaskContext(
            task_id=task.task_id,
            description=task.description,
            operation_types=[op.type for op in task.operations],
            success=result.get("success", False),
            result_summary=str(result.get("result", "")),
        )

        assert orch._last_task_context is not None
        assert orch._last_task_context.task_id == "t1"
        assert orch._last_task_context.success is True
        assert orch._last_task_context.operation_types == ["grasp_object"]

    def test_capture_scene_accepts_context_kwarg(self):
        """_capture_scene signature must accept last_task_context without error."""
        import inspect
        from autort.AutoRTLoop import AutoRTOrchestrator

        sig = inspect.signature(AutoRTOrchestrator._capture_scene)
        assert "last_task_context" in sig.parameters
        param = sig.parameters["last_task_context"]
        assert param.default is None
