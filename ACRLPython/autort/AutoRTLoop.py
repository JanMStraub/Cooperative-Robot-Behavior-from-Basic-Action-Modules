#!/usr/bin/env python3
"""AutoRT main loop: scene capture → task generation → safety filter → select → execute."""

import logging
import time
import numpy as np
from typing import List, Optional, Dict, Any

from autort.DataModels import (
    SceneDescription,
    GroundedObject,
    ProposedTask,
    ExecutedTaskContext,
)
from autort.TaskGenerator import TaskGenerator
from autort.RobotConstitution import RobotConstitution
from autort.TaskSelector import TaskSelector
from operations.Registry import get_global_registry
from operations.WorldState import get_world_state
from config import AutoRT as config
from config.Vision import DEFAULT_CAMERA_ID

logger = logging.getLogger(__name__)


class AutoRTOrchestrator:
    """Runs the AutoRT loop: perceive → generate → filter → select → execute."""

    def __init__(
        self,
        robot_ids: Optional[List[str]] = None,
        human_in_loop: Optional[bool] = None,
        autonomous: bool = False,
        loop_delay_seconds: Optional[float] = None,
        strategy: str = "balanced",
    ):
        self.robot_ids = robot_ids or config.DEFAULT_ROBOTS
        self.human_in_loop = (
            human_in_loop if human_in_loop is not None else config.HUMAN_IN_LOOP_DEFAULT
        )
        if autonomous:
            self.human_in_loop = False
        self.loop_delay = loop_delay_seconds or config.LOOP_DELAY_SECONDS
        self.strategy = strategy

        self.registry = get_global_registry()
        self.world_state = get_world_state()
        self.task_generator = TaskGenerator(config)
        self.constitution = RobotConstitution(config)
        self.task_selector = TaskSelector()

        from orchestrators.SequenceExecutor import SequenceExecutor

        self._executor = SequenceExecutor()

        self._running = False
        self._last_task_context: Optional[ExecutedTaskContext] = None

    def start(self):
        self._running = True
        logger.info(
            f"AutoRT starting: robots={self.robot_ids}, "
            f"human_in_loop={self.human_in_loop}, strategy={self.strategy}"
        )

        iteration = 0
        while self._running:
            iteration += 1
            logger.info(f"--- AutoRT iteration {iteration} ---")

            try:
                self._run_one_iteration()
            except KeyboardInterrupt:
                logger.info("AutoRT stopped by user")
                self._running = False
                break
            except Exception as e:
                logger.error(f"AutoRT iteration failed: {e}", exc_info=True)

            if self._running:
                time.sleep(self.loop_delay)

    def stop(self):
        self._running = False

    def _run_one_iteration(self):
        scene = self._capture_scene(last_task_context=self._last_task_context)
        if not scene.objects:
            logger.info("No objects detected, skipping iteration")
            return

        logger.info(f"Scene: {len(scene.objects)} objects detected")

        candidates = self.task_generator.generate_tasks(
            scene,
            robot_ids=self.robot_ids,
            num_tasks=config.MAX_TASK_CANDIDATES,
            include_collaborative=(
                len(self.robot_ids) > 1 and config.ENABLE_COLLABORATIVE_TASKS
            ),
        )

        if not candidates:
            logger.warning("No tasks generated")
            return

        logger.info(f"Generated {len(candidates)} task candidates")

        approved = []
        for task in candidates:
            verdict = self.constitution.evaluate_task(task, scene)
            if verdict.approved:
                approved.append(task)
                if verdict.warnings:
                    logger.info(
                        f"Task '{task.task_id}' approved with warnings: {verdict.warnings}"
                    )
            else:
                logger.info(
                    f"Task '{task.task_id}' rejected: {verdict.rejection_reason}"
                )

        if not approved:
            logger.warning("All tasks rejected by constitution")
            return

        logger.info(f"{len(approved)} tasks approved")

        selected = self.task_selector.select_task(approved, strategy=self.strategy)
        if selected is None:
            return

        if self.human_in_loop:
            selected = self._request_approval(selected)
            if selected is None:
                return

        logger.info(f"Executing task: {selected.description}")
        result = self._execute_task(selected)

        self.task_selector.update_history(selected, result)
        logger.info(f"Task completed: success={result.get('success', False)}")

        self._last_task_context = ExecutedTaskContext(
            task_id=selected.task_id,
            description=selected.description,
            operation_types=[op.type for op in selected.operations],
            success=result.get("success", False),
            result_summary=(
                str(result.get("result", ""))
                if result.get("success")
                else str(result.get("error", "unknown error"))
            ),
        )

    def _capture_scene(
        self, last_task_context: Optional[ExecutedTaskContext] = None
    ) -> SceneDescription:
        grounded_objects = []

        try:
            detection_result = self.registry.execute_operation_by_name(
                "detect_object_stereo",
                selection="all",
                camera_id=DEFAULT_CAMERA_ID,
            )

            if detection_result.success and detection_result.result:
                detections = detection_result.result.get("detections", [])
                for det in detections:
                    grounded_objects.append(
                        GroundedObject(
                            object_id=det.get(
                                "object_id", f"obj_{len(grounded_objects)}"
                            ),
                            color=det.get("color", "unknown"),
                            position=(det["x"], det["y"], det["z"]),
                            confidence=det.get("confidence", 0.0),
                            graspable=det.get("is_graspable", True),
                        )
                    )
        except Exception as e:
            logger.warning(f"Stereo detection failed: {e}")

        for obj_state in self.world_state.get_all_objects():
            # Avoid duplicates - skip objects already detected by stereo within 5cm
            already_detected = any(
                np.linalg.norm(np.array(g.position) - np.array(obj_state.position))
                < 0.05
                for g in grounded_objects
            )
            if not already_detected:
                grounded_objects.append(
                    GroundedObject(
                        object_id=obj_state.object_id,
                        color=obj_state.color,
                        position=obj_state.position,
                        confidence=obj_state.confidence,
                        graspable=obj_state.is_graspable,
                    )
                )

        scene_summary = ""
        if config.USE_VLM_REASONING:
            try:
                analysis_result = self.registry.execute_operation_by_name(
                    "analyze_scene",
                    prompt="Describe the robot workspace. List spatial relationships between objects "
                    "and suggest manipulation priorities. Keep under 100 words.",
                    camera_id="MainCamera",
                )
                if analysis_result.success and analysis_result.result:
                    scene_summary = analysis_result.result.get("analysis", "")
            except Exception as e:
                logger.warning(f"Scene analysis failed: {e}")

        if not scene_summary:
            labels = [obj.color for obj in grounded_objects]
            scene_summary = f"Detected {len(grounded_objects)} objects: {labels}"

        robot_states = {}
        for rid in self.robot_ids:
            state = self.world_state.get_robot_state(rid)
            if state:
                robot_states[rid] = {
                    "position": state.position,
                    "gripper_state": state.gripper_state,
                    "is_moving": state.is_moving,
                }

        return SceneDescription(
            timestamp=time.time(),
            objects=grounded_objects,
            scene_summary=scene_summary,
            robot_states=robot_states,
            last_task_context=last_task_context,
        )

    def _execute_task(self, task: ProposedTask) -> Dict[str, Any]:
        try:
            commands = []
            for op in task.operations:
                params = {"robot_id": op.robot_id, **op.parameters}
                commands.append({"operation": op.type, "params": params})

            result = self._executor.execute_sequence(commands)
            return {
                "success": (
                    result.get("success", False) if isinstance(result, dict) else False
                ),
                "result": result,
            }

        except Exception as e:
            logger.error(f"Task execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    def _request_approval(self, task: ProposedTask) -> Optional[ProposedTask]:
        print(f"\n{'=' * 60}")
        print(f"PROPOSED TASK: {task.description}")
        print(f"Robots: {task.required_robots}")
        print(f"Complexity: {task.estimated_complexity}/10")
        print(f"Operations: {len(task.operations)} steps")
        for i, op in enumerate(task.operations):
            print(f"  {i+1}. [{op.robot_id}] {op.type}({op.parameters})")
        print(f"Reasoning: {task.reasoning}")
        print(f"{'=' * 60}")

        try:
            response = input("Execute this task? [y/N/skip]: ").strip().lower()
            if response == "y":
                return task
        except EOFError:
            pass

        return None
