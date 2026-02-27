"""
AutoRT Orchestration Loop

Main loop that composes existing operations for scene understanding
and uses new modules for task generation, safety filtering, and selection.
"""

import logging
import time
import numpy as np
from typing import List, Optional, Dict, Any

from autort.DataModels import SceneDescription, GroundedObject, ProposedTask
from autort.TaskGenerator import TaskGenerator
from autort.RobotConstitution import RobotConstitution
from autort.TaskSelector import TaskSelector
from operations.Registry import get_global_registry
from operations.WorldState import get_world_state
from config import AutoRT as config

logger = logging.getLogger(__name__)


class AutoRTOrchestrator:
    """
    Main AutoRT loop. Composes existing operations for scene understanding
    and uses new modules for task generation, safety filtering, and selection.

    Scene capture uses existing operations:
    - detect_object_stereo(selection="all") → 3D-grounded detections
    - analyze_scene() → VLM scene reasoning (optional)
    - WorldState.get_all_objects() → tracked object states
    """

    def __init__(
        self,
        robot_ids: Optional[List[str]] = None,
        human_in_loop: Optional[bool] = None,
        autonomous: bool = False,
        loop_delay_seconds: Optional[float] = None,
        strategy: str = "balanced",
    ):
        """
        Initialize AutoRT orchestrator.

        Args:
            robot_ids: Robot IDs to use (default from config)
            human_in_loop: Require human approval (default from config)
            autonomous: Override human_in_loop to False
            loop_delay_seconds: Pause between loop iterations (default from config)
            strategy: Task selection strategy ("balanced", "explore", "exploit", "random")
        """
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

        self._running = False

    def start(self):
        """Run continuous task generation loop"""
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
        """Stop the AutoRT loop"""
        self._running = False

    def _run_one_iteration(self):
        """Execute one full iteration of the AutoRT loop"""
        # 1. Capture scene using existing operations
        scene = self._capture_scene()
        if not scene.objects:
            logger.info("No objects detected, skipping iteration")
            return

        logger.info(f"Scene: {len(scene.objects)} objects detected")

        # 2. Generate task candidates
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

        # 3. Filter through constitution
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

        # 4. Select best task
        selected = self.task_selector.select_task(approved, strategy=self.strategy)
        if selected is None:
            return

        # 5. Human approval (ON by default)
        if self.human_in_loop:
            selected = self._request_approval(selected)
            if selected is None:
                return

        # 6. Execute via existing SequenceExecutor
        logger.info(f"Executing task: {selected.description}")
        result = self._execute_task(selected)

        # 7. Update history for future selection
        self.task_selector.update_history(selected, result)
        logger.info(f"Task completed: success={result.get('success', False)}")

    def _capture_scene(self) -> SceneDescription:
        """
        Capture scene state by composing existing operations.

        Uses:
        - detect_object_stereo(selection="all") for 3D object detection
        - analyze_scene() for optional VLM reasoning
        - WorldState for robot positions
        """
        grounded_objects = []

        # Step 1: Run stereo detection for all visible objects
        try:
            detection_result = self.registry.execute_operation_by_name(
                "detect_object_stereo",
                selection="all",
                camera_id="StereoCamera",
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

        # Step 2: Supplement with WorldState tracked objects
        for obj_state in self.world_state.get_all_objects():
            # Avoid duplicates (objects already detected by stereo)
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

        # Step 3: Optional VLM scene reasoning
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

        # Step 4: Gather robot states
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
        )

    def _execute_task(self, task: ProposedTask) -> Dict[str, Any]:
        """
        Execute task via existing SequenceExecutor.

        Converts ProposedTask operations to SequenceExecutor format.
        """
        try:
            # Import lazily to avoid circular dependencies
            from orchestrators.SequenceExecutor import SequenceExecutor

            executor = SequenceExecutor()

            # Convert operations to executor format
            # SequenceExecutor expects: {"operation": "...", "params": {"robot_id": "...", ...}}
            commands = []
            for op in task.operations:
                # Merge robot_id into parameters (params should include robot_id)
                params = {"robot_id": op.robot_id, **op.parameters}
                commands.append({"operation": op.type, "params": params})

            result = executor.execute_sequence(commands)
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
        """Display task and wait for user approval via console"""
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
