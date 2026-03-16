#!/usr/bin/env python3
"""
AutoRT Robot Constitution

Two-layer safety system: Semantic (LLM) + Kinematic (Code)
"""

import json
import logging
import numpy as np
from typing import List, Tuple
from openai import OpenAI

from autort.DataModels import ProposedTask, TaskVerdict, SceneDescription
from operations.WorldState import get_world_state

logger = logging.getLogger(__name__)


class BoundingBox:
    """Axis-aligned bounding box for workspace bounds checking"""

    def __init__(
        self,
        min_corner: Tuple[float, float, float],
        max_corner: Tuple[float, float, float],
    ):
        self.min_corner = min_corner
        self.max_corner = max_corner

    def contains(self, point: Tuple[float, float, float]) -> bool:
        """Check if a 3D point is within the bounding box"""
        return all(
            self.min_corner[i] <= point[i] <= self.max_corner[i] for i in range(3)
        )


class RobotConstitution:
    """
    Two-Layer Safety System:

    LAYER 1: SEMANTIC SAFETY (LLM-based)
    - Detects intent violations (harm, damage, unethical actions)
    - Uses LLM to understand natural language descriptions

    LAYER 2: KINEMATIC SAFETY (Code-based)
    - Validates physical feasibility and safety
    - Checks workspace bounds, live robot positions, velocity limits
    - Uses WorldState for current robot positions (not just planned targets)
    """

    def __init__(self, config):
        """
        Initialize RobotConstitution.

        Args:
            config: AutoRT config module with safety settings
        """
        self.config = config
        self.llm_client = OpenAI(base_url=config.LM_STUDIO_URL, api_key="not-needed")
        self.model = config.SAFETY_VALIDATION_MODEL
        self.world_state = get_world_state()

        # Semantic safety rules (checked by LLM)
        self.semantic_rules = [
            "Do not harm humans or animals",
            "Do not throw objects at living beings",
            "Do not damage expensive or fragile equipment",
            "Do not perform unethical or dangerous actions",
            "Do not move at unsafe speeds near obstacles",
        ]

        # Kinematic safety limits (checked by code)
        bounds = config.WORKSPACE_BOUNDS
        self.workspace_bounds = BoundingBox(
            min_corner=bounds["min_corner"], max_corner=bounds["max_corner"]
        )
        self.max_velocity = config.MAX_VELOCITY
        self.min_robot_separation = config.MIN_ROBOT_SEPARATION
        self.max_gripper_force = config.MAX_GRIPPER_FORCE

    def evaluate_task(self, task: ProposedTask, scene: SceneDescription) -> TaskVerdict:
        """
        Evaluate task through both safety layers.

        Returns:
            TaskVerdict with approval status and reasons
        """
        # LAYER 1: Semantic Safety (LLM)
        semantic_verdict = self._evaluate_semantic_safety(task)
        if not semantic_verdict.approved:
            return semantic_verdict

        # LAYER 2: Kinematic Safety (Code)
        kinematic_verdict = self._evaluate_kinematic_safety(task, scene)
        if not kinematic_verdict.approved:
            return kinematic_verdict

        # All checks passed
        return TaskVerdict(
            approved=True,
            violations=[],
            warnings=kinematic_verdict.warnings,  # Pass through kinematic warnings
        )

    def _evaluate_semantic_safety(self, task: ProposedTask) -> TaskVerdict:
        """
        LAYER 1: Use LLM to check for semantic violations.

        Purpose: Detect harmful intent that code cannot catch.
        Example: "Throw the cube at the camera" (harmful intent)
        """
        rules_str = "\n".join(f"- {rule}" for rule in self.semantic_rules)

        prompt = f"""You are a robot safety validator.

TASK DESCRIPTION:
{task.description}

TASK OPERATIONS:
{json.dumps([op.model_dump() for op in task.operations], indent=2)}

SAFETY RULES:
{rules_str}

QUESTION:
Does this task violate any safety rule?

Respond in JSON format ONLY:
{{
  "violates": true or false,
  "rule_violated": "name of rule" or null,
  "reason": "brief explanation"
}}"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            raw = response.choices[0].message.content

            if raw is None:
                raise ValueError("LLM returned empty response")

            # Strip markdown if present
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            result = json.loads(raw)

            if result.get("violates", False):
                return TaskVerdict(
                    approved=False,
                    violations=[f"Semantic safety: {result.get('reason', 'Unknown')}"],
                    rejection_reason=f"Violates rule: {result.get('rule_violated', 'Unknown')}",
                )

            return TaskVerdict(approved=True)

        except Exception as e:
            logger.error(f"Semantic safety check failed: {e}")
            # Fail-safe: Reject on error
            return TaskVerdict(
                approved=False,
                violations=["Semantic safety check error"],
                rejection_reason=f"LLM safety validation failed: {e}",
            )

    def _evaluate_kinematic_safety(
        self, task: ProposedTask, scene: SceneDescription
    ) -> TaskVerdict:
        """
        LAYER 2: Code-based physics and bounds checking.

        Uses WorldState for live robot positions in addition to
        checking planned target positions.

        Checks:
        1. Workspace bounds: All targets reachable
        2. Robot collision: Live positions + planned targets maintain separation
        3. Velocity limits: Safe motion speeds
        4. Force limits: Gripper force within safe range
        """
        violations = []
        warnings = []

        # Collect planned target positions per robot
        robot_planned_targets = {}

        for i, op in enumerate(task.operations):
            # Workspace bounds check
            if op.type in ("move_to_coordinate", "move_from_a_to_b"):
                target_pos = op.parameters.get("target_position")
                if target_pos:
                    target_tuple = tuple(target_pos)
                    robot_planned_targets.setdefault(op.robot_id, []).append(
                        target_tuple
                    )

                    if not self.workspace_bounds.contains(target_tuple):
                        violations.append(
                            f"Op {i} ({op.type}): Target {target_tuple} outside workspace "
                            f"bounds {self.workspace_bounds.min_corner} to {self.workspace_bounds.max_corner}"
                        )

            # Velocity check (if specified)
            if "velocity" in op.parameters:
                vel = op.parameters["velocity"]
                if vel > self.max_velocity:
                    violations.append(
                        f"Op {i} ({op.type}): Velocity {vel} m/s exceeds limit {self.max_velocity} m/s"
                    )

            # Gripper force check
            if op.type == "control_gripper":
                force = op.parameters.get("force", 0.0)
                if force > self.max_gripper_force:
                    violations.append(
                        f"Op {i}: Gripper force {force}N exceeds limit {self.max_gripper_force}N"
                    )

        # Robot collision check: planned targets + live positions
        if len(task.required_robots) > 1:
            collision_violations = self._check_robot_collisions(
                task.required_robots, robot_planned_targets
            )
            violations.extend(collision_violations)

        if violations:
            return TaskVerdict(
                approved=False,
                violations=violations,
                warnings=warnings,
                rejection_reason="Kinematic safety violations detected",
            )

        return TaskVerdict(approved=True, violations=[], warnings=warnings)

    def _check_robot_collisions(
        self, robot_ids: List[str], planned_targets: dict
    ) -> List[str]:
        """
        Check robot collision risk using both planned targets AND live WorldState positions.

        Checks pairwise distances between:
        1. All planned target positions across robots
        2. Each robot's planned targets vs other robots' current positions (from WorldState)
        """
        violations = []

        # Gather all positions per robot (live + planned)
        robot_positions = {}
        for rid in robot_ids:
            positions = list(planned_targets.get(rid, []))

            # Add current live position from WorldState
            live_pos = self.world_state.get_robot_position(rid)
            if live_pos:
                positions.append(live_pos)

            robot_positions[rid] = positions

        # Pairwise collision check
        ids = list(robot_positions.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                r1, r2 = ids[i], ids[j]
                for p1 in robot_positions[r1]:
                    for p2 in robot_positions[r2]:
                        dist = np.linalg.norm(np.array(p1) - np.array(p2))
                        if dist < self.min_robot_separation:
                            violations.append(
                                f"Collision risk: {r1} and {r2} within "
                                f"{dist:.3f}m (min: {self.min_robot_separation}m)"
                            )
        return violations
