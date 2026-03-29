#!/usr/bin/env python3
"""
Per-Robot LLM Agent for Negotiation
=====================================

Each robot gets its own LLM agent that can:
1. Analyze tasks from its perspective (capabilities, workspace, reachability)
2. Propose coordinated plans with parallel groups and synchronization
3. Evaluate counter-proposals from other robots

The agent uses LM Studio for LLM inference, following the same pattern
as CommandParser._parse_with_llm().
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import requests

from config.Servers import LMSTUDIO_BASE_URL, DEFAULT_LMSTUDIO_MODEL, LLM_THINKING_BUDGET, LLM_THINKING_ENABLED, SYSTEM_PROMPT_BASE
from core.LLMUtils import extract_json as _extract_json_util
from config.Negotiation import (
    AGENT_LLM_TIMEOUT,
    NEGOTIATION_TEMPERATURE,
    USE_STRUCTURED_OUTPUT,
)
from config.Robot import (
    ROBOT_BASE_POSITIONS,
    ROBOT_WORKSPACE_ASSIGNMENTS,
    WORKSPACE_REGIONS,
    MAX_ROBOT_REACH,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class TaskAnalysis:
    """
    Result of a robot analyzing a task from its own perspective.

    Attributes:
        robot_id: Which robot performed the analysis
        can_contribute: Whether this robot can help with the task
        capabilities: What this robot can do for the task
        constraints: Limitations (workspace, reach, current state)
        suggested_role: What role this robot should play
        requires_collaboration: Whether other robots are needed
        confidence: Confidence in the analysis (0.0-1.0)
    """

    robot_id: str
    can_contribute: bool = True
    capabilities: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    suggested_role: str = ""
    requires_collaboration: bool = False
    confidence: float = 0.5


@dataclass
class PlanProposal:
    """
    A proposed coordinated plan from a robot agent.

    Attributes:
        proposer_id: Robot that proposed this plan
        reasoning: Why this plan was chosen
        commands: List of commands in SequenceExecutor format
        round_number: Which negotiation round this came from
        estimated_duration_s: Estimated execution time
    """

    proposer_id: str
    reasoning: str = ""
    commands: List[Dict[str, Any]] = field(default_factory=list)
    round_number: int = 1
    estimated_duration_s: float = 0.0


@dataclass
class ProposalEvaluation:
    """
    A robot's evaluation of another robot's plan proposal.

    Attributes:
        evaluator_id: Robot that evaluated the proposal
        accept: Whether the robot accepts the proposal
        concerns: List of concerns about the proposal
        suggested_changes: Modifications to improve the plan
        confidence: Confidence in the evaluation (0.0-1.0)
    """

    evaluator_id: str
    accept: bool = False
    concerns: List[str] = field(default_factory=list)
    suggested_changes: List[str] = field(default_factory=list)
    confidence: float = 0.5


# ============================================================================
# Robot LLM Agent
# ============================================================================


class RobotLLMAgent:
    """
    Per-robot LLM agent for task analysis, plan proposal, and evaluation.

    Each instance represents one robot's perspective and uses LM Studio
    for reasoning about multi-robot coordination.
    """

    def __init__(
        self,
        robot_id: str,
        lm_studio_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        """
        Initialize a robot LLM agent.

        Args:
            robot_id: ID of the robot this agent represents
            lm_studio_url: LM Studio base URL (default from config)
            model: Model name (default from config)
            temperature: LLM temperature (default from config)
        """
        self.robot_id = robot_id
        self.lm_studio_url = lm_studio_url or LMSTUDIO_BASE_URL
        self.model = model or DEFAULT_LMSTUDIO_MODEL
        self.temperature = (
            temperature if temperature is not None else NEGOTIATION_TEMPERATURE
        )

        # Robot config
        self.base_position = ROBOT_BASE_POSITIONS.get(robot_id, (0, 0, 0))
        self.workspace = ROBOT_WORKSPACE_ASSIGNMENTS.get(robot_id, "unknown")
        self.max_reach = MAX_ROBOT_REACH

    def analyze_task(
        self,
        task: str,
        world_state_snapshot: Dict[str, Any],
        available_operations: List[str],
    ) -> TaskAnalysis:
        """
        Phase 1: Analyze a task from this robot's perspective.

        Determines what this robot can contribute, its constraints,
        and whether collaboration is needed.

        Args:
            task: Natural language task description
            world_state_snapshot: Current world state
            available_operations: List of available operation names

        Returns:
            TaskAnalysis with this robot's assessment
        """
        context = self._build_agent_context(world_state_snapshot)
        ops_str = ", ".join(available_operations)

        workspace_side = self._get_workspace_label()
        system_prompt = (
            SYSTEM_PROMPT_BASE
            + f" You are {self.robot_id}, the {workspace_side} robot arm. "
            f"Analyze tasks from your own spatial perspective — only claim capabilities "
            f"within your workspace bounds. Respond only with a JSON object."
        )
        user_prompt = f"""Analyze this task from your perspective as {self.robot_id}.

{context}

Available operations: {ops_str}

Task: "{task}"

IMPORTANT: Set "can_contribute" to true if you can play ANY part in this task — even as one half of a collaborative pair. Only set it to false if this robot is completely irrelevant to the task (e.g. wrong workspace, wrong tool).

Respond with JSON:
{{
    "can_contribute": true/false,
    "capabilities": ["what you can do for this task"],
    "constraints": ["your limitations"],
    "suggested_role": "brief role description (e.g. 'grasper', 'receiver', 'stabilizer')",
    "requires_collaboration": true/false,
    "confidence": 0.0-1.0
}}

Output only valid JSON."""

        response = self._call_llm(system_prompt, user_prompt)
        if response is None:
            logger.warning(f"[{self.robot_id}] LLM analysis failed, returning default")
            return TaskAnalysis(robot_id=self.robot_id)

        logger.info(f"[{self.robot_id}] Raw analysis response: {response[:300]}")

        try:
            data = self._extract_json(response)
            if data is None:
                return TaskAnalysis(robot_id=self.robot_id)

            return TaskAnalysis(
                robot_id=self.robot_id,
                can_contribute=data.get("can_contribute", True),
                capabilities=data.get("capabilities", []),
                constraints=data.get("constraints", []),
                suggested_role=data.get("suggested_role", ""),
                requires_collaboration=data.get("requires_collaboration", False),
                confidence=data.get("confidence", 0.5),
            )
        except Exception as e:
            logger.error(f"[{self.robot_id}] Error parsing analysis: {e}")
            return TaskAnalysis(robot_id=self.robot_id)

    def propose_plan(
        self,
        task: str,
        other_analyses: List[TaskAnalysis],
        world_state: Dict[str, Any],
        round_number: int = 1,
        available_operations: Optional[List[str]] = None,
    ) -> PlanProposal:
        """
        Phase 2: Propose a coordinated plan considering other robots' analyses.

        Args:
            task: Natural language task description
            other_analyses: Analyses from other robots
            world_state: Current world state snapshot
            round_number: Current negotiation round
            available_operations: List of valid operation names from the registry

        Returns:
            PlanProposal with commands in SequenceExecutor format
        """
        context = self._build_agent_context(world_state)

        # Build summary of other robots' analyses
        analyses_summary = ""
        for analysis in other_analyses:
            analyses_summary += (
                f"\n{analysis.robot_id}: can_contribute={analysis.can_contribute}, "
                f"role='{analysis.suggested_role}', "
                f"capabilities={analysis.capabilities}, "
                f"constraints={analysis.constraints}"
            )

        # Build operations section for the prompt
        if available_operations:
            ops_str = "\n".join(f"  - {op}" for op in available_operations)
            ops_section = f"\nAvailable operations (use ONLY these exact names):\n{ops_str}\n"
        else:
            logger.warning(
                f"[{self.robot_id}] propose_plan called without available_operations; "
                f"LLM may hallucinate operation names"
            )
            ops_section = ""

        workspace_side = self._get_workspace_label()
        system_prompt = (
            SYSTEM_PROMPT_BASE
            + f" You are {self.robot_id}, the {workspace_side} robot arm, proposing a "
            f"multi-robot coordination plan. Assign operations to robots based on workspace "
            f"proximity. Every signal must have a matching wait_for_signal. "
            f"Respond only with a JSON object."
        )
        user_prompt = f"""Propose a coordinated plan for this task. This is negotiation round {round_number}.

{context}
{ops_section}
Other robots' analyses:{analyses_summary}

Task: "{task}"

Create a plan using these rules:
- Each command needs: "operation", "params" (with "robot_id"), optionally "parallel_group", "capture_var"
- Same parallel_group = concurrent execution
- Use signal/wait_for_signal for synchronization between robots
- Every signal must have a matching wait_for_signal
- IMPORTANT: Use ONLY operation names from the available operations list above
- CRITICAL: This is a MULTI-ROBOT task. The plan MUST include operations assigned to EVERY participating robot. A plan that only assigns work to one robot will be rejected. Each robot must have at least one command.

Respond with JSON:
{{
    "reasoning": "why this plan works",
    "commands": [
        {{"parallel_group": 1, "operation": "op_name", "params": {{"robot_id": "Robot1", ...}}}},
        ...
    ],
    "estimated_duration_s": 10.0
}}

Output only valid JSON."""

        response = self._call_llm(system_prompt, user_prompt)
        if response is None:
            logger.warning(
                f"[{self.robot_id}] LLM proposal failed, returning empty plan"
            )
            return PlanProposal(proposer_id=self.robot_id, round_number=round_number)

        try:
            data = self._extract_json(response)
            if data is None:
                return PlanProposal(
                    proposer_id=self.robot_id, round_number=round_number
                )

            commands = data.get("commands", data.get("plan", []))
            return PlanProposal(
                proposer_id=self.robot_id,
                reasoning=data.get("reasoning", ""),
                commands=commands,
                round_number=round_number,
                estimated_duration_s=data.get("estimated_duration_s", 0.0),
            )
        except Exception as e:
            logger.error(f"[{self.robot_id}] Error parsing proposal: {e}")
            return PlanProposal(proposer_id=self.robot_id, round_number=round_number)

    def evaluate_proposal(
        self,
        proposal: PlanProposal,
        task: str,
        world_state: Dict[str, Any],
    ) -> ProposalEvaluation:
        """
        Phase 3: Evaluate another robot's plan proposal.

        Args:
            proposal: Plan proposal to evaluate
            task: Original task description
            world_state: Current world state snapshot

        Returns:
            ProposalEvaluation with accept/reject and concerns
        """
        context = self._build_agent_context(world_state)

        commands_json = json.dumps(proposal.commands, indent=2)

        workspace_side = self._get_workspace_label()
        system_prompt = (
            SYSTEM_PROMPT_BASE
            + f" You are {self.robot_id}, the {workspace_side} robot arm, evaluating a plan "
            f"proposed by {proposal.proposer_id}. Be conservative: flag any operation that "
            f"exceeds your workspace bounds or creates collision risk. "
            f"Respond only with a JSON object."
        )
        user_prompt = f"""Evaluate this plan from your perspective as {self.robot_id}.

{context}

IMPORTANT operation semantics — do NOT raise concerns about missing coordinates for these:
- orient_gripper_for_handoff_receive(robot_id, object_id, source_robot_id): computes gripper orientation from WorldState automatically. No coordinate params needed or expected.
- receive_handoff(robot_id, object_id, source_robot_id): computes target position from WorldState automatically. No coordinate params needed or expected.
- grasp_object_for_handoff(robot_id, object_id, receiving_robot_id): same — positions computed internally.
Only flag missing coordinates for operations like move_to_coordinate that explicitly require them.

Task: "{task}"
Proposed by: {proposal.proposer_id}
Reasoning: {proposal.reasoning}

Plan:
{commands_json}

Check:
1. Are your assigned actions within your workspace and reach?
2. Is synchronization correct (signal/wait pairs match)?
3. Are there collision risks?
4. Is the plan efficient?

Respond with JSON:
{{
    "accept": true/false,
    "concerns": ["list of concerns — omit concerns about missing coords on handoff ops"],
    "suggested_changes": ["list of suggested modifications"],
    "confidence": 0.0-1.0
}}

Output only valid JSON."""

        response = self._call_llm(system_prompt, user_prompt)
        if response is None:
            logger.warning(
                f"[{self.robot_id}] LLM evaluation failed, rejecting proposal"
            )
            return ProposalEvaluation(
                evaluator_id=self.robot_id, accept=False, confidence=0.3
            )

        try:
            data = self._extract_json(response)
            if data is None:
                return ProposalEvaluation(
                    evaluator_id=self.robot_id, accept=False, confidence=0.3
                )

            return ProposalEvaluation(
                evaluator_id=self.robot_id,
                accept=data.get("accept", False),
                concerns=data.get("concerns", []),
                suggested_changes=data.get("suggested_changes", []),
                confidence=data.get("confidence", 0.5),
            )
        except Exception as e:
            logger.error(f"[{self.robot_id}] Error parsing evaluation: {e}")
            return ProposalEvaluation(
                evaluator_id=self.robot_id, accept=False, confidence=0.3
            )

    def _get_workspace_label(self) -> str:
        """
        Return a human-readable workspace side label for this robot.

        Uses ROBOT_WORKSPACE_ASSIGNMENTS config to determine left/right/unknown,
        avoiding fragile substring matches on robot_id strings.

        Returns:
            Descriptive label such as "left (X < 0)", "right (X > 0)", or
            "workspace '<name>'" for unrecognized assignments.
        """
        workspace = ROBOT_WORKSPACE_ASSIGNMENTS.get(self.robot_id, "")
        if "left" in workspace.lower():
            return "left (X < 0)"
        if "right" in workspace.lower():
            return "right (X > 0)"
        return f"workspace '{workspace}'" if workspace else "workspace (unknown)"

    def _build_agent_context(self, world_state_snapshot: Dict[str, Any]) -> str:
        """
        Build context string describing this robot's situation.

        Args:
            world_state_snapshot: Current world state

        Returns:
            Formatted context string for LLM prompt
        """
        # Robot identity and workspace
        workspace_bounds = WORKSPACE_REGIONS.get(self.workspace, {})
        shared_bounds = WORKSPACE_REGIONS.get("shared_zone", {})
        context = f"""Your identity: {self.robot_id}
Base position: {self.base_position}
Assigned workspace: {self.workspace}
Workspace bounds: {workspace_bounds}
Shared zone (reachable by ALL robots): {shared_bounds}
Max reach: {self.max_reach}m
NOTE: Objects in the shared zone are reachable by both robots. Set can_contribute=true if the target object is in your workspace OR the shared zone."""

        # Robot state from world state
        robot_states = world_state_snapshot.get("robots", {})
        my_state = robot_states.get(self.robot_id, {})
        if my_state:
            context += f"\nCurrent position: {my_state.get('position', 'unknown')}"
            context += f"\nGripper state: {my_state.get('gripper_state', 'unknown')}"
            context += f"\nIs moving: {my_state.get('is_moving', False)}"

        # Objects in scene — each annotated with its zone so the model can
        # determine reachability without doing numeric comparisons itself.
        objects = world_state_snapshot.get("objects", {})
        if objects:
            context += "\nObjects in scene:"
            for obj_id, obj_data in objects.items():
                pos = obj_data.get("position", "unknown")
                color = obj_data.get("color", "unknown")
                zone = self._classify_position_zone(pos)
                context += f"\n  - {obj_id}: color={color}, position={pos}, zone={zone}"

        return context

    def _classify_position_zone(self, position) -> str:
        """
        Classify a world position into a named workspace zone.

        Args:
            position: Position as tuple/list (x, y, z) or "unknown"

        Returns:
            Zone label: "left_workspace", "right_workspace", "shared_zone", or "unknown"
        """
        if not isinstance(position, (list, tuple)) or len(position) < 1:
            return "unknown"
        x = position[0]
        for zone_name, bounds in WORKSPACE_REGIONS.items():
            x_min = bounds.get("x_min", float("-inf"))
            x_max = bounds.get("x_max", float("inf"))
            if x_min <= x <= x_max:
                return zone_name
        return "unknown"

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        Call LM Studio for LLM inference.

        Follows the same pattern as CommandParser._parse_with_llm().

        Args:
            system_prompt: System message for the LLM
            user_prompt: User message with the actual query

        Returns:
            LLM response text or None if failed
        """
        try:
            payload: dict = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": 8192,  # Must cover thinking budget + actual JSON response
                **({"thinking": {"type": "enabled", "budget_tokens": LLM_THINKING_BUDGET}} if LLM_THINKING_ENABLED else {}),
            }
            # Structured output forces the model to emit valid JSON directly,
            # eliminating prose wrapping and Markdown fences.  Kept as opt-in
            # so callers can disable it for models that don't support the flag.
            if USE_STRUCTURED_OUTPUT:
                payload["response_format"] = {"type": "json_object"}

            response = requests.post(
                f"{self.lm_studio_url}/chat/completions",
                json=payload,
                timeout=AGENT_LLM_TIMEOUT,
            )

            if response.status_code != 200:
                logger.error(
                    f"[{self.robot_id}] LLM request failed: {response.status_code}"
                )
                return None

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            logger.debug(f"[{self.robot_id}] LLM response: {content[:200]}")
            return content

        except requests.exceptions.Timeout:
            logger.error(
                f"[{self.robot_id}] LLM request timed out after {AGENT_LLM_TIMEOUT}s"
            )
            return None
        except requests.exceptions.ConnectionError:
            logger.error(
                f"[{self.robot_id}] Cannot connect to LM Studio at {self.lm_studio_url}"
            )
            return None
        except Exception as e:
            logger.error(f"[{self.robot_id}] LLM call error: {e}")
            return None

    def _extract_json(self, content: str) -> Optional[Dict]:
        """
        Extract JSON from LLM response text. Delegates to core.LLMUtils.

        Args:
            content: Raw LLM response

        Returns:
            Parsed JSON dict or None
        """
        result = _extract_json_util(content)
        if result is None:
            logger.error(f"[{self.robot_id}] Failed to extract JSON from response")
        return result
