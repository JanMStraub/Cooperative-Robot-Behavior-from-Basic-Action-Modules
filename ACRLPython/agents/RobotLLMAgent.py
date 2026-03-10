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
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import requests

from config.Servers import LMSTUDIO_BASE_URL, DEFAULT_LMSTUDIO_MODEL
from config.Negotiation import AGENT_LLM_TIMEOUT, NEGOTIATION_TEMPERATURE, USE_STRUCTURED_OUTPUT
from config.Memory import MEMORY_ENABLED
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
        self.temperature = temperature if temperature is not None else NEGOTIATION_TEMPERATURE

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

        system_prompt = (
            f"You are {self.robot_id}, a robot arm analyzing a task. "
            f"Respond with a JSON object describing your assessment."
        )
        user_prompt = f"""Analyze this task from your perspective as {self.robot_id}.

{context}

Available operations: {ops_str}

Task: "{task}"

Respond with JSON:
{{
    "can_contribute": true/false,
    "capabilities": ["what you can do for this task"],
    "constraints": ["your limitations"],
    "suggested_role": "brief role description",
    "requires_collaboration": true/false,
    "confidence": 0.0-1.0
}}

Output only valid JSON."""

        response = self._call_llm(system_prompt, user_prompt)
        if response is None:
            logger.warning(f"[{self.robot_id}] LLM analysis failed, returning default")
            return TaskAnalysis(robot_id=self.robot_id)

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
    ) -> PlanProposal:
        """
        Phase 2: Propose a coordinated plan considering other robots' analyses.

        Args:
            task: Natural language task description
            other_analyses: Analyses from other robots
            world_state: Current world state snapshot
            round_number: Current negotiation round

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

        system_prompt = (
            f"You are {self.robot_id}, proposing a multi-robot coordination plan. "
            f"Output a JSON plan that all robots can execute."
        )
        user_prompt = f"""Propose a plan for this task. This is negotiation round {round_number}.

{context}

Other robots' analyses:{analyses_summary}

Task: "{task}"

Create a plan using these rules:
- Each command needs: "operation", "params" (with "robot_id"), optionally "parallel_group", "capture_var"
- Same parallel_group = concurrent execution
- Use signal/wait_for_signal for synchronization between robots
- Every signal must have a matching wait_for_signal

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
            logger.warning(f"[{self.robot_id}] LLM proposal failed, returning empty plan")
            return PlanProposal(proposer_id=self.robot_id, round_number=round_number)

        try:
            data = self._extract_json(response)
            if data is None:
                return PlanProposal(proposer_id=self.robot_id, round_number=round_number)

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

        system_prompt = (
            f"You are {self.robot_id}, evaluating a plan proposed by {proposal.proposer_id}. "
            f"Determine if the plan is safe and effective for your role."
        )
        user_prompt = f"""Evaluate this plan from your perspective as {self.robot_id}.

{context}

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
    "concerns": ["list of concerns"],
    "suggested_changes": ["list of suggested modifications"],
    "confidence": 0.0-1.0
}}

Output only valid JSON."""

        response = self._call_llm(system_prompt, user_prompt)
        if response is None:
            logger.warning(f"[{self.robot_id}] LLM evaluation failed, accepting by default")
            return ProposalEvaluation(evaluator_id=self.robot_id, accept=True, confidence=0.3)

        try:
            data = self._extract_json(response)
            if data is None:
                return ProposalEvaluation(evaluator_id=self.robot_id, accept=True, confidence=0.3)

            return ProposalEvaluation(
                evaluator_id=self.robot_id,
                accept=data.get("accept", True),
                concerns=data.get("concerns", []),
                suggested_changes=data.get("suggested_changes", []),
                confidence=data.get("confidence", 0.5),
            )
        except Exception as e:
            logger.error(f"[{self.robot_id}] Error parsing evaluation: {e}")
            return ProposalEvaluation(evaluator_id=self.robot_id, accept=True, confidence=0.3)

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
        context = f"""Your identity: {self.robot_id}
Base position: {self.base_position}
Assigned workspace: {self.workspace}
Workspace bounds: {workspace_bounds}
Max reach: {self.max_reach}m"""

        # Robot state from world state
        robot_states = world_state_snapshot.get("robots", {})
        my_state = robot_states.get(self.robot_id, {})
        if my_state:
            context += f"\nCurrent position: {my_state.get('position', 'unknown')}"
            context += f"\nGripper state: {my_state.get('gripper_state', 'unknown')}"
            context += f"\nIs moving: {my_state.get('is_moving', False)}"

        # Objects in scene
        objects = world_state_snapshot.get("objects", {})
        if objects:
            context += "\nObjects in scene:"
            for obj_id, obj_data in objects.items():
                pos = obj_data.get("position", "unknown")
                color = obj_data.get("color", "unknown")
                context += f"\n  - {obj_id}: color={color}, position={pos}"

        # Cross-session memory (operation outcomes from past sessions)
        if MEMORY_ENABLED:
            try:
                from core.MemoryManager import get_memory_manager
                memory_text = get_memory_manager().read_memory(self.robot_id)
                if memory_text:
                    context += f"\n\n## Memory (past sessions)\n{memory_text}"
            except Exception as e:
                logger.debug(f"[{self.robot_id}] Could not load memory: {e}")

        return context

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
                "max_tokens": 3000,
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
                logger.error(f"[{self.robot_id}] LLM request failed: {response.status_code}")
                return None

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            logger.debug(f"[{self.robot_id}] LLM response: {content[:200]}")
            return content

        except requests.exceptions.Timeout:
            logger.error(f"[{self.robot_id}] LLM request timed out after {AGENT_LLM_TIMEOUT}s")
            return None
        except requests.exceptions.ConnectionError:
            logger.error(f"[{self.robot_id}] Cannot connect to LM Studio at {self.lm_studio_url}")
            return None
        except Exception as e:
            logger.error(f"[{self.robot_id}] LLM call error: {e}")
            return None

    def _extract_json(self, content: str) -> Optional[Dict]:
        """
        Extract JSON from LLM response text.

        Args:
            content: Raw LLM response

        Returns:
            Parsed JSON dict or None
        """
        # Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try markdown code block
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try finding JSON object in text
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.error(f"[{self.robot_id}] Failed to extract JSON from response")
        return None
