#!/usr/bin/env python3
"""Per-robot LLM agent for multi-robot negotiation (task analysis, plan proposal, evaluation)."""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import requests

from config.Servers import (
    LMSTUDIO_BASE_URL,
    DEFAULT_LMSTUDIO_MODEL,
    LLM_THINKING_BUDGET,
    LLM_THINKING_ENABLED,
    SYSTEM_PROMPT_BASE,
)
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


@dataclass
class TaskAnalysis:
    """Robot's self-assessment of a task."""

    robot_id: str
    can_contribute: bool = True
    capabilities: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    suggested_role: str = ""
    requires_collaboration: bool = False
    confidence: float = 0.5


@dataclass
class PlanProposal:
    """Coordinated plan proposed by a robot agent."""

    proposer_id: str
    reasoning: str = ""
    commands: List[Dict[str, Any]] = field(default_factory=list)
    round_number: int = 1
    estimated_duration_s: float = 0.0


@dataclass
class ProposalEvaluation:
    """Robot's accept/reject verdict on a peer's plan proposal."""

    evaluator_id: str
    accept: bool = False
    concerns: List[str] = field(default_factory=list)
    suggested_changes: List[str] = field(default_factory=list)
    confidence: float = 0.5


class RobotLLMAgent:
    """Per-robot LLM agent for task analysis, plan proposal, and evaluation."""

    def __init__(
        self,
        robot_id: str,
        lm_studio_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
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
        """Phase 1: Analyze task from this robot's perspective."""
        context = self._build_agent_context(world_state_snapshot)
        ops_str = ", ".join(available_operations)

        workspace_side = self._get_workspace_label()
        system_prompt = (
            SYSTEM_PROMPT_BASE
            + f" You are {self.robot_id}, the {workspace_side} robot arm. Analyze tasks from your own spatial perspective and only claim capabilities within your workspace bounds."
        )
        user_prompt = f"""{context}

Operations: {ops_str}

Task: "{task}"

Set can_contribute=true if you can play ANY part (even as one half of a pair). Only false if completely irrelevant.

JSON: {{"can_contribute":bool,"capabilities":[],"constraints":[],"suggested_role":"","requires_collaboration":bool,"confidence":0.0}}"""

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
        """Phase 2: Propose a coordinated plan considering other robots' analyses."""
        context = self._build_agent_context(world_state)

        analyses_summary = ""
        for analysis in other_analyses:
            analyses_summary += f"\n{analysis.robot_id}: role='{analysis.suggested_role}' contribute={analysis.can_contribute}, collab={analysis.requires_collaboration}"

        if available_operations:
            ops_str = "\n".join(f"  - {op}" for op in available_operations)
            ops_section = (
                f"\nAvailable operations (use ONLY these exact names):\n{ops_str}\n"
            )
        else:
            logger.warning(
                f"[{self.robot_id}] propose_plan called without available_operations; LLM may hallucinate operation names"
            )
            ops_section = ""

        workspace_side = self._get_workspace_label()
        system_prompt = (
            SYSTEM_PROMPT_BASE
            + f" You are {self.robot_id}, the {workspace_side} robot arm, proposing a multi-robot coordination plan. Assign operations to robots based on workspace proximity. Every signal must have a matching wait_for_signal."
        )
        user_prompt = f"""Round {round_number}: propose a coordinated plan.

{context}
{ops_section}
Other robots:{analyses_summary}

Task: "{task}"

Rules: each command needs operation+params(robot_id), optional parallel_group/capture_var. Every signal needs matching wait_for_signal. Every participating robot must have >= 1 command.

JSON: {{"reasoning":"","commands":[{{"parallel_group":1,"operation":"","params":{{"robot_id":""}}}}],"estimated_duration_s":0.0}}"""

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
        """Phase 3: Evaluate another robot's plan proposal."""
        context = self._build_agent_context(world_state)

        commands_json = json.dumps(proposal.commands, indent=2)

        workspace_side = self._get_workspace_label()
        system_prompt = (
            SYSTEM_PROMPT_BASE
            + f" You are {self.robot_id}, the {workspace_side} robot arm, evaluating a plan proposed by {proposal.proposer_id}. Be conservative: flag any operation that exceeds your workspace bounds or creates collision risk."
        )
        user_prompt = f"""{context}

Note: grasp_object and receive_handoff compute positions internally -> do NOT flag missing coords for these.

Task: "{task}"
Proposed by {proposal.proposer_id}: {proposal.reasoning}

Plan: {commands_json}

Check: (1) your actions within reach? (2) signal/wait pairs match? (3) collision risks? (4) efficient?

JSON: {{"accept":bool,"concerns":[],"suggested_changes":[],"confidence":0.0}}"""

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
        """Return workspace side label — uses config assignments, not fragile robot_id substring matching."""
        workspace = ROBOT_WORKSPACE_ASSIGNMENTS.get(self.robot_id, "")
        if "left" in workspace.lower():
            return "left (X < 0)"
        if "right" in workspace.lower():
            return "right (X > 0)"
        return f"workspace '{workspace}'" if workspace else "workspace (unknown)"

    def _build_agent_context(self, world_state_snapshot: Dict[str, Any]) -> str:
        """Build LLM context string for this robot's spatial situation."""
        wb = WORKSPACE_REGIONS.get(self.workspace, {})
        sz = WORKSPACE_REGIONS.get("shared_zone", {})
        context = (
            f"Robot: {self.robot_id} | Base: {self.base_position} | Reach: {self.max_reach}m\n"
            f"Workspace: {self.workspace} x=[{wb.get('x_min')},{wb.get('x_max')}] z=[{wb.get('z_min')},{wb.get('z_max')}]\n"
            f"Shared zone (both robots): x=[{sz.get('x_min')},{sz.get('x_max')}] -> objects here are reachable by either robot"
        )

        # Robot state from world state
        robot_states = world_state_snapshot.get("robots", {})
        my_state = robot_states.get(self.robot_id, {})
        if my_state:
            context += f"\nCurrent position: {my_state.get('position', 'unknown')}"
            context += f"\nGripper state: {my_state.get('gripper_state', 'unknown')}"
            context += f"\nIs moving: {my_state.get('is_moving', False)}"

        # Objects in scene - each annotated with its zone so the model can
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
        """Classify world position into named workspace zone via WORKSPACE_REGIONS X bounds."""
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
        """Call LM Studio. Returns response text or None on failure."""
        try:
            payload: dict = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": 8192,  # Must cover thinking budget + actual JSON response
                **(
                    {
                        "thinking": {
                            "type": "enabled",
                            "budget_tokens": LLM_THINKING_BUDGET,
                        }
                    }
                    if LLM_THINKING_ENABLED
                    else {}
                ),
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
        """Extract JSON from LLM response. Delegates to core.LLMUtils."""
        result = _extract_json_util(content)
        if result is None:
            logger.error(f"[{self.robot_id}] Failed to extract JSON from response")
        return result
