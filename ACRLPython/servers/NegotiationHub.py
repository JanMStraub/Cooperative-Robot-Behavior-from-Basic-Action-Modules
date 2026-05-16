#!/usr/bin/env python3
"""
Multi-robot negotiation coordinator. Not a TCP server — called directly by SequenceExecutor.
Output matches CommandParser.parse() format, consumable by SequenceExecutor.execute_sequence().
"""

import re
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.SingletonBase import SingletonBase

from config.Negotiation import (
    MAX_NEGOTIATION_ROUNDS,
    NEGOTIATION_TIMEOUT,
    VERIFY_NEGOTIATED_PLANS,
    MAX_PLAN_LENGTH,
)
from config.Robot import ROBOT_BASE_POSITIONS
from agents.RobotLLMAgent import RobotLLMAgent, TaskAnalysis, PlanProposal

logger = logging.getLogger(__name__)


class NegotiationState(Enum):
    """States of a negotiation session."""

    IDLE = "idle"
    ANALYZING = "analyzing"
    PROPOSING = "proposing"
    EVALUATING = "evaluating"
    CONSENSUS = "consensus"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class NegotiationSession:
    """Tracks state of an active negotiation."""

    session_id: str
    task: str
    robot_ids: List[str]
    state: NegotiationState = NegotiationState.IDLE
    analyses: Dict[str, TaskAnalysis] = field(default_factory=dict)
    proposals: List[PlanProposal] = field(default_factory=list)
    current_round: int = 0
    start_time: float = field(default_factory=time.time)


@dataclass
class NegotiationResult:
    """Final result of a negotiation."""

    success: bool = False
    commands: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    rounds_taken: int = 0
    duration_s: float = 0.0
    state: NegotiationState = NegotiationState.IDLE


class NegotiationHub(SingletonBase):
    """
    Central coordinator for multi-robot negotiation.

    Singleton that manages robot LLM agents and orchestrates the
    negotiation protocol. Thread-safe for concurrent access.
    """

    def _singleton_init(self):
        """Initialize the negotiation hub (called once by SingletonBase)."""
        self._agents: Dict[str, RobotLLMAgent] = {}
        self._active_session: Optional[NegotiationSession] = None
        self._last_round_count: int = 0
        logger.info("NegotiationHub initialized")

    def get_last_round_count(self) -> int:
        """Return the number of negotiation rounds used in the last negotiate() call."""
        return self._last_round_count

    def _get_or_create_agent(self, robot_id: str) -> RobotLLMAgent:
        if robot_id not in self._agents:
            self._agents[robot_id] = RobotLLMAgent(robot_id)
            logger.info(f"Created LLM agent for {robot_id}")
        return self._agents[robot_id]

    def needs_negotiation(self, command_text: str, robot_id: str = "Robot1") -> bool:
        # Re-read config dynamically to support runtime changes
        import config.Negotiation as neg_cfg

        if not neg_cfg.NEGOTIATION_ENABLED:
            return False

        text_lower = command_text.lower()

        # Check collaboration keywords
        for keyword in neg_cfg.COLLABORATION_KEYWORDS:
            if keyword in text_lower:
                logger.info(f"Negotiation triggered by keyword: '{keyword}'")
                return True

        # Check for explicit multi-robot references (word-boundary match prevents
        # "Robot1" from matching inside "Robot10", "Robot1x", etc.)
        robot_refs = sum(
            1
            for rid in ROBOT_BASE_POSITIONS
            if re.search(r"\b" + re.escape(rid.lower()) + r"\b", text_lower)
        )
        if robot_refs >= 2:
            logger.info(f"Negotiation triggered: {robot_refs} robot references found")
            return True

        return False

    def negotiate(
        self,
        task_description: str,
        robot_ids: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        spatial_context: Optional[Dict[str, Any]] = None,
    ) -> "NegotiationResult":
        """
        Run the full negotiation protocol.

        spatial_context: Optional KG-derived dict (e.g. handoff candidates) logged for
            future agent prompt injection.
        """
        timeout = timeout or NEGOTIATION_TIMEOUT
        robot_ids = robot_ids or list(ROBOT_BASE_POSITIONS.keys())

        # Create session
        session_id = f"neg_{int(time.time() * 1000)}"
        session = NegotiationSession(
            session_id=session_id,
            task=task_description,
            robot_ids=robot_ids,
        )
        self._active_session = session

        logger.info(
            f"Starting negotiation {session_id}: task='{task_description[:80]}', "
            f"robots={robot_ids}"
        )
        if spatial_context:
            logger.debug(f"Negotiation spatial context: {spatial_context}")

        start_time = time.time()
        result = NegotiationResult()
        self._last_round_count = 0

        try:
            # Get world state snapshot
            world_state = self._get_world_state_snapshot()

            # Phase 1: Analysis
            session.state = NegotiationState.ANALYZING
            analysis_ok = self._run_analysis_phase(session, world_state)
            if not analysis_ok:
                result.state = NegotiationState.FAILED
                result.reasoning = "Analysis phase failed"
                return result

            # Negotiation rounds
            for round_num in range(1, MAX_NEGOTIATION_ROUNDS + 1):
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    logger.warning(f"Negotiation timed out after {elapsed:.1f}s")
                    result.state = NegotiationState.TIMEOUT
                    result.rounds_taken = round_num
                    result.duration_s = elapsed
                    return result

                session.current_round = round_num

                # Refresh world state and re-run analysis each round so agents
                # incorporate any state changes and previous round outcomes.
                if round_num > 1:
                    world_state = self._get_world_state_snapshot()
                    analysis_ok = self._run_analysis_phase(session, world_state)
                    if not analysis_ok:
                        result.state = NegotiationState.FAILED
                        result.reasoning = "Analysis phase failed"
                        return result

                # Phase 2: Proposal
                session.state = NegotiationState.PROPOSING
                proposal = self._run_proposal_phase(session, world_state)
                if proposal is None or not proposal.commands:
                    logger.warning(f"Round {round_num}: no valid proposal generated")
                    continue

                session.proposals.append(proposal)

                # Phase 3: Evaluation
                session.state = NegotiationState.EVALUATING
                accepted = self._run_evaluation_phase(session, proposal, world_state)

                if accepted:
                    # BUG 2 FIX: Validate raw LLM output BEFORE normalization.
                    # _normalize_commands() silently drops commands missing "operation",
                    # so validating after normalization misses those malformed entries.
                    if VERIFY_NEGOTIATED_PLANS:
                        valid, validation_errors = self._validate_plan(
                            proposal.commands
                        )
                        if not valid:
                            logger.warning(
                                f"Round {round_num}: plan validation failed: {validation_errors}"
                            )
                            continue

                    # Only normalize after validation passes
                    commands = self._normalize_commands(proposal.commands, robot_ids)

                    session.state = NegotiationState.CONSENSUS
                    result.success = True
                    result.commands = commands
                    result.reasoning = proposal.reasoning
                    result.rounds_taken = round_num
                    result.duration_s = time.time() - start_time
                    result.state = NegotiationState.CONSENSUS

                    logger.info(
                        f"Negotiation {session_id} reached consensus in round {round_num} "
                        f"({result.duration_s:.1f}s, {len(commands)} commands)"
                    )
                    return result

                logger.info(f"Round {round_num}: no consensus, continuing")

            # No consensus after all rounds
            result.state = NegotiationState.FAILED
            result.rounds_taken = MAX_NEGOTIATION_ROUNDS
            result.duration_s = time.time() - start_time
            result.reasoning = "No consensus reached after max rounds"
            logger.warning(
                f"Negotiation {session_id} failed after {MAX_NEGOTIATION_ROUNDS} rounds"
            )
            return result

        except Exception as e:
            logger.error(f"Negotiation error: {e}", exc_info=True)
            result.state = NegotiationState.FAILED
            result.reasoning = f"Error: {str(e)}"
            result.duration_s = time.time() - start_time
            return result
        finally:
            self._last_round_count = result.rounds_taken
            self._active_session = None

    def _run_analysis_phase(
        self, session: NegotiationSession, world_state: Dict[str, Any]
    ) -> bool:
        """Run analysis phase: each robot analyzes the task in parallel."""
        try:
            from core.Imports import get_global_registry

            registry = get_global_registry()
            available_ops = [op.name for op in registry.get_all_operations()]
        except Exception:
            available_ops = []
            logger.warning("Cannot get operation names for analysis")

        with ThreadPoolExecutor(max_workers=len(session.robot_ids)) as executor:
            futures = {}
            for robot_id in session.robot_ids:
                agent = self._get_or_create_agent(robot_id)
                future = executor.submit(
                    agent.analyze_task, session.task, world_state, available_ops
                )
                futures[future] = robot_id

            for future in as_completed(futures):
                robot_id = futures[future]
                try:
                    analysis = future.result()
                    session.analyses[robot_id] = analysis
                    logger.info(
                        f"[{robot_id}] Analysis: can_contribute={analysis.can_contribute}, "
                        f"role='{analysis.suggested_role}'"
                    )
                except Exception as e:
                    logger.error(f"[{robot_id}] Analysis failed: {e}")
                    session.analyses[robot_id] = TaskAnalysis(
                        robot_id=robot_id, can_contribute=False
                    )

        # Check if any robot can contribute.
        # If all robots returned can_contribute=False, promote them all: for
        # inherently joint tasks (handoff, stabilize) the model often interprets
        # "can_contribute" as "can complete solo" and returns False even though
        # both robots are needed.  Negotiation was triggered precisely because
        # multiple robots are involved, so a unanimous False is a model artefact,
        # not a genuine capability gap.
        contributors = [a for a in session.analyses.values() if a.can_contribute]
        if not contributors:
            all_analyses = list(session.analyses.values())
            logger.warning(
                f"All {len(all_analyses)} robots returned can_contribute=False — "
                f"promoting all to contributors (likely a solo-vs-collaborative "
                f"misinterpretation by the model)"
            )
            for a in all_analyses:
                a.can_contribute = True
            contributors = all_analyses

        return True

    def _run_proposal_phase(
        self, session: NegotiationSession, world_state: Dict[str, Any]
    ) -> Optional[PlanProposal]:
        """Proposal phase: one robot proposes a plan. Proposer chosen round-robin."""
        contributors = [rid for rid, a in session.analyses.items() if a.can_contribute]
        if not contributors:
            return None

        proposer_idx = (session.current_round - 1) % len(contributors)
        proposer_id = contributors[proposer_idx]

        other_analyses = [
            a for rid, a in session.analyses.items() if rid != proposer_id
        ]

        # BUG 4 FIX: Fetch available operations from the registry so the LLM
        # proposal prompt lists valid operation names. Without this, the LLM
        # hallucinates names (e.g. "move_arm") that fail NegotiationVerifier checks.
        try:
            from core.Imports import get_global_registry

            registry = get_global_registry()
            available_ops = [op.name for op in registry.get_all_operations()]
        except Exception:
            available_ops = []
            logger.warning("Cannot get operation names for proposal phase")

        agent = self._get_or_create_agent(proposer_id)
        proposal = agent.propose_plan(
            session.task,
            other_analyses,
            world_state,
            session.current_round,
            available_operations=available_ops,
        )

        logger.info(
            f"[{proposer_id}] Proposed plan: {len(proposal.commands)} commands, "
            f"reasoning='{proposal.reasoning[:100]}'"
        )

        # Sanity-check: reject single-robot plans for multi-robot sessions
        # before wasting an evaluation round-trip.
        if len(session.robot_ids) > 1 and proposal.commands:
            robots_in_plan = {
                cmd.get("params", {}).get("robot_id") or cmd.get("robot")
                for cmd in proposal.commands
            }
            robots_in_plan.discard(None)
            missing = set(session.robot_ids) - robots_in_plan
            if missing:
                logger.warning(
                    f"[{proposer_id}] Proposal missing commands for {missing} — "
                    f"discarding (single-robot plan for multi-robot task)"
                )
                return None

        return proposal

    def _run_evaluation_phase(
        self,
        session: NegotiationSession,
        proposal: PlanProposal,
        world_state: Dict[str, Any],
    ) -> bool:
        """
        Evaluation phase: other robots evaluate the proposal. Consensus = all accept.

        Only robots with can_contribute=True evaluate; querying others wastes LLM tokens.
        """
        evaluator_ids = [
            rid
            for rid, analysis in session.analyses.items()
            if rid != proposal.proposer_id and analysis.can_contribute
        ]

        if not evaluator_ids:
            return True  # Only one robot, auto-accept

        all_accept = True
        with ThreadPoolExecutor(max_workers=len(evaluator_ids)) as executor:
            futures = {}
            for robot_id in evaluator_ids:
                agent = self._get_or_create_agent(robot_id)
                future = executor.submit(
                    agent.evaluate_proposal, proposal, session.task, world_state
                )
                futures[future] = robot_id

            for future in as_completed(futures):
                robot_id = futures[future]
                try:
                    evaluation = future.result()
                    logger.info(
                        f"[{robot_id}] Evaluation: accept={evaluation.accept}, "
                        f"concerns={evaluation.concerns}"
                    )
                    if not evaluation.accept:
                        all_accept = False
                except Exception as e:
                    logger.error(f"[{robot_id}] Evaluation failed: {e}")
                    all_accept = False

        return all_accept

    def _get_world_state_snapshot(self) -> Dict[str, Any]:
        """Serialize current world state for LLM context."""
        snapshot = {"robots": {}, "objects": {}}

        try:
            from core.Imports import get_world_state

            ws = get_world_state()

            for robot_id in ROBOT_BASE_POSITIONS:
                state = ws.get_robot_state(robot_id)
                if state:
                    snapshot["robots"][robot_id] = {
                        "position": state.position,
                        "gripper_state": state.gripper_state,
                        "is_moving": state.is_moving,
                        "is_initialized": state.is_initialized,
                    }

            for obj in ws.get_all_objects():
                snapshot["objects"][obj.object_id] = {
                    "position": obj.position,
                    "color": obj.color,
                    "type": obj.object_type,
                    "grasped_by": obj.grasped_by,
                }

        except Exception as e:
            logger.warning(f"Cannot get world state snapshot: {e}")

        return snapshot

    def _normalize_commands(
        self, commands: List[Dict[str, Any]], robot_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Normalize command format for SequenceExecutor. Ensures operation, params.robot_id, capture_var, parallel_group."""
        normalized = []
        default_robot = robot_ids[0] if robot_ids else "Robot1"

        for cmd in commands:
            operation = cmd.get("operation", "")
            if not operation:
                continue

            params = cmd.get("params", {})

            if "robot_id" not in params:
                params["robot_id"] = cmd.get("robot", default_robot)

            entry = {"operation": operation, "params": params}

            if "capture_var" in cmd:
                entry["capture_var"] = cmd["capture_var"]
            if "parallel_group" in cmd:
                entry["parallel_group"] = cmd["parallel_group"]

            normalized.append(entry)

        return normalized

    def _validate_plan(self, commands: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Structural validation of the negotiated plan. Returns (valid, errors)."""
        errors = []

        if len(commands) > MAX_PLAN_LENGTH:
            errors.append(
                f"Plan too long: {len(commands)} commands (max {MAX_PLAN_LENGTH})"
            )

        if not commands:
            errors.append("Empty plan")
            return False, errors

        # Use NegotiationVerifier for detailed checks
        try:
            from operations.NegotiationVerifier import NegotiationVerifier

            verifier = NegotiationVerifier()
            result = verifier.verify_plan(commands)
            errors.extend(result.errors)
            return result.valid, errors
        except Exception as e:
            logger.warning(f"Plan verification error: {e}")
            # Fall back to basic checks
            return len(errors) == 0, errors
