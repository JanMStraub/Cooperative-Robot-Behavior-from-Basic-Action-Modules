#!/usr/bin/env python3
"""
Tests for the Multi-Robot Negotiation System.

Tests the NegotiationHub, RobotLLMAgent, NegotiationVerifier,
and config with mocked LLM calls.
"""

import pytest
import sys
import os
import json
import time
from unittest.mock import patch, MagicMock
from dataclasses import asdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Config Tests
# ============================================================================


class TestNegotiationConfig:
    """Tests for config/Negotiation.py defaults and env overrides."""

    def test_default_values(self):
        """Test default config values are set correctly."""
        from config.Negotiation import (
            NEGOTIATION_ENABLED,
            MAX_NEGOTIATION_ROUNDS,
            AGENT_LLM_TIMEOUT,
            NEGOTIATION_TIMEOUT,
            NEGOTIATION_TEMPERATURE,
            COLLABORATION_KEYWORDS,
            VERIFY_NEGOTIATED_PLANS,
            MAX_PLAN_LENGTH,
        )

        assert NEGOTIATION_ENABLED is True
        assert MAX_NEGOTIATION_ROUNDS == 3
        assert AGENT_LLM_TIMEOUT == 30.0
        assert NEGOTIATION_TIMEOUT == 120.0
        assert NEGOTIATION_TEMPERATURE == 0.3
        assert isinstance(COLLABORATION_KEYWORDS, list)
        assert "both" in COLLABORATION_KEYWORDS
        assert "together" in COLLABORATION_KEYWORDS
        assert VERIFY_NEGOTIATED_PLANS is True
        assert MAX_PLAN_LENGTH == 50

    def test_collaboration_keywords_coverage(self):
        """Test that collaboration keywords cover common phrases."""
        from config.Negotiation import COLLABORATION_KEYWORDS

        expected_keywords = [
            "both", "together", "cooperate", "collaborate",
            "coordinate", "simultaneously", "handoff",
        ]
        for kw in expected_keywords:
            assert kw in COLLABORATION_KEYWORDS, f"Missing keyword: {kw}"

    @patch.dict(os.environ, {"NEGOTIATION_ENABLED": "false"})
    def test_env_override_disabled(self):
        """Test that NEGOTIATION_ENABLED can be disabled via env var."""
        # Need to reimport to pick up env var
        import importlib
        import config.Negotiation as neg_config
        importlib.reload(neg_config)

        assert neg_config.NEGOTIATION_ENABLED is False

        # Restore
        os.environ.pop("NEGOTIATION_ENABLED", None)
        importlib.reload(neg_config)

    @patch.dict(os.environ, {"MAX_NEGOTIATION_ROUNDS": "5"})
    def test_env_override_rounds(self):
        """Test that MAX_NEGOTIATION_ROUNDS can be overridden via env var."""
        import importlib
        import config.Negotiation as neg_config
        importlib.reload(neg_config)

        assert neg_config.MAX_NEGOTIATION_ROUNDS == 5

        os.environ.pop("MAX_NEGOTIATION_ROUNDS", None)
        importlib.reload(neg_config)


# ============================================================================
# Robot LLM Agent Tests
# ============================================================================


def _mock_llm_response(content):
    """Create a mock requests.post response with given content."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    return mock_resp


class TestRobotLLMAgent:
    """Tests for agents/RobotLLMAgent.py."""

    def test_agent_creation(self):
        """Test agent initialization with robot config."""
        from agents.RobotLLMAgent import RobotLLMAgent

        agent = RobotLLMAgent("Robot1")
        assert agent.robot_id == "Robot1"
        assert agent.base_position == (-0.475, 0.0, 0.0)
        assert agent.workspace == "left_workspace"

    def test_agent_creation_robot2(self):
        """Test agent initialization for Robot2."""
        from agents.RobotLLMAgent import RobotLLMAgent

        agent = RobotLLMAgent("Robot2")
        assert agent.robot_id == "Robot2"
        assert agent.base_position == (0.475, 0.0, 0.0)
        assert agent.workspace == "right_workspace"

    def test_build_agent_context(self):
        """Test context string includes robot identity and world state."""
        from agents.RobotLLMAgent import RobotLLMAgent

        agent = RobotLLMAgent("Robot1")
        world_state = {
            "robots": {
                "Robot1": {
                    "position": (-0.3, 0.2, 0.0),
                    "gripper_state": "open",
                    "is_moving": False,
                }
            },
            "objects": {
                "red_cube": {
                    "position": (0.0, 0.1, 0.0),
                    "color": "red",
                }
            },
        }

        context = agent._build_agent_context(world_state)
        assert "Robot1" in context
        assert "left_workspace" in context
        assert "red_cube" in context
        assert "red" in context

    @patch("agents.RobotLLMAgent.requests.post")
    def test_analyze_task(self, mock_post):
        """Test task analysis with mocked LLM."""
        from agents.RobotLLMAgent import RobotLLMAgent

        mock_post.return_value = _mock_llm_response(json.dumps({
            "can_contribute": True,
            "capabilities": ["move to object", "grasp object"],
            "constraints": ["limited to left workspace"],
            "suggested_role": "primary manipulator",
            "requires_collaboration": True,
            "confidence": 0.8,
        }))

        agent = RobotLLMAgent("Robot1")
        analysis = agent.analyze_task(
            "Both robots lift the heavy cube together",
            {"robots": {}, "objects": {}},
            ["move_to_coordinate", "control_gripper"],
        )

        assert analysis.robot_id == "Robot1"
        assert analysis.can_contribute is True
        assert analysis.requires_collaboration is True
        assert analysis.confidence == 0.8
        assert "move to object" in analysis.capabilities
        mock_post.assert_called_once()

    @patch("agents.RobotLLMAgent.requests.post")
    def test_propose_plan(self, mock_post):
        """Test plan proposal with mocked LLM."""
        from agents.RobotLLMAgent import RobotLLMAgent, TaskAnalysis

        mock_post.return_value = _mock_llm_response(json.dumps({
            "reasoning": "Robot1 detects, Robot2 waits then both move",
            "commands": [
                {
                    "parallel_group": 1,
                    "operation": "detect_object_stereo",
                    "params": {"robot_id": "Robot1", "color": "red"},
                    "capture_var": "target",
                },
                {
                    "parallel_group": 2,
                    "operation": "move_to_coordinate",
                    "params": {"robot_id": "Robot1", "position": "$target"},
                },
                {
                    "parallel_group": 2,
                    "operation": "move_to_coordinate",
                    "params": {"robot_id": "Robot2", "x": 0.0, "y": 0.3, "z": 0.0},
                },
            ],
            "estimated_duration_s": 15.0,
        }))

        agent = RobotLLMAgent("Robot1")
        other_analysis = TaskAnalysis(
            robot_id="Robot2",
            can_contribute=True,
            suggested_role="support",
        )

        proposal = agent.propose_plan(
            "Both robots approach the red cube",
            [other_analysis],
            {"robots": {}, "objects": {}},
            round_number=1,
        )

        assert proposal.proposer_id == "Robot1"
        assert len(proposal.commands) == 3
        assert proposal.round_number == 1
        assert "detect" in proposal.reasoning.lower() or "robot" in proposal.reasoning.lower()

    @patch("agents.RobotLLMAgent.requests.post")
    def test_evaluate_proposal_accept(self, mock_post):
        """Test proposal evaluation (accept)."""
        from agents.RobotLLMAgent import RobotLLMAgent, PlanProposal

        mock_post.return_value = _mock_llm_response(json.dumps({
            "accept": True,
            "concerns": [],
            "suggested_changes": [],
            "confidence": 0.9,
        }))

        agent = RobotLLMAgent("Robot2")
        proposal = PlanProposal(
            proposer_id="Robot1",
            reasoning="test plan",
            commands=[{"operation": "move_to_coordinate", "params": {"robot_id": "Robot2", "x": 0, "y": 0.3, "z": 0}}],
        )

        evaluation = agent.evaluate_proposal(proposal, "test task", {"robots": {}, "objects": {}})
        assert evaluation.evaluator_id == "Robot2"
        assert evaluation.accept is True
        assert evaluation.confidence == 0.9

    @patch("agents.RobotLLMAgent.requests.post")
    def test_evaluate_proposal_reject(self, mock_post):
        """Test proposal evaluation (reject with concerns)."""
        from agents.RobotLLMAgent import RobotLLMAgent, PlanProposal

        mock_post.return_value = _mock_llm_response(json.dumps({
            "accept": False,
            "concerns": ["target is out of my workspace"],
            "suggested_changes": ["move target to shared zone"],
            "confidence": 0.7,
        }))

        agent = RobotLLMAgent("Robot2")
        proposal = PlanProposal(
            proposer_id="Robot1",
            reasoning="test plan",
            commands=[],
        )

        evaluation = agent.evaluate_proposal(proposal, "test task", {"robots": {}, "objects": {}})
        assert evaluation.accept is False
        assert len(evaluation.concerns) == 1
        assert "workspace" in evaluation.concerns[0]

    @patch("agents.RobotLLMAgent.requests.post")
    def test_llm_timeout_fallback(self, mock_post):
        """Test graceful handling of LLM timeout."""
        import requests as req
        mock_post.side_effect = req.exceptions.Timeout()

        from agents.RobotLLMAgent import RobotLLMAgent

        agent = RobotLLMAgent("Robot1")
        analysis = agent.analyze_task("test task", {"robots": {}, "objects": {}}, [])

        # Should return default analysis
        assert analysis.robot_id == "Robot1"
        assert analysis.confidence == 0.5

    @patch("agents.RobotLLMAgent.requests.post")
    def test_llm_connection_error_fallback(self, mock_post):
        """Test graceful handling of LLM connection error."""
        import requests as req
        mock_post.side_effect = req.exceptions.ConnectionError()

        from agents.RobotLLMAgent import RobotLLMAgent

        agent = RobotLLMAgent("Robot1")
        analysis = agent.analyze_task("test task", {"robots": {}, "objects": {}}, [])
        assert analysis.robot_id == "Robot1"

    @patch("agents.RobotLLMAgent.requests.post")
    def test_json_extraction_from_markdown(self, mock_post):
        """Test JSON extraction from markdown code block."""
        from agents.RobotLLMAgent import RobotLLMAgent

        mock_post.return_value = _mock_llm_response(
            "Here is the analysis:\n```json\n"
            '{"can_contribute": true, "capabilities": ["test"], '
            '"constraints": [], "suggested_role": "helper", '
            '"requires_collaboration": false, "confidence": 0.6}\n```'
        )

        agent = RobotLLMAgent("Robot1")
        analysis = agent.analyze_task("test", {"robots": {}, "objects": {}}, [])
        assert analysis.can_contribute is True
        assert analysis.confidence == 0.6


# ============================================================================
# Negotiation Hub Tests
# ============================================================================


class TestNegotiationHub:
    """Tests for servers/NegotiationHub.py."""

    def setup_method(self):
        """Reset singleton between tests."""
        from servers.NegotiationHub import NegotiationHub
        NegotiationHub._instance = None

    def test_singleton(self):
        """Test NegotiationHub is a singleton."""
        from servers.NegotiationHub import NegotiationHub

        hub1 = NegotiationHub()
        hub2 = NegotiationHub()
        assert hub1 is hub2

    def test_needs_negotiation_keyword(self):
        """Test collaboration keyword detection."""
        from servers.NegotiationHub import NegotiationHub

        hub = NegotiationHub()
        assert hub.needs_negotiation("Both robots lift the cube together") is True
        assert hub.needs_negotiation("Cooperate to move the object") is True
        assert hub.needs_negotiation("Simultaneously grasp the beam") is True

    def test_needs_negotiation_multi_robot_ref(self):
        """Test multi-robot reference detection."""
        from servers.NegotiationHub import NegotiationHub

        hub = NegotiationHub()
        assert hub.needs_negotiation("Robot1 detects and Robot2 grasps") is True

    def test_needs_negotiation_single_robot(self):
        """Test single-robot commands don't trigger negotiation."""
        from servers.NegotiationHub import NegotiationHub

        hub = NegotiationHub()
        assert hub.needs_negotiation("Move to (0.3, 0.2, 0.1)") is False
        assert hub.needs_negotiation("Robot1 close the gripper") is False
        assert hub.needs_negotiation("Detect the red cube") is False

    @patch.dict(os.environ, {"NEGOTIATION_ENABLED": "false"})
    def test_needs_negotiation_disabled(self):
        """Test negotiation disabled via config."""
        import importlib
        import config.Negotiation as neg_config
        importlib.reload(neg_config)

        from servers.NegotiationHub import NegotiationHub
        NegotiationHub._instance = None
        hub = NegotiationHub()

        # Even with keywords, should return False
        assert hub.needs_negotiation("Both robots cooperate") is False

        os.environ.pop("NEGOTIATION_ENABLED", None)
        importlib.reload(neg_config)

    @patch("agents.RobotLLMAgent.requests.post")
    def test_negotiate_success(self, mock_post):
        """Test successful negotiation with mocked LLM."""
        from servers.NegotiationHub import NegotiationHub, NegotiationState

        # Set up LLM responses for analysis, proposal, and evaluation
        responses = [
            # Robot1 analysis
            _mock_llm_response(json.dumps({
                "can_contribute": True,
                "capabilities": ["grasp"],
                "constraints": [],
                "suggested_role": "left gripper",
                "requires_collaboration": True,
                "confidence": 0.8,
            })),
            # Robot2 analysis
            _mock_llm_response(json.dumps({
                "can_contribute": True,
                "capabilities": ["grasp"],
                "constraints": [],
                "suggested_role": "right gripper",
                "requires_collaboration": True,
                "confidence": 0.8,
            })),
            # Robot1 proposal
            _mock_llm_response(json.dumps({
                "reasoning": "Both robots move to cube and grasp",
                "commands": [
                    {
                        "parallel_group": 1,
                        "operation": "move_to_coordinate",
                        "params": {"robot_id": "Robot1", "x": -0.1, "y": 0.2, "z": 0.0},
                    },
                    {
                        "parallel_group": 1,
                        "operation": "move_to_coordinate",
                        "params": {"robot_id": "Robot2", "x": 0.1, "y": 0.2, "z": 0.0},
                    },
                    {
                        "parallel_group": 2,
                        "operation": "control_gripper",
                        "params": {"robot_id": "Robot1", "open_gripper": False},
                    },
                    {
                        "parallel_group": 2,
                        "operation": "control_gripper",
                        "params": {"robot_id": "Robot2", "open_gripper": False},
                    },
                ],
                "estimated_duration_s": 10.0,
            })),
            # Robot2 evaluation (accept)
            _mock_llm_response(json.dumps({
                "accept": True,
                "concerns": [],
                "suggested_changes": [],
                "confidence": 0.9,
            })),
        ]
        mock_post.side_effect = responses

        NegotiationHub._instance = None
        hub = NegotiationHub()
        result = hub.negotiate("Both robots grasp the cube together")

        assert result.success is True
        assert result.state == NegotiationState.CONSENSUS
        assert len(result.commands) == 4
        assert result.rounds_taken == 1

    @patch("agents.RobotLLMAgent.requests.post")
    def test_negotiate_no_consensus(self, mock_post):
        """Test negotiation failure when robots reject proposals."""
        from servers.NegotiationHub import NegotiationHub, NegotiationState

        # Analysis responses (both can contribute)
        analysis_response = _mock_llm_response(json.dumps({
            "can_contribute": True,
            "capabilities": ["move"],
            "constraints": [],
            "suggested_role": "helper",
            "requires_collaboration": True,
            "confidence": 0.7,
        }))

        # Proposal with empty commands (triggers continue)
        proposal_response = _mock_llm_response(json.dumps({
            "reasoning": "plan",
            "commands": [],
            "estimated_duration_s": 5.0,
        }))

        # Cycle through: 2 analyses + (proposal per round * 3 rounds)
        responses = [analysis_response, analysis_response]
        for _ in range(3):
            responses.append(proposal_response)

        mock_post.side_effect = responses

        NegotiationHub._instance = None
        hub = NegotiationHub()
        result = hub.negotiate("Both robots lift", timeout=10.0)

        assert result.success is False
        assert result.state == NegotiationState.FAILED

    @patch("agents.RobotLLMAgent.requests.post")
    def test_negotiate_timeout(self, mock_post):
        """Test negotiation timeout."""
        from servers.NegotiationHub import NegotiationHub, NegotiationState

        # Make LLM calls very slow
        def slow_response(*args, **kwargs):
            time.sleep(0.5)
            return _mock_llm_response(json.dumps({
                "can_contribute": True,
                "capabilities": [],
                "constraints": [],
                "suggested_role": "slow",
                "requires_collaboration": True,
                "confidence": 0.5,
            }))

        mock_post.side_effect = slow_response

        NegotiationHub._instance = None
        hub = NegotiationHub()
        result = hub.negotiate("Both robots cooperate", timeout=0.1)

        # Should timeout since analyses take >0.1s total
        assert result.success is False
        assert result.state in (NegotiationState.TIMEOUT, NegotiationState.FAILED)


# ============================================================================
# Negotiation Verifier Tests
# ============================================================================


class TestNegotiationVerifier:
    """Tests for operations/NegotiationVerifier.py."""

    def test_empty_plan(self):
        """Test empty plan is invalid."""
        from operations.NegotiationVerifier import NegotiationVerifier

        verifier = NegotiationVerifier()
        result = verifier.verify_plan([])
        assert result.valid is False
        assert any("Empty" in e for e in result.errors)

    def test_valid_plan(self):
        """Test a valid simple plan passes verification."""
        from operations.NegotiationVerifier import NegotiationVerifier

        commands = [
            {
                "operation": "move_to_coordinate",
                "params": {"robot_id": "Robot1", "x": -0.3, "y": 0.2, "z": 0.0},
                "parallel_group": 1,
            },
            {
                "operation": "control_gripper",
                "params": {"robot_id": "Robot1", "open_gripper": False},
                "parallel_group": 2,
            },
        ]

        verifier = NegotiationVerifier()
        result = verifier.verify_plan(commands)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_unmatched_wait_for_signal(self):
        """Test unmatched wait_for_signal is detected."""
        from operations.NegotiationVerifier import NegotiationVerifier

        commands = [
            {
                "operation": "wait_for_signal",
                "params": {"robot_id": "Robot2", "event_name": "cube_gripped"},
                "parallel_group": 1,
            },
        ]

        verifier = NegotiationVerifier()
        result = verifier.verify_plan(commands)
        assert result.valid is False
        assert any("wait_for_signal" in e and "cube_gripped" in e for e in result.errors)

    def test_matched_signal_wait_pair(self):
        """Test matched signal/wait pair passes."""
        from operations.NegotiationVerifier import NegotiationVerifier

        commands = [
            {
                "operation": "signal",
                "params": {"robot_id": "Robot1", "event_name": "ready"},
                "parallel_group": 1,
            },
            {
                "operation": "wait_for_signal",
                "params": {"robot_id": "Robot2", "event_name": "ready"},
                "parallel_group": 1,
            },
        ]

        verifier = NegotiationVerifier()
        result = verifier.verify_plan(commands)
        # Should not have signal-related errors
        signal_errors = [e for e in result.errors if "signal" in e.lower()]
        assert len(signal_errors) == 0

    def test_variable_used_before_definition(self):
        """Test variable usage before definition is detected."""
        from operations.NegotiationVerifier import NegotiationVerifier

        commands = [
            {
                "operation": "move_to_coordinate",
                "params": {"robot_id": "Robot1", "position": "$target"},
                "parallel_group": 1,
            },
            {
                "operation": "detect_object_stereo",
                "params": {"robot_id": "Robot1", "color": "red"},
                "capture_var": "target",
                "parallel_group": 2,
            },
        ]

        verifier = NegotiationVerifier()
        result = verifier.verify_plan(commands)
        assert result.valid is False
        assert any("$target" in e and "before definition" in e for e in result.errors)

    def test_variable_defined_then_used(self):
        """Test variable defined before usage passes."""
        from operations.NegotiationVerifier import NegotiationVerifier

        commands = [
            {
                "operation": "detect_object_stereo",
                "params": {"robot_id": "Robot1", "color": "red"},
                "capture_var": "target",
                "parallel_group": 1,
            },
            {
                "operation": "move_to_coordinate",
                "params": {"robot_id": "Robot1", "position": "$target"},
                "parallel_group": 2,
            },
        ]

        verifier = NegotiationVerifier()
        result = verifier.verify_plan(commands)
        # Should not have variable-related errors
        var_errors = [e for e in result.errors if "$target" in e]
        assert len(var_errors) == 0

    def test_parallel_group_collision_check(self):
        """Test spatial safety detects close concurrent targets."""
        from operations.NegotiationVerifier import NegotiationVerifier

        commands = [
            {
                "operation": "move_to_coordinate",
                "params": {"robot_id": "Robot1", "x": 0.0, "y": 0.2, "z": 0.0},
                "parallel_group": 1,
            },
            {
                "operation": "move_to_coordinate",
                "params": {"robot_id": "Robot2", "x": 0.05, "y": 0.2, "z": 0.0},
                "parallel_group": 1,
            },
        ]

        verifier = NegotiationVerifier()
        result = verifier.verify_plan(commands)
        # Should detect close targets (0.05m < MIN_ROBOT_SEPARATION of 0.2m)
        assert result.valid is False
        assert result.safety_check is False

    def test_safe_parallel_targets(self):
        """Test safe concurrent targets pass spatial check."""
        from operations.NegotiationVerifier import NegotiationVerifier

        commands = [
            {
                "operation": "move_to_coordinate",
                "params": {"robot_id": "Robot1", "x": -0.3, "y": 0.2, "z": 0.0},
                "parallel_group": 1,
            },
            {
                "operation": "move_to_coordinate",
                "params": {"robot_id": "Robot2", "x": 0.3, "y": 0.2, "z": 0.0},
                "parallel_group": 1,
            },
        ]

        verifier = NegotiationVerifier()
        result = verifier.verify_plan(commands)
        # Targets are 0.6m apart, should be safe
        spatial_errors = [e for e in result.errors if "apart" in e]
        assert len(spatial_errors) == 0

    def test_invalid_parallel_group_type(self):
        """Test non-integer parallel_group is detected."""
        from operations.NegotiationVerifier import NegotiationVerifier

        commands = [
            {
                "operation": "move_to_coordinate",
                "params": {"robot_id": "Robot1", "x": 0, "y": 0.2, "z": 0},
                "parallel_group": "first",
            },
        ]

        verifier = NegotiationVerifier()
        result = verifier.verify_plan(commands)
        assert result.valid is False
        assert any("integer" in e for e in result.errors)


# ============================================================================
# Integration Tests
# ============================================================================


class TestSequenceExecutorNegotiation:
    """Tests for negotiation integration in SequenceExecutor."""

    def setup_method(self):
        """Reset singleton between tests."""
        from servers.NegotiationHub import NegotiationHub
        NegotiationHub._instance = None

    def test_negotiate_if_needed_returns_none_for_simple_command(self):
        """Test that simple commands bypass negotiation."""
        from orchestrators.SequenceExecutor import SequenceExecutor

        executor = SequenceExecutor(check_completion=False, enable_verification=False)
        result = executor.negotiate_if_needed("Move to (0.3, 0.2, 0.1)", "Robot1")
        assert result is None

    @patch("agents.RobotLLMAgent.requests.post")
    def test_negotiate_if_needed_triggers_for_collaboration(self, mock_post):
        """Test that collaboration commands trigger negotiation."""
        from orchestrators.SequenceExecutor import SequenceExecutor

        # Mock LLM responses for full negotiation
        responses = [
            # Robot1 analysis
            _mock_llm_response(json.dumps({
                "can_contribute": True, "capabilities": ["grasp"],
                "constraints": [], "suggested_role": "left",
                "requires_collaboration": True, "confidence": 0.8,
            })),
            # Robot2 analysis
            _mock_llm_response(json.dumps({
                "can_contribute": True, "capabilities": ["grasp"],
                "constraints": [], "suggested_role": "right",
                "requires_collaboration": True, "confidence": 0.8,
            })),
            # Proposal
            _mock_llm_response(json.dumps({
                "reasoning": "Both robots cooperate",
                "commands": [
                    {"parallel_group": 1, "operation": "move_to_coordinate",
                     "params": {"robot_id": "Robot1", "x": -0.1, "y": 0.2, "z": 0}},
                    {"parallel_group": 1, "operation": "move_to_coordinate",
                     "params": {"robot_id": "Robot2", "x": 0.1, "y": 0.2, "z": 0}},
                ],
                "estimated_duration_s": 10.0,
            })),
            # Evaluation (accept)
            _mock_llm_response(json.dumps({
                "accept": True, "concerns": [],
                "suggested_changes": [], "confidence": 0.9,
            })),
        ]
        mock_post.side_effect = responses

        # Reset hub singleton
        from servers.NegotiationHub import NegotiationHub
        NegotiationHub._instance = None

        executor = SequenceExecutor(check_completion=False, enable_verification=False)
        result = executor.negotiate_if_needed(
            "Both robots lift the cube together", "Robot1"
        )

        assert result is not None
        assert len(result) == 2
        assert result[0]["operation"] == "move_to_coordinate"

    def test_negotiate_if_needed_handles_import_error(self):
        """Test graceful handling when negotiation module unavailable."""
        from orchestrators.SequenceExecutor import SequenceExecutor

        executor = SequenceExecutor(check_completion=False, enable_verification=False)

        # Even if import fails internally, should return None gracefully
        with patch("core.Imports.get_negotiation_hub", side_effect=ImportError("test")):
            result = executor.negotiate_if_needed("Both robots cooperate", "Robot1")
            assert result is None


# ============================================================================
# Core Imports Tests
# ============================================================================


class TestCoreImportsNegotiation:
    """Tests for get_negotiation_hub in core/Imports.py."""

    def setup_method(self):
        """Reset singleton between tests."""
        from servers.NegotiationHub import NegotiationHub
        NegotiationHub._instance = None

    def test_get_negotiation_hub_returns_hub(self):
        """Test get_negotiation_hub returns a NegotiationHub instance."""
        from core.Imports import get_negotiation_hub
        from servers.NegotiationHub import NegotiationHub

        hub = get_negotiation_hub()
        assert hub is not None
        assert isinstance(hub, NegotiationHub)

    @patch.dict(os.environ, {"NEGOTIATION_ENABLED": "false"})
    def test_get_negotiation_hub_disabled(self):
        """Test get_negotiation_hub returns None when disabled."""
        import importlib
        import config.Negotiation as neg_config
        importlib.reload(neg_config)

        from core.Imports import get_negotiation_hub

        hub = get_negotiation_hub()
        assert hub is None

        os.environ.pop("NEGOTIATION_ENABLED", None)
        importlib.reload(neg_config)
