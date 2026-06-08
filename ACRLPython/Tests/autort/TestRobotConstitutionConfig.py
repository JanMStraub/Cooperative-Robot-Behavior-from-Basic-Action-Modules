import os
import importlib
import pytest
from unittest.mock import Mock, patch


def _make_config(rules=None):
    config = Mock()
    config.LM_STUDIO_URL = "http://localhost:1234/v1"
    config.SAFETY_VALIDATION_MODEL = "test-model"
    config.WORKSPACE_BOUNDS = {
        "min_corner": (-1.0, -1.0, 0.0),
        "max_corner": (1.0, 1.0, 1.5),
    }
    config.MAX_VELOCITY = 2.0
    config.MIN_ROBOT_SEPARATION = 0.2
    config.MAX_GRIPPER_FORCE = 50.0
    if rules is not None:
        config.SEMANTIC_SAFETY_RULES = rules
    else:
        # Simulate a config object that has no SEMANTIC_SAFETY_RULES attribute
        del config.SEMANTIC_SAFETY_RULES
    return config


@pytest.fixture
def constitution(request):
    rules = getattr(
        request,
        "param",
        [
            "Do not harm humans or animals",
            "Do not damage expensive or fragile equipment",
        ],
    )
    config = _make_config(rules)
    from autort.RobotConstitution import RobotConstitution

    with patch("autort.RobotConstitution.get_world_state", return_value=Mock()):
        with patch("autort.RobotConstitution.OpenAI"):
            return RobotConstitution(config)


class TestRuleLoading:
    def test_rules_loaded_from_config(self):
        from autort.RobotConstitution import RobotConstitution

        rules = ["Rule A", "Rule B", "Rule C"]
        config = _make_config(rules)
        with patch("autort.RobotConstitution.get_world_state", return_value=Mock()):
            with patch("autort.RobotConstitution.OpenAI"):
                rc = RobotConstitution(config)
        assert rc.semantic_rules == rules

    def test_rules_are_copy_not_reference(self):
        from autort.RobotConstitution import RobotConstitution

        rules = ["Rule A"]
        config = _make_config(rules)
        with patch("autort.RobotConstitution.get_world_state", return_value=Mock()):
            with patch("autort.RobotConstitution.OpenAI"):
                rc = RobotConstitution(config)
        config.SEMANTIC_SAFETY_RULES.append("Injected after init")
        assert "Injected after init" not in rc.semantic_rules

    def test_fallback_to_defaults_when_no_attribute(self):
        from autort.RobotConstitution import RobotConstitution

        config = _make_config(rules=None)  # attribute deleted
        with patch("autort.RobotConstitution.get_world_state", return_value=Mock()):
            with patch("autort.RobotConstitution.OpenAI"):
                rc = RobotConstitution(config)
        assert len(rc.semantic_rules) == 5
        assert "Do not harm humans or animals" in rc.semantic_rules


class TestAddRule:
    def test_add_rule_appends(self, constitution):
        initial_len = len(constitution.semantic_rules)
        constitution.add_rule("Do not touch the green zone")
        assert len(constitution.semantic_rules) == initial_len + 1
        assert "Do not touch the green zone" in constitution.semantic_rules

    def test_add_rule_deduplication(self, constitution):
        constitution.add_rule("Do not harm humans or animals")
        count = constitution.semantic_rules.count("Do not harm humans or animals")
        assert count == 1

    def test_add_empty_rule_ignored(self, constitution):
        initial = list(constitution.semantic_rules)
        constitution.add_rule("")
        assert constitution.semantic_rules == initial

    def test_add_rule_used_in_semantic_prompt(self, constitution):
        constitution.add_rule("Do not enter zone B")
        prompt_text = "\n".join(f"- {r}" for r in constitution.semantic_rules)
        assert "Do not enter zone B" in prompt_text


class TestExtraRulesEnvVar:
    def test_extra_rules_appended(self, monkeypatch):
        monkeypatch.setenv(
            "AUTORT_EXTRA_SAFETY_RULES",
            "Do not touch the green zone; Do not exceed table height",
        )
        import config.AutoRT as autort_cfg

        importlib.reload(autort_cfg)
        assert "Do not touch the green zone" in autort_cfg.SEMANTIC_SAFETY_RULES
        assert "Do not exceed table height" in autort_cfg.SEMANTIC_SAFETY_RULES
        # Base rules still present
        assert "Do not harm humans or animals" in autort_cfg.SEMANTIC_SAFETY_RULES

    def test_no_extra_rules_by_default(self, monkeypatch):
        monkeypatch.delenv("AUTORT_EXTRA_SAFETY_RULES", raising=False)
        import config.AutoRT as autort_cfg

        importlib.reload(autort_cfg)
        assert len(autort_cfg.SEMANTIC_SAFETY_RULES) == 5
