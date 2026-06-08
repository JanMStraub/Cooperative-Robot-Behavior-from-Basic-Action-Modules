import pytest

from rag.ConfidenceScorer import (
    get_confidence_level,
    calculate_parameter_match_score,
    calculate_metadata_match_score,
    calculate_reliability_score,
    compute_confidence_score,
    apply_confidence_boosting,
    get_category_min_score,
    ConfidenceLevel,
    WEIGHTS,
)


class TestConfidenceLevel:

    def test_high_confidence(self):
        assert get_confidence_level(0.9) == ConfidenceLevel.HIGH
        assert get_confidence_level(0.75) == ConfidenceLevel.HIGH

    def test_medium_confidence(self):
        assert get_confidence_level(0.6) == ConfidenceLevel.MEDIUM
        assert get_confidence_level(0.5) == ConfidenceLevel.MEDIUM

    def test_low_confidence(self):
        assert get_confidence_level(0.4) == ConfidenceLevel.LOW
        assert get_confidence_level(0.25) == ConfidenceLevel.LOW

    def test_uncertain_confidence(self):
        assert get_confidence_level(0.1) == ConfidenceLevel.UNCERTAIN
        assert get_confidence_level(0.0) == ConfidenceLevel.UNCERTAIN


class TestParameterMatchScore:

    def test_no_parameters(self):
        score = calculate_parameter_match_score("move robot", [])
        assert score == 0.5

    def test_parameter_match(self):
        score = calculate_parameter_match_score(
            "move robot with x and y coordinates", ["x", "y", "z", "robot_id"]
        )
        assert score > 0.5

    def test_no_parameter_match(self):
        score = calculate_parameter_match_score(
            "pick up the cube", ["x", "y", "z", "speed"]
        )
        assert score == 0.5

    def test_partial_parameter_match(self):
        score = calculate_parameter_match_score(
            "move to coordinate with speed", ["x", "y", "z", "speed", "robot_id"]
        )
        assert score >= 0.3


class TestMetadataMatchScore:

    def test_no_filters(self):
        metadata = {"category": "navigation", "complexity": "basic"}
        score = calculate_metadata_match_score(metadata)
        assert score == 0.5

    def test_category_match(self):
        metadata = {"category": "navigation"}
        score = calculate_metadata_match_score(metadata, category_filter="navigation")
        assert score > 0.5

    def test_category_mismatch(self):
        metadata = {"category": "manipulation"}
        score = calculate_metadata_match_score(metadata, category_filter="navigation")
        assert score < 0.5

    def test_complexity_match(self):
        metadata = {"complexity": "basic"}
        score = calculate_metadata_match_score(metadata, complexity_filter="basic")
        assert score > 0.5


class TestReliabilityScore:

    def test_high_reliability(self):
        metadata = {"success_rate": 0.98}
        score = calculate_reliability_score(metadata)
        assert score == 0.98

    def test_default_reliability(self):
        metadata = {}
        score = calculate_reliability_score(metadata)
        assert score == 0.95

    def test_low_reliability(self):
        metadata = {"success_rate": 0.5}
        score = calculate_reliability_score(metadata)
        assert score == 0.5


class TestComputeConfidenceScore:

    def test_compute_score_structure(self):
        result = compute_confidence_score(
            similarity_score=0.8,
            metadata={"category": "navigation", "parameters": ["x", "y", "z"]},
            query_text="move to position x y",
        )

        assert "final_score" in result
        assert "confidence_level" in result
        assert "breakdown" in result
        assert "weights" in result

        assert "similarity" in result["breakdown"]
        assert "metadata_match" in result["breakdown"]
        assert "parameter_match" in result["breakdown"]
        assert "reliability" in result["breakdown"]

    def test_score_range(self):
        result = compute_confidence_score(
            similarity_score=0.9,
            metadata={"category": "navigation", "success_rate": 0.98},
            query_text="move robot",
        )

        assert result["final_score"] >= 0.0
        assert result["final_score"] <= 1.0

    def test_high_similarity_high_confidence(self):
        result = compute_confidence_score(
            similarity_score=0.95,
            metadata={
                "category": "navigation",
                "success_rate": 0.99,
                "parameters": ["x", "y", "z"],
            },
            query_text="move robot to x y z coordinate",
        )

        assert result["final_score"] > 0.7

    def test_low_similarity_low_confidence(self):
        result = compute_confidence_score(
            similarity_score=0.3,
            metadata={"category": "navigation"},
            query_text="random text",
        )

        assert result["final_score"] < 0.5


class TestApplyConfidenceBoosting:

    def test_empty_results(self):
        results = apply_confidence_boosting([])
        assert results == []

    def test_results_enhanced(self):
        results = [
            {
                "operation_id": "op1",
                "score": 0.8,
                "metadata": {"category": "navigation", "parameters": ["x", "y"]},
            },
            {
                "operation_id": "op2",
                "score": 0.6,
                "metadata": {"category": "manipulation", "parameters": ["gripper"]},
            },
        ]

        enhanced = apply_confidence_boosting(results, query_text="move to x y")

        for result in enhanced:
            assert "confidence" in result
            assert "final_score" in result["confidence"]

    def test_results_reordered(self):
        results = [
            {
                "operation_id": "op1",
                "score": 0.5,
                "metadata": {
                    "category": "navigation",
                    "parameters": ["x", "y", "z"],
                    "success_rate": 0.99,
                },
            },
            {
                "operation_id": "op2",
                "score": 0.7,
                "metadata": {
                    "category": "perception",
                    "parameters": [],
                    "success_rate": 0.5,
                },
            },
        ]

        enhanced = apply_confidence_boosting(
            results, query_text="move to x y z", category_filter="navigation"
        )

        assert len(enhanced) == 2
        assert enhanced[0]["operation_id"] == "op1"


class TestCategoryMinScores:

    def test_navigation_threshold(self):
        score = get_category_min_score("navigation")
        assert score == 0.6

    def test_manipulation_threshold(self):
        score = get_category_min_score("manipulation")
        assert score == 0.55

    def test_unknown_category(self):
        score = get_category_min_score("unknown_category")
        assert score == 0.5


class TestWeights:

    def test_weights_sum_to_one(self):
        total = sum(WEIGHTS.values())
        assert abs(total - 1.0) < 10**-7

    def test_all_weights_present(self):
        expected_weights = [
            "similarity",
            "metadata_match",
            "parameter_match",
            "reliability",
        ]
        for weight in expected_weights:
            assert weight in WEIGHTS
