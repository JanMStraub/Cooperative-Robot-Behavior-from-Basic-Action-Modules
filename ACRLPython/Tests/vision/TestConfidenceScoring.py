#!/usr/bin/env python3
"""
Tests for Confidence Scoring System
====================================

Unit tests for the multi-factor confidence scoring in the RAG system.
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


class TestConfidenceLevel(unittest.TestCase):
    """Test confidence level classification"""

    def test_high_confidence(self):
        """Test high confidence threshold"""
        self.assertEqual(get_confidence_level(0.9), ConfidenceLevel.HIGH)
        self.assertEqual(get_confidence_level(0.75), ConfidenceLevel.HIGH)

    def test_medium_confidence(self):
        """Test medium confidence threshold"""
        self.assertEqual(get_confidence_level(0.6), ConfidenceLevel.MEDIUM)
        self.assertEqual(get_confidence_level(0.5), ConfidenceLevel.MEDIUM)

    def test_low_confidence(self):
        """Test low confidence threshold"""
        self.assertEqual(get_confidence_level(0.4), ConfidenceLevel.LOW)
        self.assertEqual(get_confidence_level(0.25), ConfidenceLevel.LOW)

    def test_uncertain_confidence(self):
        """Test uncertain confidence threshold"""
        self.assertEqual(get_confidence_level(0.1), ConfidenceLevel.UNCERTAIN)
        self.assertEqual(get_confidence_level(0.0), ConfidenceLevel.UNCERTAIN)


class TestParameterMatchScore(unittest.TestCase):
    """Test parameter matching scoring"""

    def test_no_parameters(self):
        """Test scoring when operation has no parameters"""
        score = calculate_parameter_match_score("move robot", [])
        self.assertEqual(score, 0.5)  # Neutral score

    def test_parameter_match(self):
        """Test scoring when query mentions parameter names"""
        score = calculate_parameter_match_score(
            "move robot with x and y coordinates", ["x", "y", "z", "robot_id"]
        )
        self.assertGreater(score, 0.5)  # Should boost score

    def test_no_parameter_match(self):
        """Test scoring when query doesn't mention parameters"""
        score = calculate_parameter_match_score(
            "pick up the cube", ["x", "y", "z", "speed"]
        )
        self.assertEqual(score, 0.3)  # Low score for no matches

    def test_partial_parameter_match(self):
        """Test scoring with partial parameter matches"""
        score = calculate_parameter_match_score(
            "move to coordinate with speed", ["x", "y", "z", "speed", "robot_id"]
        )
        # Should match 'speed' and possibly partial from coordinate
        self.assertGreaterEqual(score, 0.3)


class TestMetadataMatchScore(unittest.TestCase):
    """Test metadata matching scoring"""

    def test_no_filters(self):
        """Test scoring when no filters provided"""
        metadata = {"category": "navigation", "complexity": "basic"}
        score = calculate_metadata_match_score(metadata)
        self.assertEqual(score, 0.5)  # Neutral

    def test_category_match(self):
        """Test scoring when category matches filter"""
        metadata = {"category": "navigation"}
        score = calculate_metadata_match_score(metadata, category_filter="navigation")
        self.assertGreater(score, 0.5)  # Boost for match

    def test_category_mismatch(self):
        """Test scoring when category doesn't match filter"""
        metadata = {"category": "manipulation"}
        score = calculate_metadata_match_score(metadata, category_filter="navigation")
        self.assertLess(score, 0.5)  # Penalty for mismatch

    def test_complexity_match(self):
        """Test scoring when complexity matches filter"""
        metadata = {"complexity": "basic"}
        score = calculate_metadata_match_score(metadata, complexity_filter="basic")
        self.assertGreater(score, 0.5)


class TestReliabilityScore(unittest.TestCase):
    """Test reliability scoring"""

    def test_high_reliability(self):
        """Test high reliability operation"""
        metadata = {"success_rate": 0.98}
        score = calculate_reliability_score(metadata)
        self.assertEqual(score, 0.98)

    def test_default_reliability(self):
        """Test default reliability when not specified"""
        metadata = {}
        score = calculate_reliability_score(metadata)
        self.assertEqual(score, 0.95)  # Default

    def test_low_reliability(self):
        """Test low reliability operation"""
        metadata = {"success_rate": 0.5}
        score = calculate_reliability_score(metadata)
        self.assertEqual(score, 0.5)


class TestComputeConfidenceScore(unittest.TestCase):
    """Test overall confidence score computation"""

    def test_compute_score_structure(self):
        """Test that compute_confidence_score returns correct structure"""
        result = compute_confidence_score(
            similarity_score=0.8,
            metadata={"category": "navigation", "parameters": ["x", "y", "z"]},
            query_text="move to position x y",
        )

        self.assertIn("final_score", result)
        self.assertIn("confidence_level", result)
        self.assertIn("breakdown", result)
        self.assertIn("weights", result)

        self.assertIn("similarity", result["breakdown"])
        self.assertIn("metadata_match", result["breakdown"])
        self.assertIn("parameter_match", result["breakdown"])
        self.assertIn("reliability", result["breakdown"])

    def test_score_range(self):
        """Test that scores are in valid range"""
        result = compute_confidence_score(
            similarity_score=0.9,
            metadata={"category": "navigation", "success_rate": 0.98},
            query_text="move robot",
        )

        self.assertGreaterEqual(result["final_score"], 0.0)
        self.assertLessEqual(result["final_score"], 1.0)

    def test_high_similarity_high_confidence(self):
        """Test that high similarity leads to high confidence"""
        result = compute_confidence_score(
            similarity_score=0.95,
            metadata={
                "category": "navigation",
                "success_rate": 0.99,
                "parameters": ["x", "y", "z"],
            },
            query_text="move robot to x y z coordinate",
        )

        self.assertGreater(result["final_score"], 0.7)

    def test_low_similarity_low_confidence(self):
        """Test that low similarity leads to lower confidence"""
        result = compute_confidence_score(
            similarity_score=0.3,
            metadata={"category": "navigation"},
            query_text="random text",
        )

        self.assertLess(result["final_score"], 0.5)


class TestApplyConfidenceBoosting(unittest.TestCase):
    """Test confidence boosting on search results"""

    def test_empty_results(self):
        """Test boosting with empty results"""
        results = apply_confidence_boosting([])
        self.assertEqual(results, [])

    def test_results_enhanced(self):
        """Test that results are enhanced with confidence"""
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
            self.assertIn("confidence", result)
            self.assertIn("final_score", result["confidence"])

    def test_results_reordered(self):
        """Test that results are re-sorted by confidence score"""
        # Lower similarity but better metadata match
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

        # Results should be reordered based on confidence
        self.assertEqual(len(enhanced), 2)
        # The navigation operation should score higher due to better metadata match
        self.assertEqual(enhanced[0]["operation_id"], "op1")


class TestCategoryMinScores(unittest.TestCase):
    """Test category-specific minimum score thresholds"""

    def test_navigation_threshold(self):
        """Test navigation category threshold"""
        score = get_category_min_score("navigation")
        self.assertEqual(score, 0.6)

    def test_manipulation_threshold(self):
        """Test manipulation category threshold"""
        score = get_category_min_score("manipulation")
        self.assertEqual(score, 0.55)

    def test_unknown_category(self):
        """Test unknown category falls back to default"""
        score = get_category_min_score("unknown_category")
        self.assertEqual(score, 0.5)


class TestWeights(unittest.TestCase):
    """Test scoring weights configuration"""

    def test_weights_sum_to_one(self):
        """Test that weights sum to 1.0"""
        total = sum(WEIGHTS.values())
        self.assertAlmostEqual(total, 1.0)

    def test_all_weights_present(self):
        """Test that all expected weights are present"""
        expected_weights = [
            "similarity",
            "metadata_match",
            "parameter_match",
            "reliability",
        ]
        for weight in expected_weights:
            self.assertIn(weight, WEIGHTS)


if __name__ == "__main__":
    unittest.main()
