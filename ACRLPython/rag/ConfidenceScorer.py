#!/usr/bin/env python3
"""Multi-factor confidence scoring for RAG system."""

from typing import Dict, Any, List, Optional
from enum import Enum
import re


class ConfidenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


CONFIDENCE_TIERS = {
    "high": 0.75,
    "medium": 0.5,
    "low": 0.25,
}

CATEGORY_MIN_SCORES = {
    "navigation": 0.6,
    "manipulation": 0.55,
    "perception": 0.50,
    "coordination": 0.60,
    "state_query": 0.55,
}

WEIGHTS = {
    "similarity": 0.4,
    "metadata_match": 0.3,
    "parameter_match": 0.2,
    "reliability": 0.1,
}


def get_confidence_level(score: float) -> ConfidenceLevel:
    if score >= CONFIDENCE_TIERS["high"]:
        return ConfidenceLevel.HIGH
    elif score >= CONFIDENCE_TIERS["medium"]:
        return ConfidenceLevel.MEDIUM
    elif score >= CONFIDENCE_TIERS["low"]:
        return ConfidenceLevel.LOW
    else:
        return ConfidenceLevel.UNCERTAIN


def calculate_parameter_match_score(
    query_text: str, parameter_names: List[str]
) -> float:
    if not parameter_names:
        return 0.5  # Neutral score for operations without parameters

    query_lower = query_text.lower()
    query_terms = set(re.findall(r"\w+", query_lower))

    param_terms = set()
    for param in parameter_names:
        parts = param.lower().split("_")
        param_terms.update(parts)
        param_terms.add(param.lower().replace("_", ""))

    matches = len(param_terms & query_terms)
    if matches == 0:
        return 0.3  # Low but not zero - query might not mention params

    return min(1.0, 0.5 + (matches / len(param_terms)) * 0.5)


def calculate_metadata_match_score(
    metadata: Dict[str, Any],
    category_filter: Optional[str] = None,
    complexity_filter: Optional[str] = None,
) -> float:
    if not category_filter and not complexity_filter:
        return 0.5  # Neutral when no filters

    score = 0.5

    if category_filter:
        if metadata.get("category") == category_filter:
            score += 0.3
        else:
            score -= 0.2

    if complexity_filter:
        if metadata.get("complexity") == complexity_filter:
            score += 0.2
        else:
            score -= 0.1

    return max(0.0, min(1.0, score))


def calculate_reliability_score(metadata: Dict[str, Any]) -> float:
    success_rate = metadata.get("success_rate", 0.95)
    return float(success_rate)


def compute_confidence_score(
    similarity_score: float,
    metadata: Dict[str, Any],
    query_text: str = "",
    category_filter: Optional[str] = None,
    complexity_filter: Optional[str] = None,
) -> Dict[str, Any]:
    parameters = metadata.get("parameters", [])
    if isinstance(parameters, list):
        if parameters and isinstance(parameters[0], dict):
            param_names = [p.get("name", "") for p in parameters]
        else:
            param_names = parameters  # Already a list of names
    else:
        param_names = []

    param_score = calculate_parameter_match_score(query_text, param_names)
    metadata_score = calculate_metadata_match_score(
        metadata, category_filter, complexity_filter
    )
    reliability_score = calculate_reliability_score(metadata)

    final_score = (
        WEIGHTS["similarity"] * similarity_score
        + WEIGHTS["metadata_match"] * metadata_score
        + WEIGHTS["parameter_match"] * param_score
        + WEIGHTS["reliability"] * reliability_score
    )

    final_score = max(0.0, min(1.0, final_score))

    confidence_level = get_confidence_level(final_score)

    return {
        "final_score": final_score,
        "confidence_level": confidence_level.value,
        "breakdown": {
            "similarity": similarity_score,
            "metadata_match": metadata_score,
            "parameter_match": param_score,
            "reliability": reliability_score,
        },
        "weights": WEIGHTS,
    }


def get_category_min_score(category: str) -> float:
    return CATEGORY_MIN_SCORES.get(category.lower(), 0.5)


def apply_confidence_boosting(
    results: List[Dict[str, Any]],
    query_text: str = "",
    category_filter: Optional[str] = None,
    complexity_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    enhanced_results = []

    for result in results:
        similarity = result.get("score", 0.0)
        metadata = result.get("metadata", {})

        confidence = compute_confidence_score(
            similarity_score=similarity,
            metadata=metadata,
            query_text=query_text,
            category_filter=category_filter,
            complexity_filter=complexity_filter,
        )

        enhanced_result = result.copy()
        enhanced_result["confidence"] = confidence
        enhanced_result["score"] = confidence["final_score"]
        enhanced_results.append(enhanced_result)

    enhanced_results.sort(key=lambda x: x["score"], reverse=True)

    return enhanced_results
