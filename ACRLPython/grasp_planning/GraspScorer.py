"""
Grasp candidate scoring and ranking.

This module scores grasp candidates using weighted multi-criteria evaluation,
including IK quality, approach preference, grasp depth, and stability.

Ported from Unity's GraspScorer.cs for compatibility.
"""

import logging
import numpy as np
from typing import List, Tuple, Optional

from .GraspCandidate import GraspCandidate
from .GraspConfig import GraspConfig

# Import quaternion and vector utilities
try:
    from utils.QuaternionMath import quaternion_angle
    from utils.VectorMath import vector_normalize, vector_dot, vector_cross, vector_distance
except ImportError:
    from ..utils.QuaternionMath import quaternion_angle
    from ..utils.VectorMath import vector_normalize, vector_dot, vector_cross, vector_distance


logger = logging.getLogger(__name__)


class GraspScorer:
    """
    Scores and ranks grasp candidates using weighted multi-criteria evaluation.

    Criteria:
    - IK quality: How reachable is the grasp pose
    - Approach preference: User-configured preference for approach type
    - Grasp depth: Deviation from target depth
    - Stability: Geometric stability based on grasp geometry
    - Antipodal quality: Quality of opposing contact points

    All scores are normalized to [0, 1] range.

    Attributes:
        config: Grasp planning configuration with scoring weights
    """

    def __init__(self, config: Optional[GraspConfig] = None):
        """
        Initialize scorer with configuration.

        Args:
            config: Grasp planning configuration (uses default if None)
        """
        self.config = config or GraspConfig.create_default()

    def score_and_rank(
        self,
        candidates: List[GraspCandidate],
        object_size: Tuple[float, float, float],
        gripper_position: Tuple[float, float, float],
        gripper_rotation: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[GraspCandidate]:
        """
        Score all candidates and return sorted list (best first).

        Pre-computes object_size array and ideal pre-grasp distance once to
        avoid redundant conversions and calculations inside _score_candidate.

        Args:
            candidates: List of candidates to score
            object_size: Size of target object (width, height, depth) in meters
            gripper_position: Current gripper position (x, y, z)
            gripper_rotation: Current gripper rotation (x, y, z, w) - optional

        Returns:
            Sorted list of candidates (highest score first)
        """
        # Pre-compute shared values used by every candidate scoring call
        obj_size_array = np.array(object_size)
        ideal_distance = self._calculate_pre_grasp_distance(obj_size_array)

        # Score each candidate
        for candidate in candidates:
            self._score_candidate(
                candidate, object_size, gripper_position, gripper_rotation,
                obj_size_array=obj_size_array, ideal_distance=ideal_distance,
            )

        # Sort by total score (descending)
        sorted_candidates = sorted(
            candidates, key=lambda c: c.total_score, reverse=True
        )

        logger.debug(
            f"Scored {len(candidates)} candidates. "
            f"Best score: {sorted_candidates[0].total_score:.3f}"
        )

        return sorted_candidates

    def _score_candidate(
        self,
        candidate: GraspCandidate,
        object_size: Tuple[float, float, float],
        gripper_position: Tuple[float, float, float],
        gripper_rotation: Optional[Tuple[float, float, float, float]] = None,
        obj_size_array: Optional[np.ndarray] = None,
        ideal_distance: Optional[float] = None,
    ) -> None:
        """
        Compute scores for a single candidate.

        Updates candidate's score fields in-place.

        Args:
            candidate: Candidate to score (modified in-place)
            object_size: Size of target object
            gripper_position: Current gripper position
            gripper_rotation: Current gripper rotation (optional)
            obj_size_array: Pre-computed np.array(object_size) to avoid redundant
                conversion per candidate (computed here if not provided)
            ideal_distance: Pre-computed ideal pre-grasp distance to avoid redundant
                calculation per candidate (computed here if not provided)
        """
        if obj_size_array is None:
            obj_size_array = np.array(object_size)
        if ideal_distance is None:
            ideal_distance = self._calculate_pre_grasp_distance(obj_size_array)

        # Compute individual scores
        candidate.ik_score = self._compute_ik_score(candidate, gripper_position)
        approach_score = self._compute_approach_score(candidate)
        depth_score = self._compute_depth_score(candidate, obj_size_array)
        stability_score = self._compute_stability_score(candidate, obj_size_array, ideal_distance)

        # Compute orientation consistency (default to 1.0 if no current rotation)
        if gripper_rotation is not None:
            orientation_consistency = self._compute_orientation_consistency_score(
                candidate, gripper_rotation
            )
        else:
            orientation_consistency = 1.0

        # Weighted sum of scores
        candidate.total_score = (
            candidate.ik_score * self.config.ik_score_weight
            + approach_score * self.config.approach_score_weight
            + depth_score * self.config.depth_score_weight
            + stability_score * self.config.stability_score_weight
            + candidate.antipodal_score * self.config.antipodal_score_weight
        )

        # Apply orientation consistency multiplier
        candidate.total_score *= orientation_consistency

    def _compute_ik_score(
        self, candidate: GraspCandidate, gripper_position: Tuple[float, float, float]
    ) -> float:
        """
        Compute IK quality score.

        Higher score for positions closer to current gripper (easier to reach).
        If IK validation was performed, use validation quality instead.

        Args:
            candidate: Candidate to evaluate
            gripper_position: Current gripper position

        Returns:
            Normalized score [0, 1]
        """
        # If IK was validated by MoveIt, use that score
        if candidate.ik_validated:
            return np.clip(candidate.ik_score, 0.0, 1.0)

        # Otherwise, estimate based on distance to current position
        distance = vector_distance(
            np.array(candidate.pre_grasp_position), np.array(gripper_position)
        )
        max_reach = self.config.max_reach_distance

        normalized_distance = np.clip(distance / max_reach, 0.0, 1.0)
        return 1.0 - normalized_distance

    def _compute_approach_score(self, candidate: GraspCandidate) -> float:
        """
        Compute approach preference score.

        Uses configured preference weights for each approach type.

        Args:
            candidate: Candidate to evaluate

        Returns:
            Normalized score [0, 1]
        """
        weight = self.config.get_approach_weight(candidate.approach_type)

        # Normalize weight to [0, 1] (max weight is 2.0 in config)
        return np.clip(weight / 2.0, 0.0, 1.0)

    def _compute_depth_score(
        self, candidate: GraspCandidate, object_size: np.ndarray
    ) -> float:
        """
        Compute grasp depth score.

        Penalizes deviations from target depth using Gaussian distribution.
        Object-size-aware scoring.

        Args:
            candidate: Candidate to evaluate
            object_size: Object size as np.ndarray (pre-computed by caller)

        Returns:
            Normalized score [0, 1]
        """
        avg_object_size = np.mean(object_size)
        target_depth = self.config.target_grasp_depth * avg_object_size
        actual_depth = candidate.grasp_depth

        deviation = abs(actual_depth - target_depth)
        sigma = avg_object_size * 0.15

        # Gaussian score: exp(-deviation^2 / (2*sigma^2))
        score = np.exp(-(deviation**2) / (2.0 * sigma**2))

        return float(score)

    def _compute_stability_score(
        self,
        candidate: GraspCandidate,
        object_size: np.ndarray,
        ideal_distance: float,
    ) -> float:
        """
        Compute stability score based on grasp geometry.

        Higher for grasps aligned with object center and gravity.
        Enhanced with center-of-mass alignment, contact area, and edge avoidance.

        Args:
            candidate: Candidate to evaluate
            object_size: Object size as np.ndarray (pre-computed by caller)
            ideal_distance: Pre-computed ideal pre-grasp distance (avoids
                redundant _calculate_pre_grasp_distance call per candidate)

        Returns:
            Normalized score [0, 1]
        """
        score = 1.0

        # Gravity alignment: prefer downward approaches
        approach_dir = np.array(candidate.approach_direction)
        gravity_alignment = abs(vector_dot(approach_dir, np.array([0.0, 1.0, 0.0])))
        score *= 0.65 + 0.35 * gravity_alignment

        # Gripper compatibility: can gripper grasp this object?
        if not self.config.gripper_geometry.can_grasp(object_size):
            score *= 0.5

        # Distance ratio: prefer distances close to ideal
        if ideal_distance > 1e-6:
            distance_ratio = abs(candidate.approach_distance - ideal_distance) / ideal_distance
            score *= np.clip(1.0 - distance_ratio, 0.0, 1.0)

        # Center alignment: prefer grasps centered on object
        grasp_pos = np.array(candidate.grasp_position)
        contact_pos = np.array(candidate.contact_point_estimate)
        grasp_offset = grasp_pos - contact_pos
        object_size_magnitude = np.linalg.norm(object_size)
        if object_size_magnitude > 1e-6:
            center_deviation = np.linalg.norm(grasp_offset) / object_size_magnitude
            center_score = np.exp(-center_deviation * 2.0)
            score *= 0.7 + 0.3 * center_score

        # Contact area: larger contact area = more stable
        contact_area_score = self._estimate_contact_area(candidate, object_size)
        score *= 0.8 + 0.2 * contact_area_score

        # Edge avoidance: penalize grasps near edges
        edge_score = self._compute_edge_avoidance_score(candidate, object_size)
        score *= edge_score

        return float(np.clip(score, 0.0, 1.0))

    def _estimate_contact_area(
        self, candidate: GraspCandidate, object_size: np.ndarray
    ) -> float:
        """
        Estimate contact area between gripper and object.

        Larger contact areas provide more stable grasps.

        Args:
            candidate: Candidate to evaluate
            object_size: Size of target object

        Returns:
            Normalized contact area score [0, 1]
        """
        approach_dir = np.array(candidate.approach_direction)
        abs_approach = np.abs(approach_dir)

        # Determine which face of the object is being grasped
        if abs_approach[0] > 0.9:  # X-axis approach
            contact_area = object_size[1] * object_size[2]  # Y * Z face
        elif abs_approach[1] > 0.9:  # Y-axis approach
            contact_area = object_size[0] * object_size[2]  # X * Z face
        else:  # Z-axis approach
            contact_area = object_size[0] * object_size[1]  # X * Y face

        # Compare with gripper finger area
        gripper_area = (
            self.config.gripper_geometry.finger_length
            * self.config.gripper_geometry.finger_width
        )

        # Ratio of smaller to larger area
        area_ratio = min(contact_area, gripper_area) / max(contact_area, gripper_area)

        return float(area_ratio)

    def _compute_edge_avoidance_score(
        self, candidate: GraspCandidate, object_size: np.ndarray
    ) -> float:
        """
        Compute edge avoidance score.

        Penalizes grasps near object edges for better stability.

        Args:
            candidate: Candidate to evaluate
            object_size: Object size as np.ndarray (pre-computed by caller)

        Returns:
            Edge avoidance score [0, 1] (higher = farther from edges)
        """
        grasp_pos = np.array(candidate.grasp_position)
        contact_pos = np.array(candidate.contact_point_estimate)
        relative_pos = grasp_pos - contact_pos

        # Normalize position relative to object half-extents
        normalized_pos = np.abs(relative_pos) / (object_size * 0.5)

        # Find minimum distance to any edge
        min_dist_to_edge = np.min(normalized_pos)

        # Penalize if very close to edge (> 80% of half-extent)
        if min_dist_to_edge > 0.8:
            edge_penalty = (min_dist_to_edge - 0.8) / 0.2
            return 1.0 - edge_penalty * 0.3

        return 1.0

    def _compute_orientation_consistency_score(
        self,
        candidate: GraspCandidate,
        current_gripper_rotation: Tuple[float, float, float, float],
    ) -> float:
        """
        Compute orientation consistency score.

        Prevents "180-degree flip" scenarios by penalizing large rotations
        from current gripper orientation.

        Args:
            candidate: Candidate to evaluate
            current_gripper_rotation: Current gripper rotation quaternion

        Returns:
            Consistency score [0, 1] (higher = smaller rotation needed)
        """
        delta_angle = quaternion_angle(
            current_gripper_rotation, candidate.grasp_rotation
        )

        # Piecewise scoring:
        # 0-45°: perfect (1.0)
        # 45-90°: linear decrease (1.0 -> 0.5)
        # 90-180°: quadratic decrease (0.5 -> 0.1)
        if delta_angle <= 45.0:
            return 1.0
        elif delta_angle <= 90.0:
            t = (delta_angle - 45.0) / 45.0
            return 1.0 - t * 0.5
        else:
            t = (delta_angle - 90.0) / 90.0
            return 0.5 - t * t * 0.4

    def _calculate_pre_grasp_distance(self, obj_size: np.ndarray) -> float:
        """
        Calculate ideal pre-grasp distance from object size.

        Args:
            obj_size: Object size array

        Returns:
            Pre-grasp distance in meters
        """
        avg_size = np.mean(obj_size)
        distance = avg_size * self.config.pre_grasp_distance_factor
        return float(
            np.clip(
                distance,
                self.config.min_pre_grasp_distance,
                self.config.max_pre_grasp_distance,
            )
        )

    def filter_by_min_score(
        self, candidates: List[GraspCandidate], min_score: float
    ) -> List[GraspCandidate]:
        """
        Filter candidates below a minimum score threshold.

        Args:
            candidates: Scored candidates
            min_score: Minimum total score threshold

        Returns:
            Filtered list of candidates
        """
        filtered = [c for c in candidates if c.total_score >= min_score]
        logger.debug(
            f"Filtered {len(candidates)} candidates to {len(filtered)} "
            f"(min_score={min_score:.2f})"
        )
        return filtered

    def get_top_n(
        self, candidates: List[GraspCandidate], count: int
    ) -> List[GraspCandidate]:
        """
        Get top N candidates from scored list.

        Args:
            candidates: Scored and sorted candidates
            count: Number of candidates to return

        Returns:
            Top N candidates
        """
        return candidates[:count]

    def normalize_scores(self, candidates: List[GraspCandidate]) -> None:
        """
        Normalize all candidate scores to 0-1 range within the list.

        Useful for comparing candidates across different scenarios.
        Modifies candidates in-place.

        Args:
            candidates: Candidates to normalize (modified in-place)
        """
        if not candidates:
            return

        scores = [c.total_score for c in candidates]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        # Avoid division by zero
        if score_range < 0.001:
            return

        # Normalize each candidate
        for candidate in candidates:
            candidate.total_score = (candidate.total_score - min_score) / score_range

        logger.debug(f"Normalized {len(candidates)} candidate scores to [0, 1]")
