#!/usr/bin/env python3
"""
End-to-end grasp planning orchestrator.

This module provides the main GraspPlanner class that coordinates the entire
grasp planning pipeline: candidate generation, IK validation, scoring, and
selection of the best grasp pose.

Integrates with MoveIt via ROSBridge for IK validation.
"""

import logging
import time
from typing import List, Tuple, Optional, Dict

from .GraspCandidate import GraspCandidate
from .GraspConfig import GraspConfig
from .GraspCandidateGenerator import GraspCandidateGenerator
from .GraspScorer import GraspScorer


logger = logging.getLogger(__name__)


class GraspPlanner:
    """
    End-to-end grasp planning pipeline orchestrator.

    Coordinates candidate generation, IK validation, scoring, and selection.
    Provides high-level interface for grasp planning operations.

    Attributes:
        config: Grasp planning configuration
        generator: Candidate generator
        scorer: Candidate scorer
    """

    def __init__(self, config: Optional[GraspConfig] = None):
        """
        Initialize grasp planner.

        Args:
            config: Grasp planning configuration (uses default if None)
        """
        self.config = config or GraspConfig.create_default()
        self.generator = GraspCandidateGenerator(self.config)
        self.scorer = GraspScorer(self.config)

    def plan_grasp(
        self,
        object_position: Tuple[float, float, float],
        object_rotation: Tuple[float, float, float, float],
        object_size: Tuple[float, float, float],
        robot_id: str,
        gripper_position: Tuple[float, float, float],
        gripper_rotation: Optional[Tuple[float, float, float, float]] = None,
        use_moveit_ik: bool = True,
        preferred_approach: Optional[str] = None,
        min_score: float = 0.3,
        max_candidates: int = 5,
    ) -> Optional[GraspCandidate]:
        """
        Plan a grasp for a target object.

        Full pipeline:
        1. Generate grasp candidates (8-24 depending on config)
        2. Optionally validate IK via MoveIt
        3. Score and rank candidates
        4. Return best candidate above minimum score

        Args:
            object_position: Object center position (x, y, z) in world coordinates
            object_rotation: Object rotation quaternion (x, y, z, w)
            object_size: Object dimensions (width, height, depth) in meters
            robot_id: Robot identifier for IK validation
            gripper_position: Current gripper position (x, y, z)
            gripper_rotation: Current gripper rotation (x, y, z, w) - optional
            use_moveit_ik: Whether to validate IK via MoveIt (default: True)
            preferred_approach: Preferred approach type ("top", "front", "side") or None
            min_score: Minimum acceptable grasp score (default: 0.3)
            max_candidates: Maximum candidates to validate (default: 5)

        Returns:
            Best grasp candidate or None if no valid grasps found
        """
        start_time = time.time()

        logger.info(
            f"Planning grasp for object at {object_position}, "
            f"size={object_size}, robot={robot_id}"
        )

        # Filter config if preferred approach is specified.
        # State is saved before and restored after generation so reusing this
        # planner instance on a subsequent call without preferred_approach still
        # sees all approaches enabled.
        saved_state = None
        if preferred_approach is not None:
            saved_state = self._save_approach_state()
            self._filter_approaches(preferred_approach)

        # Step 1: Generate candidates
        try:
            candidates = self.generator.generate_candidates(
                object_position, object_rotation, object_size, gripper_position
            )
        finally:
            if saved_state is not None:
                self._restore_approach_state(saved_state)

        if not candidates:
            logger.warning("No grasp candidates generated")
            return None

        logger.info(f"Generated {len(candidates)} grasp candidates")

        # Step 2: Validate IK via MoveIt (if enabled)
        if use_moveit_ik:
            candidates = self._validate_ik_with_moveit(
                candidates, robot_id, max_candidates
            )

            if not candidates:
                logger.warning("No candidates passed IK validation")
                return None

            logger.info(f"{len(candidates)} candidates passed IK validation")

        # Step 3: Score and rank candidates
        ranked_candidates = self.scorer.score_and_rank(
            candidates, object_size, gripper_position, gripper_rotation
        )

        # Step 4: Filter by minimum score
        valid_candidates = self.scorer.filter_by_min_score(ranked_candidates, min_score)

        if not valid_candidates:
            logger.warning(
                f"No candidates above minimum score ({min_score:.2f}). "
                f"Best score was {ranked_candidates[0].total_score:.2f}"
            )
            return None

        # Return best candidate
        best_candidate = valid_candidates[0]

        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        logger.info(
            f"Grasp planning complete in {elapsed_time:.1f}ms. "
            f"Best candidate: {best_candidate.approach_type} approach, "
            f"score={best_candidate.total_score:.3f}"
        )

        return best_candidate

    def _validate_ik_with_moveit(
        self,
        candidates: List[GraspCandidate],
        robot_id: str,
        max_candidates: int = 5,
    ) -> List[GraspCandidate]:
        """
        Validate IK for candidates using MoveIt.

        Only validates top N candidates to save time.

        Args:
            candidates: List of candidates to validate
            robot_id: Robot identifier
            max_candidates: Maximum candidates to validate

        Returns:
            List of candidates that passed IK validation
        """
        try:
            from ros2.ROSBridge import ROSBridge

            bridge = ROSBridge.get_instance()

            if not bridge.is_connected:
                logger.warning("ROSBridge not connected, skipping IK validation")
                return candidates

            # Limit number of candidates to validate (performance optimization)
            candidates_to_validate = candidates[:max_candidates]

            # Prepare candidate data for validation
            candidate_data = []
            for candidate in candidates_to_validate:
                candidate_data.append(
                    {
                        "position": {
                            "x": candidate.grasp_position[0],
                            "y": candidate.grasp_position[1],
                            "z": candidate.grasp_position[2],
                        },
                        "rotation": {
                            "x": candidate.grasp_rotation[0],
                            "y": candidate.grasp_rotation[1],
                            "z": candidate.grasp_rotation[2],
                            "w": candidate.grasp_rotation[3],
                        },
                    }
                )

            # Call IK validation service
            result = bridge.validate_grasp_candidates(candidate_data, robot_id)

            if not result or not result.get("success"):
                logger.error(
                    f"IK validation failed: {result.get('error', 'Unknown error') if result else 'Unknown error'}"
                )
                return candidates

            # Update candidates with IK validation results
            validation_results = result.get("results", [])
            validated_candidates = []

            for i, (candidate, (is_valid, quality_score)) in enumerate(
                zip(candidates_to_validate, validation_results)
            ):
                candidate.ik_validated = is_valid
                candidate.ik_score = quality_score

                if is_valid:
                    validated_candidates.append(candidate)

            # Add remaining unvalidated candidates (if any)
            if len(candidates) > max_candidates:
                validated_candidates.extend(candidates[max_candidates:])

            return validated_candidates

        except ImportError:
            logger.warning("ROSBridge not available, skipping IK validation")
            return candidates
        except Exception as e:
            logger.error(f"Error during IK validation: {e}", exc_info=True)
            return candidates

    def _filter_approaches(self, preferred_approach: str) -> None:
        """
        Filter configuration to only use preferred approach.

        Saves original enabled/preference_weight state, disables all approaches
        except the preferred one, generates candidates, then restores state so
        subsequent calls on the same planner instance are unaffected.

        The actual save/restore lifecycle is managed by plan_grasp via
        _save_approach_state and _restore_approach_state; this method only
        applies the filter.

        Args:
            preferred_approach: Approach type to prefer ("top", "front", "side")
        """
        matched = False
        for approach_settings in self.config.enabled_approaches:
            if approach_settings.approach_type == preferred_approach:
                approach_settings.enabled = True
                approach_settings.preference_weight = 2.0  # Max weight
                matched = True
            else:
                approach_settings.enabled = False

        if not matched:
            logger.warning(
                f"preferred_approach='{preferred_approach}' matched no enabled approach; "
                f"all approaches have been disabled and no candidates will be generated."
            )

    def _save_approach_state(self) -> list:
        """
        Save current enabled/preference_weight state of all approach settings.

        Returns:
            List of (enabled, preference_weight) tuples in config order
        """
        return [
            (s.enabled, s.preference_weight) for s in self.config.enabled_approaches
        ]

    def _restore_approach_state(self, saved_state: list) -> None:
        """
        Restore enabled/preference_weight state saved by _save_approach_state.

        Args:
            saved_state: List of (enabled, preference_weight) tuples
        """
        for approach_settings, (enabled, weight) in zip(
            self.config.enabled_approaches, saved_state
        ):
            approach_settings.enabled = enabled
            approach_settings.preference_weight = weight

    def plan_multi_grasp(
        self,
        object_position: Tuple[float, float, float],
        object_rotation: Tuple[float, float, float, float],
        object_size: Tuple[float, float, float],
        robot_id: str,
        gripper_position: Tuple[float, float, float],
        num_candidates: int = 3,
        **kwargs,
    ) -> List[GraspCandidate]:
        """
        Plan multiple grasp candidates for a target object.

        Useful for fallback strategies or multi-robot coordination.

        Args:
            object_position: Object center position
            object_rotation: Object rotation
            object_size: Object dimensions
            robot_id: Robot identifier
            gripper_position: Current gripper position
            num_candidates: Number of candidates to return
            **kwargs: Additional arguments passed to plan_grasp

        Returns:
            List of top N grasp candidates
        """
        # Generate and score all candidates
        candidates = self.generator.generate_candidates(
            object_position, object_rotation, object_size, gripper_position
        )

        if not candidates:
            return []

        # Validate IK if requested
        if kwargs.get("use_moveit_ik", True):
            candidates = self._validate_ik_with_moveit(
                candidates, robot_id, max_candidates=num_candidates * 2
            )

        # Score and rank
        ranked_candidates = self.scorer.score_and_rank(
            candidates,
            object_size,
            gripper_position,
            kwargs.get("gripper_rotation"),
        )

        # Return top N
        return self.scorer.get_top_n(ranked_candidates, num_candidates)

    def get_statistics(self, candidates: List[GraspCandidate]) -> Dict:
        """
        Get statistics about a set of grasp candidates.

        Useful for debugging and analysis.

        Args:
            candidates: List of candidates

        Returns:
            Dictionary with statistics
        """
        if not candidates:
            return {"count": 0}

        scores = [c.total_score for c in candidates]
        ik_valid_count = sum(1 for c in candidates if c.ik_validated)
        approach_counts = {}

        for candidate in candidates:
            approach_type = candidate.approach_type
            approach_counts[approach_type] = approach_counts.get(approach_type, 0) + 1

        return {
            "count": len(candidates),
            "ik_validated": ik_valid_count,
            "ik_valid_percent": (ik_valid_count / len(candidates)) * 100,
            "score_mean": sum(scores) / len(scores),
            "score_min": min(scores),
            "score_max": max(scores),
            "approach_counts": approach_counts,
        }
