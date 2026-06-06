#!/usr/bin/env python3
"""Grasp planning pipeline: candidate generation, IK validation, scoring."""

import logging
import time
from typing import List, Tuple, Optional, Dict

from .GraspCandidate import GraspCandidate
from .GraspConfig import GraspConfig
from .GraspCandidateGenerator import GraspCandidateGenerator
from .GraspScorer import GraspScorer

logger = logging.getLogger(__name__)


class GraspPlanner:

    def __init__(self, config: Optional[GraspConfig] = None):
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
        """Plan a grasp. Returns best candidate above min_score, or None."""
        start_time = time.time()

        logger.info(
            f"Planning grasp for object at {object_position}, "
            f"size={object_size}, robot={robot_id}"
        )

        # Save/restore approach state so subsequent calls without preferred_approach still see all approaches.
        saved_state = None
        if preferred_approach is not None:
            saved_state = self._save_approach_state()
            # Map directional top-down variants to "top" for geometric planning
            _approach_for_filter = (
                "top"
                if preferred_approach in ("left_side", "right_side")
                else preferred_approach
            )
            self._filter_approaches(_approach_for_filter)

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

        if use_moveit_ik:
            candidates = self._validate_ik_with_moveit(
                candidates, robot_id, max_candidates
            )

            if not candidates:
                logger.warning("No candidates passed IK validation")
                return None

            logger.info(f"{len(candidates)} candidates passed IK validation")

        ranked_candidates = self.scorer.score_and_rank(
            candidates, object_size, gripper_position, gripper_rotation
        )

        valid_candidates = self.scorer.filter_by_min_score(ranked_candidates, min_score)

        if not valid_candidates:
            logger.warning(
                f"No candidates above minimum score ({min_score:.2f}). "
                f"Best score was {ranked_candidates[0].total_score:.2f}"
            )
            return None

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
        """Validate IK via MoveIt. Only checks top N to keep it fast."""
        try:
            from ros2.ROSBridge import ROSBridge

            bridge = ROSBridge.get_instance()

            if not bridge.is_connected:
                logger.warning("ROSBridge not connected, skipping IK validation")
                return candidates

            candidates_to_validate = candidates[:max_candidates]
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

            result = bridge.validate_grasp_candidates(candidate_data, robot_id)

            if not result or not result.get("success"):
                logger.error(
                    f"IK validation failed: {result.get('error', 'Unknown error') if result else 'Unknown error'}"
                )
                return candidates

            validation_results = result.get("results", [])
            validated_candidates = []

            for i, (candidate, (is_valid, quality_score)) in enumerate(
                zip(candidates_to_validate, validation_results)
            ):
                candidate.ik_validated = is_valid
                candidate.ik_score = quality_score

                if is_valid:
                    validated_candidates.append(candidate)

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
        return [
            (s.enabled, s.preference_weight) for s in self.config.enabled_approaches
        ]

    def _restore_approach_state(self, saved_state: list) -> None:
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
        """Plan multiple candidates. Useful for fallback/multi-robot strategies."""
        candidates = self.generator.generate_candidates(
            object_position, object_rotation, object_size, gripper_position
        )

        if not candidates:
            return []

        if kwargs.get("use_moveit_ik", True):
            candidates = self._validate_ik_with_moveit(
                candidates, robot_id, max_candidates=num_candidates * 2
            )

        ranked_candidates = self.scorer.score_and_rank(
            candidates,
            object_size,
            gripper_position,
            kwargs.get("gripper_rotation"),
        )

        return self.scorer.get_top_n(ranked_candidates, num_candidates)

    def get_statistics(self, candidates: List[GraspCandidate]) -> Dict:
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
