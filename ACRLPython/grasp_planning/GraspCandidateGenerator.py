"""
Grasp candidate generation with adaptive positioning.

This module generates multiple grasp candidates per approach type (top/front/side)
with randomized variations in angle, distance, and depth. The algorithm is
ported from Unity's GraspCandidateGenerator.cs for compatibility.

Key features:
- Multiple candidates per approach (default: 8)
- Object-size-adaptive positioning
- Random cone sampling for approach variation
- URDF-aware gripper orientation calculation
- Antipodal grasp quality estimation
"""

import logging
import numpy as np
from typing import List, Tuple, Optional

from .GraspCandidate import GraspCandidate
from .GraspConfig import GraspConfig

# Import quaternion and vector utilities
try:
    from utils.QuaternionMath import (
        quaternion_from_euler,
        quaternion_multiply,
        quaternion_rotate_vector,
        quaternion_inverse,
        quaternion_from_axis_angle,
        quaternion_identity,
    )
    from utils.VectorMath import (
        vector_normalize,
        vector_cross,
        vector_dot,
    )
except ImportError:
    from ..utils.QuaternionMath import (
        quaternion_from_euler,
        quaternion_multiply,
        quaternion_rotate_vector,
        quaternion_inverse,
        quaternion_from_axis_angle,
        quaternion_identity,
    )
    from ..utils.VectorMath import (
        vector_normalize,
        vector_cross,
        vector_dot,
    )


logger = logging.getLogger(__name__)


class GraspCandidateGenerator:
    """
    Generates multiple grasp candidates per approach type with adaptive positioning.

    Optimized for object-size-adaptive candidate generation with randomized variations.
    Matches Unity's GraspCandidateGenerator.cs algorithm.

    Attributes:
        config: Grasp planning configuration
        random: NumPy random number generator (for reproducibility)
    """

    # Approach axes in object-local space (Unity convention: Y=up, X=right, Z=forward)
    APPROACH_AXES = {
        "top": np.array([0.0, 1.0, 0.0]),
        "side": np.array([1.0, 0.0, 0.0]),
        "front": np.array([0.0, 0.0, 1.0]),
    }

    def __init__(self, config: Optional[GraspConfig] = None, seed: Optional[int] = None):
        """
        Initialize grasp candidate generator.

        Args:
            config: Grasp configuration (uses default if None)
            seed: Random seed for reproducibility (None = random)
        """
        self.config = config or GraspConfig.create_default()
        self.random = np.random.RandomState(seed)

    def generate_candidates(
        self,
        object_position: Tuple[float, float, float],
        object_rotation: Tuple[float, float, float, float],
        object_size: Tuple[float, float, float],
        gripper_position: Optional[Tuple[float, float, float]] = None,
    ) -> List[GraspCandidate]:
        """
        Generate all grasp candidates for a target object.

        Generates candidates for all enabled approach types (top/front/side)
        with randomized variations in angle, distance, and depth.

        Args:
            object_position: Object center position (x, y, z) in world coordinates
            object_rotation: Object rotation quaternion (x, y, z, w)
            object_size: Object dimensions (width, height, depth) in meters
            gripper_position: Current gripper position (optional, for scoring)

        Returns:
            List of GraspCandidate instances
        """
        candidates = []

        obj_pos = np.array(object_position)
        obj_rot = object_rotation
        obj_size = np.array(object_size)

        # Calculate base pre-grasp distance from object size
        base_pre_grasp_dist = self._calculate_pre_grasp_distance(obj_size)

        # Generate candidates for each enabled approach
        for approach_settings in self.config.enabled_approaches:
            if not approach_settings.enabled:
                continue

            approach_type = approach_settings.approach_type

            approach_candidates = self._generate_candidates_for_approach(
                approach_type,
                obj_pos,
                obj_rot,
                obj_size,
                base_pre_grasp_dist,
            )

            candidates.extend(approach_candidates)

        logger.debug(
            f"Generated {len(candidates)} grasp candidates "
            f"({self.config.candidates_per_approach} per approach)"
        )

        return candidates

    def _generate_candidates_for_approach(
        self,
        approach: str,
        obj_pos: np.ndarray,
        obj_rot: Tuple[float, float, float, float],
        obj_size: np.ndarray,
        base_pre_grasp_dist: float,
    ) -> List[GraspCandidate]:
        """
        Generate grasp candidates for a specific approach type.

        Args:
            approach: Approach type ("top", "front", "side")
            obj_pos: Object position (x, y, z)
            obj_rot: Object rotation quaternion (x, y, z, w)
            obj_size: Object size (width, height, depth)
            base_pre_grasp_dist: Base pre-grasp distance

        Returns:
            List of GraspCandidate instances for this approach
        """
        candidates = []

        # Get dimension along approach axis (constant per approach type)
        dimension_on_axis = self._get_dimension_on_axis(approach, obj_size)

        # Generate N candidates with variations.
        # _get_approach_basis is called inside the loop so that side_sign and
        # front_sign are re-sampled per candidate, giving each candidate an
        # independent chance of coming from either side (±X) or either end
        # (±Z). Calling it once outside the loop would lock all candidates for
        # this approach batch to the same side, halving effective diversity.
        for i in range(self.config.candidates_per_approach):
            # Re-sample approach basis per candidate for directional diversity
            approach_axis_world, approach_tangent_world = self._get_approach_basis(
                approach, obj_rot
            )

            # Sample variations
            dist_var = self._sample_distance_variation(base_pre_grasp_dist)
            angle_var = self._sample_angle_variation()
            depth_var = self._sample_depth_variation(obj_size)

            # Perturb approach direction
            perturbed_approach_dir = self._perturb_direction(
                approach_axis_world, approach_tangent_world
            )

            # Calculate grasp point
            grasp_point = obj_pos + perturbed_approach_dir * (
                dimension_on_axis * 0.5 + depth_var
            )

            # Calculate gripper rotation
            grasp_rotation = self._calculate_gripper_rotation(
                approach, perturbed_approach_dir, approach_tangent_world, obj_rot, angle_var
            )

            # Calculate pre-grasp position
            pre_grasp_pos = grasp_point + perturbed_approach_dir * dist_var

            # Calculate retreat position
            if self.config.enable_retreat:
                retreat_dist = self._calculate_retreat_distance(obj_size)
                retreat_pos = grasp_point + perturbed_approach_dir * retreat_dist
            else:
                retreat_pos = grasp_point

            # Create candidate
            candidate = GraspCandidate.create(
                tuple(pre_grasp_pos),
                grasp_rotation,
                tuple(grasp_point),
                grasp_rotation,
                approach,
            )

            # Set additional properties
            candidate.retreat_position = tuple(retreat_pos)
            candidate.retreat_rotation = grasp_rotation
            candidate.approach_distance = dist_var
            candidate.grasp_depth = depth_var
            candidate.contact_point_estimate = tuple(grasp_point)
            candidate.approach_direction = tuple(perturbed_approach_dir)

            # Compute antipodal score
            candidate.antipodal_score = self._compute_antipodal_score(
                grasp_point, grasp_rotation, obj_pos, obj_size, approach
            )

            candidates.append(candidate)

        return candidates

    def _get_approach_basis(
        self, approach: str, obj_rot: Tuple[float, float, float, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the primary approach axis and a reference tangent in world space.

        Respects the object's rotation by transforming local axes to world space.

        Args:
            approach: Approach type ("top", "front", "side")
            obj_rot: Object rotation quaternion (x, y, z, w)

        Returns:
            Tuple of (approach_axis_world, tangent_world) as normalized vectors
        """
        if approach == "top":
            # Top: approach from above (Y-axis), tangent is right (X-axis)
            axis_local = np.array([0.0, 1.0, 0.0])
            tangent_local = np.array([1.0, 0.0, 0.0])
        elif approach == "side":
            # Side: approach from left or right (X-axis), tangent is up (Y-axis)
            side_sign = 1.0 if self.random.rand() > 0.5 else -1.0
            axis_local = np.array([side_sign, 0.0, 0.0])
            tangent_local = np.array([0.0, 1.0, 0.0])
        elif approach == "front":
            # Front: approach from front or back (Z-axis), tangent is up (Y-axis)
            front_sign = 1.0 if self.random.rand() > 0.5 else -1.0
            axis_local = np.array([0.0, 0.0, front_sign])
            tangent_local = np.array([0.0, 1.0, 0.0])
        else:
            # Default to top
            axis_local = np.array([0.0, 1.0, 0.0])
            tangent_local = np.array([1.0, 0.0, 0.0])

        # Transform to world space using object rotation
        axis_world = quaternion_rotate_vector(obj_rot, axis_local)
        tangent_world = quaternion_rotate_vector(obj_rot, tangent_local)

        return vector_normalize(axis_world), vector_normalize(tangent_world)

    def _get_dimension_on_axis(self, approach: str, obj_size: np.ndarray) -> float:
        """
        Get object dimension along approach axis.

        Args:
            approach: Approach type ("top", "front", "side")
            obj_size: Object size (width, height, depth)

        Returns:
            Dimension in meters
        """
        # Unity convention: size = (width=X, height=Y, depth=Z)
        if approach == "top":
            return float(obj_size[1])  # height
        elif approach == "side":
            return float(obj_size[0])  # width
        elif approach == "front":
            return float(obj_size[2])  # depth
        else:
            return float(obj_size[1])  # default to height

    def _perturb_direction(
        self, main_axis: np.ndarray, tangent: np.ndarray
    ) -> np.ndarray:
        """
        Perturb a direction vector within a small cone.

        Uses random cone sampling to generate varied approach directions.

        Args:
            main_axis: Primary approach direction
            tangent: Tangent vector for perturbation

        Returns:
            Perturbed direction vector (normalized)
        """
        # Random angle within cone (in degrees)
        perturbation_angle = (self.random.rand() * 2.0 - 1.0) * self.config.angle_variation_range
        perturbation_roll = self.random.rand() * 360.0

        # Convert to radians
        angle_rad = np.radians(perturbation_angle)
        roll_rad = np.radians(perturbation_roll)

        # Rotate around tangent (cone angle)
        rot_tangent = quaternion_from_axis_angle(tangent, angle_rad)
        perturbed = quaternion_rotate_vector(rot_tangent, main_axis)

        # Roll around main axis
        roll_quat = quaternion_from_axis_angle(main_axis, roll_rad)
        final = quaternion_rotate_vector(roll_quat, perturbed)

        return vector_normalize(final)

    def _calculate_gripper_rotation(
        self,
        approach: str,
        approach_dir: np.ndarray,
        tangent: np.ndarray,
        obj_rot: Tuple[float, float, float, float],
        angle_var: float,
    ) -> Tuple[float, float, float, float]:
        """
        Calculate gripper rotation for approach type.

        Accounts for URDF gripper coordinate frame (90° Z-rotation baked in).
        Matches Unity's Quaternion.Euler conventions.

        Args:
            approach: Approach type ("top", "front", "side")
            approach_dir: World-space approach direction
            tangent: World-space tangent for roll reference
            obj_rot: Object rotation quaternion (x, y, z, w)
            angle_var: Angle variation in degrees

        Returns:
            Gripper rotation quaternion (x, y, z, w)
        """
        # Calculate base rotation in object-local space
        if approach == "top":
            # Top: point gripper down with 90° Z-offset for URDF
            base_rotation = quaternion_from_euler(
                np.radians(180.0 + angle_var), 0.0, np.radians(90.0)
            )
        elif approach == "side":
            # Side: determine left vs right from local approach direction
            local_approach_dir = quaternion_rotate_vector(
                quaternion_inverse(obj_rot), approach_dir
            )
            side_angle = -90.0 if local_approach_dir[0] > 0 else 90.0
            base_rotation = quaternion_from_euler(
                np.radians(angle_var), np.radians(side_angle), 0.0
            )
        elif approach == "front":
            # Front: determine forward vs backward from local approach direction
            local_front_dir = quaternion_rotate_vector(
                quaternion_inverse(obj_rot), approach_dir
            )
            front_angle = 180.0 if local_front_dir[2] > 0 else 0.0
            base_rotation = quaternion_from_euler(
                np.radians(angle_var), np.radians(front_angle), 0.0
            )
        else:
            base_rotation = quaternion_identity()

        # Transform to world space
        world_rotation = quaternion_multiply(obj_rot, base_rotation)

        return world_rotation

    def _compute_antipodal_score(
        self,
        grasp_pos: np.ndarray,
        grasp_rot: Tuple[float, float, float, float],
        obj_center: np.ndarray,
        obj_size: np.ndarray,
        approach: str,
    ) -> float:
        """
        Compute antipodal grasp quality score.

        Approach-aware: Top approaches use centering-only scoring since traditional
        antipodal (opposing contact points) concept doesn't apply to top-down grasps.

        Args:
            grasp_pos: Grasp position
            grasp_rot: Grasp rotation quaternion
            obj_center: Object center position
            obj_size: Object size (width, height, depth)
            approach: Approach type

        Returns:
            Antipodal score [0, 1]
        """
        to_center = obj_center - grasp_pos

        if approach == "top":
            # Top approach: use horizontal centering + vertical alignment
            horizontal_offset = np.array([to_center[0], to_center[2]])  # X-Z plane
            max_horizontal_extent = max(obj_size[0], obj_size[2]) * 0.5

            if max_horizontal_extent > 1e-6:
                centering_score = 1.0 - np.clip(
                    np.linalg.norm(horizontal_offset) / max_horizontal_extent, 0.0, 1.0
                )
            else:
                centering_score = 1.0

            # Vertical alignment (pointing down)
            vertical_alignment = abs(vector_dot(vector_normalize(to_center), np.array([0.0, -1.0, 0.0])))
            vertical_score = np.clip(vertical_alignment, 0.0, 1.0)

            return 0.3 + (centering_score * 0.4) + (vertical_score * 0.3)

        # Side/Front approach: use approach alignment + centering
        closing_axis = quaternion_rotate_vector(grasp_rot, np.array([1.0, 0.0, 0.0]))
        approach_axis = quaternion_rotate_vector(grasp_rot, np.array([0.0, 0.0, 1.0]))

        alignment_dot = vector_dot(vector_normalize(to_center), approach_axis)
        pointing_score = np.clip(alignment_dot, 0.0, 1.0)

        dist_from_center_line = np.linalg.norm(vector_cross(to_center, approach_axis))
        max_extent = max(obj_size[0], obj_size[2]) * 0.5

        if max_extent > 1e-6:
            side_centering_score = 1.0 - np.clip(dist_from_center_line / max_extent, 0.0, 1.0)
        else:
            side_centering_score = 1.0

        return (pointing_score * 0.6) + (side_centering_score * 0.4)

    def _calculate_pre_grasp_distance(self, obj_size: np.ndarray) -> float:
        """
        Calculate pre-grasp distance from object size.

        Args:
            obj_size: Object size (width, height, depth)

        Returns:
            Pre-grasp distance in meters
        """
        avg_size = np.mean(obj_size)
        distance = avg_size * self.config.pre_grasp_distance_factor
        return float(np.clip(
            distance,
            self.config.min_pre_grasp_distance,
            self.config.max_pre_grasp_distance,
        ))

    def _calculate_retreat_distance(self, obj_size: np.ndarray) -> float:
        """
        Calculate retreat distance from object size.

        Args:
            obj_size: Object size (width, height, depth)

        Returns:
            Retreat distance in meters
        """
        avg_size = np.mean(obj_size)
        return float(avg_size * self.config.retreat_distance_factor)

    def _sample_distance_variation(self, base_dist: float) -> float:
        """
        Sample distance variation for pre-grasp distance.

        Args:
            base_dist: Base pre-grasp distance

        Returns:
            Varied distance in meters
        """
        variation = (self.random.rand() * 2.0 - 1.0) * self.config.distance_variation_range
        distance = base_dist * (1.0 + variation)
        return np.clip(
            distance,
            self.config.min_pre_grasp_distance,
            self.config.max_pre_grasp_distance,
        )

    def _sample_angle_variation(self) -> float:
        """
        Sample angle variation for gripper rotation.

        Returns:
            Angle variation in degrees
        """
        return (self.random.rand() * 2.0 - 1.0) * self.config.angle_variation_range

    def _sample_depth_variation(self, obj_size: np.ndarray) -> float:
        """
        Sample depth variation for grasp penetration.

        Args:
            obj_size: Object size (width, height, depth)

        Returns:
            Depth variation in meters
        """
        avg_size = np.mean(obj_size)
        base_depth = self.config.target_grasp_depth * avg_size
        variation = avg_size * self.config.depth_variation_range
        return float(base_depth + (self.random.rand() * variation * 2.0 - variation))
