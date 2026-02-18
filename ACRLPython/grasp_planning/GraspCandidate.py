"""
Grasp candidate data structures.

This module defines the core data structures for grasp planning, including
grasp candidate poses and gripper geometry specifications.

Matches Unity's GraspCandidate.cs for compatibility.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np


@dataclass
class GripperGeometry:
    """
    Gripper geometry specification for grasp validation.

    Used to check if object fits between gripper fingers.
    All dimensions are in meters.

    Attributes:
        max_width: Maximum opening width of gripper fingers
        finger_pad_width: Width of each gripper finger pad
        finger_pad_depth: Depth of gripper finger pads
        finger_length: Length of gripper finger
        finger_width: Width of gripper finger
        gripper_center_offset: Offset from wrist center to gripper center (x, y, z)
    """

    max_width: float = 0.08  # 8cm max opening
    finger_pad_width: float = 0.015  # 1.5cm finger pad
    finger_pad_depth: float = 0.02  # 2cm depth
    finger_length: float = 0.04  # 4cm finger length
    finger_width: float = 0.01  # 1cm finger width
    gripper_center_offset: Tuple[float, float, float] = (0.0, 0.0, 0.05)  # 5cm forward

    def can_grasp(self, object_size: Tuple[float, float, float]) -> bool:
        """
        Check if an object of given size can be grasped by this gripper.

        Args:
            object_size: Size of the target object (width, height, depth) in meters

        Returns:
            True if object can fit between gripper fingers
        """
        min_dimension = min(object_size)
        max_dimension = max(object_size)

        # Object must be small enough to fit in gripper opening
        # and large enough to make contact with finger pads
        return (
            max_dimension < self.max_width
            and min_dimension > self.finger_pad_width * 0.1
        )


@dataclass
class GraspCandidate:
    """
    Extended grasp candidate with scoring metadata, retreat pose, and validation flags.

    Used in MoveIt2-inspired grasp planning pipeline for multi-criteria evaluation.
    All positions are in world coordinates (meters), rotations are quaternions (x, y, z, w).

    Attributes:
        pre_grasp_position: Position to approach from (x, y, z)
        pre_grasp_rotation: Orientation at pre-grasp (x, y, z, w)
        grasp_position: Final grasp position (x, y, z)
        grasp_rotation: Final grasp orientation (x, y, z, w)
        approach_type: Approach direction ("top", "front", "side")
        retreat_position: Position to retreat to after grasp (x, y, z)
        retreat_rotation: Orientation at retreat (x, y, z, w)
        approach_distance: Distance from pre-grasp to grasp (meters)
        grasp_depth: Depth of gripper penetration into object (meters)
        contact_point_estimate: Estimated contact point (x, y, z)
        approach_direction: Unit vector of approach direction (x, y, z)
        antipodal_score: Antipodal grasp quality score [0, 1]
        ik_validated: True if IK solution found
        ik_score: IK quality score [0, 1]
        collision_validated: True if collision-free
        total_score: Weighted sum of all scores
        use_simplified_execution: True to skip pre-grasp/retreat waypoints
    """

    pre_grasp_position: Tuple[float, float, float]
    pre_grasp_rotation: Tuple[float, float, float, float]
    grasp_position: Tuple[float, float, float]
    grasp_rotation: Tuple[float, float, float, float]
    approach_type: str  # "top", "front", "side"

    retreat_position: Optional[Tuple[float, float, float]] = None
    retreat_rotation: Optional[Tuple[float, float, float, float]] = None

    approach_distance: float = 0.0
    grasp_depth: float = 0.5
    contact_point_estimate: Optional[Tuple[float, float, float]] = None
    approach_direction: Optional[Tuple[float, float, float]] = None

    antipodal_score: float = 0.0

    ik_validated: bool = False
    ik_score: float = 0.0
    collision_validated: bool = False

    total_score: float = 0.0

    use_simplified_execution: bool = False

    @property
    def is_valid(self) -> bool:
        """
        Check if candidate is valid (both IK and collision validated).

        Returns:
            True if candidate passed all validation checks
        """
        return self.ik_validated and self.collision_validated

    @staticmethod
    def create(
        pre_grasp: Tuple[float, float, float],
        pre_grasp_rot: Tuple[float, float, float, float],
        grasp: Tuple[float, float, float],
        grasp_rot: Tuple[float, float, float, float],
        approach: str,
    ) -> "GraspCandidate":
        """
        Create a basic grasp candidate with minimal information.

        Calculates approach direction and distance automatically.
        Sets retreat position to 10cm beyond pre-grasp along approach direction.

        Args:
            pre_grasp: Pre-grasp position (x, y, z)
            pre_grasp_rot: Pre-grasp rotation (x, y, z, w)
            grasp: Grasp position (x, y, z)
            grasp_rot: Grasp rotation (x, y, z, w)
            approach: Approach type ("top", "front", "side")

        Returns:
            Initialized GraspCandidate
        """
        # Calculate approach direction (pre-grasp -> grasp)
        pre_grasp_np = np.array(pre_grasp)
        grasp_np = np.array(grasp)

        approach_vec = pre_grasp_np - grasp_np
        approach_dist = float(np.linalg.norm(approach_vec))

        if approach_dist > 1e-6:
            approach_dir = approach_vec / approach_dist
        else:
            approach_dir = np.array([0.0, 0.0, 1.0])

        # Calculate retreat position (10cm beyond pre-grasp)
        retreat_pos = pre_grasp_np + approach_dir * 0.1

        return GraspCandidate(
            pre_grasp_position=pre_grasp,
            pre_grasp_rotation=pre_grasp_rot,
            grasp_position=grasp,
            grasp_rotation=grasp_rot,
            retreat_position=tuple(retreat_pos),
            retreat_rotation=grasp_rot,
            approach_type=approach,
            approach_distance=approach_dist,
            grasp_depth=0.5,
            contact_point_estimate=grasp,
            approach_direction=tuple(approach_dir),
            antipodal_score=0.0,
            ik_validated=False,
            ik_score=0.0,
            collision_validated=False,
            total_score=0.0,
            use_simplified_execution=False,
        )

    def to_dict(self) -> dict:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the grasp candidate
        """
        return {
            "pre_grasp_position": {
                "x": self.pre_grasp_position[0],
                "y": self.pre_grasp_position[1],
                "z": self.pre_grasp_position[2],
            },
            "pre_grasp_rotation": {
                "x": self.pre_grasp_rotation[0],
                "y": self.pre_grasp_rotation[1],
                "z": self.pre_grasp_rotation[2],
                "w": self.pre_grasp_rotation[3],
            },
            "grasp_position": {
                "x": self.grasp_position[0],
                "y": self.grasp_position[1],
                "z": self.grasp_position[2],
            },
            "grasp_rotation": {
                "x": self.grasp_rotation[0],
                "y": self.grasp_rotation[1],
                "z": self.grasp_rotation[2],
                "w": self.grasp_rotation[3],
            },
            "approach_type": self.approach_type,
            "approach_distance": self.approach_distance,
            "grasp_depth": self.grasp_depth,
            "antipodal_score": self.antipodal_score,
            "ik_validated": self.ik_validated,
            "ik_score": self.ik_score,
            "collision_validated": self.collision_validated,
            "total_score": self.total_score,
            "is_valid": self.is_valid,
        }
