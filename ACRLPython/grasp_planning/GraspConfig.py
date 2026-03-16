#!/usr/bin/env python3
"""
Grasp planning configuration.

This module defines configuration settings for the grasp planning pipeline,
including candidate generation, scoring weights, and validation parameters.

Matches Unity's GraspConfig.cs for compatibility.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
from .GraspCandidate import GripperGeometry


@dataclass
class GraspApproachSettings:
    """
    Settings for a specific grasp approach type.

    Attributes:
        approach_type: Approach direction ("top", "front", "side")
        enabled: True if this approach should be used
        preference_weight: Preference multiplier (higher = more preferred)
    """

    approach_type: str  # "top", "front", "side"
    enabled: bool = True
    preference_weight: float = 1.0


@dataclass
class GraspConfig:
    """
    Configuration for MoveIt2-inspired grasp planning pipeline.

    Controls candidate generation, filtering, scoring, and execution.
    All distances are in meters, angles in degrees.

    Attributes:
        candidates_per_approach: Number of candidates to generate per approach type
        enabled_approaches: List of enabled approach types with weights
        pre_grasp_distance_factor: Multiplier for object size to determine pre-grasp distance
        min_pre_grasp_distance: Minimum pre-grasp distance (meters)
        max_pre_grasp_distance: Maximum pre-grasp distance (meters)
        enable_retreat: Enable post-grasp retreat motion
        retreat_distance_factor: Multiplier for object size to determine retreat distance
        retreat_direction: Direction for retreat motion (typically upward)
        gripper_geometry: Gripper geometry for validation
        target_grasp_depth: Target grasp depth as fraction of object size [0-1]
        ik_score_weight: Weight for IK quality score
        approach_score_weight: Weight for approach preference score
        depth_score_weight: Weight for grasp depth score
        stability_score_weight: Weight for stability score
        antipodal_score_weight: Weight for antipodal grasp quality score
        enable_collision_checking: Enable collision checking along approach path
        collision_check_waypoints: Number of waypoints to check along approach
        collision_check_radius: Radius of sphere for collision checking (meters)
        enable_ik_validation: Enable IK validation for candidate filtering
        max_ik_validation_iterations: Maximum IK iterations for validation
        ik_validation_threshold: Distance threshold for IK convergence (meters)
        ik_rotation_tolerance: Rotation tolerance for IK convergence (degrees)
        max_joint_step_per_iteration: Maximum joint step per IK iteration (radians)
        max_reach_distance: Maximum reach distance for quick rejection (meters)
        angle_variation_range: Angular variation for candidate generation (degrees)
        distance_variation_range: Distance variation factor for candidate generation
        depth_variation_range: Depth variation factor for candidate generation
        max_pipeline_time_ms: Maximum time budget for full pipeline (milliseconds)
    """

    # Candidate generation
    candidates_per_approach: int = 8
    enabled_approaches: List[GraspApproachSettings] = field(
        default_factory=lambda: [
            GraspApproachSettings("top", True, 1.2),
            GraspApproachSettings("front", True, 1.0),
            GraspApproachSettings("side", True, 0.8),
        ]
    )

    # Approach distances
    pre_grasp_distance_factor: float = 1.5
    min_pre_grasp_distance: float = 0.05
    max_pre_grasp_distance: float = 0.15

    # Retreat motion
    enable_retreat: bool = True
    retreat_distance_factor: float = 2.0
    retreat_direction: Tuple[float, float, float] = (
        0.0,
        1.0,
        0.0,
    )  # Unity's Vector3.up

    # Gripper settings
    gripper_geometry: GripperGeometry = field(default_factory=GripperGeometry)
    target_grasp_depth: float = 0.5

    # Scoring weights
    ik_score_weight: float = 1.0
    approach_score_weight: float = 0.8
    depth_score_weight: float = 1.0
    stability_score_weight: float = 1.2
    antipodal_score_weight: float = 1.0

    # Collision checking
    enable_collision_checking: bool = True
    collision_check_waypoints: int = 5
    collision_check_radius: float = 0.03

    # IK validation
    enable_ik_validation: bool = True
    max_ik_validation_iterations: int = 50
    ik_validation_threshold: float = 0.005
    ik_rotation_tolerance: float = 20.0
    max_joint_step_per_iteration: float = 0.2
    max_reach_distance: float = 0.75

    # Candidate variation ranges
    angle_variation_range: float = 15.0  # degrees
    distance_variation_range: float = 0.3  # factor
    depth_variation_range: float = 0.2  # factor

    # Performance
    max_pipeline_time_ms: int = 200

    def get_approach_weight(self, approach: str) -> float:
        """
        Get preference weight for a specific approach type.

        Args:
            approach: Approach type ("top", "front", "side")

        Returns:
            Preference weight (0.0 if disabled)
        """
        for settings in self.enabled_approaches:
            if settings.approach_type == approach:
                return settings.preference_weight if settings.enabled else 0.0
        return 0.0

    def is_approach_enabled(self, approach: str) -> bool:
        """
        Check if a specific approach type is enabled.

        Args:
            approach: Approach type ("top", "front", "side")

        Returns:
            True if enabled
        """
        for settings in self.enabled_approaches:
            if settings.approach_type == approach:
                return settings.enabled
        return False

    @staticmethod
    def create_default() -> "GraspConfig":
        """
        Create default AR4 grasp configuration.

        Returns:
            GraspConfig with default AR4 values
        """
        return GraspConfig()

    @staticmethod
    def create_fast() -> "GraspConfig":
        """
        Create fast grasp configuration with reduced candidates.

        Returns:
            GraspConfig optimized for speed
        """
        config = GraspConfig()
        config.candidates_per_approach = 4
        config.max_pipeline_time_ms = 100
        config.enable_collision_checking = False
        return config

    @staticmethod
    def create_precise() -> "GraspConfig":
        """
        Create precise grasp configuration with more candidates.

        Returns:
            GraspConfig optimized for quality
        """
        config = GraspConfig()
        config.candidates_per_approach = 12
        config.ik_validation_threshold = 0.002
        config.max_pipeline_time_ms = 500
        return config
