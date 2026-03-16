#!/usr/bin/env python3
"""
Grasp planning system for ROS-MoveIt integration.

This package provides grasp planning capabilities that integrate with MoveIt
for collision-aware trajectory planning. The system generates grasp candidates,
validates them using MoveIt's IK service, scores them using multi-criteria
evaluation, and selects the best grasp for execution.

Key components:
- GraspCandidate: Data structure for grasp poses
- GraspConfig: Configuration for candidate generation and scoring
- GraspCandidateGenerator: Generates grasp candidates with approach variation
- GraspScorer: Multi-criteria scoring system
- GraspPlanner: End-to-end orchestration
"""

from .GraspCandidate import GraspCandidate
from .GraspConfig import GraspConfig
from .GraspCandidateGenerator import GraspCandidateGenerator
from .GraspScorer import GraspScorer
from .GraspPlanner import GraspPlanner

__all__ = [
    "GraspCandidate",
    "GraspConfig",
    "GraspCandidateGenerator",
    "GraspScorer",
    "GraspPlanner",
]
