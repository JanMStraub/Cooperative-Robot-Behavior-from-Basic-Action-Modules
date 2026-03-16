#!/usr/bin/env python3
"""
Knowledge Graph Schema Definitions
===================================

Dataclass models for graph nodes. Uses Python dataclasses (not Pydantic)
to avoid adding dependencies.

Node Types:
- RobotNode: Represents a robot with position, workspace, gripper state
- ObjectNode: Represents a detected object with liveness tracking
- RegionNode: Represents a workspace region with allocation info
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class RobotNode:
    """
    Robot node in knowledge graph.

    Attributes:
        node_id: Unique node identifier (e.g., "Robot1")
        node_type: Always "robot"
        position: End effector position (x, y, z) in world coordinates
        workspace_region: Current workspace region (e.g., "left_workspace")
        gripper_state: "open", "closed", or "unknown"
        is_moving: True if robot is currently moving
        confidence: Detection/tracking confidence (0.0 - 1.0)
        timestamp: Time of last update
    """

    node_id: str
    node_type: str = "robot"
    position: Optional[Tuple[float, float, float]] = None
    workspace_region: Optional[str] = None
    gripper_state: str = "unknown"
    is_moving: bool = False
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        """Convert to dictionary for NetworkX node attributes."""
        return {
            "node_type": self.node_type,
            "position": self.position,
            "workspace_region": self.workspace_region,
            "gripper_state": self.gripper_state,
            "is_moving": self.is_moving,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


@dataclass
class ObjectNode:
    """
    Object node in knowledge graph.

    Attributes:
        node_id: Unique node identifier (e.g., "RedCube")
        node_type: Always "object"
        position: Object position (x, y, z) in world coordinates
        color: Object color (e.g., "red", "blue")
        object_type: Object type (e.g., "cube", "sphere")
        is_graspable: True if object can be grasped
        grasped_by: Robot ID if currently grasped, None otherwise
        confidence: Detection confidence (0.0 - 1.0)
        stale: True if confidence < threshold (not seen recently)
        timestamp: Time of last detection
    """

    node_id: str
    node_type: str = "object"
    position: Optional[Tuple[float, float, float]] = None
    color: str = "unknown"
    object_type: str = "unknown"
    is_graspable: bool = True
    grasped_by: Optional[str] = None
    confidence: float = 1.0
    stale: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        """Convert to dictionary for NetworkX node attributes."""
        return {
            "node_type": self.node_type,
            "position": self.position,
            "color": self.color,
            "object_type": self.object_type,
            "is_graspable": self.is_graspable,
            "grasped_by": self.grasped_by,
            "confidence": self.confidence,
            "stale": self.stale,
            "timestamp": self.timestamp,
        }


@dataclass
class RegionNode:
    """
    Workspace region node in knowledge graph.

    Attributes:
        node_id: Unique node identifier (e.g., "left_workspace")
        node_type: Always "region"
        bounds: Dictionary with x_min, x_max, y_min, y_max, z_min, z_max
        allocated_to: Robot ID if region is allocated, None otherwise
    """

    node_id: str
    node_type: str = "region"
    bounds: Optional[dict] = None
    allocated_to: Optional[str] = None

    def to_dict(self):
        """Convert to dictionary for NetworkX node attributes."""
        return {
            "node_type": self.node_type,
            "bounds": self.bounds,
            "allocated_to": self.allocated_to,
        }
