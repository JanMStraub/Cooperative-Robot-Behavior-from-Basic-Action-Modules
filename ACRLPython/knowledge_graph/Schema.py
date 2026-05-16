#!/usr/bin/env python3
"""Dataclass node models for the knowledge graph (no Pydantic dependency)."""

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class RobotNode:

    node_id: str
    node_type: str = "robot"
    position: Optional[Tuple[float, float, float]] = None
    workspace_region: Optional[str] = None
    gripper_state: str = "unknown"
    is_moving: bool = False
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
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

    node_id: str
    node_type: str = "region"
    bounds: Optional[dict] = None
    allocated_to: Optional[str] = None

    def to_dict(self):
        return {
            "node_type": self.node_type,
            "bounds": self.bounds,
            "allocated_to": self.allocated_to,
        }
