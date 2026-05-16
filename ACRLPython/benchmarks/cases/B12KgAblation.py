#!/usr/bin/env python3
"""B12: Knowledge Graph Ablation — spatial context enrichment vs none."""

from __future__ import annotations

from typing import List

# Objects pre-loaded into synthetic KG for this ablation.
# IDs match Unity GameObjects under the Targets hierarchy.
KG_OBJECTS = (
    "red_cube",
    "blue_cube",
    "yellow_cube",
    "purple_cube",
    "orange_cube",
    "cyan_cube",
    "green_cube",
    "magenta_cube",
)
KG_ROBOT_NEARBY = "Robot2"


def get_tasks(config=None) -> List[str]:
    """
    Return NL tasks that exercise spatial/handoff reasoning.

    With KG enabled, _get_spatial_context() injects reachability data, so the
    LLM should reference the exact object IDs from KG_OBJECTS. Without it,
    the LLM has only the command text and may hallucinate object identifiers.

    """

    return [
        f"Robot2: Detect the nearest object and move to it.",
        f"Robot1: Grasp the red cube and hand it to the other robot.",
        f"Robot1: Find the reachable object closest to you and lift it.",
        f"Robot1: Pass the red cube to Robot2.",
    ]


def populate_synthetic_kg(robot_id: str) -> None:
    """
    Insert synthetic spatial state into the KG singleton so _get_spatial_context
    returns meaningful reachability data during parse-only ablation.

    Adds all KG_OBJECTS as reachable ObjectNodes spread across the workspace,
    and adds KG_ROBOT_NEARBY as a RobotNode 0.5m away.

    """
    from knowledge_graph._singleton import get_knowledge_graph

    kg = get_knowledge_graph()

    # Spread objects across the workspace surface (y=0.0 = desk height)
    positions = [
        (0.20, 0.0, 0.10),
        (0.25, 0.0, -0.10),
        (0.30, 0.0, 0.20),
        (-0.20, 0.0, 0.10),
        (-0.25, 0.0, -0.10),
        (0.10, 0.0, -0.20),
        (-0.10, 0.0, 0.25),
        (0.0, 0.0, 0.0),
    ]

    for obj_id, pos in zip(KG_OBJECTS, positions):
        color = obj_id.split("_")[0]
        kg.add_node(obj_id, node_type="object", color=color, position=pos)
        kg.add_edge(
            robot_id,
            obj_id,
            "CAN_REACH",
            distance=round((pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2) ** 0.5, 2),
        )

    kg.add_node(KG_ROBOT_NEARBY, node_type="robot", position=(0.5, 0.0, 0.0))
    kg.add_edge(robot_id, KG_ROBOT_NEARBY, "NEAR", distance=0.5)


def clear_synthetic_kg() -> None:
    """Remove synthetic nodes added by populate_synthetic_kg."""
    from knowledge_graph._singleton import get_knowledge_graph

    kg = get_knowledge_graph()
    for obj_id in KG_OBJECTS:
        kg.remove_node(obj_id)
    kg.remove_node(KG_ROBOT_NEARBY)
