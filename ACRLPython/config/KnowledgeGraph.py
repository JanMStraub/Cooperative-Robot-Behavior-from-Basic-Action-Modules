"""
Knowledge Graph Configuration
==============================

Controls the NetworkX-based knowledge graph for spatial reasoning
and multi-hop relationship queries.

Environment variable overrides:
    KNOWLEDGE_GRAPH_ENABLED=true  — activate the knowledge graph
    KG_NEAR_THRESHOLD=0.15        — override the NEAR edge distance (meters)
"""

import os

# Master switch: enable/disable knowledge graph (off by default)
KNOWLEDGE_GRAPH_ENABLED = os.environ.get("KNOWLEDGE_GRAPH_ENABLED", "false").lower() in (
    "true",
    "1",
    "yes",
)

# Distance threshold for NEAR edges (meters)
# Objects/robots closer than this threshold will be connected by a NEAR edge.
KG_NEAR_THRESHOLD = float(os.environ.get("KG_NEAR_THRESHOLD", "0.1"))
