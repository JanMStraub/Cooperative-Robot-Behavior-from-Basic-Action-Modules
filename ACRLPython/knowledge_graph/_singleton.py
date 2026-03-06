"""
Knowledge Graph Singleton Holders
===================================

Module-level singletons for KnowledgeGraph and GraphQueryEngine.
Python's module cache guarantees these live for the entire process lifetime,
so any caller that imports get_query_engine() gets the same instance.

Usage:
    from knowledge_graph._singleton import get_query_engine, get_knowledge_graph
"""

from typing import Optional

from .Core import KnowledgeGraph
from .QueryEngine import GraphQueryEngine

_kg: Optional[KnowledgeGraph] = None
_query_engine: Optional[GraphQueryEngine] = None


def get_query_engine() -> GraphQueryEngine:
    """
    Get (or lazily create) the GraphQueryEngine singleton.

    Returns:
        GraphQueryEngine wrapping the shared KnowledgeGraph instance
    """
    global _kg, _query_engine
    if _query_engine is None:
        _kg = KnowledgeGraph()
        _query_engine = GraphQueryEngine(_kg)
    return _query_engine


def get_knowledge_graph() -> KnowledgeGraph:
    """
    Get (or lazily create) the KnowledgeGraph singleton.

    Ensures _query_engine is also initialized (they share the same graph).

    Returns:
        KnowledgeGraph instance used by the singleton query engine
    """
    get_query_engine()  # ensures _kg is initialised
    assert _kg is not None
    return _kg
