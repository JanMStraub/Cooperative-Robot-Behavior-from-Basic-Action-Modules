#!/usr/bin/env python3
"""
Knowledge Graph Singleton Holders
===================================

Module-level singletons for KnowledgeGraph and GraphQueryEngine.
Python's module cache guarantees these live for the entire process lifetime,
so any caller that imports get_query_engine() gets the same instance.

Usage:
    from knowledge_graph._singleton import get_query_engine, get_knowledge_graph
"""

import threading
from typing import Optional

from .Core import KnowledgeGraph
from .QueryEngine import GraphQueryEngine

_kg: Optional[KnowledgeGraph] = None
_query_engine: Optional[GraphQueryEngine] = None
_init_lock = threading.Lock()


def get_query_engine() -> GraphQueryEngine:
    """
    Get (or lazily create) the GraphQueryEngine singleton.

    Uses double-checked locking so concurrent callers don't race to create
    two separate KnowledgeGraph instances.

    Returns:
        GraphQueryEngine wrapping the shared KnowledgeGraph instance
    """
    global _kg, _query_engine
    if _query_engine is None:
        with _init_lock:
            if _query_engine is None:  # double-checked locking
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
    if _kg is None:
        raise RuntimeError(
            "KnowledgeGraph singleton was not initialized. Call get_query_engine() first."
        )
    return _kg
