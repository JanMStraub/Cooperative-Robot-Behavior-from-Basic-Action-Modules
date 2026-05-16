#!/usr/bin/env python3
"""Module-level singletons for KnowledgeGraph and GraphQueryEngine."""

import threading
from typing import Optional

from .Core import KnowledgeGraph
from .QueryEngine import GraphQueryEngine

_kg: Optional[KnowledgeGraph] = None
_query_engine: Optional[GraphQueryEngine] = None
_init_lock = threading.Lock()


def get_query_engine() -> GraphQueryEngine:
    """Lazily create GraphQueryEngine singleton. Double-checked locking prevents races."""
    global _kg, _query_engine
    if _query_engine is None:
        with _init_lock:
            if _query_engine is None:  # double-checked locking
                _kg = KnowledgeGraph()
                _query_engine = GraphQueryEngine(_kg)
    return _query_engine


def get_knowledge_graph() -> KnowledgeGraph:
    """Return shared KnowledgeGraph instance, initializing via get_query_engine() if needed."""
    get_query_engine()  # ensures _kg is initialised
    if _kg is None:
        raise RuntimeError(
            "KnowledgeGraph singleton was not initialized. Call get_query_engine() first."
        )
    return _kg
