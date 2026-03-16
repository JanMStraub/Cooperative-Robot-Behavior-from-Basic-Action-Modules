#!/usr/bin/env python3
"""
SingletonBase - Thread-safe singleton base class.

Provides double-checked locking for all singleton managers in the ACRL
system. Subclasses override ``_singleton_init()`` instead of ``__init__``.

Usage::

    class MyManager(SingletonBase):
        def _singleton_init(self):
            self._data = []
            ...
"""

import threading


class SingletonBase:
    """
    Thread-safe singleton base using double-checked locking.

    Subclasses should override ``_singleton_init()`` to perform one-time
    initialisation instead of ``__init__``. The base class guarantees
    ``_singleton_init()`` is called exactly once, even under concurrent
    construction.
    """

    _instance = None
    _singleton_lock = threading.RLock()

    def __new__(cls):
        """Return the singleton instance, creating it on first call."""
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._singleton_initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self):
        """Guard-wrapped init: delegates to _singleton_init() exactly once."""
        if self._singleton_initialized:
            return
        with self._singleton_lock:
            if self._singleton_initialized:
                return
            self._singleton_initialized = True
            self._singleton_init()

    def _singleton_init(self):
        """Override in subclasses to perform one-time initialisation."""
        pass
