#!/usr/bin/env python3
"""
Dry-run mock for OperationRegistry.

Intercepts get_operation_by_name() on the global registry singleton and returns
_MockOperation wrappers that return OperationResult without contacting Unity.

No pytest dependency — safe for standalone use.
"""

from __future__ import annotations

import random
import time
from typing import Callable, Optional

PROFILES = ("always_succeed", "10pct_failure", "detect_fails")


class _MockOperation:
    """
    Wraps a real BasicOperation, replacing execute() with a configurable stub.

    Copies all attributes from the real operation so that SequenceExecutor's
    category/name checks (e.g. for Reflexion) still work correctly.
    """

    def __init__(self, real_op, profile: str) -> None:
        """
        Initialise mock from real operation.

        Args:
            real_op: The real BasicOperation instance.
            profile: One of PROFILES controlling stub behaviour.
        """
        self.__dict__.update(real_op.__dict__)
        self._profile = profile

    def execute(self, **kwargs):
        """
        Return a stubbed OperationResult.

        Simulate ~50ms latency. detect_* ops return rich result dicts so that
        variable chaining ($target.x / $target.y / $target.z / $target.color)
        resolves correctly in dry-run.
        """
        from operations.Base import OperationResult

        time.sleep(0.05)
        name: str = getattr(self, "name", "")

        if self._profile == "always_succeed":
            if "detect" in name:
                return OperationResult.success_result(
                    {"x": 0.3, "y": 0.15, "z": 0.1, "color": "red_cube", "confidence": 0.95}
                )
            return OperationResult.success_result({"mock": True})

        if self._profile == "10pct_failure":
            if random.random() < 0.10:
                return OperationResult.error_result(
                    "MOCK_FAIL", "Simulated 10% random failure", []
                )
            return OperationResult.success_result({"mock": True})

        if self._profile == "detect_fails":
            if "detect" in name:
                return OperationResult.error_result(
                    "DETECTION_FAILED", "Mock: detection always fails in this profile", []
                )
            return OperationResult.success_result({"mock": True})

        return OperationResult.success_result({"mock": True})


def install_mock(profile: str = "always_succeed") -> Callable:
    """
    Monkeypatch get_operation_by_name on the global registry singleton.

    Args:
        profile: Stub profile — one of PROFILES.

    Returns:
        The original get_operation_by_name method for later restoration.
    """
    from core.Imports import get_global_registry

    if profile not in PROFILES:
        raise ValueError(f"Unknown mock profile '{profile}'. Choose from: {PROFILES}")

    registry = get_global_registry()
    original = registry.get_operation_by_name

    def _patched(name: str) -> Optional[_MockOperation]:
        real = original(name)
        return _MockOperation(real, profile) if real is not None else None

    registry.get_operation_by_name = _patched  # type: ignore[method-assign]
    return original


def restore_mock(original: Callable) -> None:
    """
    Restore get_operation_by_name to original implementation.

    Args:
        original: The callable returned by install_mock().
    """
    from core.Imports import get_global_registry

    get_global_registry().get_operation_by_name = original
