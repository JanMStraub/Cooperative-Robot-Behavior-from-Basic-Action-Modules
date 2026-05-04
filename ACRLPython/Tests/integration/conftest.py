#!/usr/bin/env python3
"""
Integration test conftest
=========================

Adds the `helpers/` sub-directory to sys.path so that all integration tests
can import shared utilities with a flat import::

    from backend_client import BackendClient, backend_available

This avoids the need for __init__.py files in the test directories (which the
project intentionally omits — see memory/MEMORY.md) while still providing a
single source of truth for the BackendClient.
"""

import sys
from pathlib import Path

import pytest

_helpers_dir = Path(__file__).parent / "helpers"
if str(_helpers_dir) not in sys.path:
    sys.path.insert(0, str(_helpers_dir))


@pytest.fixture(autouse=True, scope="session")
def reset_simulation_at_session_start():
    """Reset simulation once at the start of the integration test session.

    Ensures robots are at home positions and scene objects are at their initial
    positions before any test runs, regardless of prior state.
    Skipped silently if the backend is not available (unit-test runs).
    """
    from backend_client import backend_available, reset_simulation  # type: ignore[import]

    if backend_available():
        reset_simulation(timeout=20.0)
    yield
