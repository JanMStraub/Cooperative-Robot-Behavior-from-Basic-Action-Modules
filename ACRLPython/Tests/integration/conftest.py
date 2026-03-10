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

_helpers_dir = Path(__file__).parent / "helpers"
if str(_helpers_dir) not in sys.path:
    sys.path.insert(0, str(_helpers_dir))
