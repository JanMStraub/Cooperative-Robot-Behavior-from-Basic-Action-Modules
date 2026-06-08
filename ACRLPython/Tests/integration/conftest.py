import sys
from pathlib import Path

import pytest

# helpers/ has no __init__.py; add it directly so tests can do:
#   from BackendClient import BackendClient, backend_available
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
    from BackendClient import backend_available, reset_simulation  # type: ignore[import]

    if backend_available():
        reset_simulation(timeout=20.0)
    yield
