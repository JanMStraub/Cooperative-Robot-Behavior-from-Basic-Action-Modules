"""Grasp sub-package - public API."""

try:
    from ._dispatcher import GRASP_OBJECT_OPERATION, grasp_object  # type: ignore[import]
    from ._handoff import RECEIVE_HANDOFF_OPERATION, receive_handoff  # type: ignore[import]
except ImportError:
    from operations.grasp._dispatcher import GRASP_OBJECT_OPERATION, grasp_object  # type: ignore[no-redef]
    from operations.grasp._handoff import RECEIVE_HANDOFF_OPERATION, receive_handoff  # type: ignore[no-redef]

__all__ = [
    "grasp_object",
    "receive_handoff",
    "GRASP_OBJECT_OPERATION",
    "RECEIVE_HANDOFF_OPERATION",
]
