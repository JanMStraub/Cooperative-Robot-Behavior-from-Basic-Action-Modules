"""Backward-compatible re-export — implementation in operations/grasp/."""

try:
    from .grasp import (
        GRASP_OBJECT_OPERATION,
        RECEIVE_HANDOFF_OPERATION,
        grasp_object,
        receive_handoff,
    )
    from .grasp._dispatcher import _get_command_broadcaster
    from .grasp._vgn import _grasp_via_vgn, _grasp_via_vgn_with_ros
    from .grasp._helpers import GRASP_TCP_OFFSET
except ImportError:
    from operations.grasp import (  # type: ignore[no-redef]
        GRASP_OBJECT_OPERATION,
        RECEIVE_HANDOFF_OPERATION,
        grasp_object,
        receive_handoff,
    )
    from operations.grasp._dispatcher import _get_command_broadcaster  # type: ignore[no-redef]
    from operations.grasp._vgn import _grasp_via_vgn, _grasp_via_vgn_with_ros  # type: ignore[no-redef]
    from operations.grasp._helpers import GRASP_TCP_OFFSET  # type: ignore[no-redef]

# _build_segmentation_mask: used by VGNClient; was re-exported from here for test patching
try:
    from .GraspUtils import _build_segmentation_mask  # noqa: F401
except ImportError:
    from operations.GraspUtils import _build_segmentation_mask  # type: ignore[no-redef]  # noqa: F401

__all__ = [
    "grasp_object",
    "receive_handoff",
    "GRASP_OBJECT_OPERATION",
    "RECEIVE_HANDOFF_OPERATION",
    "_get_command_broadcaster",
    "_grasp_via_vgn",
    "_grasp_via_vgn_with_ros",
    "GRASP_TCP_OFFSET",
    "_build_segmentation_mask",
]
