"""Backward-compatible re-export - implementation in operations/grasp/."""

try:
    from .grasp import (
        GRASP_OBJECT_OPERATION,
        RECEIVE_HANDOFF_OPERATION,
        grasp_object,
        receive_handoff,
    )
    from .grasp._dispatcher import _get_command_broadcaster
    from .grasp._vgn import _grasp_via_vgn, _grasp_via_vgn_with_ros
    from config.Robot import GRASP_TCP_OFFSET
except ImportError:
    from operations.grasp import (  # type: ignore[no-redef]
        GRASP_OBJECT_OPERATION,
        RECEIVE_HANDOFF_OPERATION,
        grasp_object,
        receive_handoff,
    )
    from operations.grasp._dispatcher import _get_command_broadcaster  # type: ignore[no-redef]
    from operations.grasp._vgn import _grasp_via_vgn, _grasp_via_vgn_with_ros  # type: ignore[no-redef]
    from config.Robot import GRASP_TCP_OFFSET  # type: ignore[no-redef]


def _compute_handoff_approach_vector(
    object_position: tuple,
    object_dimensions: tuple,
    receiving_robot_position: tuple,
) -> tuple:
    """Return a unit axis vector for the handoff approach direction.

    Picks the wider horizontal axis (X vs Z) and approaches from the side
    opposite the receiving robot so the gripper clears the object extent.
    """
    x_dim, _, z_dim = object_dimensions
    if x_dim >= z_dim:
        sign = -1.0 if receiving_robot_position[0] >= object_position[0] else 1.0
        return (sign, 0.0, 0.0)
    else:
        sign = -1.0 if receiving_robot_position[2] >= object_position[2] else 1.0
        return (0.0, 0.0, sign)


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
    "_compute_handoff_approach_vector",
]
