#!/usr/bin/env python3
"""Converts grasp poses from right-handed OpenCV/camera frame to Unity left-handed world frame (pure NumPy)."""

import logging
from typing import List

import numpy as np

from core.LoggingSetup import setup_logging

setup_logging(__name__)
logger = logging.getLogger(__name__)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def _quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qvec = q[:3]
    w = q[3]
    t = 2.0 * np.cross(qvec, v)
    return v + w * t + np.cross(qvec, t)


def _normalise_quat(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transform_grasp_poses_to_unity(
    grasps: List[dict],
    camera_position: List[float],
    camera_rotation: List[float],
) -> List[dict]:
    """Transform VGNClient grasp poses from right-handed camera frame to Unity world frame. Malformed grasps silently skipped."""
    cam_pos = np.array(camera_position, dtype=np.float64)
    cam_rot_raw = np.array(camera_rotation, dtype=np.float64)
    if cam_rot_raw.shape[0] == 3:
        # Legacy path: Unity sent Euler angles (degrees) instead of a quaternion.
        # Convert XYZ Euler (degrees) → quaternion [x, y, z, w].
        import math

        rx, ry, rz = [math.radians(a) for a in cam_rot_raw]
        cx, sx = math.cos(rx / 2), math.sin(rx / 2)
        cy, sy = math.cos(ry / 2), math.sin(ry / 2)
        cz, sz = math.cos(rz / 2), math.sin(rz / 2)
        cam_rot_raw = np.array(
            [
                sx * cy * cz + cx * sy * sz,
                cx * sy * cz - sx * cy * sz,
                cx * cy * sz + sx * sy * cz,
                cx * cy * cz - sx * sy * sz,
            ],
            dtype=np.float64,
        )
        logger.warning(
            "camera_rotation had 3 components (Euler angles); converted to quaternion. "
            "Update StereoCameraController to send rotation.xyzw instead of eulerAngles."
        )
    cam_rot = _normalise_quat(cam_rot_raw)

    transformed: List[dict] = []

    for i, grasp in enumerate(grasps):
        try:
            pos_cam = np.array(grasp["position"], dtype=np.float64)
            rot_cam_raw = np.array(grasp["rotation"], dtype=np.float64)
            if rot_cam_raw.shape[0] == 3:
                # Grasp rotation arrived as 3-element Euler angles (degrees) instead of
                # a quaternion [x, y, z, w].  Convert using the same path used for
                # camera_rotation above.
                import math

                rx, ry, rz = [math.radians(a) for a in rot_cam_raw]
                cx, sx = math.cos(rx / 2), math.sin(rx / 2)
                cy, sy = math.cos(ry / 2), math.sin(ry / 2)
                cz, sz = math.cos(rz / 2), math.sin(rz / 2)
                rot_cam_raw = np.array(
                    [
                        sx * cy * cz + cx * sy * sz,
                        cx * sy * cz - sx * cy * sz,
                        cx * cy * sz + sx * sy * cz,
                        cx * cy * cz - sx * sy * sz,
                    ],
                    dtype=np.float64,
                )
                logger.warning(
                    f"Grasp #{i} rotation had 3 components (Euler angles); converted to quaternion."
                )
            rot_cam = _normalise_quat(rot_cam_raw)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            logger.warning(f"Skipping malformed grasp #{i}: {exc}")
            continue

        # Step 1: RH → LH handedness flip: negate X axis.
        # Camera frame is right-handed (OpenCV: X-right, Y-down, Z-forward).
        # Unity world frame is left-handed (X-right, Y-up, Z-forward).
        # Negating X converts between the two conventions.
        pos_de = pos_cam * np.array([-1.0, 1.0, 1.0])
        rot_de = rot_cam * np.array([-1.0, 1.0, 1.0, 1.0])
        rot_de = _normalise_quat(rot_de)

        # Step 2: Rotate the flipped position by the camera quaternion and add
        # the camera world translation.
        pos_world = _quat_rotate_vector(cam_rot, pos_de) + cam_pos

        # Step 3: Compose rotations
        rot_world = _normalise_quat(_quat_multiply(cam_rot, rot_de))

        # Step 4: Transform approach direction (if present)
        approach_cam_raw = grasp.get("approach_direction")
        if approach_cam_raw is not None:
            approach_cam = np.array(approach_cam_raw, dtype=np.float64)
            approach_flipped = approach_cam * np.array([-1.0, 1.0, 1.0])
            approach_world = _quat_rotate_vector(cam_rot, approach_flipped)
            approach_list = approach_world.tolist()
        else:
            # Derive from quaternion Z-axis = approach direction
            z_axis_local = np.array([0.0, 0.0, 1.0])
            approach_world = _quat_rotate_vector(rot_world, z_axis_local)
            approach_list = approach_world.tolist()

        transformed.append(
            {
                "position": pos_world.tolist(),
                "rotation": rot_world.tolist(),
                "approach_direction": approach_list,
                "score": float(grasp.get("score", 0.0)),
                "width": float(grasp.get("width", 0.05)),
            }
        )

    logger.debug(
        f"Transformed {len(transformed)}/{len(grasps)} grasp poses to Unity frame"
    )
    return transformed
