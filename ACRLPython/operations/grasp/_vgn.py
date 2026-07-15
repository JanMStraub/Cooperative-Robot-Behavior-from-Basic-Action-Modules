#!/usr/bin/env python3
"""VGN-based grasp paths: TCP-only and VGN+ROS combined."""

import logging
import math
import time
from typing import List, Optional

from core.LoggingSetup import setup_logging

from ..Base import OperationResult
from ._helpers import (
    _execute_grasp_with_follow_target,
    _side_offset_world_xz,
    _yaw_from_world_state_or_robot,
)

setup_logging(__name__)
logger = logging.getLogger(__name__)

try:
    from ...config.Robot import (
        GRASP_DESCENT_ACCELERATION_SCALING,
        GRASP_DESCENT_VELOCITY_SCALING,
        GRASP_TCP_OFFSET,
        PRE_GRASP_CLEARANCE_Y,
        PRE_GRASP_HOVER_OFFSET,
        PREGRASP_ACCELERATION_SCALING,
        PREGRASP_VELOCITY_SCALING,
        VGN_MIN_Y_APPROACH,
    )
except ImportError:
    from config.Robot import (  # type: ignore[no-redef]
        GRASP_DESCENT_ACCELERATION_SCALING,
        GRASP_DESCENT_VELOCITY_SCALING,
        GRASP_TCP_OFFSET,
        PRE_GRASP_CLEARANCE_Y,
        PRE_GRASP_HOVER_OFFSET,
        PREGRASP_ACCELERATION_SCALING,
        PREGRASP_VELOCITY_SCALING,
        VGN_MIN_Y_APPROACH,
    )

try:
    from ...config.Vision import YOLO_MODEL_PATH as _YOLO_MODEL_PATH
except ImportError:
    from config.Vision import YOLO_MODEL_PATH as _YOLO_MODEL_PATH  # type: ignore[no-redef]

try:
    from ...core.Imports import get_command_broadcaster as _get_command_broadcaster
except ImportError:
    from core.Imports import get_command_broadcaster as _get_command_broadcaster  # type: ignore[no-redef]


def _grasp_via_vgn(
    robot_id: str,
    object_id: str,
    preferred_approach: str,
    use_advanced_planning: bool,
    pre_grasp_distance: float,
    enable_retreat: bool,
    retreat_distance: float,
    request_id: int,
    custom_approach_vector: "Optional[List[float]]" = None,
) -> "Optional[OperationResult]":
    """VGN TCP path: point cloud → YOLO bbox → VGNClient → Unity precomputed_candidates. None if unavailable."""
    import numpy as np

    try:
        from config.Servers import VGN_TOP_K
    except ImportError:
        VGN_TOP_K = 20

    from operations.GraspFrameTransform import transform_grasp_poses_to_unity
    from operations.PointCloudOperations import generate_point_cloud
    from operations.VGNClient import VGNClient

    client = VGNClient()
    if not client.is_available():
        logger.info("[VGN] Model unavailable - will use geometric fallback")
        return None

    pc_result = generate_point_cloud(robot_id=robot_id, request_id=request_id)
    if not pc_result.success:
        logger.warning(
            f"[VGN] generate_point_cloud failed ({pc_result.error}), using geometric fallback"
        )
        return None

    pc = pc_result.result
    assert pc is not None
    points_list = pc["points"]
    colors_list = pc["colors"]
    cam_pos = pc["camera_position"]
    cam_rot = pc["camera_rotation"]
    fov = pc["fov"]

    points_np = np.array(points_list, dtype=np.float32)
    colors_np = np.array(colors_list, dtype=np.uint8) if colors_list else None

    yolo_bbox: tuple = (0, 0, 0, 0)
    image_np: "Optional[np.ndarray]" = None
    img_w = 640
    img_h = 480
    try:
        from core.Imports import get_unified_image_storage
        from vision.YOLODetector import YOLODetector

        _storage = get_unified_image_storage()
        _stereo = _storage.get_latest_stereo()
        if _stereo is not None:
            _, _left_img, _, _ = _stereo
            if _left_img is not None:
                img_h, img_w = _left_img.shape[:2]
                _detector = YOLODetector(model_path=_YOLO_MODEL_PATH)
                _det_result = _detector.detect_objects(_left_img, camera_id="main")
                obj_id_lower = object_id.lower().replace("_", " ")
                for _obj in _det_result.detections:
                    color_field = getattr(_obj, "color", "").lower()
                    if obj_id_lower in color_field or color_field in obj_id_lower:
                        yolo_bbox = (
                            int(_obj.bbox_x),
                            int(_obj.bbox_y),
                            int(_obj.bbox_w),
                            int(_obj.bbox_h),
                        )
                        logger.debug(f"[VGN] YOLO bbox for {object_id}: {yolo_bbox}")
                        break
    except Exception as exc:
        logger.debug(f"[VGN] Could not get YOLO bbox (non-fatal): {exc}")

    try:
        from core.Imports import get_unified_image_storage

        storage = get_unified_image_storage()
        stereo = storage.get_latest_stereo()
        if stereo is not None:
            _, left_img, _, _ = stereo
            image_np = left_img
    except Exception as exc:
        logger.debug(
            f"[VGN] Could not retrieve stereo image for VLM (non-fatal): {exc}"
        )

    if image_np is None:
        image_np = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    grasps = client.predict_grasps(
        points=points_np,
        colors=colors_np,
        image=image_np,
        yolo_bbox=yolo_bbox,
        object_label=object_id,
        image_width=img_w,
        image_height=img_h,
        fov=fov,
        top_k=VGN_TOP_K,
        cam_pos=cam_pos,
        cam_rot=cam_rot,
    )
    if not grasps:
        logger.info("[VGN] Returned no candidates - using geometric fallback")
        return None

    logger.info(f"[VGN] Candidates received: {len(grasps)}")

    if grasps and grasps[0].get("_world_frame"):
        world_grasps = grasps
        logger.info("[VGN] Grasps already in Unity world frame - skipping transform")
    else:
        world_grasps = transform_grasp_poses_to_unity(grasps, cam_pos, cam_rot)
    if not world_grasps:
        logger.warning(
            "[VGN] Frame transform produced no valid poses - using geometric fallback"
        )
        return None

    if custom_approach_vector is not None:
        cav = np.array(custom_approach_vector, dtype=np.float64)
        mag = np.linalg.norm(cav)
        if mag > 1e-6:
            cav_unit = cav / mag
            aligned = [
                g
                for g in world_grasps
                if np.dot(np.array(g["approach_direction"]), cav_unit) > 0.0
            ]
            world_grasps = aligned if aligned else world_grasps
            world_grasps.sort(
                key=lambda g: (
                    g.get("score", 0.0)
                    * np.dot(np.array(g["approach_direction"]), cav_unit)
                ),
                reverse=True,
            )
            logger.info(
                f"[VGN] custom_approach_vector filtered {len(world_grasps)} candidates "
                f"(from {len(grasps)} raw)"
            )

    hover = pre_grasp_distance if pre_grasp_distance > 0 else PRE_GRASP_HOVER_OFFSET
    candidates = []
    for g in world_grasps:
        pos = g["position"]
        rot = g["rotation"]
        approach = g["approach_direction"]

        # approach_direction points toward object (VGN convention) → subtract to place hover behind grasp.
        pre_pos = [
            pos[0] - approach[0] * hover,
            pos[1] - approach[1] * hover,
            pos[2] - approach[2] * hover,
        ]

        candidates.append(
            {
                "pre_grasp_position": {
                    "x": pre_pos[0],
                    "y": pre_pos[1],
                    "z": pre_pos[2],
                },
                "pre_grasp_rotation": {
                    "x": rot[0],
                    "y": rot[1],
                    "z": rot[2],
                    "w": rot[3],
                },
                "grasp_position": {"x": pos[0], "y": pos[1], "z": pos[2]},
                "grasp_rotation": {"x": rot[0], "y": rot[1], "z": rot[2], "w": rot[3]},
                "approach_direction": {
                    "x": approach[0],
                    "y": approach[1],
                    "z": approach[2],
                },
                "grasp_depth": 0.5,
                "antipodal_score": g.get("score", 0.0),
                "vgn_score": g.get("score", 0.0),
                "approach_type": preferred_approach,
            }
        )

    parameters = {
        "object_id": object_id,
        "use_advanced_planning": use_advanced_planning,
        "preferred_approach": preferred_approach.lower(),
        "pre_grasp_distance": pre_grasp_distance,
        "enable_retreat": enable_retreat,
        "retreat_distance": retreat_distance,
        "precomputed_candidates": candidates,
    }

    command = {
        "command_type": "grasp_object",
        "target_type": "robot",
        "robot_id": robot_id,
        "parameters": parameters,
        "request_id": request_id,
    }

    broadcaster = _get_command_broadcaster()
    if broadcaster is None:
        return OperationResult.error_result(
            "COMMUNICATION_ERROR",
            "CommandBroadcaster not available",
            ["Check Unity is connected to CommandServer"],
        )

    logger.info(
        f"[VGN] Sending grasp_object: {robot_id} -> {object_id} "
        f"({len(candidates)} candidates)"
    )
    success = broadcaster.send_command(command, request_id)
    if success:
        return OperationResult.success_result(
            {
                "command_sent": True,
                "robot_id": robot_id,
                "object_id": object_id,
                "request_id": request_id,
                "vgn_candidates": len(candidates),
            }
        )
    return OperationResult.error_result(
        "COMMUNICATION_ERROR",
        "Failed to send VGN grasp command to Unity",
        ["Ensure CommandServer is running"],
    )


def _grasp_via_vgn_with_ros(
    bridge,
    robot_id: str,
    object_id: str,
    preferred_approach: str,
    pre_grasp_distance: float,
    request_id: int,
    world_state,
    custom_approach_vector: "Optional[List[float]]" = None,
    grasp_yaw_override: "Optional[float]" = None,
    allow_parallel: bool = False,
) -> "Optional[OperationResult]":
    """VGN 6-DOF poses + MoveIt (highest-priority path). None if unavailable; error result if arm descended but gripper failed."""
    import numpy as np

    try:
        from config.Servers import VGN_TOP_K
    except ImportError:
        VGN_TOP_K = 20

    from operations.GraspFrameTransform import transform_grasp_poses_to_unity
    from operations.PointCloudOperations import generate_point_cloud
    from operations.VGNClient import VGNClient

    client = VGNClient()
    if not client.is_available():
        logger.info("[VGN+ROS] Model unavailable - falling back to geometric ROS")
        return None

    pc_result = generate_point_cloud(robot_id=robot_id, request_id=request_id)
    if not pc_result.success:
        logger.warning(
            f"[VGN+ROS] generate_point_cloud failed ({pc_result.error}), "
            "falling back to geometric ROS"
        )
        return None

    pc = pc_result.result
    assert pc is not None
    points_np = np.array(pc["points"], dtype=np.float32)
    colors_np = np.array(pc["colors"], dtype=np.uint8) if pc.get("colors") else None
    cam_pos = pc["camera_position"]
    cam_rot = pc["camera_rotation"]
    fov = pc["fov"]
    # Use the stereo image dimensions that were actually used for reconstruction
    img_w = pc.get("image_width", 640)
    img_h = pc.get("image_height", 480)

    # SGBM depth unreliable on Unity surfaces (~1.8x overestimate) → use WorldState for position.
    # VGN used only for orientation/approach direction.
    yolo_bbox: tuple = (0, 0, 0, 0)
    image_np: "Optional[np.ndarray]" = None
    det_img_w = img_w
    det_img_h = img_h
    _detection_depth_m: "Optional[float]" = None  # stereo bbox depth for VGN hint

    _detected_world_pos: "Optional[List[float]]" = None
    _original_center: "Optional[List[float]]" = None
    _object_dimensions = None
    try:
        ws_pos = world_state.get_object_position(object_id)
        if ws_pos is not None:
            _detected_world_pos = list(ws_pos)
            _original_center = list(ws_pos)
            logger.info(
                f"[VGN+ROS] Using WorldState position for '{object_id}': "
                f"{[round(v, 3) for v in _detected_world_pos]}"
            )
    except Exception:
        pass
    try:
        _object_dimensions = world_state.get_object_dimensions(object_id)
    except Exception:
        pass

    try:
        from core.Imports import get_unified_image_storage
        from vision.YOLODetector import YOLODetector

        _storage = get_unified_image_storage()
        _stereo = _storage.get_latest_stereo()
        if _stereo is not None:
            _, _left_img, _, _ = _stereo
            if _left_img is not None:
                det_img_h, det_img_w = _left_img.shape[:2]
                _detector = YOLODetector(model_path=_YOLO_MODEL_PATH)
                _det_result = _detector.detect_objects(_left_img, camera_id="main")
                obj_id_norm = object_id.lower().replace(" ", "_")
                for _obj in _det_result.detections:
                    color_field = getattr(_obj, "color", "").lower().replace(" ", "_")
                    if obj_id_norm in color_field or color_field in obj_id_norm:
                        yolo_bbox = (
                            int(_obj.bbox_x),
                            int(_obj.bbox_y),
                            int(_obj.bbox_w),
                            int(_obj.bbox_h),
                        )
                        _dm = getattr(_obj, "depth_m", None)
                        if _dm is not None:
                            _detection_depth_m = float(_dm)
                        break
    except Exception as exc:
        logger.debug(f"[VGN] Could not get YOLO bbox (non-fatal): {exc}")

    if yolo_bbox != (0, 0, 0, 0) and (det_img_w != img_w or det_img_h != img_h):
        scale_x = img_w / det_img_w
        scale_y = img_h / det_img_h
        bx, by, bw, bh = yolo_bbox
        yolo_bbox = (
            int(bx * scale_x),
            int(by * scale_y),
            int(bw * scale_x),
            int(bh * scale_y),
        )
        logger.info(
            f"[VGN] Scaled bbox {det_img_w}x{det_img_h}→{img_w}x{img_h}: {yolo_bbox}"
        )
    if yolo_bbox == (0, 0, 0, 0):
        logger.warning(
            f"[VGN] No valid bbox found for '{object_id}' - masking will use all points"
        )

    try:
        from core.Imports import get_unified_image_storage

        storage = get_unified_image_storage()
        stereo = storage.get_latest_stereo()
        if stereo is not None:
            _, left_img, _, _ = stereo
            image_np = left_img
            pass  # img_w/img_h already set from point cloud result
    except Exception as exc:
        logger.debug(f"[VGN+ROS] Stereo image retrieval (non-fatal): {exc}")

    if image_np is None:
        image_np = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    logger.info(
        f"[VGN] Calling predict_grasps: image_width={img_w}, image_height={img_h}, fov={fov}, bbox={yolo_bbox}, points_shape={points_np.shape}"
    )
    logger.info(
        f"[VGN] Point cloud sample (first 3): {points_np[:3].tolist()}, X range=[{points_np[:,0].min():.3f},{points_np[:,0].max():.3f}], Y=[{points_np[:,1].min():.3f},{points_np[:,1].max():.3f}], Z=[{points_np[:,2].min():.3f},{points_np[:,2].max():.3f}]"
    )
    grasps = client.predict_grasps(
        points=points_np,
        colors=colors_np,
        image=image_np,
        yolo_bbox=yolo_bbox,
        object_label=object_id,
        image_width=img_w,
        image_height=img_h,
        fov=fov,
        top_k=VGN_TOP_K,
        cam_pos=cam_pos,
        cam_rot=cam_rot,
        object_world_pos=_detected_world_pos,
        detection_depth_m=_detection_depth_m,
        object_dimensions=_object_dimensions,
    )
    if not grasps:
        logger.info("[VGN+ROS] No candidates returned - falling back to geometric ROS")
        return None

    if grasps and grasps[0].get("_world_frame"):
        world_grasps = grasps
        logger.info(
            "[VGN+ROS] Grasps already in Unity world frame - skipping transform"
        )
    else:
        world_grasps = transform_grasp_poses_to_unity(grasps, cam_pos, cam_rot)
    if not world_grasps:
        logger.warning(
            "[VGN+ROS] Frame transform produced no valid poses - falling back"
        )
        return None

    if custom_approach_vector is not None:
        cav = np.array(custom_approach_vector, dtype=np.float64)
        mag = np.linalg.norm(cav)
        if mag > 1e-6:
            cav_unit = cav / mag
            aligned = [
                g
                for g in world_grasps
                if np.dot(np.array(g["approach_direction"]), cav_unit) > 0.0
            ]
            world_grasps = aligned if aligned else world_grasps
            world_grasps.sort(
                key=lambda g: (
                    g.get("score", 0.0)
                    * np.dot(np.array(g["approach_direction"]), cav_unit)
                ),
                reverse=True,
            )
            logger.info(
                f"[VGN+ROS] custom_approach_vector filtered {len(world_grasps)} candidates "
                f"(from {len(grasps)} raw)"
            )
    _y_approaches = sorted(
        [g["approach_direction"][1] for g in world_grasps], reverse=True
    )
    logger.info(
        f"[VGN+ROS] Approach Y distribution (top 5): {[round(v,2) for v in _y_approaches[:5]]}"
    )
    _approach_lower = preferred_approach.lower() if preferred_approach else "top"
    if _approach_lower == "side":
        # Side approach: prefer candidates with high horizontal (X) component, low Y.
        _MIN_X_APPROACH = 0.3
        side_candidates = [
            g
            for g in world_grasps
            if abs(g.get("approach_direction", [0, 0, 0])[0]) >= _MIN_X_APPROACH
        ]
        if side_candidates:
            top = max(side_candidates, key=lambda g: g.get("score", 0.0))
            logger.info(
                f"[VGN+ROS] Selected side grasp "
                f"(X_approach={top['approach_direction'][0]:.2f}) from "
                f"{len(side_candidates)}/{len(world_grasps)} candidates"
            )
        else:
            top = max(
                world_grasps,
                key=lambda g: abs(g.get("approach_direction", [0, 0, 0])[0]),
            )
            logger.warning(
                f"[VGN+ROS] No grasp with |X_approach| >= {_MIN_X_APPROACH} - "
                f"using most-horizontal candidate (X_approach={top['approach_direction'][0]:.2f})"
            )
    elif _approach_lower in ("left_side", "right_side"):
        top = max(world_grasps, key=lambda g: g.get("score", 0.0))
        logger.info(
            f"[VGN+ROS] Selected {_approach_lower} grasp (best score) from {len(world_grasps)} candidates"
        )
        if _detected_world_pos and _object_dimensions:
            _lx, _, _lz = _object_dimensions
            _obj_yaw_deg = 0.0
            try:
                if world_state is not None:
                    _rot = world_state.get_object_rotation(object_id)
                    if _rot is not None:
                        _obj_yaw_deg = _rot[1]
            except Exception:
                pass
            _side_sign = 1.0 if _approach_lower == "left_side" else -1.0
            _dx, _dz = _side_offset_world_xz(_lx, _lz, _obj_yaw_deg, _side_sign)
            _detected_world_pos = list(_detected_world_pos)
            _detected_world_pos[0] += _dx
            _detected_world_pos[2] += _dz
            logger.info(
                f"[VGN+ROS] {_approach_lower}: offset ({_dx:+.3f}, {_dz:+.3f})m "
                f"(obj_yaw={_obj_yaw_deg:.1f}°) → target=({_detected_world_pos[0]:.3f}, "
                f"{_detected_world_pos[2]:.3f})"
            )
    elif _approach_lower == "front":
        _MIN_Z_APPROACH = 0.3
        front_candidates = [
            g
            for g in world_grasps
            if abs(g.get("approach_direction", [0, 0, 0])[2]) >= _MIN_Z_APPROACH
        ]
        if front_candidates:
            top = max(front_candidates, key=lambda g: g.get("score", 0.0))
            logger.info(
                f"[VGN+ROS] Selected front grasp "
                f"(Z_approach={top['approach_direction'][2]:.2f}) from "
                f"{len(front_candidates)}/{len(world_grasps)} candidates"
            )
        else:
            top = max(
                world_grasps,
                key=lambda g: abs(g.get("approach_direction", [0, 0, 0])[2]),
            )
            logger.warning(
                f"[VGN+ROS] No grasp with |Z_approach| >= {_MIN_Z_APPROACH} - "
                f"using most-frontal candidate"
            )
    else:
        # Default: top-down - prefer upward Y approach to avoid table collisions.
        top_down_candidates = [
            g
            for g in world_grasps
            if g.get("approach_direction", [0, 0, 0])[1] >= VGN_MIN_Y_APPROACH
        ]
        if top_down_candidates:
            top = max(top_down_candidates, key=lambda g: g.get("score", 0.0))
            logger.info(
                f"[VGN+ROS] Selected top-down-feasible grasp "
                f"(Y_approach={top['approach_direction'][1]:.2f}) from "
                f"{len(top_down_candidates)}/{len(world_grasps)} candidates"
            )
        else:
            top = max(
                world_grasps, key=lambda g: g.get("approach_direction", [0, 0, 0])[1]
            )
            logger.warning(
                f"[VGN+ROS] No grasp with Y_approach >= {VGN_MIN_Y_APPROACH} - "
                f"using most-top-down candidate (Y_approach={top['approach_direction'][1]:.2f})"
            )
    pos = top["position"]
    approach = top["approach_direction"]
    logger.info(
        f"[VGN+ROS] Top grasp world_pos={[round(v,3) for v in pos]}, approach={[round(v,3) for v in approach]}, cam_pos={cam_pos}, cam_rot={cam_rot}"
    )

    # Stereo depth scale error ~1.8x on synthetic Unity surfaces → WorldState position is more accurate.
    if _detected_world_pos:
        dp = _detected_world_pos
        logger.info(
            f"[VGN+ROS] Overriding VGN pos {[round(v,3) for v in pos]} with "
            f"DepthEstimator pos {[round(v,3) for v in dp]} for '{object_id}'"
        )
        pos = dp

    # Fail fast before MoveIt if grasp position outside robot reach.
    try:
        from operations.SpatialPredicates import (
            target_within_reach as _twr,
            warn_if_target_outside_workspace as _warn_ws,
        )

        _warn_ws(robot_id, pos[0], pos[1], pos[2])
        _reachable, _reach_reason = _twr(robot_id, pos[0], pos[1], pos[2])
        if not _reachable:
            logger.warning(
                f"[VGN+ROS] Grasp position {[round(v,3) for v in pos]} unreachable "
                f"for {robot_id}: {_reach_reason} - falling back to geometric ROS"
            )
            return None
    except Exception:
        pass  # non-fatal: SpatialPredicates unavailable

    hover = pre_grasp_distance if pre_grasp_distance > 0 else PRE_GRASP_HOVER_OFFSET
    _approach_lower = preferred_approach.lower() if preferred_approach else "top"
    _is_top_down_approach = _approach_lower not in ("side", "front")

    # VGN rarely predicts near-vertical grasps for table cubes (typical Y~0.45 → twisted wrist).
    # Use proven top_down_orientation unless VGN approach |Y| >= 0.7 (genuinely top-down).
    _TOP_DOWN_Y_THRESHOLD = 0.7
    _vgn_approach_y = approach[1]  # Unity Y = up
    if abs(_vgn_approach_y) >= _TOP_DOWN_Y_THRESHOLD:
        pre_approach = approach

        if grasp_yaw_override is not None:
            yaw_unity = grasp_yaw_override
            if yaw_unity > math.pi / 2:
                yaw_unity -= math.pi
            elif yaw_unity < -math.pi / 2:
                yaw_unity += math.pi
            yaw_source = f"override ({math.degrees(yaw_unity):.1f}°)"
        else:
            yaw_unity, yaw_source = _yaw_from_world_state_or_robot(
                robot_id, object_id, pos, world_state
            )

        half = yaw_unity / 2.0
        qy_z = math.sin(half)
        qy_w = math.cos(half)
        bx, by, bz, bw = 0.9999, 0.0, 0.0, 0.0087
        ox = qy_w * bx - qy_z * by
        oy = qy_w * by + qy_z * bx
        oz = qy_w * bz + qy_z * bw
        ow = qy_w * bw - qy_z * bz
        mag = math.sqrt(ox * ox + oy * oy + oz * oz + ow * ow)
        orientation = {
            "x": ox / mag,
            "y": oy / mag,
            "z": oz / mag,
            "w": ow / mag,
        }
        logger.info(
            f"[VGN+ROS] Top-down + yaw={math.degrees(yaw_unity):.1f}° "
            f"from {yaw_source} "
            f"(VGN approach |Y|={abs(_vgn_approach_y):.2f} >= {_TOP_DOWN_Y_THRESHOLD}), "
            f"orientation={orientation}"
        )
    else:
        # Side/front: use VGN approach direction. Table top-down fallback: straight up.
        if not _is_top_down_approach:
            pre_approach = approach
        else:
            pre_approach = [0.0, 1.0, 0.0]

        if grasp_yaw_override is not None:
            yaw_unity = grasp_yaw_override
            yaw_source = f"override ({math.degrees(grasp_yaw_override):.1f}°)"
        else:
            yaw_unity, yaw_source = _yaw_from_world_state_or_robot(
                robot_id, object_id, pos, world_state
            )

        # 180° gripper symmetry: grasping at θ and θ+π identical → minimise wrist travel.
        if yaw_unity > math.pi / 2:
            yaw_unity -= math.pi
        elif yaw_unity < -math.pi / 2:
            yaw_unity += math.pi
        # q_final = q_yaw_ros * q_topdown
        # q_topdown ≈ (0.9999, 0, 0, 0.0087): 179° around ROS X = gripper down
        # q_yaw_ros = (0, 0, sin(θ/2), cos(θ/2)): yaw around ROS Z
        half = yaw_unity / 2.0
        qy_z = math.sin(half)
        qy_w = math.cos(half)
        bx, by, bz, bw = 0.9999, 0.0, 0.0, 0.0087
        ox = qy_w * bx - qy_z * by
        oy = qy_w * by + qy_z * bx
        oz = qy_w * bz + qy_z * bw
        ow = qy_w * bw - qy_z * bz
        mag = math.sqrt(ox * ox + oy * oy + oz * oz + ow * ow)
        orientation = {
            "x": ox / mag,
            "y": oy / mag,
            "z": oz / mag,
            "w": ow / mag,
        }
        logger.info(
            f"[VGN+ROS] Top-down + yaw={math.degrees(yaw_unity):.1f}° "
            f"from {yaw_source} "
            f"(VGN approach |Y|={abs(_vgn_approach_y):.2f} < {_TOP_DOWN_Y_THRESHOLD}), "
            f"orientation={orientation}"
        )

    pre_grasp_pos = {
        "x": pos[0] + pre_approach[0] * hover,
        "y": pos[1] + pre_approach[1] * hover,
        "z": pos[2] + pre_approach[2] * hover,
    }
    # GRASP_TCP_OFFSET along approach: fingers stop at surface, don't drive through.
    grasp_pos = {
        "x": pos[0] + pre_approach[0] * GRASP_TCP_OFFSET,
        "y": pos[1] + pre_approach[1] * GRASP_TCP_OFFSET,
        "z": pos[2] + pre_approach[2] * GRASP_TCP_OFFSET,
    }

    # Clearance waypoint: top-down only - side/front don't sweep through table-height space.
    clearance_pos = {"x": pos[0], "y": PRE_GRASP_CLEARANCE_Y, "z": pos[2]}
    _is_top_down_approach = _approach_lower not in ("side", "front")
    if _is_top_down_approach and pre_grasp_pos["y"] < PRE_GRASP_CLEARANCE_Y:
        # Pre-grasp is below clearance height - insert the waypoint.
        logger.info(f"[VGN+ROS] Clearance waypoint for {robot_id}: {clearance_pos}")
        clearance_result = bridge.plan_and_execute(
            position=clearance_pos,
            orientation=None,  # no orientation needed at clearance height - constraint causes slow planning
            planning_time=3.0,
            robot_id=robot_id,
            max_velocity_scaling=PREGRASP_VELOCITY_SCALING,
            max_acceleration_scaling=PREGRASP_ACCELERATION_SCALING,
            constrain_joint4=True,
            allow_parallel=allow_parallel,
        )
        if not clearance_result or not clearance_result.get("success"):
            cl_err = (
                clearance_result.get("error", "Unknown")
                if clearance_result
                else "No response"
            )
            logger.warning(
                f"[VGN+ROS] Clearance waypoint failed ({cl_err}) - "
                "proceeding directly to pre-grasp"
            )
        else:
            time.sleep(0.2)

    # No orientation for side/front pre-grasp - shrinks IK solution space near workspace boundaries.
    _pre_grasp_orientation = orientation if _is_top_down_approach else None
    logger.info(f"[VGN+ROS] Moving to pre-grasp for {robot_id}: {pre_grasp_pos}")
    pre_result = bridge.plan_and_execute(
        position=pre_grasp_pos,
        orientation=_pre_grasp_orientation,
        planning_time=10.0,
        robot_id=robot_id,
        max_velocity_scaling=PREGRASP_VELOCITY_SCALING,
        max_acceleration_scaling=PREGRASP_ACCELERATION_SCALING,
        constrain_joint4=True,
        allow_parallel=allow_parallel,
    )
    if not pre_result or not pre_result.get("success"):
        pre_err = pre_result.get("error", "Unknown") if pre_result else "No response"
        logger.info(
            f"[VGN+ROS] Pre-grasp with orientation failed ({pre_err}) - "
            "retrying without orientation constraint"
        )
        # Position-only: constrain_joint6 prevents free-spin; constrain_joint4 prevents long-arc IK.
        pre_result = bridge.plan_and_execute(
            position=pre_grasp_pos,
            orientation=None,
            planning_time=10.0,
            robot_id=robot_id,
            max_velocity_scaling=PREGRASP_VELOCITY_SCALING,
            max_acceleration_scaling=PREGRASP_ACCELERATION_SCALING,
            constrain_joint6=True,
            constrain_joint4=True,
            allow_parallel=allow_parallel,
        )
    if not pre_result or not pre_result.get("success"):
        pre_err = pre_result.get("error", "Unknown") if pre_result else "No response"
        logger.warning(
            f"[VGN+ROS] Pre-grasp planning failed ({pre_err}) - "
            "falling back to geometric ROS"
        )
        return None

    # Settle pause: arm needs ~0.4s after OMPL trajectory to damp PD oscillation below 1°/s;
    # 0.15s was too short - residual motion caused ROSMotionClient to plan the Cartesian
    # descent from a still-moving start state, shifting the first waypoint and inducing rotation.
    time.sleep(0.4)

    logger.info(f"[VGN+ROS] Cartesian descent for {robot_id}: {grasp_pos}")
    descent_result = bridge.plan_cartesian_descent(
        position=grasp_pos,
        orientation=orientation,
        robot_id=robot_id,
        max_velocity_scaling=GRASP_DESCENT_VELOCITY_SCALING,
        max_acceleration_scaling=GRASP_DESCENT_ACCELERATION_SCALING,
        allow_parallel=allow_parallel,
    )
    if not descent_result or not descent_result.get("success"):
        descent_err = (
            descent_result.get("error", "Unknown") if descent_result else "No response"
        )
        logger.warning(
            f"[VGN+ROS] Cartesian descent failed ({descent_err}) - "
            "falling back to geometric ROS"
        )
        return None

    # Arm has descended - do NOT return None from here; return error result.
    # XZ offset from object centre to the approach-side grasp target. Passed to
    # follow_target so drift correction targets the same side (e.g. left_side Z offset)
    # instead of the raw object centre, which may collide with a bracing partner robot.
    _approach_offset_xz = (0.0, 0.0)
    if _original_center and _detected_world_pos:
        _approach_offset_xz = (
            _detected_world_pos[0] - _original_center[0],
            _detected_world_pos[2] - _original_center[2],
        )

    grasp_ok, grasp_fail_reason = _execute_grasp_with_follow_target(
        bridge=bridge,
        robot_id=robot_id,
        object_id=object_id,
        planned_position=grasp_pos,
        orientation=orientation,
        tcp_y_offset=GRASP_TCP_OFFSET,
        world_state=world_state,
        approach_offset_xz=_approach_offset_xz,
        allow_parallel=allow_parallel,
    )
    if not grasp_ok:
        return OperationResult.error_result(
            "GRASP_EXECUTION_FAILED",
            f"VGN+ROS grasp failed for {robot_id}: {grasp_fail_reason}",
            [
                "Check gripper hardware/simulation state",
                "Verify GripperContactSensor is active",
                "If corrective move failed, check for workspace collision with partner robot",
            ],
        )

    return OperationResult.success_result(
        {
            "robot_id": robot_id,
            "object_id": object_id,
            "request_id": request_id,
            "vgn_candidates": len(world_grasps),
            "status": "vgn_ros_executed",
            "timestamp": time.time(),
        }
    )
