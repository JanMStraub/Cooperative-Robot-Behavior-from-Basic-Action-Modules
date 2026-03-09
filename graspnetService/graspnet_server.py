"""
Contact-GraspNet FastAPI Inference Service
==========================================

Wraps the NVlabs Contact-GraspNet model (TF 2.2, Python 3.7) behind a simple
HTTP API so the ACRL Python backend can request 6-DOF grasp poses without a
direct TensorFlow dependency in the main process.

Endpoints
---------
GET  /health           -> liveness check
POST /predict_grasps   -> point cloud in, ranked grasp poses out (camera frame)

Poses are returned in camera frame, right-handed (Z-forward, Y-up).
GraspFrameTransform in the ACRL backend handles the Unity world-space conversion.

NVlabs tested stack: Python 3.7, TensorFlow 2.2, CUDA 11.1
"""

import sys
import os
import time
import logging
from typing import List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Contact-GraspNet Inference Service", version="1.0.0")

# ---------------------------------------------------------------------------
# Model loading (done once at startup)
# ---------------------------------------------------------------------------

_estimator = None
_sess = None
_device_name = "cpu"

# NVlabs inference uses a global_config dict and a persistent tf.Session
_global_config = None
_grasp_estimator = None
_tf_sess = None  # kept alive for the process lifetime


def _load_model():
    """Load Contact-GraspNet weights into GPU memory using the NVlabs TF1-style API."""
    global _grasp_estimator, _global_config, _device_name, _tf_sess

    try:
        import tensorflow.compat.v1 as tf

        tf.disable_eager_execution()

        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            _device_name = "GPU:0"
        else:
            _device_name = "CPU"
            logger.warning("No GPU found — running on CPU (inference will be slow)")

        # NVlabs uses a config_utils module in the repo root
        sys.path.insert(0, "/app")
        sys.path.insert(0, "/app/contact_graspnet")

        # Mock scene_renderer before importing the estimator — it pulls in
        # pyrender → pyglet → GLX at class-definition time, which crashes in
        # a headless container. The inference path never uses SceneRenderer.
        import types

        _mock_scene_renderer = types.ModuleType("scene_renderer")
        _mock_scene_renderer.SceneRenderer = type("SceneRenderer", (), {})
        sys.modules.setdefault("scene_renderer", _mock_scene_renderer)

        import config_utils
        from contact_grasp_estimator import GraspEstimator

        checkpoint_dir = "checkpoints/scene_test_2048_bs3_hor_sigma_001"
        if not os.path.isdir(checkpoint_dir):
            logger.error(
                f"Checkpoint directory not found: {checkpoint_dir}\n"
                "Mount your weights volume and ensure the folder exists."
            )
            return

        _global_config = config_utils.load_config(
            "contact_graspnet/config.yaml", batch_size=1, arg_configs=[]
        )
        _grasp_estimator = GraspEstimator(_global_config)
        _grasp_estimator.build_network()

        # Saver must be created after build_network() so all TF variables exist
        saver = tf.train.Saver(save_relative_paths=True)

        # Session is kept alive — weights live in GPU memory for all requests
        _tf_sess = tf.Session()
        _grasp_estimator.load_weights(_tf_sess, saver, checkpoint_dir)
        logger.info(
            f"Contact-GraspNet loaded from '{checkpoint_dir}' on {_device_name}"
        )

    except Exception as exc:
        logger.error(f"Failed to load Contact-GraspNet: {exc}", exc_info=True)
        logger.error(
            "Ensure checkpoints/ volume is mounted and TF ops were compiled during build."
        )
        _grasp_estimator = None


@app.on_event("startup")
def startup_event():
    _load_model()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class GraspRequest(BaseModel):
    points: List[List[float]]  # (N, 3) camera frame
    colors: Optional[List[List[int]]] = None  # (N, 3) uint8 RGB, optional
    segmentation_mask: Optional[List[bool]] = None  # (N,) bool, optional
    top_k: int = 20

    class Config:
        # pydantic v1 compat
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    """Liveness check — returns 200 when model is loaded."""
    return {
        "status": (
            "ok"
            if (_grasp_estimator is not None and _tf_sess is not None)
            else "model_not_loaded"
        ),
        "device": _device_name,
    }


@app.post("/predict_grasps")
def predict_grasps(req: GraspRequest):
    """Run Contact-GraspNet inference on the provided point cloud.

    Applies the optional segmentation mask before inference, then returns
    up to top_k grasp poses sorted by descending quality score.
    All poses are in camera frame (right-handed, Z-forward).
    """
    if _grasp_estimator is None or _tf_sess is None:
        return {
            "success": False,
            "grasps": [],
            "count": 0,
            "error": "Model not loaded — check server logs",
        }

    t0 = time.time()

    pts = np.array(req.points, dtype=np.float32)

    # Apply segmentation mask
    if req.segmentation_mask is not None:
        mask = np.array(req.segmentation_mask, dtype=bool)
        if mask.shape[0] == pts.shape[0]:
            pts = pts[mask]

    if pts.shape[0] < 50:
        return {
            "success": False,
            "grasps": [],
            "count": 0,
            "error": f"Too few points after masking: {pts.shape[0]} (need >= 50)",
        }

    # NVlabs model expects exactly 20 000 points
    target_n = 20_000
    if pts.shape[0] > target_n:
        idx = np.random.choice(pts.shape[0], target_n, replace=False)
        pts = pts[idx]
    elif pts.shape[0] < target_n:
        repeats = (target_n // pts.shape[0]) + 1
        pts = np.tile(pts, (repeats, 1))[:target_n]

    try:
        # NVlabs API: predict_scene_grasps returns dicts keyed by segment id
        # -1 is the full-scene key when no segmentation is used
        pred_grasps_cam, scores, contact_pts, _ = _grasp_estimator.predict_scene_grasps(
            _tf_sess,
            pts,
            pc_segments={},
            local_regions=True,
            filter_grasps=True,
            forward_passes=1,
        )
    except Exception as exc:
        logger.exception(f"Inference error: {exc}")
        return {"success": False, "grasps": [], "count": 0, "error": str(exc)}

    # Flatten all grasp proposals across segment keys
    all_grasps = []
    for key in pred_grasps_cam:
        grasp_mats = pred_grasps_cam[key]  # (M, 4, 4) homogeneous transforms
        grasp_scores = scores[key]  # (M,)
        for mat, score in zip(grasp_mats, grasp_scores):
            R = mat[:3, :3]
            t = mat[:3, 3]

            # Rotation matrix -> quaternion [x, y, z, w]
            from scipy.spatial.transform import Rotation

            quat = Rotation.from_matrix(R).as_quat().tolist()

            # Approach direction = -Z axis of the grasp frame
            approach = (-R[:, 2]).tolist()

            all_grasps.append(
                {
                    "position": t.tolist(),
                    "rotation": quat,
                    "score": float(score),
                    "width": 0.08,  # AR4 max gripper opening in metres
                    "approach_direction": approach,
                }
            )

    all_grasps.sort(key=lambda g: g["score"], reverse=True)
    top_grasps = all_grasps[: req.top_k]

    elapsed_ms = (time.time() - t0) * 1000
    logger.info(
        f"predict_grasps: {len(top_grasps)} poses returned "
        f"(total: {len(all_grasps)}, {elapsed_ms:.0f}ms)"
    )

    return {
        "success": True,
        "grasps": top_grasps,
        "count": len(top_grasps),
        "inference_time_ms": round(elapsed_ms, 1),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8766, log_level="info")
