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

import base64
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

        # Build a ConfigProto that mirrors the memory-growth setting above so the
        # session actually runs on the GPU.  A bare tf.Session() ignores the
        # set_memory_growth call and may silently fall back to CPU.
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        _tf_sess = tf.Session(config=session_config)
        _grasp_estimator.load_weights(_tf_sess, saver, checkpoint_dir)
        logger.info(
            f"Contact-GraspNet loaded from '{checkpoint_dir}' on {_device_name}"
        )

        # --- Warm-up pass ---------------------------------------------------
        # Run one dummy inference with a random 20k-point cloud so TF1 compiles
        # all CUDA kernels at startup.  Without this, the first real request pays
        # the 4-5 minute compilation cost (observed: 269 s on RTX-class GPU).
        # Subsequent calls reuse compiled kernels and complete in ~800 ms.
        logger.info("Running warm-up inference to compile CUDA kernels …")
        t_wu = time.time()
        dummy_pts = np.random.randn(20_000, 3).astype(np.float32) * 0.3
        try:
            _grasp_estimator.predict_scene_grasps(
                _tf_sess,
                dummy_pts,
                pc_segments={},
                local_regions=False,
                filter_grasps=False,
                forward_passes=1,
            )
            logger.info(
                f"Warm-up complete in {(time.time() - t_wu):.1f}s — "
                "first real request will be fast."
            )
        except Exception as wu_exc:
            logger.warning(f"Warm-up inference failed (non-fatal): {wu_exc}")
        # --------------------------------------------------------------------

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
    points: str              # base64-encoded float32 bytes, shape (N, 3)
    points_shape: List[int]  # [N, 3]
    colors: Optional[str] = None             # base64-encoded uint8 bytes, shape (N, 3)
    colors_shape: Optional[List[int]] = None  # [N, 3]
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

    Decodes base64-encoded float32 points (and optional uint8 colors) before
    inference, then returns up to top_k grasp poses sorted by descending
    quality score.  All poses are in camera frame (right-handed, Z-forward).
    """
    if _grasp_estimator is None or _tf_sess is None:
        return {
            "success": False,
            "grasps": [],
            "count": 0,
            "error": "Model not loaded — check server logs",
        }

    t0 = time.time()

    pts = np.frombuffer(base64.b64decode(req.points), dtype=np.float32).reshape(req.points_shape)
    clr = None
    if req.colors is not None and req.colors_shape is not None:
        clr = np.frombuffer(base64.b64decode(req.colors), dtype=np.uint8).reshape(req.colors_shape)

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
        # NVlabs API full-scene mode: no pc_segments, local_regions=False.
        # With local_regions=True the model crops boxes around pc_segments entries;
        # passing an empty dict produces zero iterations and zero grasps.
        # In full-scene mode results are keyed at -1.  filter_grasps must be False
        # here — when True the NVlabs code deletes the -1 key before returning,
        # discarding all predictions.
        pred_grasps_cam, scores, contact_pts, _ = _grasp_estimator.predict_scene_grasps(
            _tf_sess,
            pts,
            pc_segments={},
            local_regions=False,
            filter_grasps=False,
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
