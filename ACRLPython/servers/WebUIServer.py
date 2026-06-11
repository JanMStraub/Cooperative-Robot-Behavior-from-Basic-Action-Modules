#!/usr/bin/env python3
"""
WebUIServer.py - FastAPI Gateway for the ACRL Web Dashboard

Exposes REST and WebSocket endpoints to control the robot and visualize
system state without needing the Unity Editor.
Serves a static HTML/JS/CSS frontend.
"""

import os
import json
import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Dict, Any, List, Optional

# Shared YOLO detector instance — set via set_shared_detector() to avoid loading
# a duplicate model. Falls back to creating its own if not set.
_shared_detector = None


def set_shared_detector(detector):
    """Set the shared YOLO detector so the WebUI reuses the existing model instance."""
    global _shared_detector
    _shared_detector = detector


try:
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, StreamingResponse
    import uvicorn

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    # Provide stubs so the module can be imported without fastapi installed.
    # A clear RuntimeError is raised only when run_webui_server() is actually called.
    if TYPE_CHECKING:
        from contextlib import asynccontextmanager
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import HTMLResponse, StreamingResponse
        import uvicorn
    else:
        asynccontextmanager = None  # type: ignore
        FastAPI = WebSocket = WebSocketDisconnect = BackgroundTasks = None  # type: ignore
        StaticFiles = None  # type: ignore
        HTMLResponse = StreamingResponse = None  # type: ignore
        uvicorn = None  # type: ignore

# Try to import core and orchestrator modules
try:
    from config.Servers import DEFAULT_HOST
except ImportError:
    from ..config.Servers import DEFAULT_HOST

from core.Imports import get_command_broadcaster, get_world_state

logger = logging.getLogger(__name__)

# The running uvicorn event loop — captured at startup so background threads
# can schedule coroutines on it via run_coroutine_threadsafe().
_main_loop: Optional[asyncio.AbstractEventLoop] = None

# Fired once uvicorn's startup_event completes, so callers can wait for the
# Web UI to be fully initialised before proceeding.
_startup_complete = threading.Event()


def get_startup_event() -> threading.Event:
    """Return the event that is set when the Web UI server has fully started."""
    return _startup_complete


# SequenceQueryHandler singleton — initialized once to avoid repeated LLM model loading.
_sequence_handler: Optional[Any] = None


def _get_sequence_handler():
    """
    Return the SequenceQueryHandler singleton, initializing it on first call.

    Using a module-level singleton prevents repeated LLM model initialization
    when multiple WebSocket prompts are sent during a session.
    """
    global _sequence_handler
    if _sequence_handler is None:
        from servers.SequenceServer import SequenceQueryHandler

        _sequence_handler = SequenceQueryHandler()
        if not _sequence_handler.is_ready():
            _sequence_handler.initialize()
    return _sequence_handler


# When fastapi is absent, `app` is a _NoOpApp stub whose attribute access
# returns a no-op decorator so all @app.get / @app.websocket / app.mount calls
# silently pass at import time; run_webui_server() raises a RuntimeError before
# uvicorn is ever reached.
class _NoOpApp:
    """
    Stub FastAPI app used when fastapi is not installed.

    Every attribute access returns a decorator factory so that all @app.get(),
    @app.post(), @app.websocket(), @app.on_event() decorators silently pass
    at import time without calling into fastapi at all.
    """

    def __getattr__(self, _name):
        # Return a decorator factory: app.get("/path") returns a decorator
        # that returns the original function unchanged.
        def _decorator_factory(*_args, **_kwargs):
            def _decorator(func):
                return func

            return _decorator

        return _decorator_factory

    def mount(self, *_args, **_kwargs):
        pass


async def _startup_logic():
    """Startup logic extracted so it can be called from the lifespan handler."""
    global _main_loop
    _main_loop = asyncio.get_running_loop()
    _startup_complete.set()

    asyncio.create_task(state_broadcaster())

    try:
        from core.LoggingSetup import add_websocket_handler
    except ImportError:
        from ..core.LoggingSetup import add_websocket_handler

    def log_callback(msg: str, level: str):
        """Forward a log record to all active WebSocket clients."""
        if manager.active_connections and _main_loop:
            log_data = json.dumps({"type": "log", "message": msg, "level": level})
            asyncio.run_coroutine_threadsafe(manager.broadcast(log_data), _main_loop)

    add_websocket_handler(log_callback)

    try:
        from servers.AutoRTIntegration import AutoRTHandler

        _autort = AutoRTHandler.get_instance()

        def _autort_web_push(payload: dict):
            if manager.active_connections and _main_loop:
                asyncio.run_coroutine_threadsafe(
                    manager.broadcast(json.dumps(payload)), _main_loop
                )

        _autort.set_web_broadcast_callback(_autort_web_push)
        logger.info("AutoRT web broadcast callback registered")
    except Exception as e:
        logger.warning(f"Could not register AutoRT web callback: {e}")


if _FASTAPI_AVAILABLE:

    @asynccontextmanager
    async def _lifespan(app_instance):
        await _startup_logic()
        yield

    app = FastAPI(title="ACRL Mission Control", lifespan=_lifespan)
else:
    app = _NoOpApp()


# Track active websocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List["WebSocket"] = []

    async def connect(self, websocket: "WebSocket"):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: "WebSocket"):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.debug(f"Dropping dead WebSocket client: {e}")
                dead.append(connection)
        for connection in dead:
            self.disconnect(connection)


manager = ConnectionManager()

# Determine paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
WEBUI_DIR = os.path.join(PROJECT_ROOT, "ACRLDashboard")
UNITY_URDF_DIR = os.path.join(
    PROJECT_ROOT, "ACRLUnity", "Assets", "Prefabs", "ar4_urdf"
)

# Ensure webui dir exists
os.makedirs(WEBUI_DIR, exist_ok=True)

# Ensure benchmark results dir exists
BENCHMARK_RESULTS_DIR = os.path.join(BASE_DIR, "benchmark_results")
os.makedirs(BENCHMARK_RESULTS_DIR, exist_ok=True)


def _iter_benchmark_files():
    """Yield (file_path, relative_path) for all benchmark JSON files, walking subfolders."""
    for root, dirs, files in os.walk(BENCHMARK_RESULTS_DIR):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for file in sorted(files):
            if file.startswith("benchmark") and file.endswith(".json"):
                file_path = os.path.join(root, file)
                rel = os.path.relpath(file_path, BENCHMARK_RESULTS_DIR)
                yield file_path, rel


def _model_from_rel(rel: str) -> str:
    """Recover the model name from a result's relative path.

    Layout is ``bN/<model>/file.json`` (model x task benchmarks) or
    ``bN/file.json`` (flat ablations). Returns ``"default"`` when no model
    subdirectory is present.
    """
    parts = rel.replace("\\", "/").split("/")
    if len(parts) >= 3:
        return parts[1]
    return "default"


# Mount static files — only when fastapi is available
if _FASTAPI_AVAILABLE:
    app.mount("/static", StaticFiles(directory=WEBUI_DIR), name="static")
    if os.path.exists(UNITY_URDF_DIR):
        app.mount("/urdf", StaticFiles(directory=UNITY_URDF_DIR), name="urdf")


@app.get("/")
async def get_index():
    index_path = os.path.join(WEBUI_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        content="<h1>WebUI Not Found</h1><p>Please create index.html in the webui directory.</p>"
    )


def _check_ros_connected() -> bool:
    try:
        from ros2.ROSBridge import ROSBridge

        bridge = ROSBridge.get_instance()
        return getattr(bridge, "_connected", False)
    except Exception:
        return False


def _check_unity_connected() -> bool:
    try:
        broadcaster = get_command_broadcaster()
        if hasattr(broadcaster, "_server") and broadcaster._server is not None:
            return broadcaster._server.get_client_count() > 0
        return False
    except Exception:
        return False


def _check_camera_available() -> bool:
    try:
        from servers.ImageStorageCore import UnifiedImageStorage

        storage = UnifiedImageStorage()
        return bool(storage.get_latest_single() or storage.get_latest_stereo())
    except Exception:
        return False


def _check_llm_studio_connected() -> bool:
    """Return True if LM Studio is reachable and responding to /models."""
    try:
        from config.Servers import LMSTUDIO_BASE_URL
        import urllib.request

        url = LMSTUDIO_BASE_URL.rstrip("/")
        req = urllib.request.Request(f"{url}/models", method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


@app.get("/api/status")
async def api_status():
    """
    Return live connectivity status for all subsystems.

    Polled by the dashboard every 3 seconds to update status badges.

    The probes do blocking I/O (notably the LM Studio HTTP check), so they run in
    the default executor to avoid stalling the event loop for all other clients.
    """
    loop = asyncio.get_running_loop()
    ros, unity, camera, llm = await asyncio.gather(
        loop.run_in_executor(None, _check_ros_connected),
        loop.run_in_executor(None, _check_unity_connected),
        loop.run_in_executor(None, _check_camera_available),
        loop.run_in_executor(None, _check_llm_studio_connected),
    )
    return {
        "backend": True,
        "ros": ros,
        "unity": unity,
        "camera": camera,
        "llm": llm,
    }


@app.get("/api/world_state")
async def api_world_state():
    world_state = get_world_state()
    try:
        data = {
            "robots": [r.__dict__ for r in world_state._robot_states.values()],
            "objects": [o.__dict__ for o in world_state.get_all_objects()],
        }
        return data
    except Exception as e:
        logger.error(f"Error getting world state: {e}")
        return {"error": str(e)}


@app.get("/api/benchmarks")
async def api_get_benchmarks():
    """List all benchmark result files."""
    try:
        results = []
        if os.path.exists(BENCHMARK_RESULTS_DIR):
            for file_path, rel in _iter_benchmark_files():
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                folder = os.path.dirname(rel)
                results.append(
                    {
                        "filepath": rel,
                        "filename": os.path.basename(rel),
                        "folder": folder,
                        "benchmark_id": data.get("benchmark_id"),
                        "benchmark_name": data.get("benchmark_name"),
                        "run_id": data.get("run_id"),
                        "success": data.get("success"),
                        "total_duration_ms": data.get("total_duration_ms"),
                        "success_rate": data.get("success_rate"),
                        "ops_executed": data.get("ops_executed"),
                        "ops_succeeded": data.get("ops_succeeded"),
                        "mtime": os.path.getmtime(file_path),
                    }
                )
        # Sort by mtime descending
        results.sort(key=lambda x: x["mtime"], reverse=True)
        return {"success": True, "files": results}
    except Exception as e:
        logger.error(f"Error listing benchmarks: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/benchmarks/aggregate")
async def api_get_benchmarks_aggregate():
    """Aggregate benchmark results grouped by benchmark_id for cross-run analysis."""
    import statistics
    from collections import defaultdict

    try:
        groups: dict = defaultdict(list)
        if os.path.exists(BENCHMARK_RESULTS_DIR):
            for file_path, rel in _iter_benchmark_files():
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                bid = data.get("benchmark_id")
                if bid is not None:
                    # Tag each run with its model (JSON field for new runs, else
                    # recovered from the directory path) for the by_model breakout.
                    data["_model"] = data.get("model") or _model_from_rel(rel)
                    groups[bid].append(data)

        result = {}
        for bid, runs in sorted(groups.items()):
            success_rates = [r.get("success_rate", 0.0) for r in runs]
            durations = [r.get("total_duration_ms", 0.0) for r in runs]
            pass_count = sum(1 for r in runs if r.get("success"))

            entry: Dict[str, Any] = {
                "benchmark_id": bid,
                "benchmark_name": runs[0].get("benchmark_name", f"B{bid}"),
                "run_count": len(runs),
                "pass_count": pass_count,
                "mean_success_rate": statistics.mean(success_rates),
                "std_success_rate": (
                    statistics.stdev(success_rates) if len(success_rates) > 1 else 0.0
                ),
                "mean_duration_ms": statistics.mean(durations),
                "std_duration_ms": (
                    statistics.stdev(durations) if len(durations) > 1 else 0.0
                ),
            }

            # Ablation metrics — split by condition (enabled/disabled)
            ablation_runs = [r for r in runs if r.get("ablation")]
            if ablation_runs:
                by_condition: dict = defaultdict(list)
                for r in ablation_runs:
                    ab = r["ablation"]
                    by_condition[ab.get("condition", "unknown")].append(ab)
                entry["ablation"] = {
                    cond: {
                        "mean_success_rate": statistics.mean(
                            a.get("success_rate", 0.0) for a in abs_list
                        ),
                        "mean_hallucinated_ops": statistics.mean(
                            a.get("hallucinated_ops", 0) for a in abs_list
                        ),
                        "mean_reflexion_recoveries": statistics.mean(
                            a.get("reflexion_recoveries", 0) for a in abs_list
                        ),
                        "mean_negotiation_rounds": statistics.mean(
                            a.get("negotiation_rounds", 0) for a in abs_list
                        ),
                        "run_count": len(abs_list),
                    }
                    for cond, abs_list in by_condition.items()
                }

            # Per-operation stats across all steps in all runs
            op_buckets = defaultdict(lambda: {"durations": [], "fails": 0})
            robot_buckets: dict = defaultdict(
                lambda: {"total_duration_ms": 0.0, "step_count": 0}
            )
            for r in runs:
                for s in r.get("steps", []):
                    op = s.get("operation", "unknown")
                    dur = s.get("duration_ms", 0.0)
                    op_buckets[op]["durations"].append(dur)  # type: ignore[union-attr]
                    if not s.get("success"):
                        op_buckets[op]["fails"] += 1
                    robot_id = s.get("robot_id")
                    if robot_id:
                        robot_buckets[robot_id]["total_duration_ms"] += dur
                        robot_buckets[robot_id]["step_count"] += 1
            entry["op_stats"] = {
                op: {
                    "mean_duration_ms": statistics.mean(v["durations"]),
                    "fail_count": v["fails"],
                    "call_count": len(v["durations"]),
                }
                for op, v in op_buckets.items()
            }
            entry["per_robot_stats"] = dict(robot_buckets)

            # Per-model breakout — the primary independent variable for b1-b11.
            # Aggregating across models (as the top-level entry does) mixes models
            # of very different capability, so expose each model separately.
            model_buckets: dict = defaultdict(list)
            for r in runs:
                model_buckets[r.get("_model", "default")].append(r)
            by_model = {}
            for model, m_runs in sorted(model_buckets.items()):
                m_sr = [r.get("success_rate", 0.0) for r in m_runs]
                m_dur = [r.get("total_duration_ms", 0.0) for r in m_runs]
                by_model[model] = {
                    "run_count": len(m_runs),
                    "pass_count": sum(1 for r in m_runs if r.get("success")),
                    "mean_success_rate": statistics.mean(m_sr),
                    "std_success_rate": (
                        statistics.stdev(m_sr) if len(m_sr) > 1 else 0.0
                    ),
                    "mean_duration_ms": statistics.mean(m_dur),
                }
            entry["by_model"] = by_model

            plan_lengths = [len(r.get("parsed_plan", [])) for r in runs]
            entry["mean_plan_length"] = (
                statistics.mean(plan_lengths) if plan_lengths else 0.0
            )

            result[str(bid)] = entry

        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"Error aggregating benchmarks: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/benchmarks/export/{filepath:path}")
async def api_export_benchmark(filepath: str):
    """Download a benchmark JSON result file as an attachment."""
    try:
        from fastapi.responses import FileResponse

        base = os.path.abspath(BENCHMARK_RESULTS_DIR)
        abs_path = os.path.abspath(os.path.join(BENCHMARK_RESULTS_DIR, filepath))
        if not abs_path.startswith(base + os.sep) and abs_path != base:
            return {"success": False, "error": "Invalid path"}
        if not os.path.exists(abs_path):
            return {"success": False, "error": "File not found"}
        clean_name = os.path.basename(abs_path)
        return FileResponse(
            abs_path,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{clean_name}"'},
        )
    except Exception as e:
        logger.error(f"Error exporting benchmark {filepath}: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/benchmarks/{filepath:path}")
async def api_get_benchmark_detail(filepath: str):
    try:
        base = os.path.abspath(BENCHMARK_RESULTS_DIR)
        abs_path = os.path.abspath(os.path.join(BENCHMARK_RESULTS_DIR, filepath))
        if not abs_path.startswith(base + os.sep) and abs_path != base:
            return {"success": False, "error": "Invalid path"}
        if not os.path.exists(abs_path):
            return {"success": False, "error": "File not found"}
        with open(abs_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {"success": True, "data": data}
    except Exception as e:
        logger.error(f"Error reading benchmark file {filepath}: {e}")
        return {"success": False, "error": str(e)}


def _make_placeholder_frame(text="Waiting for Unity..."):
    """Return a minimal JPEG bytes placeholder frame for MJPEG streams."""
    import cv2
    import numpy as np

    img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(img, text, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
    _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return encoded.tobytes()


def frame_generator(stream_type="left"):
    """Generator for MJPEG streaming from UnifiedImageStorage"""
    import time

    try:
        from servers.ImageStorageCore import UnifiedImageStorage
        import cv2
    except ImportError:
        logger.error("Could not import Vision dependencies for streaming")
        return

    storage = UnifiedImageStorage()
    _placeholder = _make_placeholder_frame()

    # Reuse the shared detector if available, otherwise load one
    _detector = None
    if stream_type == "left":
        if _shared_detector is not None:
            _detector = _shared_detector
        else:
            try:
                from vision.YOLODetector import YOLODetector
                from config.Vision import YOLO_MODEL_PATH

                _detector = YOLODetector(model_path=YOLO_MODEL_PATH)
            except Exception as e:
                logger.warning(f"Could not load YOLODetector for streaming: {e}")

    while True:
        frame_bytes = None

        # Try to get stereo first as it's the primary feed
        stereo = storage.get_latest_stereo()
        if stereo:
            _, imgL, imgR, _ = stereo

            if stream_type == "depth":
                try:
                    from vision.DepthEstimator import calc_disparity
                    import numpy as np

                    imgL_gray = (
                        cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
                        if len(imgL.shape) == 3
                        else imgL
                    )
                    imgR_gray = (
                        cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
                        if len(imgR.shape) == 3
                        else imgR
                    )

                    disparity = calc_disparity(imgL_gray, imgR_gray)
                    disp_valid = np.nan_to_num(disparity, nan=0.0)

                    # Normalize and paint heat map
                    disp_normalized = np.zeros_like(disp_valid, dtype=np.uint8)
                    cv2.normalize(
                        disp_valid,
                        disp_normalized,
                        0,
                        255,
                        cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U,
                    )
                    frame = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)
                    # Mask out invalid/zero disparity areas as black instead of dark blue
                    frame[disp_valid <= 0] = [0, 0, 0]
                except Exception as e:
                    logger.debug(f"WebUI depth stream failed: {e}")
                    frame = imgR  # Fallback to right eye if disparity fails
            else:
                frame = imgL if stream_type == "left" else imgR

            # Add YOLO annotations to RGB stream (field bboxes excluded)
            if stream_type == "left" and _detector is not None:
                try:
                    res = _detector.detect_objects(frame, camera_id="webui_stream")
                    for det in res.detections:
                        if det.color.lower().startswith("field"):
                            continue
                        color = _detector._get_class_color(det.color)
                        # Box
                        cv2.rectangle(
                            frame,
                            (det.bbox_x, det.bbox_y),
                            (det.bbox_x + det.bbox_w, det.bbox_y + det.bbox_h),
                            color,
                            2,
                        )
                        # Center
                        cv2.circle(frame, (det.center_x, det.center_y), 5, color, -1)
                        # Label
                        cv2.putText(
                            frame,
                            f"{det.color} {det.confidence:.2f}",
                            (det.bbox_x, max(10, det.bbox_y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                        )
                except Exception as detect_err:
                    logger.debug(f"WebUI Stream detection skipping frame: {detect_err}")

            from config.Vision import STEREO_JPEG_QUALITY

            _, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, STEREO_JPEG_QUALITY]
            )
            frame_bytes = encoded.tobytes()

        # Fallback to single camera if no stereo
        elif stream_type == "left":
            single = storage.get_latest_single()
            if single:
                _, img, _ = single
                from config.Vision import STEREO_JPEG_QUALITY

                _, encoded = cv2.imencode(
                    ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, STEREO_JPEG_QUALITY]
                )
                frame_bytes = encoded.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + (frame_bytes or _placeholder)
            + b"\r\n"
        )

        time.sleep(0.06)  # ~15 fps


@app.get("/api/stream/rgb")
async def stream_rgb():
    """Stream left/main RGB camera feed as MJPEG"""
    return StreamingResponse(
        frame_generator("left"), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/stream/depth")
async def stream_depth():
    """Stream right/secondary camera feed as MJPEG"""
    return StreamingResponse(
        frame_generator("depth"), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/api/command")
async def api_send_command(command_data: Dict[str, Any]):
    try:
        cmd_type = command_data.get("type", "direct")

        if cmd_type == "estop":
            from servers.SequenceServer import SequenceQueryHandler
            from servers.AutoRTIntegration import AutoRTHandler

            # Abort the live executor owned by SequenceQueryHandler, not a new instance
            try:
                handler = SequenceQueryHandler()
                if handler._executor is not None:
                    handler._executor.abort()
            except Exception as e:
                logger.warning(f"E-Stop: executor abort failed: {e}")
            try:
                AutoRTHandler.get_instance().stop_loop()
            except Exception:
                pass
            # Tell Unity to immediately stop all active movements
            try:
                broadcaster = get_command_broadcaster()
                broadcaster.send_command(
                    {"command_type": "halt_all", "robot_id": "system"}
                )
            except Exception as e:
                logger.warning(f"E-Stop: halt_all send failed: {e}")
            return {
                "success": True,
                "message": "E-Stop: sequence aborted, AutoRT stopped, movements halted",
            }

        # Determine if it's an AutoRT or Direct command
        if cmd_type == "autort":
            action = command_data.get("action")
            from servers.AutoRTIntegration import AutoRTHandler

            handler = AutoRTHandler.get_instance()

            if action == "start":
                res = handler.start_loop(robot_ids=["Robot1", "Robot2"])
            elif action == "stop":
                res = handler.stop_loop()
            else:
                res = {"success": False, "error": "Unknown AutoRT action"}
            return res

        else:
            # Direct gripper/jog commands
            broadcaster = get_command_broadcaster()
            robot_id = command_data.get("robot_id")
            cmd = command_data.get("command", {})

            if robot_id:
                success = broadcaster.send_command_to_robot(robot_id, cmd)
            else:
                success = broadcaster.send_command(cmd)

            return {"success": success, "message": "Command dispatched"}

    except Exception as e:
        logger.error(f"API Command Error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/reset")
async def api_reset_simulation():
    """Trigger a simulation reset directly, bypassing the LLM pipeline."""

    def _do_reset():
        broadcaster = get_command_broadcaster()
        return broadcaster.send_command_and_wait(
            {"command_type": "reset_simulation", "robot_id": "system"},
            timeout=15.0,
        )

    try:
        completion = await asyncio.get_running_loop().run_in_executor(None, _do_reset)
        if completion is None:
            return {
                "success": False,
                "error": "Unity did not respond (timeout or not connected)",
            }
        if not completion.get("success", False):
            return {"success": False, "error": "Reset command failed in Unity"}
        return {"success": True}
    except Exception as e:
        logger.error(f"API Reset Error: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/autort/tasks")
async def api_autort_tasks():
    try:
        from servers.AutoRTIntegration import AutoRTHandler

        return AutoRTHandler.get_instance().get_pending_tasks()
    except Exception as e:
        logger.error(f"AutoRT tasks fetch error: {e}")
        return {"success": False, "tasks": [], "error": str(e)}


@app.post("/api/autort/generate")
async def api_autort_generate(body: Dict[str, Any]):
    """Trigger manual task generation and broadcast results via WebSocket."""
    try:
        from servers.AutoRTIntegration import AutoRTHandler

        handler = AutoRTHandler.get_instance()
        result = handler.generate_tasks(
            num_tasks=body.get("num_tasks"),
            robot_ids=body.get("robot_ids"),
            strategy=body.get("strategy", "balanced"),
        )
        if result.get("tasks") and manager.active_connections:
            await manager.broadcast(
                json.dumps(
                    {
                        "type": "autort_tasks",
                        "tasks": result["tasks"],
                        "loop_running": result.get("loop_running", False),
                    }
                )
            )
        return result
    except Exception as e:
        logger.error(f"AutoRT generate error: {e}")
        return {"success": False, "tasks": [], "error": str(e)}


@app.post("/api/autort/execute")
async def api_autort_execute(body: Dict[str, Any]):
    """Execute an approved task by ID."""
    try:
        task_id = body.get("task_id")
        if not task_id:
            return {"success": False, "error": "task_id is required"}
        from servers.AutoRTIntegration import AutoRTHandler

        return AutoRTHandler.get_instance().execute_task(task_id)
    except Exception as e:
        logger.error(f"AutoRT execute error: {e}")
        return {"success": False, "error": str(e)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: "WebSocket"):
    await manager.connect(websocket)
    try:
        # Send historical logs on connect
        try:
            from config.Servers import LOG_DIR
            from pathlib import Path

            log_path = Path(LOG_DIR) / "server_logs.txt"
            if log_path.exists():
                with open(log_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    # Send up to the last 200 lines of history
                    for line in lines[-200:]:
                        line = line.strip()
                        if line:
                            # Parse out a rudimentary level for coloring
                            level = "info"
                            if "[WARNING]" in line:
                                level = "warning"
                            elif "[ERROR]" in line:
                                level = "error"
                            elif "[CRITICAL]" in line:
                                level = "error"
                            await websocket.send_text(
                                json.dumps(
                                    {"type": "log", "level": level, "message": line}
                                )
                            )
        except Exception as e:
            logger.error(f"Failed to send log history: {e}")

        while True:
            # Receive commands from Web UI
            data = await websocket.receive_text()
            try:
                message = json.loads(data)

                # Handle sequence prompts
                if message.get("type") == "sequence_prompt":
                    prompt = message.get("prompt", "")
                    robot_id = message.get("robot_id", "Robot1")

                    # Log receipt
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "log",
                                "level": "info",
                                "message": f"Processing prompt: '{prompt}' for {robot_id}...",
                            }
                        )
                    )

                    # Execute in background thread so WS doesn't block.
                    # Capture loop reference here (not inside the closure) to
                    # avoid a race with _main_loop being set after thread starts.
                    _captured_loop = _main_loop

                    def _run_sequence():
                        """Execute sequence and push results to all WebSocket clients."""
                        try:
                            handler = _get_sequence_handler()
                            result = handler.execute_sequence(
                                command_text=prompt,
                                robot_id=robot_id,
                                auto_execute=True,
                            )

                            # Bridge from sync thread → async event loop
                            if _captured_loop:
                                try:
                                    asyncio.run_coroutine_threadsafe(
                                        manager.broadcast(
                                            json.dumps(
                                                {
                                                    "type": "sequence_result",
                                                    "data": result,
                                                }
                                            )
                                        ),
                                        _captured_loop,
                                    ).result(timeout=30)
                                except Exception as broadcast_err:
                                    logger.error(f"Broadcast failed: {broadcast_err}")

                        except Exception as e:
                            logger.error(f"Sequence execution error: {e}")
                            if _captured_loop:
                                try:
                                    asyncio.run_coroutine_threadsafe(
                                        manager.broadcast(
                                            json.dumps(
                                                {
                                                    "type": "log",
                                                    "level": "error",
                                                    "message": f"Execution failed: {e}",
                                                }
                                            )
                                        ),
                                        _captured_loop,
                                    ).result(timeout=30)
                                except Exception as broadcast_err:
                                    logger.error(
                                        f"Error broadcast failed: {broadcast_err}"
                                    )

                    threading.Thread(target=_run_sequence, daemon=True).start()

            except json.JSONDecodeError:
                pass

    except Exception as e:
        if (
            _FASTAPI_AVAILABLE
            and WebSocketDisconnect
            and isinstance(e, WebSocketDisconnect)
        ):
            manager.disconnect(websocket)
        else:
            logger.error(f"WebSocket error: {e}")
            manager.disconnect(websocket)


# Background task to push world state updates to WebSockets
async def state_broadcaster():
    """Periodically pushes world state to connected UI clients"""
    while True:
        if manager.active_connections:
            try:
                world_state = get_world_state()
                world_state_data = {
                    "type": "world_state",
                    "data": {
                        "robots": [
                            r.__dict__ for r in world_state._robot_states.values()
                        ],
                        "objects": [o.__dict__ for o in world_state.get_all_objects()],
                    },
                }
                await manager.broadcast(json.dumps(world_state_data))
            except Exception as e:
                logger.warning(f"state_broadcaster error: {e}")
        await asyncio.sleep(0.5)  # 2 Hz updates


def broadcast_stereo_pointcloud(points: Any, colors: Any, scene_span: float = 1.5):
    """
    Broadcasts a stereo point cloud to connected Web UI clients.

    points: (N,3) float32 Unity world, colors: (N,3) uint8 RGB, scene_span: scene diameter in metres.
    """
    if not manager.active_connections or not _main_loop:
        return
    try:
        import base64
        import numpy as np

        pts_b64 = base64.b64encode(points.astype(np.float32).tobytes()).decode("utf-8")
        clr_b64 = base64.b64encode(colors.astype(np.uint8).tobytes()).decode("utf-8")
        payload = json.dumps(
            {
                "type": "stereo_pointcloud",
                "data": {
                    "points_b64": pts_b64,
                    "colors_b64": clr_b64,
                    "scene_span": float(scene_span),
                },
            }
        )
        asyncio.run_coroutine_threadsafe(manager.broadcast(payload), _main_loop)
    except Exception as e:
        logger.error(f"Failed to broadcast stereo point cloud: {e}")


def broadcast_vgn_debug(data: Dict[str, Any]):
    """
    Broadcasts VGN point cloud and TSDF data to connected Web UI clients.
    Safely schedules the broadcast on the captured uvicorn event loop.
    """
    if manager.active_connections and _main_loop:
        try:
            payload = json.dumps({"type": "vgn_debug", "data": data})
            asyncio.run_coroutine_threadsafe(manager.broadcast(payload), _main_loop)
        except Exception as e:
            logger.error(f"Failed to broadcast VGN debug data: {e}")


def run_webui_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server blocking."""
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError(
            "WebUIServer requires fastapi and uvicorn. "
            "Install them with: pip install fastapi uvicorn"
        )
    logger.debug(f"Starting WebUIServer on http://{host}:{port}")
    assert uvicorn is not None and isinstance(app, FastAPI)
    uvicorn.run(app, host=host, port=port, log_level="warning")


def run_webui_server_background(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server in a background thread."""
    thread = threading.Thread(
        target=run_webui_server, args=(host, port), daemon=True, name="web-ui-server"
    )
    thread.start()
    return thread


if __name__ == "__main__":
    run_webui_server()
