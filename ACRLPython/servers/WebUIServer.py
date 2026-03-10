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
from typing import Dict, Any, Optional

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, StreamingResponse
    import uvicorn
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    # Provide stubs so the module can be imported without fastapi installed.
    # A clear RuntimeError is raised only when run_webui_server() is actually called.
    FastAPI = WebSocket = WebSocketDisconnect = BackgroundTasks = None  # type: ignore
    StaticFiles = StaticFiles if False else None  # type: ignore
    HTMLResponse = StreamingResponse = None  # type: ignore
    uvicorn = None  # type: ignore

# Try to import core and orchestrator modules
try:
    from config.Servers import DEFAULT_HOST
    from servers.CommandServer import get_command_broadcaster
    from operations.WorldState import get_world_state
except ImportError:
    from ..config.Servers import DEFAULT_HOST
    from ..servers.CommandServer import get_command_broadcaster
    from ..operations.WorldState import get_world_state

logger = logging.getLogger(__name__)

# The running uvicorn event loop — captured at startup so background threads
# can schedule coroutines on it via run_coroutine_threadsafe().
_main_loop: Optional[asyncio.AbstractEventLoop] = None

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

# Initialize FastAPI app — only when the dependency is available.
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

app = FastAPI(title="ACRL Mission Control") if _FASTAPI_AVAILABLE else _NoOpApp()

# Track active websocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")

manager = ConnectionManager()

# Determine paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
WEBUI_DIR = os.path.join(PROJECT_ROOT, "ACRLDashboard")

# Ensure webui dir exists
os.makedirs(WEBUI_DIR, exist_ok=True)

# Mount static files — only when fastapi is available
if _FASTAPI_AVAILABLE:
    app.mount("/static", StaticFiles(directory=WEBUI_DIR), name="static")

@app.get("/")
async def get_index():
    index_path = os.path.join(WEBUI_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>WebUI Not Found</h1><p>Please create index.html in the webui directory.</p>")

def _check_ros_connected() -> bool:
    """Return True if the ROSBridge is currently connected."""
    try:
        from ros2.ROSBridge import ROSBridge
        bridge = ROSBridge.get_instance()
        return getattr(bridge, "_connected", False)
    except Exception:
        return False


def _check_unity_connected() -> bool:
    """Return True if at least one Unity client is connected to CommandServer."""
    try:
        broadcaster = get_command_broadcaster()
        return getattr(broadcaster, "_connected", False)
    except Exception:
        return False


def _check_camera_available() -> bool:
    """Return True if at least one camera frame has been received."""
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
        url = LMSTUDIO_BASE_URL.rstrip('/')
        req = urllib.request.Request(f"{url}/models", method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


@app.get("/api/status")
async def api_status():
    """
    Return live connectivity status for all subsystems.

    Polled by the dashboard every 5 seconds to update status badges.
    """
    return {
        "backend": True,
        "ros": _check_ros_connected(),
        "unity": _check_unity_connected(),
        "camera": _check_camera_available(),
        "llm": _check_llm_studio_connected(),
    }


@app.get("/api/world_state")
async def api_world_state():
    """Get the current world state"""
    world_state = get_world_state()
    # WorldState has method to dump all
    try:
        data = {
            "robots": world_state.get_all_robot_states(),
            "objects": world_state.get_all_object_states()
        }
        return data
    except Exception as e:
        logger.error(f"Error getting world state: {e}")
        return {"error": str(e)}

async def frame_generator(stream_type="left"):
    """Generator for MJPEG streaming from UnifiedImageStorage"""
    try:
        from servers.ImageStorageCore import UnifiedImageStorage
        import cv2
    except ImportError:
        logger.error("Could not import Vision dependencies for streaming")
        return
        
    storage = UnifiedImageStorage()
    
    while True:
        frame_bytes = None
        
        # Try to get stereo first as it's the primary feed
        stereo = storage.get_latest_stereo()
        if stereo:
            _, imgL, imgR, _ = stereo
            frame = imgL if stream_type == "left" else imgR
            _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = encoded.tobytes()
            
        # Fallback to single camera if no stereo
        elif stream_type == "left":
            single = storage.get_latest_single()
            if single:
                _, img, _ = single
                _, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = encoded.tobytes()
                
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        await asyncio.sleep(0.06)  # ~15 fps

@app.get("/api/stream/rgb")
async def stream_rgb():
    """Stream left/main RGB camera feed as MJPEG"""
    return StreamingResponse(frame_generator("left"), 
                           media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/stream/depth")
async def stream_depth():
    """Stream right/secondary camera feed as MJPEG"""
    return StreamingResponse(frame_generator("right"), 
                           media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/api/command")
async def api_send_command(command_data: Dict[str, Any]):
    """Send a command directly to the CommandBroadcaster or AutoRT"""
    try:
        cmd_type = command_data.get("type", "direct")
        
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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
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
                    await websocket.send_text(json.dumps({
                        "type": "log",
                        "level": "info",
                        "message": f"Processing prompt: '{prompt}' for {robot_id}..."
                    }))
                    
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
                                auto_execute=True
                            )

                            # Bridge from sync thread → async event loop
                            if _captured_loop:
                                try:
                                    asyncio.run_coroutine_threadsafe(
                                        manager.broadcast(json.dumps({
                                            "type": "sequence_result",
                                            "data": result
                                        })),
                                        _captured_loop
                                    ).result(timeout=30)
                                except Exception as broadcast_err:
                                    logger.error(f"Broadcast failed: {broadcast_err}")

                        except Exception as e:
                            logger.error(f"Sequence execution error: {e}")
                            if _captured_loop:
                                try:
                                    asyncio.run_coroutine_threadsafe(
                                        manager.broadcast(json.dumps({
                                            "type": "log",
                                            "level": "error",
                                            "message": f"Execution failed: {e}"
                                        })),
                                        _captured_loop
                                    ).result(timeout=30)
                                except Exception as broadcast_err:
                                    logger.error(f"Error broadcast failed: {broadcast_err}")
                            
                    threading.Thread(target=_run_sequence, daemon=True).start()
                    
            except json.JSONDecodeError:
                pass
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Background task to push world state updates to WebSockets
async def state_broadcaster():
    """Periodically pushes world state to connected UI clients"""
    while True:
        if manager.active_connections:
            try:
                world_state_data = {
                    "type": "world_state",
                    "data": {
                        "robots": get_world_state().get_all_robot_states(),
                        "objects": get_world_state().get_all_object_states()
                    }
                }
                await manager.broadcast(json.dumps(world_state_data))
            except Exception:
                pass
        await asyncio.sleep(0.5)  # 2 Hz updates

# Fast background hook
@app.on_event("startup")
async def startup_event():
    """Capture the running event loop and start background tasks."""
    global _main_loop
    _main_loop = asyncio.get_running_loop()

    asyncio.create_task(state_broadcaster())

    # Attach to root logger — route all backend log records to connected UIs.
    try:
        from core.LoggingSetup import add_websocket_handler
    except ImportError:
        from ..core.LoggingSetup import add_websocket_handler

    def log_callback(msg: str, level: str):
        """
        Forward a log record to all active WebSocket clients.

        Called from arbitrary threads by the logging system. Uses
        run_coroutine_threadsafe to safely schedule the broadcast on the
        captured uvicorn event loop without dropping records.
        """
        if manager.active_connections and _main_loop:
            log_data = json.dumps({"type": "log", "message": msg, "level": level})
            asyncio.run_coroutine_threadsafe(
                manager.broadcast(log_data), _main_loop
            )

    add_websocket_handler(log_callback)

def run_webui_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server blocking."""
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError(
            "WebUIServer requires fastapi and uvicorn. "
            "Install them with: pip install fastapi uvicorn"
        )
    logger.info(f"Starting WebUIServer on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")

def run_webui_server_background(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server in a background thread."""
    thread = threading.Thread(
        target=run_webui_server,
        args=(host, port),
        daemon=True,
        name="web-ui-server"
    )
    thread.start()
    return thread

if __name__ == "__main__":
    run_webui_server()
