#!/usr/bin/env python3
"""
RunRobotController.py - Unified orchestrator for robot control

Starts all required servers in a single process:
- ImageServer (ports 5005, 5006) - receives images
- CommandServer (port 5010) - sends commands, receives completions
- SequenceServer (port 5013) - processes command sequences
- WorldStateServer (port 5014) - receives robot/object state updates
- AutoRTServer (port 5015) - autonomous task generation

Usage:
    python -m orchestrators.RunRobotController

    # With options
    python -m orchestrators.RunRobotController --model gemma-3-12b
"""

import argparse
import signal
import threading
import logging
from typing import Optional

# Import config
try:
    from config.Servers import (
        DEFAULT_HOST,
        STREAMING_SERVER_PORT,
        STEREO_DETECTION_PORT,
        LLM_RESULTS_PORT,
        SEQUENCE_SERVER_PORT,
        WORLD_STATE_PORT,
        AUTORT_SERVER_PORT,
        DEFAULT_LMSTUDIO_MODEL,
        LMSTUDIO_BASE_URL,
    )
    from config.Vision import (
        ENABLE_VISION_STREAMING,
        YOLO_MODEL_PATH,
        ENABLE_VISION_VISUALIZATION,
        VISION_STREAM_FPS,
        ENABLE_OBJECT_TRACKING,
        SHARED_VISION_STATE_ENABLED,
    )
    from config.KnowledgeGraph import KNOWLEDGE_GRAPH_ENABLED
    from core.LoggingSetup import setup_logging, enable_file_logging

    try:
        from config.ROS import ROS_ENABLED, AUTO_CONNECT_ROS
        from ros2.ROSBridge import ROSBridge
    except ImportError:
        ROS_ENABLED = False
        AUTO_CONNECT_ROS = False
        ROSBridge = None  # type: ignore
except ImportError:
    from ..config.Servers import (
        DEFAULT_HOST,
        STREAMING_SERVER_PORT,
        STEREO_DETECTION_PORT,
        LLM_RESULTS_PORT,
        SEQUENCE_SERVER_PORT,
        WORLD_STATE_PORT,
        AUTORT_SERVER_PORT,
        DEFAULT_LMSTUDIO_MODEL,
        LMSTUDIO_BASE_URL,
    )
    from ..config.Vision import (
        ENABLE_VISION_STREAMING,
        YOLO_MODEL_PATH,
        ENABLE_VISION_VISUALIZATION,
        VISION_STREAM_FPS,
        ENABLE_OBJECT_TRACKING,
        SHARED_VISION_STATE_ENABLED,
    )
    from ..config.KnowledgeGraph import KNOWLEDGE_GRAPH_ENABLED
    from ..core.LoggingSetup import setup_logging, enable_file_logging

    try:
        from ..config.ROS import ROS_ENABLED, AUTO_CONNECT_ROS
        from ..ros2.ROSBridge import ROSBridge
    except ImportError:
        ROS_ENABLED = False
        AUTO_CONNECT_ROS = False
        ROSBridge = None  # type: ignore

# Import servers - handle both direct execution and package import
try:
    from ..servers.ImageServer import run_image_server_background
    from ..servers.CommandServer import (
        run_command_server_background,
        get_command_broadcaster,
    )
    from ..servers.SequenceServer import run_sequence_server_background
    from ..servers.WorldStateServer import WorldStateServer
    from ..servers.AutoRTServer import AutoRTServer
except ImportError:
    # Running as python -m orchestrators.RunRobotController
    from servers.ImageServer import run_image_server_background
    from servers.CommandServer import (
        run_command_server_background,
        get_command_broadcaster,
    )
    from servers.SequenceServer import run_sequence_server_background
    from servers.WorldStateServer import WorldStateServer
    from servers.AutoRTServer import AutoRTServer

# Setup centralized logging (do this early before any logging calls)
logger = setup_logging(__name__)


class RobotController:
    """
    Unified robot controller that manages all servers.

    Provides a single entry point for starting the entire Python backend.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        single_port: int = STREAMING_SERVER_PORT,
        stereo_port: int = STEREO_DETECTION_PORT,
        command_port: int = LLM_RESULTS_PORT,
        sequence_port: int = SEQUENCE_SERVER_PORT,
        world_state_port: int = WORLD_STATE_PORT,
        autort_port: int = AUTORT_SERVER_PORT,
        model: str = DEFAULT_LMSTUDIO_MODEL,
        check_completion: bool = True,
        env: str = "sim",
        web_port: Optional[int] = None,
    ):
        """
        Initialize the robot controller.

        Args:
            host: Host to bind servers to
            single_port: Port for single camera images
            stereo_port: Port for stereo image pairs
            command_port: Port for commands/results (bidirectional)
            sequence_port: Port for sequence execution
            world_state_port: Port for world state streaming
            autort_port: Port for AutoRT task generation
            model: LLM model for parsing
            check_completion: Whether to wait for Unity completion signals
            env: Execution environment — "sim" (Unity) or "real" (physical robot)
            web_port: If set, start the Web UI server on this port
        """
        self._host = host
        self._single_port = single_port
        self._stereo_port = stereo_port
        self._command_port = command_port
        self._sequence_port = sequence_port
        self._world_state_port = world_state_port
        self._autort_port = autort_port
        self._model = model
        self._check_completion = check_completion
        self._env = env
        self._web_port = web_port

        self._image_server = None
        self._command_server = None
        self._sequence_server = None
        self._world_state_server = None
        self._autort_server = None
        self._vision_processor = None
        self._graph_builder = None
        self._web_server_thread = None
        self._running = False
        self._stop_event = threading.Event()

    def _wire_world_state_to_rag(self):
        """
        Wire WorldState into RAG QueryEngine for context-aware operation selection.

        This enables the RAG system to:
        - Filter operations based on reachability
        - Boost operations targeting reachable objects
        - Downrank operations for stale objects or objects grasped by other robots
        - Include world state context in LLM prompts
        """
        try:
            from operations.WorldState import get_world_state

            # Import SequenceQueryHandler to access the singleton
            try:
                from servers.SequenceServer import SequenceQueryHandler
            except ImportError:
                from ..servers.SequenceServer import SequenceQueryHandler

            world_state = get_world_state()

            # Get the command parser from SequenceQueryHandler singleton
            # The parser is created by SequenceQueryHandler, not SequenceServer
            if self._sequence_server is not None:
                handler = SequenceQueryHandler()
                if handler.is_ready() and handler._parser is not None:
                    command_parser = handler._parser
                    if (
                        hasattr(command_parser, "rag")
                        and command_parser.rag is not None
                    ):
                        if (
                            hasattr(command_parser.rag, "query_engine")
                            and command_parser.rag.query_engine is not None
                        ):
                            command_parser.rag.query_engine.set_world_state(world_state)
                            logger.info(
                                "✓ WorldState wired into RAG QueryEngine for context-aware search"
                            )
                        else:
                            logger.debug(
                                "RAG query_engine not available, skipping WorldState wiring"
                            )
                    else:
                        logger.debug(
                            "RAG system not initialized in CommandParser, skipping WorldState wiring"
                        )
                else:
                    logger.debug(
                        "SequenceQueryHandler not ready, skipping WorldState wiring"
                    )
            else:
                logger.debug(
                    "SequenceServer not initialized, skipping WorldState wiring"
                )

        except Exception as e:
            logger.warning(f"Failed to wire WorldState into RAG: {e}")
            logger.debug(
                "This is non-critical - RAG will work without world state context"
            )

    def _wire_world_state_callbacks(self):
        """
        Wire WorldStateServer callbacks to trigger confidence decay on state updates.

        Registers a callback that updates object confidence based on which objects
        are currently detected in each frame.
        """
        try:
            from operations.WorldState import get_world_state

            world_state = get_world_state()

            # Define callback to sync Unity state into the operations WorldState singleton
            def on_state_update(state_data):
                """Called on each world state update from Unity."""
                try:
                    # Forward robot states into operations WorldState singleton
                    for robot in state_data.get("robots", []):
                        robot_id = robot.get("robot_id")
                        if robot_id:
                            world_state.update_robot_state(robot_id, robot)

                    # Trigger confidence decay based on currently visible objects
                    objects = state_data.get("objects", [])
                    seen_object_ids = {
                        obj.get("object_id") for obj in objects if obj.get("object_id")
                    }
                    world_state.decay_object_confidence(seen_object_ids)

                except Exception as e:
                    logger.error(f"Error in state update callback: {e}")

            # Register callback with WorldStateServer
            if self._world_state_server:
                self._world_state_server.register_update_callback(on_state_update)
                logger.info(
                    "✓ WorldStateServer callback registered for confidence decay"
                )
            else:
                logger.debug(
                    "WorldStateServer not available, skipping callback registration"
                )

        except Exception as e:
            logger.warning(f"Failed to wire WorldStateServer callbacks: {e}")
            logger.debug(
                "This is non-critical - confidence decay will not be automatic"
            )

    def _wire_knowledge_graph(self):
        """
        Wire GraphBuilder into WorldStateServer as an update callback.

        Creates KnowledgeGraph and GraphBuilder singletons, then registers
        GraphBuilder.on_state_update as a WorldStateServer callback so the
        graph stays synchronized with every Unity state push.

        Only runs if KNOWLEDGE_GRAPH_ENABLED is True.
        """
        if not KNOWLEDGE_GRAPH_ENABLED:
            logger.debug("Knowledge graph disabled (KNOWLEDGE_GRAPH_ENABLED=false)")
            return

        try:
            from knowledge_graph._singleton import get_knowledge_graph
            from knowledge_graph.GraphBuilder import GraphBuilder
            from operations.WorldState import get_world_state

            kg = get_knowledge_graph()
            world_state = get_world_state()
            self._graph_builder = GraphBuilder(kg, world_state)

            if self._world_state_server:
                self._world_state_server.register_update_callback(
                    self._graph_builder.on_state_update
                )
                logger.info(
                    "✓ KnowledgeGraph wired — graph updates on every world state push"
                )
            else:
                logger.debug(
                    "WorldStateServer not available, skipping KnowledgeGraph wiring"
                )

        except Exception as e:
            logger.warning(f"Failed to wire KnowledgeGraph: {e}")
            logger.debug(
                "Non-critical — knowledge graph will not be updated automatically"
            )

    def _auto_connect_ros(self):
        """
        Connect to ROS bridge on startup if AUTO_CONNECT_ROS is enabled.

        Runs in a background thread so a missing Docker container does not
        block the Python backend from starting.
        """
        if not ROS_ENABLED or not AUTO_CONNECT_ROS:
            return

        if ROSBridge is None:
            logger.warning("AUTO_CONNECT_ROS=True but ROSBridge could not be imported")
            return

        def _connect():
            bridge = ROSBridge.get_instance()  # type: ignore
            success = bridge.connect()
            if success:
                logger.info("✓ ROS bridge connected (AUTO_CONNECT_ROS)")
            else:
                logger.warning(
                    "ROS bridge auto-connect failed — Docker may not be running. "
                    "Motion commands will fall back to Unity IK."
                )

        thread = threading.Thread(target=_connect, name="ros-auto-connect", daemon=True)
        thread.start()

    def start(self):
        """Start all servers."""
        if self._running:
            logger.warning("RobotController already running")
            return

        # Start ImageServer (ports 5005, 5006)
        logger.info(
            f"Starting ImageServer (single: {self._single_port}, stereo: {self._stereo_port})"
        )
        self._image_server = run_image_server_background(
            single_port=self._single_port,
            stereo_port=self._stereo_port,
            host=self._host,
        )

        # Start CommandServer (port 5010) - bidirectional for commands and completions
        logger.info(f"Starting CommandServer (port: {self._command_port})")
        self._command_server = run_command_server_background(
            port=self._command_port, host=self._host
        )

        # Initialize and start SequenceServer (port 5013)
        logger.info(f"Starting SequenceServer (port: {self._sequence_port})")
        self._sequence_server = run_sequence_server_background(
            lm_studio_url=LMSTUDIO_BASE_URL,
            model=self._model,
            check_completion=self._check_completion,
        )

        # Start WorldStateServer (port 5014) only in sim mode.
        # In real mode, world state is populated by perception operations only.
        from core.TCPServerBase import ServerConfig

        if self._env == "sim":
            logger.info(f"Starting WorldStateServer (port: {self._world_state_port})")
            world_state_config = ServerConfig(
                host=self._host, port=self._world_state_port
            )
            self._world_state_server = WorldStateServer(config=world_state_config)
            self._world_state_server.start()
        else:
            logger.info(
                "Real env: WorldStateServer disabled — WorldState populated by perception only"
            )

        # Start AutoRTServer (port 5015) - autonomous task generation
        logger.info(f"Starting AutoRTServer (port: {self._autort_port})")
        autort_config = ServerConfig(host=self._host, port=self._autort_port)
        self._autort_server = AutoRTServer(config=autort_config)
        self._autort_server.start()

        # Share resources between servers
        broadcaster = get_command_broadcaster()
        # SequenceExecutor will use this for sending commands

        # Wire WorldState into RAG for context-aware operation selection
        self._wire_world_state_to_rag()

        # Wire WorldStateServer callbacks and knowledge graph only when the
        # server is running (sim mode); in real mode there is no server to wire.
        if self._env == "sim":
            self._wire_world_state_callbacks()
            self._wire_knowledge_graph()

        # Initialize hardware and camera singletons for the selected environment.
        # This call seeds the module-level cache so all subsequent lazy accessors
        # via core.Imports return the correct adapter for --env sim or --env real.
        from core.Imports import get_hardware_interface, get_camera_provider

        hw = get_hardware_interface(env=self._env)
        cam = get_camera_provider(env=self._env)
        logger.info(f"✓ HardwareInterface: {type(hw).__name__}")
        logger.info(f"✓ CameraProvider:    {type(cam).__name__}")

        # Auto-connect to ROS bridge if enabled
        self._auto_connect_ros()

        # Start Web UI server if requested
        if self._web_port:
            from servers.WebUIServer import run_webui_server_background

            self._web_server_thread = run_webui_server_background(
                host=self._host, port=self._web_port
            )
            logger.info(
                f"  Web UI:                 http://{self._host}:{self._web_port}"
            )

        self._running = True

        # All servers started — now safe to open the log file
        enable_file_logging()

        # Initialize vision streaming if enabled
        if ENABLE_VISION_STREAMING:
            try:
                import os
                import platform
                from vision.YOLODetector import YOLODetector
                from vision.VisionProcessor import VisionProcessor

                # Check if YOLO model exists
                if not os.path.exists(YOLO_MODEL_PATH):
                    logger.warning(
                        f"YOLO model not found at {YOLO_MODEL_PATH}. "
                        "Vision streaming disabled. Please ensure the model file exists."
                    )
                else:
                    logger.info(
                        f"Initializing VisionProcessor with YOLO model: {YOLO_MODEL_PATH}"
                    )

                    # Initialize YOLO detector
                    detector = YOLODetector(model_path=YOLO_MODEL_PATH)

                    # Determine if we should use main thread (macOS with visualization)
                    use_main_thread = (
                        platform.system() == "Darwin" and ENABLE_VISION_VISUALIZATION
                    )

                    if use_main_thread:
                        logger.info(
                            "macOS detected with visualization enabled - "
                            "VisionProcessor will run in main thread (blocking)"
                        )

                    # Create vision processor with config
                    self._vision_processor = VisionProcessor(
                        detector=detector,
                        fps=VISION_STREAM_FPS,
                        enable_tracking=ENABLE_OBJECT_TRACKING,
                        enable_shared_state=SHARED_VISION_STATE_ENABLED,
                        enable_visualization=ENABLE_VISION_VISUALIZATION,
                        use_main_thread=use_main_thread,
                    )

                    if use_main_thread:
                        # Will run in main thread (blocking) - start it in wait() method
                        logger.info(
                            "VisionProcessor initialized (will start in main thread)"
                        )
                    else:
                        # Start in background thread (non-blocking)
                        self._vision_processor.start()
                        logger.info("VisionProcessor started in background thread")

            except Exception as e:
                logger.error(
                    f"Failed to initialize VisionProcessor: {e}", exc_info=True
                )
                logger.warning("Continuing without vision streaming")
                self._vision_processor = None

        logger.info("=" * 60)
        logger.info("RobotController started successfully!")
        logger.info("=" * 60)
        logger.info(f"  Environment:            {self._env}")
        logger.info(f"  Image Server (single):  {self._host}:{self._single_port}")
        logger.info(f"  Image Server (stereo):  {self._host}:{self._stereo_port}")
        logger.info(f"  Command Server:         {self._host}:{self._command_port}")
        logger.info(f"  Sequence Server:        {self._host}:{self._sequence_port}")
        if self._env == "sim":
            logger.info(
                f"  World State Server:     {self._host}:{self._world_state_port}"
            )
        else:
            logger.info(f"  World State Server:     Disabled (real env)")
        logger.info(f"  AutoRT Server:          {self._host}:{self._autort_port}")
        logger.info(f"  LLM Model:              {self._model}")
        if self._web_port:
            logger.info(
                f"  Web UI:                 http://{self._host}:{self._web_port}"
            )
        if ENABLE_VISION_STREAMING and self._vision_processor:
            logger.info(f"  Vision Streaming:       Enabled ({VISION_STREAM_FPS} FPS)")
            if ENABLE_VISION_VISUALIZATION:
                logger.info(f"  Visualization:          Enabled (press 'q' to close)")
        if KNOWLEDGE_GRAPH_ENABLED:
            logger.info(f"  Knowledge Graph:        Enabled")
        else:
            logger.info(
                f"  Knowledge Graph:        Disabled (set KNOWLEDGE_GRAPH_ENABLED=true to enable)"
            )
        logger.info("=" * 60)

    def stop(self):
        """Stop all servers."""
        if not self._running:
            return

        logger.info("Stopping RobotController...")

        # Mark as stopped first to prevent re-entry
        self._running = False
        self._stop_event.set()

        # Stop vision processor first (may need to close OpenCV windows)
        try:
            if self._vision_processor:
                self._vision_processor.stop()
                logger.info("VisionProcessor stopped")
        except Exception as e:
            logger.error(f"Error stopping VisionProcessor: {e}")

        try:
            if self._image_server:
                self._image_server.stop()
        except Exception as e:
            logger.error(f"Error stopping ImageServer: {e}")

        try:
            if self._command_server:
                self._command_server.stop()
        except Exception as e:
            logger.error(f"Error stopping CommandServer: {e}")

        try:
            if self._sequence_server:
                self._sequence_server.stop()
        except Exception as e:
            logger.error(f"Error stopping SequenceServer: {e}")

        try:
            if self._world_state_server:
                self._world_state_server.stop()
                logger.info("WorldStateServer stopped")
        except Exception as e:
            logger.error(f"Error stopping WorldStateServer: {e}")

        try:
            if self._autort_server:
                self._autort_server.stop()
                logger.info("AutoRTServer stopped")
        except Exception as e:
            logger.error(f"Error stopping AutoRTServer: {e}")

        logger.info("RobotController stopped")

    def is_running(self) -> bool:
        """Check if controller is running."""
        return self._running

    def wait(self):
        """Wait for controller to stop (blocking)."""
        try:
            # If VisionProcessor is configured for main thread, run it now (blocking)
            if (
                self._vision_processor
                and hasattr(self._vision_processor, "use_main_thread")
                and self._vision_processor.use_main_thread
            ):
                logger.info(
                    "Starting VisionProcessor in main thread (blocking until 'q' or Ctrl+C)"
                )
                self._vision_processor.run()  # Blocking call
                # When run() returns, stop the controller
                self.stop()
            else:
                # Block until stop() sets the event (wakes immediately on shutdown,
                # unlike a polling sleep loop).  We still log camera status
                # periodically by using a short timeout on each wait() call.
                while not self._stop_event.wait(timeout=1.0):
                    if self._image_server:
                        storage = self._image_server.get_storage()
                        cameras = storage.get_all_camera_ids()
                        if cameras:
                            logger.debug(f"Active cameras: {cameras}")

        except KeyboardInterrupt:
            pass


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified Robot Controller - Start all Python servers"
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind to")
    parser.add_argument(
        "--model",
        default=DEFAULT_LMSTUDIO_MODEL,
        help="LLM model for command parsing",
    )
    parser.add_argument(
        "--no-completion-check",
        action="store_true",
        help="Don't wait for Unity completion signals",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--env",
        choices=["sim", "real"],
        default="sim",
        help="Execution environment: sim (Unity) or real (physical robot)",
    )
    parser.add_argument(
        "--web",
        type=int,
        default=None,
        metavar="PORT",
        help="Start Web UI on this port (e.g. --web 8000)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create controller
    controller = RobotController(
        host=args.host,
        model=args.model,
        check_completion=not args.no_completion_check,
        env=args.env,
        web_port=args.web,
    )

    # Handle shutdown signals
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        controller.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start and wait
    controller.start()
    controller.wait()


if __name__ == "__main__":
    main()
