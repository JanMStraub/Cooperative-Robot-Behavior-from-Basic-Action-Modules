#!/usr/bin/env python3
"""
RunRobotController.py - Unified orchestrator for robot control

Starts all required servers in a single process:
- ImageServer (ports 5005, 5006) - receives images
- CommandServer (port 5010) - sends commands, receives completions
- SequenceServer (port 5013) - processes command sequences
- WorldStateServer (port 5014) - receives robot/object state updates

Usage:
    python -m orchestrators.RunRobotController

    # With options
    python -m orchestrators.RunRobotController --model gemma-3-12b
"""

import argparse
import signal
import threading
import logging

# Import config
try: 
    from config.Servers import (
        DEFAULT_HOST,
        STREAMING_SERVER_PORT,
        STEREO_DETECTION_PORT,
        LLM_RESULTS_PORT,
        SEQUENCE_SERVER_PORT,
        WORLD_STATE_PORT,
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
    from core.LoggingSetup import setup_logging
except ImportError:
    from ..config.Servers import (
        DEFAULT_HOST,
        STREAMING_SERVER_PORT,
        STEREO_DETECTION_PORT,
        LLM_RESULTS_PORT,
        SEQUENCE_SERVER_PORT,
        WORLD_STATE_PORT,
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
    from ..core.LoggingSetup import setup_logging

# Import servers - handle both direct execution and package import
try:
    from ..servers.ImageServer import run_image_server_background
    from ..servers.CommandServer import (
        run_command_server_background,
        get_command_broadcaster,
    )
    from ..servers.SequenceServer import run_sequence_server_background
    from ..servers.WorldStateServer import WorldStateServer
except ImportError:
    # Running as python -m orchestrators.RunRobotController
    from servers.ImageServer import run_image_server_background
    from servers.CommandServer import (
        run_command_server_background,
        get_command_broadcaster,
    )
    from servers.SequenceServer import run_sequence_server_background
    from servers.WorldStateServer import WorldStateServer

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
        model: str = DEFAULT_LMSTUDIO_MODEL,
        check_completion: bool = True,
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
            model: LLM model for parsing
            check_completion: Whether to wait for Unity completion signals
        """
        self._host = host
        self._single_port = single_port
        self._stereo_port = stereo_port
        self._command_port = command_port
        self._sequence_port = sequence_port
        self._world_state_port = world_state_port
        self._model = model
        self._check_completion = check_completion

        self._image_server = None
        self._command_server = None
        self._sequence_server = None
        self._world_state_server = None
        self._vision_processor = None
        self._running = False
        self._stop_event = threading.Event()

    def start(self):
        """Start all servers."""
        if self._running:
            logger.warning("RobotController already running")
            return

        logger.info("=" * 60)
        logger.info("Starting RobotController - Unified Robot Control Backend")
        logger.info("=" * 60)

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

        # Start WorldStateServer (port 5014) - receives robot/object state updates
        logger.info(f"Starting WorldStateServer (port: {self._world_state_port})")
        from core.TCPServerBase import ServerConfig

        world_state_config = ServerConfig(host=self._host, port=self._world_state_port)
        self._world_state_server = WorldStateServer(config=world_state_config)
        self._world_state_server.start()

        # Share resources between servers
        broadcaster = get_command_broadcaster()
        # SequenceExecutor will use this for sending commands

        self._running = True

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
        logger.info(f"  Image Server (single):  {self._host}:{self._single_port}")
        logger.info(f"  Image Server (stereo):  {self._host}:{self._stereo_port}")
        logger.info(f"  Command Server:         {self._host}:{self._command_port}")
        logger.info(f"  Sequence Server:        {self._host}:{self._sequence_port}")
        logger.info(f"  LLM Model:              {self._model}")
        if ENABLE_VISION_STREAMING and self._vision_processor:
            logger.info(f"  Vision Streaming:       Enabled ({VISION_STREAM_FPS} FPS)")
            if ENABLE_VISION_VISUALIZATION:
                logger.info(f"  Visualization:          Enabled (press 'q' to close)")
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

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create controller
    controller = RobotController(
        host=args.host, model=args.model, check_completion=not args.no_completion_check
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
