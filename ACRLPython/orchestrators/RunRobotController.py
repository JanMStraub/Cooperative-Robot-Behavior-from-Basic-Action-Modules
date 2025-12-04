#!/usr/bin/env python3
"""
RunRobotController.py - Unified orchestrator for robot control

Starts all required servers in a single process:
- ImageServer (ports 5005, 5006) - receives images
- CommandServer (port 5010) - sends commands, receives completions
- SequenceServer (port 5013) - processes command sequences

Usage:
    python -m orchestrators.RunRobotController

    # With options
    python -m orchestrators.RunRobotController --model gemma-3-12b
"""

import argparse
import logging
import signal
import time
import threading

# Import config
try:
    import LLMConfig as cfg
except ImportError:
    from .. import LLMConfig as cfg

# Import servers
from servers.ImageServer import ImageServer, run_image_server_background
from servers.CommandServer import CommandServer, run_command_server_background, get_command_broadcaster
from servers.SequenceServer import SequenceServer, SequenceQueryHandler, run_sequence_server_background

# Import vision operations to register them
from operations import VisionOperations

logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=cfg.LOG_FORMAT)
logger = logging.getLogger(__name__)


class RobotController:
    """
    Unified robot controller that manages all servers.

    Provides a single entry point for starting the entire Python backend.
    """

    def __init__(
        self,
        host: str = cfg.DEFAULT_HOST,
        single_port: int = cfg.STREAMING_SERVER_PORT,
        stereo_port: int = cfg.STEREO_DETECTION_PORT,
        command_port: int = cfg.LLM_RESULTS_PORT,
        sequence_port: int = cfg.SEQUENCE_SERVER_PORT,
        model: str = cfg.DEFAULT_LMSTUDIO_MODEL,
        check_completion: bool = True
    ):
        """
        Initialize the robot controller.

        Args:
            host: Host to bind servers to
            single_port: Port for single camera images
            stereo_port: Port for stereo image pairs
            command_port: Port for commands/results (bidirectional)
            sequence_port: Port for sequence execution
            model: LLM model for parsing
            check_completion: Whether to wait for Unity completion signals
        """
        self._host = host
        self._single_port = single_port
        self._stereo_port = stereo_port
        self._command_port = command_port
        self._sequence_port = sequence_port
        self._model = model
        self._check_completion = check_completion

        self._image_server = None
        self._command_server = None
        self._sequence_server = None
        self._running = False

    def start(self):
        """Start all servers."""
        if self._running:
            logger.warning("RobotController already running")
            return

        logger.info("=" * 60)
        logger.info("Starting RobotController - Unified Robot Control Backend")
        logger.info("=" * 60)

        # Start ImageServer (ports 5005, 5006)
        logger.info(f"Starting ImageServer (single: {self._single_port}, stereo: {self._stereo_port})")
        self._image_server = ImageServer(
            single_port=self._single_port,
            stereo_port=self._stereo_port,
            host=self._host
        )
        self._image_server.start()

        # Start CommandServer (port 5010) - bidirectional for commands and completions
        logger.info(f"Starting CommandServer (port: {self._command_port})")
        self._command_server = run_command_server_background(
            port=self._command_port,
            host=self._host
        )

        # Initialize and start SequenceServer (port 5013)
        logger.info(f"Starting SequenceServer (port: {self._sequence_port})")
        self._sequence_server = run_sequence_server_background(
            lm_studio_url=cfg.LMSTUDIO_BASE_URL,
            model=self._model,
            check_completion=self._check_completion
        )

        # Share resources between servers
        broadcaster = get_command_broadcaster()
        # SequenceExecutor will use this for sending commands

        self._running = True

        logger.info("=" * 60)
        logger.info("RobotController started successfully!")
        logger.info("=" * 60)
        logger.info(f"  Image Server (single):  {self._host}:{self._single_port}")
        logger.info(f"  Image Server (stereo):  {self._host}:{self._stereo_port}")
        logger.info(f"  Command Server:         {self._host}:{self._command_port}")
        logger.info(f"  Sequence Server:        {self._host}:{self._sequence_port}")
        logger.info(f"  LLM Model:              {self._model}")
        logger.info("=" * 60)

    def stop(self):
        """Stop all servers."""
        if not self._running:
            return

        logger.info("Stopping RobotController...")

        if self._image_server:
            self._image_server.stop()

        if self._command_server:
            self._command_server.stop()

        if self._sequence_server:
            self._sequence_server.stop()

        self._running = False
        logger.info("RobotController stopped")

    def is_running(self) -> bool:
        """Check if controller is running."""
        return self._running

    def wait(self):
        """Wait for controller to stop (blocking)."""
        try:
            while self._running:
                time.sleep(1)

                # Log status periodically
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
    parser.add_argument("--host", default=cfg.DEFAULT_HOST, help="Host to bind to")
    parser.add_argument("--model", default=cfg.DEFAULT_LMSTUDIO_MODEL,
                       help="LLM model for command parsing")
    parser.add_argument("--no-completion-check", action="store_true",
                       help="Don't wait for Unity completion signals")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create controller
    controller = RobotController(
        host=args.host,
        model=args.model,
        check_completion=not args.no_completion_check
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
