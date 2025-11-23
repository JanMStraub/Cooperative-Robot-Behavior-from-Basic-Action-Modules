#!/usr/bin/env python3
"""
RunSequenceServer.py - Entry point for running the Sequence Server

This orchestrator starts all required servers for multi-command sequence execution:
- ResultsServer (port 5010) - Sends commands to Unity
- StatusServer (port 5012) - Receives completion signals from Unity
- SequenceServer (port 5013) - Receives compound commands from Unity

Usage:
    python -m orchestrators.RunSequenceServer [--model MODEL] [--test]

Example:
    python -m orchestrators.RunSequenceServer --model gemma-3-12b
"""

import argparse
import signal
import time
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import LLMConfig as cfg
from servers.ResultsServer import run_results_server_background
from servers.StatusServer import run_status_server_background
from servers.SequenceServer import run_sequence_server_background, SequenceQueryHandler
from servers.StreamingServer import run_streaming_server_background

logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=cfg.LOG_FORMAT)
logger = logging.getLogger(__name__)


def run_test_sequence():
    """Run a test sequence to verify the system is working."""
    handler = SequenceQueryHandler()

    test_commands = [
        ("move to (0.3, 0.2, 0.1) and close the gripper", "Robot1"),
        ("open gripper then move to (0, 0, 0.3)", "Robot1"),
        ("move to x=0.1, y=0.2, z=0.15, then close gripper, then move to (0, 0, 0.4)", "Robot1"),
    ]

    logger.info("=" * 60)
    logger.info("Running test sequences")
    logger.info("=" * 60)

    for command, robot_id in test_commands:
        logger.info(f"\nTest: '{command}'")
        result = handler.execute_sequence(command, robot_id)

        if result["success"]:
            logger.info(f"✓ Success! Parsed {len(result.get('parsed_commands', []))} commands:")
            for cmd in result.get("parsed_commands", []):
                logger.info(f"  - {cmd['operation']}: {cmd['params']}")
        else:
            logger.error(f"Failed: {result.get('error')}")

    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run SequenceServer for multi-command execution"
    )
    parser.add_argument(
        "--model",
        default=cfg.DEFAULT_LMSTUDIO_MODEL,
        help=f"LM Studio model for command parsing (default: {cfg.DEFAULT_LMSTUDIO_MODEL})"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test sequences after startup"
    )
    parser.add_argument(
        "--no-results-server",
        action="store_true",
        help="Don't start ResultsServer (if already running)"
    )
    parser.add_argument(
        "--no-status-server",
        action="store_true",
        help="Don't start StatusServer (if already running)"
    )
    parser.add_argument(
        "--no-streaming-server",
        action="store_true",
        help="Don't start StreamingServer (if already running)"
    )
    parser.add_argument(
        "--no-completion-check",
        action="store_true",
        help="Disable waiting for Unity completion signals (for testing)"
    )
    args = parser.parse_args()

    # Print startup banner
    logger.info("=" * 60)
    logger.info("Sequence Server")
    logger.info("=" * 60)
    logger.info(f"Port:  {cfg.SEQUENCE_SERVER_PORT}")
    logger.info(f"Model: {args.model}")
    logger.info("=" * 60)

    # Start servers
    streaming_server = None
    if not args.no_streaming_server:
        streaming_server = run_streaming_server_background(
            cfg.get_streaming_config()  # type: ignore[arg-type]
        )
        logger.info(f"StreamingServer started on port {cfg.STREAMING_SERVER_PORT}")
        time.sleep(0.5)

    results_server = None
    if not args.no_results_server:
        results_server = run_results_server_background(
            cfg.get_results_config()  # type: ignore[arg-type]
        )
        time.sleep(0.5)

    status_server = None
    if not args.no_status_server:
        status_server = run_status_server_background(
            cfg.get_status_config()  # type: ignore[arg-type]
        )
        time.sleep(0.5)

    # Determine if completion checking should be enabled
    check_completion = not args.no_completion_check
    if not check_completion:
        logger.info("Completion checking DISABLED - commands will fire without waiting")

    sequence_server = run_sequence_server_background(
        cfg.get_sequence_config(),
        model=args.model,
        setup_signals=False,
        check_completion=check_completion
    )
    time.sleep(0.5)

    # Run test if requested
    if args.test:
        time.sleep(1)  # Wait for servers to be ready
        run_test_sequence()

    # Set up signal handlers
    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            logger.warning("Force shutdown...")
            sys.exit(1)
        shutdown_requested = True
        logger.info("\nShutdown requested...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Servers started, waiting for commands...")

    # Main loop
    try:
        while not shutdown_requested:
            time.sleep(1)

            # Check server health (only SequenceServer returns server object)
            if not sequence_server.is_running():
                logger.error("SequenceServer stopped unexpectedly!")
                break

            # results_server and status_server are daemon threads - they stop automatically
            if results_server and not results_server.is_alive():
                logger.error("ResultsServer stopped unexpectedly!")
                break

            if status_server and not status_server.is_alive():
                logger.error("StatusServer stopped unexpectedly!")
                break

            if streaming_server and not streaming_server.is_alive():
                logger.error("StreamingServer stopped unexpectedly!")
                break

    except KeyboardInterrupt:
        pass
    finally:
        # Shutdown
        logger.info("Shutting down servers...")

        if sequence_server:
            sequence_server.stop()

        # results_server and status_server are daemon threads - they stop automatically
        # when main thread exits

        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
