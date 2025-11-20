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
from rag import RAGSystem

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
            logger.error(f"✗ Failed: {result.get('error')}")

    logger.info("\n" + "=" * 60)


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
        "--rebuild-index",
        action="store_true",
        help="Rebuild RAG index before starting"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Starting Sequence Server Orchestrator")
    logger.info("=" * 60)

    # Initialize RAG system if needed
    logger.info("Initializing RAG system...")
    try:
        rag = RAGSystem()
        if args.rebuild_index or not rag.is_ready():
            logger.info("Building RAG index...")
            rag.index_operations(rebuild=args.rebuild_index)
        logger.info("✓ RAG system ready")
    except Exception as e:
        logger.warning(f"RAG initialization failed: {e}. Continuing without RAG validation.")

    # Start ResultsServer for sending commands to Unity
    results_server = None
    if not args.no_results_server:
        logger.info(f"Starting ResultsServer on port {cfg.LLM_RESULTS_PORT}...")
        results_server = run_results_server_background(
            cfg.get_results_config()  # type: ignore[arg-type]
        )
        time.sleep(0.5)
        logger.info("✓ ResultsServer started")

    # Start StatusServer for receiving completion signals from Unity
    status_server = None
    if not args.no_status_server:
        logger.info(f"Starting StatusServer on port {cfg.STATUS_SERVER_PORT}...")
        status_server = run_status_server_background(
            cfg.get_status_config()  # type: ignore[arg-type]
        )
        time.sleep(0.5)
        logger.info("✓ StatusServer started")

    # Start SequenceServer
    logger.info(f"Starting SequenceServer on port {cfg.SEQUENCE_SERVER_PORT}...")
    sequence_server = run_sequence_server_background(
        cfg.get_sequence_config(),
        model=args.model,
        setup_signals=False
    )
    time.sleep(0.5)
    logger.info("✓ SequenceServer started")

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

    # Print status
    logger.info("")
    logger.info("=" * 60)
    logger.info("Sequence Server is running")
    logger.info("=" * 60)
    logger.info(f"  SequenceServer: port {cfg.SEQUENCE_SERVER_PORT}")
    if results_server:
        logger.info(f"  ResultsServer:  port {cfg.LLM_RESULTS_PORT}")
    if status_server:
        logger.info(f"  StatusServer:   port {cfg.STATUS_SERVER_PORT}")
    logger.info(f"  Model:          {args.model}")
    logger.info("")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)

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
