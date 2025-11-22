#!/usr/bin/env python3
"""
RunStatusServer.py - Orchestrator for robot status query server

Starts the StatusServer on port 5012 to handle bidirectional status queries
from Unity. Unity can query robot status and receive real-time state information.

StatusServer connects to ResultsServer (port 5010) as a TCP client to send
status requests to Unity, bypassing singleton issues across processes.

Prerequisites:
- ResultsServer must be running on port 5010 (usually started by RunAnalyzer)

Usage:
    python -m LLMCommunication.orchestrators.RunStatusServer [--port PORT] [--host HOST]

Example:
    python -m LLMCommunication.orchestrators.RunStatusServer
"""

import argparse
import logging
import socket
import sys

# Import config - try both import styles
try:
    import LLMConfig as cfg
except ImportError:
    from .. import LLMConfig as cfg

# Import servers
try:
    from servers.StatusServer import run_status_server
    from core.TCPServerBase import ServerConfig
except ImportError:
    from ..servers.StatusServer import run_status_server
    from ..core.TCPServerBase import ServerConfig

# Configure logging
logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=cfg.LOG_FORMAT)


def main():
    """
    Main entry point for status server orchestrator.
    """
    parser = argparse.ArgumentParser(
        description="Robot Status Server - Handles bidirectional status queries from Unity"
    )
    parser.add_argument(
        "--host", type=str, default=cfg.DEFAULT_HOST, help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=cfg.STATUS_SERVER_PORT,
        help=f"Port to bind to (default: {cfg.STATUS_SERVER_PORT})",
    )

    args = parser.parse_args()

    # Print startup banner
    logging.info("=" * 60)
    logging.info("Status Server")
    logging.info("=" * 60)
    logging.info(f"Port: {args.port}")
    logging.info(f"Results: {cfg.RESULTS_SERVER_PORT}")
    logging.info("=" * 60)

    # Verify ResultsServer is available
    results_available = False
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.settimeout(2.0)
        result = test_socket.connect_ex((args.host, cfg.RESULTS_SERVER_PORT))
        test_socket.close()
        if result == 0:
            results_available = True
    except Exception as e:
        logging.warning(f"Error checking ResultsServer: {e}")

    if not results_available:
        logging.error("ResultsServer not running on port 5010")
        logging.error("Start RunAnalyzer first")
        return 1

    # Create status server config
    status_config = ServerConfig(host=args.host, port=args.port)

    try:
        run_status_server(status_config)
    except KeyboardInterrupt:
        logging.info("\nShutting down StatusServer...")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
