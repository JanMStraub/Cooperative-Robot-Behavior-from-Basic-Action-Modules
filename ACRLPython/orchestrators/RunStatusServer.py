#!/usr/bin/env python3
"""
RunStatusServer.py - Orchestrator for robot status query server

Starts the StatusServer on port 5012 to handle bidirectional status queries
from Unity. Unity can query robot status and receive real-time state information.

Usage:
    python -m LLMCommunication.orchestrators.RunStatusServer [--port PORT] [--host HOST]

Example:
    python -m LLMCommunication.orchestrators.RunStatusServer
"""

import argparse
import logging

# Import config - try both import styles
try:
    import LLMConfig as cfg
except ImportError:
    from .. import LLMConfig as cfg

# Import server
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
    print("=" * 70)
    print("Robot Status Server")
    print("=" * 70)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print()
    print("This server handles bidirectional status queries from Unity:")
    print("  1. Unity sends status query → StatusServer (port 5012)")
    print("  2. StatusServer requests status → Unity via ResultsServer (port 5010)")
    print("  3. Unity gathers robot state → sends back to StatusServer")
    print("  4. StatusServer returns status → original Unity client")
    print()
    print("Prerequisites:")
    print("  - Unity must be running with:")
    print("    • PythonCommandHandler (handles 'get_robot_status' commands)")
    print("    • UnifiedPythonReceiver (connected to port 5010)")
    print("    • StatusClient (connected to this server on port 5012)")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 70)
    print()

    # Create server config
    server_config = ServerConfig(host=args.host, port=args.port)

    try:
        # Start status server (blocking)
        run_status_server(server_config)
    except KeyboardInterrupt:
        print("\nShutting down status server...")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
