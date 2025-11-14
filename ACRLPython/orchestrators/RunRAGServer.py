#!/usr/bin/env python3
"""
RunRAGServer.py - Orchestrator for RAG server

Starts the RAG server and handles graceful shutdown.
Provides semantic search over robot operations for Unity.
"""

import sys
import logging

# Import dependencies
# Import config - try both import styles
try:
    import LLMConfig as cfg
except ImportError:
    from .. import LLMConfig as cfg
from servers.RAGServer import run_rag_server
from core.TCPServerBase import ServerConfig

# Configure logging
logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=cfg.LOG_FORMAT)


def main():
    """
    Main entry point for RAG server orchestrator.

    Parses command-line arguments and starts the RAG server.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG Server for semantic search over robot operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start RAG server with default settings (port 5011)
  python -m orchestrators.RunRAGServer

  # Rebuild RAG index on startup
  python -m orchestrators.RunRAGServer --rebuild-index

  # Use custom port
  python -m orchestrators.RunRAGServer --port 5012

  # Test mode (show sample queries without starting server)
  python -m orchestrators.RunRAGServer --test
        """,
    )

    parser.add_argument(
        "--host",
        default=cfg.DEFAULT_HOST,
        help=f"Host to bind to (default: {cfg.DEFAULT_HOST})",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=cfg.RAG_SERVER_PORT,
        help=f"Port to bind to (default: {cfg.RAG_SERVER_PORT})",
    )

    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild RAG index from operations registry on startup",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (show sample queries without starting server)",
    )

    args = parser.parse_args()

    # Create server configuration
    server_config = ServerConfig(host=args.host, port=args.port)

    # Print startup banner
    logging.info("=" * 70)
    logging.info("RAG SERVER - Semantic Search for Robot Operations")
    logging.info("=" * 70)
    logging.info(f"Host:          {args.host}")
    logging.info(f"Port:          {args.port}")
    logging.info(f"Rebuild Index: {args.rebuild_index}")
    logging.info(f"Test Mode:     {args.test}")
    logging.info("=" * 70)

    if args.test:
        # Test mode - run sample queries
        logging.info("\n🧪 TEST MODE - Running sample queries\n")
        run_test_mode(args.rebuild_index)
    else:
        # Normal mode - start server
        logging.info("\n🚀 Starting RAG server...\n")
        run_rag_server(server_config, rebuild_index=args.rebuild_index)


def run_test_mode(rebuild_index: bool):
    """
    Run test mode - show sample queries without starting server.

    Args:
        rebuild_index: Whether to rebuild the RAG index
    """
    from servers.RAGServer import RAGQueryHandler

    logging.info("Initializing RAG system...")
    try:
        RAGQueryHandler.initialize(rebuild_index=rebuild_index)

        if not RAGQueryHandler.is_ready():
            logging.info("❌ ERROR: RAG system not ready")
            sys.exit(1)

        logging.info("✅ RAG system ready\n")

        # Sample queries
        test_queries = [
            "move robot to position",
            "navigate to object",
            "pick up cube",
            "detect red object",
            "get current location",
            "rotate gripper",
            "check if object is graspable",
        ]

        logging.info("Running sample queries:\n")

        for query in test_queries:
            logging.info(f"📝 Query: '{query}'")

            try:
                results = RAGQueryHandler.query(query, top_k=3)

                if results.get("num_results", 0) > 0:
                    for i, op in enumerate(results["operations"], 1):
                        score = op.get("similarity_score", 0)
                        name = op.get("name", "unknown")
                        category = op.get("category", "unknown")
                        complexity = op.get("complexity", "unknown")

                        logging.info(
                            f"  {i}. {name} (score={score:.3f}, category={category}, complexity={complexity})"
                        )
                else:
                    logging.info("  ⚠️ No results found")

            except Exception as e:
                logging.info(f"  ❌ Error: {e}")

            print()

        logging.info("=" * 70)
        logging.info("Test completed successfully!")
        logging.info("=" * 70)
        logging.info("\nTo start the server, run without --test flag:")
        logging.info(f"  python -m orchestrators.RunRAGServer")

    except Exception as e:
        logging.info(f"❌ ERROR: Failed to initialize RAG system: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\n\n👋 Shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
