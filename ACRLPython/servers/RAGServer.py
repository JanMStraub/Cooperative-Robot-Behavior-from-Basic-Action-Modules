#!/usr/bin/env python3
"""
RAGServer.py - Semantic search server for robot operations

Provides natural language search over robot operations using RAG.
Unity can query for relevant operations based on task descriptions.
"""

import socket
import logging
import sys
from typing import Dict, Optional

# Import dependencies
# Import config - try both import styles
try:
    import LLMConfig as cfg
except ImportError:
    from .. import LLMConfig as cfg

# Import base classes - try both import styles
try:
    from core.TCPServerBase import TCPServerBase, ServerConfig
    from core.UnityProtocol import UnityProtocol
except ImportError:
    from ..core.TCPServerBase import TCPServerBase, ServerConfig
    from ..core.UnityProtocol import UnityProtocol

# Import RAG system
try:
    from rag import RAGSystem
except ImportError:
    from ..rag import RAGSystem

# Configure logging
logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=cfg.LOG_FORMAT)


class RAGQueryHandler:
    """
    Singleton for handling RAG queries.

    Maintains a single RAGSystem instance shared across all client connections.
    """

    _instance: Optional["RAGQueryHandler"] = None
    _rag_system: Optional[RAGSystem] = None

    @classmethod
    def initialize(cls, rebuild_index: bool = False):
        """
        Initialize the RAG query handler.

        Args:
            rebuild_index: If True, rebuild the RAG index from scratch
        """
        if cls._instance is None:
            cls._instance = cls()

        try:
            cls._rag_system = RAGSystem()

            # Check if index needs to be built/rebuilt
            if rebuild_index or not cls._rag_system.is_ready():
                logging.info("Building RAG index from operations registry...")
                cls._rag_system.index_operations(rebuild=rebuild_index)
                logging.info("RAG index ready")
            else:
                logging.info("Loaded cached RAG index")

        except Exception as e:
            logging.error(f"Failed to initialize RAG system: {e}")
            raise

    @classmethod
    def query(cls, query_text: str, top_k: int = 5, filters: Optional[Dict] = None) -> Dict:
        """
        Execute a RAG query.

        Args:
            query_text: Natural language query
            top_k: Number of results to return
            filters: Optional filters (category, complexity, min_score)

        Returns:
            Query results dictionary
        """
        if cls._rag_system is None:
            raise RuntimeError("RAGQueryHandler not initialized")

        try:
            # If filters are provided, use search() method which supports filtering
            # Otherwise use get_operation_context() for full context
            if filters and any(filters.get(k) for k in ["category", "complexity", "min_score"]):
                # Extract filter parameters
                category = filters.get("category")
                complexity = filters.get("complexity")
                min_score = filters.get("min_score", 0.5)

                # Execute filtered search
                search_results = cls._rag_system.search(
                    query=query_text,
                    top_k=top_k,
                    category=category,
                    complexity=complexity,
                    min_score=min_score
                )

                # Convert search results to context format
                operations = []
                for result in search_results:
                    op_data = result.get("metadata", {})
                    op_data["similarity_score"] = result.get("score", 0.0)
                    operations.append(op_data)

                results = {
                    "query": query_text,
                    "num_results": len(operations),
                    "summary": f"Found {len(operations)} relevant operations for: {query_text}",
                    "operations": operations
                }
            else:
                # No filters - use get_operation_context for full details
                results = cls._rag_system.get_operation_context(
                    query=query_text,
                    top_k=top_k
                )

            return results

        except Exception as e:
            logging.error(f"RAG query failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query_text,
                "num_results": 0,
                "operations": []
            }

    @classmethod
    def is_ready(cls) -> bool:
        """Check if RAG system is ready"""
        return cls._rag_system is not None and cls._rag_system.is_ready()


class RAGServer(TCPServerBase):
    """
    TCP server that handles RAG queries from Unity.

    Inherits connection management from TCPServerBase.
    Handles RAG query decoding and result encoding.
    """

    def __init__(self, server_config: ServerConfig):
        if server_config is None:
            server_config = cfg.get_rag_config()

        super().__init__(server_config)
        logging.info("RAGServer initialized")

    def handle_client_connection(self, client: socket.socket, address: tuple):
        """
        Handle a Unity client connection.

        Receives RAG query messages and sends back operation results.
        This method is called by TCPServerBase in a separate thread per client.

        Args:
            client: Client socket
            address: Client address tuple
        """
        logging.info(f"Unity RAG client connected from {address}")

        try:
            while self.is_running():
                # Set timeout for query receive
                client.settimeout(cfg.RAG_SERVER_TIMEOUT)

                try:
                    # Receive query message from Unity
                    query_data = self._receive_query_message(client)

                    if query_data is None:
                        # Client disconnected gracefully
                        break

                    # Execute RAG query
                    query_text = query_data["query"]
                    top_k = query_data.get("top_k", 5)
                    filters = query_data.get("filters", {})

                    logging.info(f"RAG query from {address}: '{query_text}' (top_k={top_k})")

                    # Execute query
                    results = RAGQueryHandler.query(query_text, top_k, filters)

                    # Send results back to Unity
                    response_message = UnityProtocol.encode_rag_response(results)
                    client.sendall(response_message)

                    logging.debug(
                        f"Sent {results.get('num_results', 0)} operations to {address}"
                    )

                except socket.timeout:
                    # Expected - allows checking is_running()
                    continue

        except Exception as e:
            logging.debug(f"Client connection error: {e}")

    def _receive_query_message(self, client: socket.socket) -> Optional[Dict]:
        """
        Receive a RAG query message from Unity.

        Message format: [query_len][query_text][top_k][filters_json_len][filters_json]

        Args:
            client: Client socket

        Returns:
            Query data dictionary or None if client disconnected
        """
        try:
            # Read query message using UnityProtocol
            query_data = UnityProtocol.decode_rag_query(client)
            return query_data

        except Exception as e:
            logging.debug(f"Error receiving query message: {e}")
            return None


def run_rag_server(server_config: ServerConfig, rebuild_index: bool = False, setup_signals: bool = True):
    """
    Start the RAGServer (blocking).

    Args:
        server_config: Server configuration
        rebuild_index: If True, rebuild the RAG index on startup
        setup_signals: If True, setup signal handlers (only valid in main thread)
    """
    import signal
    import time

    # Initialize RAG system
    logging.info("Initializing RAG query handler...")
    RAGQueryHandler.initialize(rebuild_index=rebuild_index)

    # Create server
    server = RAGServer(server_config)

    # Setup signal handlers (only if in main thread)
    if setup_signals:

        def signal_handler(_sig, _frame):
            logging.info("Shutdown signal received")
            server.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    try:
        server.start()
        logging.info("RAGServer ready to handle queries from Unity")

        # Keep server running
        while server.is_running():
            time.sleep(1.0)

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        server.stop()


def run_rag_server_background(server_config: ServerConfig, rebuild_index: bool = False):
    """
    Start the RAGServer in a background thread.

    Args:
        server_config: Server configuration
        rebuild_index: If True, rebuild the RAG index on startup

    Returns:
        Thread object running the server
    """
    import threading

    server_config = server_config or cfg.get_rag_config()

    # Initialize RAG in main thread (faster startup)
    logging.info("Initializing RAG query handler...")
    RAGQueryHandler.initialize(rebuild_index=rebuild_index)

    # Start server in background thread
    thread = threading.Thread(
        target=run_rag_server,
        args=(server_config, False, False),  # rebuild_index=False (already done), setup_signals=False
        daemon=True,
    )
    thread.start()
    logging.info("RAGServer started in background thread")
    return thread


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG server for Unity operation queries")
    parser.add_argument("--host", default=cfg.DEFAULT_HOST, help="Host to bind to")
    parser.add_argument(
        "--port", type=int, default=cfg.RAG_SERVER_PORT, help="Port to bind to"
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild RAG index from scratch on startup",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode with sample queries"
    )

    args = parser.parse_args()

    server_config = ServerConfig(host=args.host, port=args.port)

    if args.test:
        # Test mode - show what queries would return
        print("Initializing RAG system...")
        RAGQueryHandler.initialize(rebuild_index=args.rebuild_index)

        if not RAGQueryHandler.is_ready():
            print("ERROR: RAG system not ready")
            sys.exit(1)

        print("\nRAG system ready. Testing sample queries:\n")

        test_queries = [
            "move robot to position",
            "navigate to object",
            "pick up cube",
            "detect red object",
            "get current location",
        ]

        for query in test_queries:
            print(f"Query: '{query}'")
            results = RAGQueryHandler.query(query, top_k=3)

            if results.get("num_results", 0) > 0:
                for i, op in enumerate(results["operations"], 1):
                    print(
                        f"  {i}. {op['name']} (score: {op.get('similarity_score', 0):.3f})"
                    )
            else:
                print("  No results found")
            print()

        print("Test completed. Start server with: python -m servers.RAGServer")

    else:
        # Normal mode - start server
        run_rag_server(server_config, rebuild_index=args.rebuild_index)
