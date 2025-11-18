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

    Phase 4.1: Health Check and Initialization Validation
    - Validates RAG system components during initialization
    - Provides health check method for monitoring
    - Tracks initialization status and errors
    """

    _instance: Optional["RAGQueryHandler"] = None
    _rag_system: Optional[RAGSystem] = None
    _initialization_error: Optional[str] = None
    _last_health_check: Optional[float] = None

    @classmethod
    def initialize(cls, rebuild_index: bool = False, validate: bool = True):
        """
        Initialize the RAG query handler with validation (Phase 4.1).

        Args:
            rebuild_index: If True, rebuild the RAG index from scratch
            validate: If True, perform validation checks after initialization

        Raises:
            RuntimeError: If initialization or validation fails
        """
        if cls._instance is None:
            cls._instance = cls()

        cls._initialization_error = None

        try:
            logging.info("Initializing RAG system...")
            cls._rag_system = RAGSystem()

            # Check if index needs to be built/rebuilt
            if rebuild_index or not cls._rag_system.is_ready():
                logging.info("Building RAG index from operations registry...")
                cls._rag_system.index_operations(rebuild=rebuild_index)
                logging.info("RAG index built successfully")
            else:
                logging.info("Loaded cached RAG index")

            # Phase 4.1: Validate initialization
            if validate:
                cls._validate_initialization()

            logging.info("RAG system initialized and validated")

        except Exception as e:
            error_msg = f"Failed to initialize RAG system: {e}"
            logging.error(error_msg)
            cls._initialization_error = str(e)
            raise RuntimeError(error_msg) from e

    @classmethod
    def _validate_initialization(cls):
        """
        Validate RAG system initialization (Phase 4.1).

        Checks:
        - RAG system is ready
        - Operations registry is populated
        - Index is functional with test query

        Raises:
            RuntimeError: If validation fails
        """
        logging.info("Validating RAG system initialization...")

        # Check 1: RAG system ready
        if cls._rag_system is None:
            raise RuntimeError("RAG system is None after initialization")
        if not cls._rag_system.is_ready():
            raise RuntimeError("RAG system not ready after initialization")

        # Check 2: Test query execution
        try:
            test_result = cls._rag_system.search("test query", top_k=1)
            if not isinstance(test_result, list):
                raise RuntimeError(
                    f"RAG search returned invalid type: {type(test_result)}"
                )
            logging.info(f"Validation: Test query returned {len(test_result)} results")
        except Exception as e:
            raise RuntimeError(f"RAG test query failed: {e}") from e

        # Check 3: Operations registry populated
        try:
            from operations import get_global_registry

            registry = get_global_registry()
            all_ops = registry.get_all_operations()
            if len(all_ops) == 0:
                raise RuntimeError("Operations registry is empty")
            logging.info(
                f"Validation: Operations registry has {len(all_ops)} operations"
            )
        except Exception as e:
            raise RuntimeError(f"Operations registry check failed: {e}") from e

        logging.info("✓ RAG system validation passed")

    @classmethod
    def query(
        cls, query_text: str, top_k: int = 5, filters: Optional[Dict] = None, timeout: float = 10.0
    ) -> Dict:
        """
        Execute a RAG query with timeout support (Phase 4.2).

        Args:
            query_text: Natural language query
            top_k: Number of results to return
            filters: Optional filters (category, complexity, min_score)
            timeout: Maximum execution time in seconds (default 10s)

        Returns:
            Query results dictionary

        Phase 4.2: Timeout Support
        - Wraps query execution with timeout
        - Returns timeout error if exceeded
        - Prevents slow queries from blocking server
        """
        if cls._rag_system is None:
            raise RuntimeError("RAGQueryHandler not initialized")

        import threading
        import time

        # Container for results from timeout thread
        result_container = {"result": None, "error": None, "completed": False}

        def execute_query():
            """Execute query in separate thread for timeout support"""
            try:
                # If filters are provided, use search() method which supports filtering
                # Otherwise use get_operation_context() for full context
                if filters and any(
                    filters.get(k) for k in ["category", "complexity", "min_score"]
                ):
                    # Extract filter parameters
                    category = filters.get("category")
                    complexity = filters.get("complexity")
                    min_score = filters.get("min_score", 0.5)

                    # Type guard - already checked in line 164
                    assert cls._rag_system is not None

                    # Execute filtered search
                    search_results = cls._rag_system.search(
                        query=query_text,
                        top_k=top_k,
                        category=category,
                        complexity=complexity,
                        min_score=min_score,
                    )

                    # Convert search results to context format
                    operations = []
                    for result in search_results:
                        op_data = result.get("metadata", {})
                        op_data["similarity_score"] = result.get("score", 0.0)
                        operations.append(op_data)

                    result_container["result"] = {
                        "query": query_text,
                        "num_results": len(operations),
                        "summary": f"Found {len(operations)} relevant operations for: {query_text}",
                        "operations": operations,
                    }
                else:
                    # Type guard - already checked in line 164
                    assert cls._rag_system is not None

                    # No filters - use get_operation_context for full details
                    result_container["result"] = cls._rag_system.get_operation_context(
                        query=query_text, top_k=top_k
                    )

                result_container["completed"] = True

            except Exception as e:
                result_container["error"] = e
                result_container["completed"] = True

        # Execute query in thread with timeout
        query_thread = threading.Thread(target=execute_query, daemon=True)
        start_time = time.time()
        query_thread.start()
        query_thread.join(timeout=timeout)

        # Check if query completed
        if not result_container["completed"]:
            # Timeout occurred
            elapsed = time.time() - start_time
            logging.warning(f"RAG query timed out after {elapsed:.2f}s: '{query_text}'")
            return {
                "success": False,
                "error": f"Query timed out after {timeout}s",
                "error_code": "TIMEOUT",
                "query": query_text,
                "num_results": 0,
                "operations": [],
            }

        # Check if error occurred
        if result_container["error"] is not None:
            logging.error(f"RAG query failed: {result_container['error']}")
            return {
                "success": False,
                "error": str(result_container["error"]),
                "error_code": "QUERY_FAILED",
                "query": query_text,
                "num_results": 0,
                "operations": [],
            }

        # Return successful result
        return result_container["result"]

    @classmethod
    def is_ready(cls) -> bool:
        """Check if RAG system is ready"""
        return cls._rag_system is not None and cls._rag_system.is_ready()

    @classmethod
    def health_check(cls) -> Dict:
        """
        Perform health check on RAG system (Phase 4.1).

        Returns:
            Dictionary with health status:
            - healthy: bool
            - status: str ("healthy", "degraded", "unhealthy")
            - details: dict with component status
            - last_check_time: float (timestamp)
        """
        import time

        cls._last_health_check = time.time()

        health_status = {
            "healthy": False,
            "status": "unhealthy",
            "details": {},
            "last_check_time": cls._last_health_check,
        }

        try:
            # Check 1: System initialized
            if cls._rag_system is None:
                health_status["details"]["rag_system"] = "not_initialized"
                health_status["details"]["error"] = cls._initialization_error
                return health_status

            health_status["details"]["rag_system"] = "initialized"

            # Check 2: System ready
            if not cls._rag_system.is_ready():
                health_status["details"]["index"] = "not_ready"
                health_status["status"] = "degraded"
                return health_status

            health_status["details"]["index"] = "ready"

            # Check 3: Test query (quick validation)
            try:
                test_result = cls._rag_system.search("health check", top_k=1)
                health_status["details"]["search"] = "functional"
                health_status["details"]["test_result_count"] = len(test_result)
            except Exception as e:
                health_status["details"]["search"] = f"error: {e}"
                health_status["status"] = "degraded"
                return health_status

            # All checks passed
            health_status["healthy"] = True
            health_status["status"] = "healthy"

        except Exception as e:
            health_status["details"]["unexpected_error"] = str(e)

        return health_status

    @classmethod
    def get_initialization_error(cls) -> Optional[str]:
        """Get initialization error message if any (Phase 4.1)"""
        return cls._initialization_error


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
                # Set long timeout to keep connection alive while waiting for queries
                # Use 60 seconds timeout to allow periodic server state checks
                # client.settimeout(60.0)

                try:
                    # Receive query message from Unity
                    query_data = self._receive_query_message(client)

                    if query_data is None:
                        # Client disconnected gracefully or timeout
                        continue

                    # Execute RAG query
                    query_text = query_data["query"]
                    top_k = query_data.get("top_k", 5)
                    filters = query_data.get("filters", {})
                    request_id = query_data.get("request_id", 0)  # Protocol V2

                    logging.info(
                        f"[req={request_id}] RAG query from {address}: '{query_text}' (top_k={top_k})"
                    )

                    # Execute query
                    results = RAGQueryHandler.query(query_text, top_k, filters)

                    # Include request_id in results for correlation
                    results["request_id"] = request_id

                    # Send results back to Unity (Protocol V2)
                    response_message = UnityProtocol.encode_rag_response(results, request_id)
                    client.sendall(response_message)

                    logging.debug(
                        f"[req={request_id}] Sent {results.get('num_results', 0)} operations to {address}"
                    )

                except socket.timeout:
                    # Expected - allows checking is_running()
                    # Keep connection alive
                    continue
                except (ConnectionResetError, BrokenPipeError, OSError) as e:
                    # Connection lost
                    logging.debug(f"Connection lost from {address}: {e}")
                    break

        except Exception as e:
            logging.debug(f"Client connection error: {e}")
        finally:
            logging.info(f"Unity RAG client disconnected from {address}")

    def _receive_query_message(self, client: socket.socket) -> Optional[Dict]:
        """
        Receive a RAG query message from Unity (Protocol V2).

        Message format: [type:1][request_id:4][query_len:4][query_text:N][top_k:4][filters_json_len:4][filters_json:N]

        Args:
            client: Client socket

        Returns:
            Query data dictionary (includes request_id) or None if client disconnected
        """
        try:
            # Read query message using UnityProtocol (Protocol V2)
            request_id, query_data = UnityProtocol.decode_rag_query(client)

            # Add request_id to query_data for logging/tracking
            query_data["request_id"] = request_id

            return query_data

        except Exception as e:
            logging.debug(f"Error receiving query message: {e}")
            return None


def run_rag_server(
    server_config: ServerConfig, rebuild_index: bool = False, setup_signals: bool = True
):
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
        args=(
            server_config,
            False,
            False,
        ),  # rebuild_index=False (already done), setup_signals=False
        daemon=True,
    )
    thread.start()
    logging.info("RAGServer started in background thread")
    return thread


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG server for Unity operation queries"
    )
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
