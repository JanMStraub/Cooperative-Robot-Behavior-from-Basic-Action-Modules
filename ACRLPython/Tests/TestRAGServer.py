#!/usr/bin/env python3
"""
test_rag_server.py - Tests for RAG server

Tests the RAG server implementation including:
- Protocol encoding/decoding
- Query handling
- Result formatting
- Error handling
"""

import pytest
import socket
import time
import threading

from core.UnityProtocol import UnityProtocol
from core.TCPServerBase import ServerConfig
from servers.RAGServer import RAGServer, RAGQueryHandler


class TestUnityProtocolRAG:
    """Test RAG protocol encoding/decoding"""

    def test_encode_rag_query_basic(self):
        """Test encoding a basic RAG query (Protocol V2)"""
        query = "move robot to position"
        top_k = 5
        filters = {}
        request_id = 1

        message = UnityProtocol.encode_rag_query(query, top_k, filters, request_id)

        assert message is not None
        assert len(message) > 0
        assert isinstance(message, bytes)
        # Verify header contains request_id
        assert len(message) >= UnityProtocol.HEADER_SIZE

    def test_encode_rag_query_with_filters(self):
        """Test encoding a RAG query with filters (Protocol V2)"""
        query = "navigate to object"
        top_k = 3
        filters = {"category": "navigation", "min_score": 0.7}
        request_id = 2

        message = UnityProtocol.encode_rag_query(query, top_k, filters, request_id)

        assert message is not None
        assert len(message) > 0

    def test_decode_rag_query(self):
        """Test decoding a RAG query (Protocol V2)"""
        query = "pick up cube"
        top_k = 10
        filters = {"complexity": "basic"}
        request_id = 3

        # Encode
        message = UnityProtocol.encode_rag_query(query, top_k, filters, request_id)

        # Decode
        decoded_request_id, decoded = UnityProtocol.decode_rag_query(message)

        assert decoded_request_id == request_id
        assert decoded["query"] == query
        assert decoded["top_k"] == top_k
        assert decoded["filters"]["complexity"] == "basic"

    def test_encode_rag_query_empty_query_fails(self):
        """Test that empty query raises ValueError (Protocol V2)"""
        with pytest.raises(ValueError):
            UnityProtocol.encode_rag_query("", 5, {}, 0)

    def test_encode_rag_query_invalid_top_k(self):
        """Test that invalid top_k raises ValueError (Protocol V2)"""
        with pytest.raises(ValueError):
            UnityProtocol.encode_rag_query("test query", 0, {}, 0)

        with pytest.raises(ValueError):
            UnityProtocol.encode_rag_query("test query", 101, {}, 0)

    def test_encode_decode_roundtrip(self):
        """Test encode-decode roundtrip (Protocol V2)"""
        test_cases = [
            ("move to position", 5, {}, 10),
            ("detect object", 3, {"category": "perception"}, 20),
            ("grip object", 10, {"complexity": "basic", "min_score": 0.5}, 30),
        ]

        for query, top_k, filters, request_id in test_cases:
            encoded = UnityProtocol.encode_rag_query(query, top_k, filters, request_id)
            decoded_request_id, decoded = UnityProtocol.decode_rag_query(encoded)

            assert decoded_request_id == request_id
            assert decoded["query"] == query
            assert decoded["top_k"] == top_k
            assert decoded["filters"] == (filters if filters else {})

    def test_encode_rag_response(self):
        """Test encoding a RAG response (Protocol V2)"""
        response = {
            "query": "test query",
            "num_results": 2,
            "operations": [
                {
                    "name": "move_to_coordinate",
                    "category": "navigation",
                    "similarity_score": 0.95,
                },
                {
                    "name": "detect_object",
                    "category": "perception",
                    "similarity_score": 0.78,
                },
            ],
        }
        request_id = 40

        message = UnityProtocol.encode_rag_response(response, request_id)

        assert message is not None
        assert len(message) > 0
        assert isinstance(message, bytes)
        # Verify header
        assert len(message) >= UnityProtocol.HEADER_SIZE

    def test_decode_rag_response(self):
        """Test decoding a RAG response (Protocol V2)"""
        response = {
            "query": "test query",
            "num_results": 1,
            "operations": [
                {
                    "operation_id": "nav_001",
                    "name": "move_to_coordinate",
                    "similarity_score": 0.92,
                }
            ],
        }
        request_id = 50

        # Encode
        encoded = UnityProtocol.encode_rag_response(response, request_id)

        # Decode
        decoded_request_id, decoded = UnityProtocol.decode_rag_response(encoded)

        assert decoded_request_id == request_id
        assert decoded["query"] == "test query"
        assert decoded["num_results"] == 1
        assert len(decoded["operations"]) == 1
        assert decoded["operations"][0]["name"] == "move_to_coordinate"


class TestRAGQueryHandler:
    """Test RAG query handler"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize RAG system before each test"""
        try:
            RAGQueryHandler.initialize(rebuild_index=False)
        except Exception as e:
            pytest.skip(f"RAG system initialization failed: {e}")

    def test_query_handler_basic(self):
        """Test basic RAG query"""
        if not RAGQueryHandler.is_ready():
            pytest.skip("RAG system not ready")

        results = RAGQueryHandler.query("move robot to position", top_k=3)

        assert results is not None
        assert "query" in results
        assert "num_results" in results
        assert "operations" in results

    def test_query_handler_with_filters(self):
        """Test RAG query with filters"""
        if not RAGQueryHandler.is_ready():
            pytest.skip("RAG system not ready")

        filters = {"category": "navigation", "min_score": 0.3}
        results = RAGQueryHandler.query("navigate to object", top_k=5, filters=filters)

        assert results is not None
        assert isinstance(results, dict)

    def test_query_handler_error_handling(self):
        """Test that query handler handles errors gracefully"""
        # Query with invalid parameters should return error result
        results = RAGQueryHandler.query("", top_k=5)

        assert results is not None
        assert "error" in results or "num_results" in results


@pytest.mark.integration
class TestRAGServerIntegration:
    """Integration tests for RAG server (requires running server)"""

    @pytest.fixture
    def server_config(self):
        """Create server configuration for tests"""
        return ServerConfig(host="127.0.0.1", port=5999)  # Use test port

    @pytest.fixture
    def server(self, server_config):
        """Start RAG server in background"""
        # Initialize RAG system
        try:
            RAGQueryHandler.initialize(rebuild_index=False)
        except Exception as e:
            pytest.skip(f"RAG initialization failed: {e}")

        # Create and start server
        server = RAGServer(server_config)
        server_thread = threading.Thread(target=server.start, daemon=True)
        server_thread.start()

        # Wait for server to start
        time.sleep(0.5)

        yield server

        # Cleanup
        server.stop()

    def test_server_accepts_connections(self, server, server_config):
        """Test that server accepts client connections"""
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            client.connect((server_config.host, server_config.port))
            assert True  # Connection successful
        except Exception as e:
            pytest.fail(f"Failed to connect to server: {e}")
        finally:
            client.close()

    def test_server_query_roundtrip(self, server, server_config):
        """Test sending a query and receiving a response (Protocol V2)"""
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            # Connect to server
            client.connect((server_config.host, server_config.port))

            # Send query (Protocol V2 with request_id)
            query = "move to position"
            request_id = 100
            message = UnityProtocol.encode_rag_query(query, top_k=3, filters={}, request_id=request_id)
            client.sendall(message)

            # Receive response (with timeout)
            client.settimeout(5.0)

            # Read Protocol V2 header
            header_data = client.recv(UnityProtocol.HEADER_SIZE)
            if len(header_data) < UnityProtocol.HEADER_SIZE:
                pytest.fail("Incomplete response header")

            # Read response length
            length_data = client.recv(4)
            if len(length_data) < 4:
                pytest.fail("Incomplete response length")

            response_length = int.from_bytes(length_data, byteorder="little")

            # Read response data
            response_data = header_data + length_data + client.recv(response_length)

            # Decode response (returns dict, not JSON string) - Protocol V2
            decoded_request_id, response = UnityProtocol.decode_rag_response(response_data)

            # Verify response
            assert decoded_request_id == request_id
            assert isinstance(response, dict)
            assert "query" in response
            assert response["query"] == query

        except socket.timeout:
            pytest.fail("Timeout waiting for server response")
        except Exception as e:
            pytest.fail(f"Query roundtrip failed: {e}")
        finally:
            client.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
