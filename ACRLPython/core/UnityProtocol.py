#!/usr/bin/env python3
"""
UnityProtocol.py - Unity ↔ Python communication protocol

Defines the wire protocol for communication between Unity and Python servers.
This protocol is versioned and should match the Unity-side implementation.

Protocol version: 1

Message formats:
1. Image Message (Unity → Python, single camera):
   [camera_id_len:4][camera_id:N][prompt_len:4][prompt:N][image_len:4][image_data:N]

2. Stereo Image Message (Unity → Python, stereo pair):
   [cam_pair_id_len:4][cam_pair_id:N]
   [camera_L_id_len:4][camera_L_id:N]
   [camera_R_id_len:4][camera_R_id:N]
   [prompt_len:4][prompt:N]
   [image_L_len:4][image_L_data:N]
   [image_R_len:4][image_R_data:N]

3. Result Message (Python → Unity):
   [json_len:4][json_data:N]

   Result JSON may include optional "world_position" field for 3D coordinates:
   {
     "success": true,
     "detections": [
       {
         "id": 0,
         "color": "red",
         "bbox_px": {...},
         "center_px": {...},
         "confidence": 0.95,
         "world_position": {"x": 0.5, "y": 0.2, "z": 1.0}  // Optional, meters
       }
     ]
   }

4. RAG Query Message (Unity → Python):
   [query_len:4][query_text:N][top_k:4][filters_json_len:4][filters_json:N]

5. RAG Response Message (Python → Unity):
   [json_len:4][operation_context_json:N]

All integers are little-endian unsigned 32-bit (struct format 'I').
All strings are UTF-8 encoded.
"""

import struct
import json
from typing import Tuple, Optional
import logging

# Import config
# Import config - try both import styles
try:
    import LLMConfig as cfg
except ImportError:
    from .. import LLMConfig as cfg

logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=cfg.LOG_FORMAT)


class UnityProtocol:
    """Unity ↔ Python wire protocol implementation"""

    # Protocol version
    VERSION = 1

    # Limits (from config)
    MAX_STRING_LENGTH = cfg.MAX_STRING_LENGTH
    MAX_IMAGE_SIZE = cfg.MAX_IMAGE_SIZE

    # Struct formats
    INT_FORMAT = "I"  # Unsigned 32-bit integer, little-endian
    INT_SIZE = 4

    @staticmethod
    def encode_image_message(camera_id: str, prompt: str, image_bytes: bytes) -> bytes:
        """
        Encode an image message for sending to Python server.

        Format: [camera_id_len][camera_id][prompt_len][prompt][image_len][image_data]

        Args:
            camera_id: Camera identifier
            prompt: Prompt for LLM (can be empty)
            image_bytes: PNG/JPG encoded image data

        Returns:
            Encoded message bytes

        Raises:
            ValueError: If data exceeds limits
        """
        # Validate inputs
        if len(camera_id) == 0:
            raise ValueError("Camera ID cannot be empty")
        if len(camera_id) > UnityProtocol.MAX_STRING_LENGTH:
            raise ValueError(
                f"Camera ID too long: {len(camera_id)} > {UnityProtocol.MAX_STRING_LENGTH}"
            )
        if len(prompt) > UnityProtocol.MAX_STRING_LENGTH:
            raise ValueError(
                f"Prompt too long: {len(prompt)} > {UnityProtocol.MAX_STRING_LENGTH}"
            )
        if len(image_bytes) == 0:
            raise ValueError("Image data cannot be empty")
        if len(image_bytes) > UnityProtocol.MAX_IMAGE_SIZE:
            raise ValueError(
                f"Image too large: {len(image_bytes)} > {UnityProtocol.MAX_IMAGE_SIZE}"
            )

        # Encode strings
        camera_id_bytes = camera_id.encode("utf-8")
        prompt_bytes = prompt.encode("utf-8")

        # Build message
        message = bytearray()

        # Camera ID
        message.extend(struct.pack(UnityProtocol.INT_FORMAT, len(camera_id_bytes)))
        message.extend(camera_id_bytes)

        # Prompt
        message.extend(struct.pack(UnityProtocol.INT_FORMAT, len(prompt_bytes)))
        message.extend(prompt_bytes)

        # Image
        message.extend(struct.pack(UnityProtocol.INT_FORMAT, len(image_bytes)))
        message.extend(image_bytes)

        return bytes(message)

    @staticmethod
    def decode_image_message(data: bytes) -> Tuple[str, str, bytes]:
        """
        Decode an image message received from Unity.

        Args:
            data: Raw message bytes

        Returns:
            Tuple of (camera_id, prompt, image_bytes)

        Raises:
            ValueError: If message is malformed
        """
        offset = 0

        try:
            # Read camera ID
            camera_id, offset = UnityProtocol._read_string(data, offset)

            # Read prompt
            prompt, offset = UnityProtocol._read_string(data, offset)

            # Read image
            image_bytes, offset = UnityProtocol._read_bytes(data, offset)

            return camera_id, prompt, image_bytes

        except Exception as e:
            raise ValueError(f"Failed to decode image message: {e}")

    @staticmethod
    def encode_result_message(result_dict: dict) -> bytes:
        """
        Encode an LLM result message for sending to Unity.

        Format: [json_len][json_data]

        Args:
            result_dict: Dictionary containing result data

        Returns:
            Encoded message bytes
        """
        # Convert dict to JSON
        json_str = json.dumps(result_dict, ensure_ascii=False)
        json_bytes = json_str.encode("utf-8")

        # Build message
        message = bytearray()
        message.extend(struct.pack(UnityProtocol.INT_FORMAT, len(json_bytes)))
        message.extend(json_bytes)

        return bytes(message)

    @staticmethod
    def decode_result_message(data: bytes) -> dict:
        """
        Decode a result message (for testing/verification).

        Args:
            data: Raw message bytes

        Returns:
            Result dictionary

        Raises:
            ValueError: If message is malformed
        """
        offset = 0

        try:
            # Read JSON length
            if len(data) < UnityProtocol.INT_SIZE:
                raise ValueError("Message too short")

            json_length = struct.unpack(
                UnityProtocol.INT_FORMAT, data[offset : offset + UnityProtocol.INT_SIZE]
            )[0]
            offset += UnityProtocol.INT_SIZE

            # Read JSON data
            if offset + json_length > len(data):
                raise ValueError(f"JSON length {json_length} exceeds remaining data")

            json_bytes = data[offset : offset + json_length]
            json_str = json_bytes.decode("utf-8")

            return json.loads(json_str)

        except Exception as e:
            raise ValueError(f"Failed to decode result message: {e}")

    @staticmethod
    def _read_string(data: bytes, offset: int) -> Tuple[str, int]:
        """
        Read a length-prefixed string from data.

        Args:
            data: Byte array to read from
            offset: Starting offset

        Returns:
            Tuple of (string, new_offset)
        """
        # Read length
        if offset + UnityProtocol.INT_SIZE > len(data):
            raise ValueError("Not enough data for string length")

        str_length = struct.unpack(
            UnityProtocol.INT_FORMAT, data[offset : offset + UnityProtocol.INT_SIZE]
        )[0]
        offset += UnityProtocol.INT_SIZE

        # Validate length
        if str_length > UnityProtocol.MAX_STRING_LENGTH:
            raise ValueError(
                f"String length {str_length} exceeds maximum {UnityProtocol.MAX_STRING_LENGTH}"
            )

        # Read string data
        if offset + str_length > len(data):
            raise ValueError(
                f"Not enough data for string (need {str_length}, have {len(data) - offset})"
            )

        str_bytes = data[offset : offset + str_length]
        offset += str_length

        # Decode
        string = str_bytes.decode("utf-8")
        return string, offset

    @staticmethod
    def _read_bytes(data: bytes, offset: int) -> Tuple[bytes, int]:
        """
        Read a length-prefixed byte array from data.

        Args:
            data: Byte array to read from
            offset: Starting offset

        Returns:
            Tuple of (bytes, new_offset)
        """
        # Read length
        if offset + UnityProtocol.INT_SIZE > len(data):
            raise ValueError("Not enough data for bytes length")

        bytes_length = struct.unpack(
            UnityProtocol.INT_FORMAT, data[offset : offset + UnityProtocol.INT_SIZE]
        )[0]
        offset += UnityProtocol.INT_SIZE

        # Validate length
        if bytes_length > UnityProtocol.MAX_IMAGE_SIZE:
            raise ValueError(
                f"Bytes length {bytes_length} exceeds maximum {UnityProtocol.MAX_IMAGE_SIZE}"
            )

        # Read byte data
        if offset + bytes_length > len(data):
            raise ValueError(
                f"Not enough data for bytes (need {bytes_length}, have {len(data) - offset})"
            )

        byte_data = data[offset : offset + bytes_length]
        offset += bytes_length

        return byte_data, offset

    @staticmethod
    def encode_rag_query(
        query: str, top_k: int = 5, filters: Optional[dict] = None
    ) -> bytes:
        """
        Encode a RAG query message for sending to Python server.

        Format: [query_len][query_text][top_k][filters_json_len][filters_json]

        Args:
            query: Natural language query text
            top_k: Number of results to return (default 5)
            filters: Optional filters dict (category, complexity, min_score)

        Returns:
            Encoded message bytes

        Raises:
            ValueError: If data exceeds limits
        """
        # Validate inputs
        if len(query) == 0:
            raise ValueError("Query cannot be empty")
        if len(query) > UnityProtocol.MAX_STRING_LENGTH:
            raise ValueError(
                f"Query too long: {len(query)} > {UnityProtocol.MAX_STRING_LENGTH}"
            )
        if top_k < 1 or top_k > 100:
            raise ValueError(f"top_k must be between 1 and 100, got {top_k}")

        # Encode query string
        query_bytes = query.encode("utf-8")

        # Encode filters as JSON
        filters = filters or {}
        filters_json = json.dumps(filters, ensure_ascii=False)
        filters_bytes = filters_json.encode("utf-8")

        # Build message
        message = bytearray()

        # Query text
        message.extend(struct.pack(UnityProtocol.INT_FORMAT, len(query_bytes)))
        message.extend(query_bytes)

        # Top-k parameter
        message.extend(struct.pack(UnityProtocol.INT_FORMAT, top_k))

        # Filters JSON
        message.extend(struct.pack(UnityProtocol.INT_FORMAT, len(filters_bytes)))
        message.extend(filters_bytes)

        return bytes(message)

    @staticmethod
    def decode_rag_query(socket_or_data) -> dict:
        """
        Decode a RAG query message from Unity.

        Args:
            socket_or_data: Either a socket object or raw bytes

        Returns:
            Dictionary with query, top_k, and filters

        Raises:
            ValueError: If message is malformed
        """
        # Handle both socket and bytes input
        if hasattr(socket_or_data, "recv"):
            # It's a socket - read the complete message
            data = UnityProtocol._receive_complete_rag_query(socket_or_data)
        else:
            # It's already bytes
            data = socket_or_data

        offset = 0

        try:
            # Read query text
            query, offset = UnityProtocol._read_string(data, offset)

            # Read top_k
            if offset + UnityProtocol.INT_SIZE > len(data):
                raise ValueError("Not enough data for top_k")

            top_k = struct.unpack(
                UnityProtocol.INT_FORMAT, data[offset : offset + UnityProtocol.INT_SIZE]
            )[0]
            offset += UnityProtocol.INT_SIZE

            # Read filters JSON
            filters_json, offset = UnityProtocol._read_string(data, offset)

            # Parse filters
            filters = json.loads(filters_json) if filters_json else {}

            return {"query": query, "top_k": top_k, "filters": filters}

        except Exception as e:
            raise ValueError(f"Failed to decode RAG query: {e}")

    @staticmethod
    def _receive_complete_rag_query(client_socket) -> bytes:
        """
        Receive a complete RAG query message from socket.

        Args:
            client_socket: Socket to receive from

        Returns:
            Complete message bytes
        """
        data = bytearray()

        # Read query length and text
        query_len_bytes = client_socket.recv(UnityProtocol.INT_SIZE)
        if not query_len_bytes:
            raise ValueError("Connection closed")

        query_len = struct.unpack(UnityProtocol.INT_FORMAT, query_len_bytes)[0]
        data.extend(query_len_bytes)

        query_bytes = client_socket.recv(query_len)
        data.extend(query_bytes)

        # Read top_k
        top_k_bytes = client_socket.recv(UnityProtocol.INT_SIZE)
        data.extend(top_k_bytes)

        # Read filters length and JSON
        filters_len_bytes = client_socket.recv(UnityProtocol.INT_SIZE)
        data.extend(filters_len_bytes)

        filters_len = struct.unpack(UnityProtocol.INT_FORMAT, filters_len_bytes)[0]

        if filters_len > 0:
            filters_bytes = client_socket.recv(filters_len)
            data.extend(filters_bytes)

        return bytes(data)

    @staticmethod
    def encode_rag_response(operation_context: dict) -> bytes:
        """
        Encode a RAG response message for sending to Unity.

        Format: [json_len][operation_context_json]

        This is identical to encode_result_message but kept separate for clarity.

        Args:
            operation_context: Dictionary containing operation results

        Returns:
            Encoded message bytes
        """
        return UnityProtocol.encode_result_message(operation_context)

    @staticmethod
    def decode_rag_response(data: bytes) -> dict:
        """
        Decode a RAG response message (for testing/verification).

        Args:
            data: Raw message bytes

        Returns:
            Operation context dictionary

        Raises:
            ValueError: If message is malformed
        """
        return UnityProtocol.decode_result_message(data)


if __name__ == "__main__":
    # Test the protocol
    print("Testing UnityProtocol...")

    # Test image message encoding/decoding
    test_camera_id = "TestCamera"
    test_prompt = "What do you see?"
    test_image = b"FAKE_PNG_DATA_HERE" * 100  # Fake image data

    # Encode
    encoded = UnityProtocol.encode_image_message(
        test_camera_id, test_prompt, test_image
    )
    print(f"Encoded image message: {len(encoded)} bytes")

    # Decode
    decoded_id, decoded_prompt, decoded_image = UnityProtocol.decode_image_message(
        encoded
    )
    assert decoded_id == test_camera_id
    assert decoded_prompt == test_prompt
    assert decoded_image == test_image
    print("✓ Image message encode/decode works")

    # Test result message encoding/decoding
    test_result = {
        "success": True,
        "response": "I see a robot arm",
        "camera_id": "TestCamera",
        "metadata": {"model": "llava", "duration_seconds": 1.5},
    }

    # Encode
    encoded_result = UnityProtocol.encode_result_message(test_result)
    print(f"Encoded result message: {len(encoded_result)} bytes")

    # Decode
    decoded_result = UnityProtocol.decode_result_message(encoded_result)
    assert decoded_result == test_result
    print("✓ Result message encode/decode works")

    # Test error cases
    try:
        UnityProtocol.encode_image_message("", "prompt", b"data")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Empty camera ID rejected: {e}")

    try:
        UnityProtocol.encode_image_message("A" * 300, "prompt", b"data")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Camera ID too long rejected: {e}")

    print("\nAll tests passed!")
