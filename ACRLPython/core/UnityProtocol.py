#!/usr/bin/env python3
"""
UnityProtocol.py - Unity ↔ Python communication protocol

Defines the wire protocol for communication between Unity and Python servers.
This protocol is versioned and should match the Unity-side implementation.

Protocol version: 2

ALL messages now include:
- message_type (1 byte): Identifies the message type
- request_id (4 bytes): Unsigned integer for request/response correlation

Message Type Enumeration:
- 0x01: IMAGE - Single camera image
- 0x02: RESULT - JSON result (LLM, detection, etc.)
- 0x03: RAG_QUERY - Natural language query for operations
- 0x04: RAG_RESPONSE - Operation recommendations
- 0x05: STATUS_QUERY - Robot status request
- 0x06: STATUS_RESPONSE - Robot status data
- 0x07: STEREO_IMAGE - Stereo camera pair

Message formats (ALL with header):
1. Image Message (Unity → Python, single camera):
   [type:1][request_id:4][camera_id_len:4][camera_id:N][prompt_len:4][prompt:N][image_len:4][image_data:N]

2. Stereo Image Message (Unity → Python, stereo pair):
   [type:1][request_id:4][cam_pair_id_len:4][cam_pair_id:N]
   [camera_L_id_len:4][camera_L_id:N]
   [camera_R_id_len:4][camera_R_id:N]
   [prompt_len:4][prompt:N]
   [image_L_len:4][image_L_data:N]
   [image_R_len:4][image_R_data:N]

3. Result Message (Python → Unity):
   [type:1][request_id:4][json_len:4][json_data:N]

4. RAG Query Message (Unity → Python):
   [type:1][request_id:4][query_len:4][query_text:N][top_k:4][filters_json_len:4][filters_json:N]

5. RAG Response Message (Python → Unity):
   [type:1][request_id:4][json_len:4][operation_context_json:N]

6. Status Query Message (Unity → Python):
   [type:1][request_id:4][robot_id_len:4][robot_id:N][detailed:1]

7. Status Response Message (Python → Unity):
   [type:1][request_id:4][json_len:4][robot_status_json:N]

All integers are little-endian unsigned 32-bit (struct format 'I').
All strings are UTF-8 encoded.
"""

import struct
import json
from typing import Tuple, Optional
from enum import IntEnum
import logging

# Import config
try:
    from config.Servers import (
        MAX_STRING_LENGTH as _MAX_STRING_LENGTH,
        MAX_IMAGE_SIZE as _MAX_IMAGE_SIZE,
    )
except ImportError:
    from ..config.Servers import (
        MAX_STRING_LENGTH as _MAX_STRING_LENGTH,
        MAX_IMAGE_SIZE as _MAX_IMAGE_SIZE,
    )

# Configure logging
try:
    from core.LoggingSetup import setup_logging
    setup_logging(__name__)
except ImportError:
    from .LoggingSetup import setup_logging
    setup_logging(__name__)


class MessageType(IntEnum):
    """Message type enumeration for protocol V2"""

    IMAGE = 0x01
    RESULT = 0x02
    RAG_QUERY = 0x03
    RAG_RESPONSE = 0x04
    STATUS_QUERY = 0x05
    STATUS_RESPONSE = 0x06
    STEREO_IMAGE = 0x07
    SEQUENCE_QUERY = 0x08


class UnityProtocol:
    """Unity ↔ Python wire protocol implementation (Version 2)"""

    # Protocol version
    VERSION = 2

    # Limits (from config)
    MAX_STRING_LENGTH = _MAX_STRING_LENGTH
    MAX_IMAGE_SIZE = _MAX_IMAGE_SIZE

    # Struct formats
    INT_FORMAT = "I"  # Unsigned 32-bit integer, little-endian
    INT_SIZE = 4
    TYPE_SIZE = 1  # Message type byte
    HEADER_SIZE = TYPE_SIZE + INT_SIZE  # type + request_id

    @staticmethod
    def _encode_header(message_type: MessageType, request_id: int) -> bytes:
        """
        Encode message header (type + request_id).

        Args:
            message_type: Message type from MessageType enum
            request_id: Unsigned 32-bit request ID

        Returns:
            5-byte header
        """
        header = bytearray()
        header.append(message_type)  # 1 byte
        header.extend(struct.pack(UnityProtocol.INT_FORMAT, request_id))  # 4 bytes
        return bytes(header)

    @staticmethod
    def decode_header(data: bytes) -> Tuple[MessageType, int]:
        """
        Decode message header (public API).

        Args:
            data: Byte array to read from (must be exactly 5 bytes)

        Returns:
            Tuple of (message_type, request_id)

        Raises:
            ValueError: If header is malformed
        """
        if len(data) < UnityProtocol.HEADER_SIZE:
            raise ValueError(
                f"Not enough data for header (need {UnityProtocol.HEADER_SIZE}, have {len(data)})"
            )

        message_type = MessageType(data[0])
        request_id = struct.unpack(
            UnityProtocol.INT_FORMAT, data[1:5]
        )[0]

        return message_type, request_id

    @staticmethod
    def _decode_header(data: bytes, offset: int = 0) -> Tuple[MessageType, int, int]:
        """
        Decode message header from data (internal use with offset tracking).

        Args:
            data: Byte array to read from
            offset: Starting offset (default 0)

        Returns:
            Tuple of (message_type, request_id, new_offset)

        Raises:
            ValueError: If header is malformed
        """
        if len(data) - offset < UnityProtocol.HEADER_SIZE:
            raise ValueError(
                f"Not enough data for header (need {UnityProtocol.HEADER_SIZE}, have {len(data) - offset})"
            )

        message_type = MessageType(data[offset])
        offset += UnityProtocol.TYPE_SIZE

        request_id = struct.unpack(
            UnityProtocol.INT_FORMAT, data[offset : offset + UnityProtocol.INT_SIZE]
        )[0]
        offset += UnityProtocol.INT_SIZE

        return message_type, request_id, offset

    @staticmethod
    def encode_image_message(
        camera_id: str, prompt: str, image_bytes: bytes, request_id: int = 0
    ) -> bytes:
        """
        Encode an image message for sending to Python server.

        Format: [type:1][request_id:4][camera_id_len:4][camera_id:N][prompt_len:4][prompt:N][image_len:4][image_data:N]

        Args:
            camera_id: Camera identifier
            prompt: Prompt for LLM (can be empty)
            image_bytes: PNG/JPG encoded image data
            request_id: Request ID for correlation (default 0)

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

        # Header
        message.extend(UnityProtocol._encode_header(MessageType.IMAGE, request_id))

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
    def decode_image_message(data: bytes) -> Tuple[int, str, str, bytes]:
        """
        Decode an image message received from Unity.

        Args:
            data: Raw message bytes

        Returns:
            Tuple of (request_id, camera_id, prompt, image_bytes)

        Raises:
            ValueError: If message is malformed
        """
        try:
            # Decode header
            msg_type, request_id, offset = UnityProtocol._decode_header(data)

            if msg_type != MessageType.IMAGE:
                raise ValueError(f"Expected IMAGE message, got {msg_type.name}")

            # Read camera ID
            camera_id, offset = UnityProtocol._read_string(data, offset)

            # Read prompt
            prompt, offset = UnityProtocol._read_string(data, offset)

            # Read image
            image_bytes, offset = UnityProtocol._read_bytes(data, offset)

            return request_id, camera_id, prompt, image_bytes

        except Exception as e:
            raise ValueError(f"Failed to decode image message: {e}")

    @staticmethod
    def encode_result_message(result_dict: dict, request_id: int = 0) -> bytes:
        """
        Encode a result message for sending to Unity.

        Format: [type:1][request_id:4][json_len:4][json_data:N]

        Args:
            result_dict: Dictionary containing result data
            request_id: Request ID for correlation (default 0)

        Returns:
            Encoded message bytes
        """
        # Convert dict to JSON
        json_str = json.dumps(result_dict, ensure_ascii=False)
        json_bytes = json_str.encode("utf-8")

        # Build message
        message = bytearray()

        # Header
        message.extend(UnityProtocol._encode_header(MessageType.RESULT, request_id))

        # JSON data
        message.extend(struct.pack(UnityProtocol.INT_FORMAT, len(json_bytes)))
        message.extend(json_bytes)

        return bytes(message)

    @staticmethod
    def decode_result_message(data: bytes) -> Tuple[int, dict]:
        """
        Decode a result message (for testing/verification).

        Args:
            data: Raw message bytes

        Returns:
            Tuple of (request_id, result_dict)

        Raises:
            ValueError: If message is malformed
        """
        try:
            # Decode header
            msg_type, request_id, offset = UnityProtocol._decode_header(data)

            if msg_type != MessageType.RESULT:
                raise ValueError(f"Expected RESULT message, got {msg_type.name}")

            # Read JSON length
            if len(data) - offset < UnityProtocol.INT_SIZE:
                raise ValueError("Not enough data for JSON length")

            json_length = struct.unpack(
                UnityProtocol.INT_FORMAT, data[offset : offset + UnityProtocol.INT_SIZE]
            )[0]
            offset += UnityProtocol.INT_SIZE

            # Read JSON data
            if offset + json_length > len(data):
                raise ValueError(f"JSON length {json_length} exceeds remaining data")

            json_bytes = data[offset : offset + json_length]
            json_str = json_bytes.decode("utf-8")

            return request_id, json.loads(json_str)

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
        query: str, top_k: int = 5, filters: Optional[dict] = None, request_id: int = 0
    ) -> bytes:
        """
        Encode a RAG query message for sending to Python server.

        Format: [type:1][request_id:4][query_len:4][query_text:N][top_k:4][filters_json_len:4][filters_json:N]

        Args:
            query: Natural language query text
            top_k: Number of results to return (default 5)
            filters: Optional filters dict (category, complexity, min_score)
            request_id: Request ID for correlation (default 0)

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

        # Header
        message.extend(UnityProtocol._encode_header(MessageType.RAG_QUERY, request_id))

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
    def decode_rag_query(socket_or_data) -> Tuple[int, dict]:
        """
        Decode a RAG query message from Unity.

        Args:
            socket_or_data: Either a socket object or raw bytes

        Returns:
            Tuple of (request_id, query_dict) where query_dict contains query, top_k, and filters

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

        try:
            # Decode header
            msg_type, request_id, offset = UnityProtocol._decode_header(data)

            if msg_type != MessageType.RAG_QUERY:
                raise ValueError(f"Expected RAG_QUERY message, got {msg_type.name}")

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

            query_dict = {"query": query, "top_k": top_k, "filters": filters}
            return request_id, query_dict

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

        # Read header (type + request_id)
        header_bytes = client_socket.recv(UnityProtocol.HEADER_SIZE)
        if not header_bytes or len(header_bytes) < UnityProtocol.HEADER_SIZE:
            raise ValueError("Connection closed or incomplete header")
        data.extend(header_bytes)

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
    def encode_rag_response(operation_context: dict, request_id: int = 0) -> bytes:
        """
        Encode a RAG response message for sending to Unity.

        Format: [type:1][request_id:4][json_len:4][operation_context_json:N]

        Args:
            operation_context: Dictionary containing operation results
            request_id: Request ID for correlation (default 0)

        Returns:
            Encoded message bytes
        """
        # Convert dict to JSON
        json_str = json.dumps(operation_context, ensure_ascii=False)
        json_bytes = json_str.encode("utf-8")

        # Build message
        message = bytearray()

        # Header
        message.extend(
            UnityProtocol._encode_header(MessageType.RAG_RESPONSE, request_id)
        )

        # JSON data
        message.extend(struct.pack(UnityProtocol.INT_FORMAT, len(json_bytes)))
        message.extend(json_bytes)

        return bytes(message)

    @staticmethod
    def decode_rag_response(data: bytes) -> Tuple[int, dict]:
        """
        Decode a RAG response message (for testing/verification).

        Args:
            data: Raw message bytes

        Returns:
            Tuple of (request_id, operation_context_dict)

        Raises:
            ValueError: If message is malformed
        """
        try:
            # Decode header
            msg_type, request_id, offset = UnityProtocol._decode_header(data)

            if msg_type != MessageType.RAG_RESPONSE:
                raise ValueError(f"Expected RAG_RESPONSE message, got {msg_type.name}")

            # Read JSON data
            if len(data) - offset < UnityProtocol.INT_SIZE:
                raise ValueError("Not enough data for JSON length")

            json_length = struct.unpack(
                UnityProtocol.INT_FORMAT, data[offset : offset + UnityProtocol.INT_SIZE]
            )[0]
            offset += UnityProtocol.INT_SIZE

            if offset + json_length > len(data):
                raise ValueError(f"JSON length {json_length} exceeds remaining data")

            json_bytes = data[offset : offset + json_length]
            json_str = json_bytes.decode("utf-8")

            return request_id, json.loads(json_str)

        except Exception as e:
            raise ValueError(f"Failed to decode RAG response: {e}")

    @staticmethod
    def encode_status_query(
        robot_id: str, detailed: bool = False, request_id: int = 0
    ) -> bytes:
        """
        Encode a status query message for sending to Python server.

        Format: [type:1][request_id:4][robot_id_len:4][robot_id:N][detailed:1]

        Args:
            robot_id: Robot identifier (e.g., "Robot1", "AR4_Robot")
            detailed: If True, return detailed joint information
            request_id: Request ID for correlation (default 0)

        Returns:
            Encoded message bytes

        Raises:
            ValueError: If data exceeds limits
        """
        # Validate inputs
        if len(robot_id) == 0:
            raise ValueError("Robot ID cannot be empty")
        if len(robot_id) > UnityProtocol.MAX_STRING_LENGTH:
            raise ValueError(
                f"Robot ID too long: {len(robot_id)} > {UnityProtocol.MAX_STRING_LENGTH}"
            )

        # Encode robot_id string
        robot_id_bytes = robot_id.encode("utf-8")

        # Build message
        message = bytearray()

        # Header
        message.extend(
            UnityProtocol._encode_header(MessageType.STATUS_QUERY, request_id)
        )

        # Robot ID
        message.extend(struct.pack(UnityProtocol.INT_FORMAT, len(robot_id_bytes)))
        message.extend(robot_id_bytes)

        # Detailed flag (1 byte: 0 or 1)
        message.extend(struct.pack("B", 1 if detailed else 0))

        return bytes(message)

    @staticmethod
    def decode_status_query(socket_or_data) -> Tuple[int, dict]:
        """
        Decode a status query message from Unity.

        Args:
            socket_or_data: Either a socket object or raw bytes

        Returns:
            Tuple of (request_id, query_dict) where query_dict contains robot_id and detailed flag

        Raises:
            ValueError: If message is malformed
        """
        # Handle both socket and bytes input
        if hasattr(socket_or_data, "recv"):
            # It's a socket - read the complete message
            data = UnityProtocol._receive_complete_status_query(socket_or_data)
        else:
            # It's already bytes
            data = socket_or_data

        try:
            # Decode header
            msg_type, request_id, offset = UnityProtocol._decode_header(data)

            if msg_type != MessageType.STATUS_QUERY:
                raise ValueError(f"Expected STATUS_QUERY message, got {msg_type.name}")

            # Read robot_id
            robot_id, offset = UnityProtocol._read_string(data, offset)

            # Read detailed flag
            if offset + 1 > len(data):
                raise ValueError("Not enough data for detailed flag")

            detailed = struct.unpack("B", data[offset : offset + 1])[0]
            offset += 1

            query_dict = {"robot_id": robot_id, "detailed": bool(detailed)}
            return request_id, query_dict

        except Exception as e:
            raise ValueError(f"Failed to decode status query: {e}")

    @staticmethod
    def _receive_complete_status_query(client_socket) -> bytes:
        """
        Receive a complete status query message from socket.

        Args:
            client_socket: Socket to receive from

        Returns:
            Complete message bytes
        """
        data = bytearray()

        # Read header (type + request_id)
        header_bytes = client_socket.recv(UnityProtocol.HEADER_SIZE)
        if not header_bytes or len(header_bytes) < UnityProtocol.HEADER_SIZE:
            raise ValueError("Connection closed or incomplete header")
        data.extend(header_bytes)

        # Read robot_id length and text
        robot_id_len_bytes = client_socket.recv(UnityProtocol.INT_SIZE)
        if not robot_id_len_bytes:
            raise ValueError("Connection closed")

        robot_id_len = struct.unpack(UnityProtocol.INT_FORMAT, robot_id_len_bytes)[0]
        data.extend(robot_id_len_bytes)

        robot_id_bytes = client_socket.recv(robot_id_len)
        data.extend(robot_id_bytes)

        # Read detailed flag (1 byte)
        detailed_byte = client_socket.recv(1)
        data.extend(detailed_byte)

        return bytes(data)

    @staticmethod
    def encode_status_response(robot_status: dict, request_id: int = 0) -> bytes:
        """
        Encode a status response message for sending to Unity.

        Format: [type:1][request_id:4][json_len:4][robot_status_json:N]

        Args:
            robot_status: Dictionary containing robot status data
            request_id: Request ID for correlation (default 0)

        Returns:
            Encoded message bytes
        """
        # Convert dict to JSON
        json_str = json.dumps(robot_status, ensure_ascii=False)
        json_bytes = json_str.encode("utf-8")

        # Build message
        message = bytearray()

        # Header
        message.extend(
            UnityProtocol._encode_header(MessageType.STATUS_RESPONSE, request_id)
        )

        # JSON data
        message.extend(struct.pack(UnityProtocol.INT_FORMAT, len(json_bytes)))
        message.extend(json_bytes)

        return bytes(message)

    @staticmethod
    def decode_status_response(data: bytes) -> Tuple[int, dict]:
        """
        Decode a status response message (for testing/verification).

        Args:
            data: Raw message bytes

        Returns:
            Tuple of (request_id, robot_status_dict)

        Raises:
            ValueError: If message is malformed
        """
        try:
            # Decode header
            msg_type, request_id, offset = UnityProtocol._decode_header(data)

            if msg_type != MessageType.STATUS_RESPONSE:
                raise ValueError(
                    f"Expected STATUS_RESPONSE message, got {msg_type.name}"
                )

            # Read JSON data
            if len(data) - offset < UnityProtocol.INT_SIZE:
                raise ValueError("Not enough data for JSON length")

            json_length = struct.unpack(
                UnityProtocol.INT_FORMAT, data[offset : offset + UnityProtocol.INT_SIZE]
            )[0]
            offset += UnityProtocol.INT_SIZE

            if offset + json_length > len(data):
                raise ValueError(f"JSON length {json_length} exceeds remaining data")

            json_bytes = data[offset : offset + json_length]
            json_str = json_bytes.decode("utf-8")

            return request_id, json.loads(json_str)

        except Exception as e:
            raise ValueError(f"Failed to decode status response: {e}")


if __name__ == "__main__":
    # Test the protocol
    print("Testing UnityProtocol V2...")

    # Test image message encoding/decoding
    test_camera_id = "TestCamera"
    test_prompt = "What do you see?"
    test_image = b"FAKE_PNG_DATA_HERE" * 100  # Fake image data
    test_request_id = 12345

    # Encode
    encoded = UnityProtocol.encode_image_message(
        test_camera_id, test_prompt, test_image, test_request_id
    )
    print(f"Encoded image message: {len(encoded)} bytes (with 5-byte header)")

    # Decode
    request_id, decoded_id, decoded_prompt, decoded_image = (
        UnityProtocol.decode_image_message(encoded)
    )
    assert request_id == test_request_id
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
    test_result_request_id = 67890

    # Encode
    encoded_result = UnityProtocol.encode_result_message(
        test_result, test_result_request_id
    )
    print(f"Encoded result message: {len(encoded_result)} bytes (with 5-byte header)")

    # Decode
    result_request_id, decoded_result = UnityProtocol.decode_result_message(
        encoded_result
    )
    assert result_request_id == test_result_request_id
    assert decoded_result == test_result
    print("✓ Result message encode/decode works")

    # Test RAG query
    rag_query_request_id = 99999
    rag_encoded = UnityProtocol.encode_rag_query(
        "move robot to position", top_k=5, request_id=rag_query_request_id
    )
    rag_request_id, rag_query_dict = UnityProtocol.decode_rag_query(rag_encoded)
    assert rag_request_id == rag_query_request_id
    assert rag_query_dict["query"] == "move robot to position"
    print("✓ RAG query encode/decode works")

    # Test status query
    status_query_request_id = 11111
    status_encoded = UnityProtocol.encode_status_query(
        "Robot1", detailed=True, request_id=status_query_request_id
    )
    status_request_id, status_query_dict = UnityProtocol.decode_status_query(
        status_encoded
    )
    assert status_request_id == status_query_request_id
    assert status_query_dict["robot_id"] == "Robot1"
    assert status_query_dict["detailed"] == True
    print("✓ Status query encode/decode works")

    # Test error cases
    try:
        UnityProtocol.encode_image_message("", "prompt", b"data", 0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Empty camera ID rejected: {e}")

    try:
        UnityProtocol.encode_image_message("A" * 300, "prompt", b"data", 0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Camera ID too long rejected: {e}")

    # Test message type detection
    image_type, _, _ = UnityProtocol._decode_header(encoded)
    assert image_type == MessageType.IMAGE
    result_type, _, _ = UnityProtocol._decode_header(encoded_result)
    assert result_type == MessageType.RESULT
    print("✓ Message type detection works")

    print("\nAll tests passed! Protocol V2 ready.")
