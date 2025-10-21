#!/usr/bin/env python3
"""
UnityProtocol.py - Unity ↔ Python communication protocol

Defines the wire protocol for communication between Unity and Python servers.
This protocol is versioned and should match the Unity-side implementation.

Protocol version: 1

Message formats:
1. Image Message (Unity → Python):
   [camera_id_len:4][camera_id:N][prompt_len:4][prompt:N][image_len:4][image_data:N]

2. Result Message (Python → Unity):
   [json_len:4][json_data:N]

All integers are little-endian unsigned 32-bit (struct format 'I').
All strings are UTF-8 encoded.
"""

import struct
import json
from typing import Tuple, Optional
import logging

# Import config
import config as cfg

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
