#!/usr/bin/env python3
"""
Unit tests for core/UnityProtocol.py

Tests the wire protocol for Unity ↔ Python communication
"""

import pytest
import struct
import json
import sys
from pathlib import Path

# Add LLMCommunication directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "LLMCommunication"))

from LLMCommunication.core.UnityProtocol import UnityProtocol


class TestUnityProtocolConstants:
    """Test protocol constants and limits"""

    def test_protocol_version(self):
        """Test protocol version is defined"""
        assert UnityProtocol.VERSION == 1

    def test_limits(self):
        """Test protocol limits are defined"""
        assert UnityProtocol.MAX_STRING_LENGTH > 0
        assert UnityProtocol.MAX_IMAGE_SIZE > 0
        assert UnityProtocol.INT_SIZE == 4


class TestImageMessageEncoding:
    """Test encode_image_message functionality"""

    def test_encode_valid_message(self):
        """Test encoding a valid image message"""
        camera_id = "TestCamera"
        prompt = "What do you see?"
        image_bytes = b"FAKE_IMAGE_DATA" * 10

        encoded = UnityProtocol.encode_image_message(camera_id, prompt, image_bytes)

        # Check that message has correct structure
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0

        # Message should contain all components:
        # camera_id_len(4) + camera_id + prompt_len(4) + prompt + image_len(4) + image
        expected_min_length = (
            UnityProtocol.INT_SIZE  # camera_id length
            + len(camera_id.encode("utf-8"))
            + UnityProtocol.INT_SIZE  # prompt length
            + len(prompt.encode("utf-8"))
            + UnityProtocol.INT_SIZE  # image length
            + len(image_bytes)
        )
        assert len(encoded) == expected_min_length

    def test_encode_empty_prompt(self):
        """Test encoding with empty prompt"""
        camera_id = "TestCamera"
        prompt = ""
        image_bytes = b"IMAGE_DATA"

        encoded = UnityProtocol.encode_image_message(camera_id, prompt, image_bytes)
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0

    def test_encode_unicode_strings(self):
        """Test encoding with unicode characters"""
        camera_id = "Camera_日本語"
        prompt = "What do you see? 🤖"
        image_bytes = b"IMAGE"

        encoded = UnityProtocol.encode_image_message(camera_id, prompt, image_bytes)
        assert isinstance(encoded, bytes)

    def test_encode_empty_camera_id_raises(self):
        """Test that empty camera ID raises ValueError"""
        with pytest.raises(ValueError, match="Camera ID cannot be empty"):
            UnityProtocol.encode_image_message("", "prompt", b"data")

    def test_encode_empty_image_raises(self):
        """Test that empty image data raises ValueError"""
        with pytest.raises(ValueError, match="Image data cannot be empty"):
            UnityProtocol.encode_image_message("camera", "prompt", b"")

    def test_encode_oversized_camera_id_raises(self):
        """Test that oversized camera ID raises ValueError"""
        camera_id = "A" * (UnityProtocol.MAX_STRING_LENGTH + 1)
        with pytest.raises(ValueError, match="Camera ID too long"):
            UnityProtocol.encode_image_message(camera_id, "prompt", b"data")

    def test_encode_oversized_prompt_raises(self):
        """Test that oversized prompt raises ValueError"""
        prompt = "A" * (UnityProtocol.MAX_STRING_LENGTH + 1)
        with pytest.raises(ValueError, match="Prompt too long"):
            UnityProtocol.encode_image_message("camera", prompt, b"data")

    def test_encode_oversized_image_raises(self):
        """Test that oversized image raises ValueError"""
        image = b"A" * (UnityProtocol.MAX_IMAGE_SIZE + 1)
        with pytest.raises(ValueError, match="Image too large"):
            UnityProtocol.encode_image_message("camera", "prompt", image)


class TestImageMessageDecoding:
    """Test decode_image_message functionality"""

    def test_decode_valid_message(self):
        """Test decoding a valid encoded message"""
        camera_id = "TestCamera"
        prompt = "What do you see?"
        image_bytes = b"FAKE_IMAGE_DATA"

        # Encode then decode
        encoded = UnityProtocol.encode_image_message(camera_id, prompt, image_bytes)
        decoded_id, decoded_prompt, decoded_image = UnityProtocol.decode_image_message(
            encoded
        )

        assert decoded_id == camera_id
        assert decoded_prompt == prompt
        assert decoded_image == image_bytes

    def test_decode_empty_prompt(self):
        """Test decoding message with empty prompt"""
        camera_id = "Camera1"
        prompt = ""
        image_bytes = b"DATA"

        encoded = UnityProtocol.encode_image_message(camera_id, prompt, image_bytes)
        decoded_id, decoded_prompt, decoded_image = UnityProtocol.decode_image_message(
            encoded
        )

        assert decoded_id == camera_id
        assert decoded_prompt == ""
        assert decoded_image == image_bytes

    def test_decode_unicode(self):
        """Test decoding message with unicode strings"""
        camera_id = "Camera_日本語"
        prompt = "Test 🤖"
        image_bytes = b"IMAGE"

        encoded = UnityProtocol.encode_image_message(camera_id, prompt, image_bytes)
        decoded_id, decoded_prompt, decoded_image = UnityProtocol.decode_image_message(
            encoded
        )

        assert decoded_id == camera_id
        assert decoded_prompt == prompt
        assert decoded_image == image_bytes

    def test_decode_truncated_message_raises(self):
        """Test that truncated message raises ValueError"""
        camera_id = "Camera"
        prompt = "Prompt"
        image_bytes = b"DATA"

        encoded = UnityProtocol.encode_image_message(camera_id, prompt, image_bytes)

        # Truncate the message
        truncated = encoded[: len(encoded) // 2]

        with pytest.raises(ValueError, match="Failed to decode"):
            UnityProtocol.decode_image_message(truncated)

    def test_decode_empty_data_raises(self):
        """Test that empty data raises ValueError"""
        with pytest.raises(ValueError):
            UnityProtocol.decode_image_message(b"")

    def test_decode_malformed_length_raises(self):
        """Test that malformed length field raises ValueError"""
        # Create message with invalid length
        bad_data = struct.pack(UnityProtocol.INT_FORMAT, 999999999)
        bad_data += b"short_data"

        with pytest.raises(ValueError):
            UnityProtocol.decode_image_message(bad_data)


class TestResultMessageEncoding:
    """Test encode_result_message functionality"""

    def test_encode_valid_result(self):
        """Test encoding a valid result dictionary"""
        result = {
            "success": True,
            "response": "Test response",
            "camera_id": "Camera1",
            "metadata": {"model": "llava", "duration": 1.5},
        }

        encoded = UnityProtocol.encode_result_message(result)

        assert isinstance(encoded, bytes)
        assert len(encoded) > 0

        # Should start with length prefix
        json_length = struct.unpack(UnityProtocol.INT_FORMAT, encoded[:4])[0]
        assert json_length > 0

    def test_encode_empty_dict(self):
        """Test encoding empty dictionary"""
        result = {}

        encoded = UnityProtocol.encode_result_message(result)
        assert isinstance(encoded, bytes)
        assert len(encoded) >= UnityProtocol.INT_SIZE

    def test_encode_nested_dict(self):
        """Test encoding nested dictionary"""
        result = {
            "detections": [
                {"id": 0, "color": "red", "bbox": {"x": 10, "y": 20}},
                {"id": 1, "color": "blue", "bbox": {"x": 30, "y": 40}},
            ]
        }

        encoded = UnityProtocol.encode_result_message(result)
        assert isinstance(encoded, bytes)

    def test_encode_unicode_result(self):
        """Test encoding result with unicode strings"""
        result = {
            "response": "I see 日本語 characters 🤖",
            "camera_id": "Camera_日本語",
        }

        encoded = UnityProtocol.encode_result_message(result)
        assert isinstance(encoded, bytes)


class TestResultMessageDecoding:
    """Test decode_result_message functionality"""

    def test_decode_valid_result(self):
        """Test decoding a valid result message"""
        original = {
            "success": True,
            "response": "Test response",
            "camera_id": "Camera1",
        }

        encoded = UnityProtocol.encode_result_message(original)
        decoded = UnityProtocol.decode_result_message(encoded)

        assert decoded == original

    def test_decode_empty_dict(self):
        """Test decoding empty dictionary"""
        original = {}

        encoded = UnityProtocol.encode_result_message(original)
        decoded = UnityProtocol.decode_result_message(encoded)

        assert decoded == original

    def test_decode_nested_dict(self):
        """Test decoding nested dictionary"""
        original = {
            "detections": [{"id": 0, "color": "red"}, {"id": 1, "color": "blue"}],
            "metadata": {"count": 2},
        }

        encoded = UnityProtocol.encode_result_message(original)
        decoded = UnityProtocol.decode_result_message(encoded)

        assert decoded == original

    def test_decode_unicode(self):
        """Test decoding unicode strings"""
        original = {"response": "Unicode test 日本語 🤖"}

        encoded = UnityProtocol.encode_result_message(original)
        decoded = UnityProtocol.decode_result_message(encoded)

        assert decoded == original

    def test_decode_truncated_raises(self):
        """Test that truncated message raises ValueError"""
        result = {"key": "value"}
        encoded = UnityProtocol.encode_result_message(result)

        # Truncate
        truncated = encoded[: len(encoded) // 2]

        with pytest.raises(ValueError, match="Failed to decode"):
            UnityProtocol.decode_result_message(truncated)

    def test_decode_empty_data_raises(self):
        """Test that empty data raises ValueError"""
        with pytest.raises(ValueError):
            UnityProtocol.decode_result_message(b"")

    def test_decode_invalid_json_raises(self):
        """Test that invalid JSON raises ValueError"""
        # Create message with valid length but invalid JSON
        bad_json = b"not valid json"
        bad_data = struct.pack(UnityProtocol.INT_FORMAT, len(bad_json))
        bad_data += bad_json

        with pytest.raises(ValueError):
            UnityProtocol.decode_result_message(bad_data)


class TestRoundTripEncoding:
    """Test encoding and decoding round trips"""

    def test_image_message_roundtrip(self):
        """Test that image message survives encode/decode cycle"""
        camera_id = "TestCamera_日本語"
        prompt = "Describe the scene 🤖"
        image_bytes = bytes(range(256))  # All byte values

        encoded = UnityProtocol.encode_image_message(camera_id, prompt, image_bytes)
        decoded_id, decoded_prompt, decoded_image = UnityProtocol.decode_image_message(
            encoded
        )

        assert decoded_id == camera_id
        assert decoded_prompt == prompt
        assert decoded_image == image_bytes

    def test_result_message_roundtrip(self):
        """Test that result message survives encode/decode cycle"""
        result = {
            "success": True,
            "detections": [
                {"id": 0, "color": "red", "confidence": 0.95},
                {"id": 1, "color": "blue", "confidence": 0.87},
            ],
            "metadata": {
                "model": "detection_v1",
                "duration": 0.123,
                "unicode": "日本語 🤖",
            },
        }

        encoded = UnityProtocol.encode_result_message(result)
        decoded = UnityProtocol.decode_result_message(encoded)

        assert decoded == result

    def test_large_image_roundtrip(self):
        """Test encoding/decoding large image data"""
        camera_id = "Camera"
        prompt = "Test"
        # Create large but not oversized image (1MB)
        image_bytes = b"X" * (1024 * 1024)

        encoded = UnityProtocol.encode_image_message(camera_id, prompt, image_bytes)
        decoded_id, decoded_prompt, decoded_image = UnityProtocol.decode_image_message(
            encoded
        )

        assert decoded_id == camera_id
        assert decoded_prompt == prompt
        assert decoded_image == image_bytes
        assert len(decoded_image) == 1024 * 1024
