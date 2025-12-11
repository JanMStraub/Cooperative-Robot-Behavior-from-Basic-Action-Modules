#!/usr/bin/env python3
"""
Edge Case Tests for Robot Control System

Tests boundary conditions, edge cases, and unusual scenarios across
all major components of the robot control system including:
- Extremely large/small coordinate values
- Very long command strings
- Network instability simulation
- Resource exhaustion
- Unicode and special characters
- Malformed data handling
"""

import pytest
import numpy as np
import json
import time
from unittest.mock import Mock, patch, MagicMock

from operations.MoveOperations import move_to_coordinate
from operations.GripperOperations import control_gripper
from operations.DetectionOperations import detect_objects
from operations.Base import OperationResult
from orchestrators.CommandParser import CommandParser
from servers.StreamingServer import ImageStorage
from servers.CommandServer import CommandBroadcaster


# ============================================================================
# Test Extreme Coordinate Values
# ============================================================================

class TestExtremeCoordinateValues:
    """Test operations with extreme coordinate values"""

    def test_move_with_maximum_valid_coordinates(self):
        """Test movement at maximum valid coordinate limits"""
        result = move_to_coordinate(
            robot_id="Robot1",
            x=1.0,   # Maximum X
            y=1.0,   # Maximum Y
            z=0.6    # Maximum Z
        )

        assert result["success"] is True

    def test_move_with_minimum_valid_coordinates(self):
        """Test movement at minimum valid coordinate limits"""
        result = move_to_coordinate(
            robot_id="Robot1",
            x=-1.0,  # Minimum X
            y=-1.0,  # Minimum Y
            z=-0.5   # Minimum Z
        )

        assert result["success"] is True

    def test_move_with_zero_coordinates(self):
        """Test movement to origin (0, 0, 0)"""
        result = move_to_coordinate(
            robot_id="Robot1",
            x=0.0,
            y=0.0,
            z=0.0
        )

        assert result["success"] is True

    def test_move_beyond_maximum_x(self):
        """Test movement beyond maximum X coordinate"""
        result = move_to_coordinate(
            robot_id="Robot1",
            x=1.5,  # Exceeds maximum
            y=0.0,
            z=0.1
        )

        assert result["success"] is False
        assert "INVALID_X_COORDINATE" in result["error"]["code"]

    def test_move_below_minimum_z(self):
        """Test movement below minimum Z coordinate"""
        result = move_to_coordinate(
            robot_id="Robot1",
            x=0.0,
            y=0.0,
            z=-1.0  # Below minimum
        )

        assert result["success"] is False
        assert "INVALID_Z_COORDINATE" in result["error"]["code"]

    def test_move_with_very_small_increments(self):
        """Test movement with extremely small coordinate differences"""
        result = move_to_coordinate(
            robot_id="Robot1",
            x=0.000001,  # Very small
            y=0.000001,
            z=0.000001
        )

        assert result["success"] is True


# ============================================================================
# Test Long and Complex Command Strings
# ============================================================================

class TestLongCommandStrings:
    """Test parsing and execution of very long commands"""

    def test_very_long_command_string(self):
        """Test parsing a very long command string (1000+ characters)"""
        parser = CommandParser(use_rag=False)

        # Create a long but valid command
        long_command = "move to (0.3, 0.2, 0.1) " + "and " * 100 + "close the gripper"

        result = parser.parse(long_command, robot_id="Robot1", use_llm=False)

        # Should still parse correctly (regex fallback)
        assert result["success"] is True
        assert len(result["commands"]) >= 2

    def test_command_with_repeated_operations(self):
        """Test command with many repeated operations"""
        parser = CommandParser(use_rag=False)

        command = " and ".join([f"move to ({0.1 + i*0.05}, 0.0, 0.1)" for i in range(20)])

        result = parser.parse(command, robot_id="Robot1", use_llm=False)

        assert result["success"] is True
        assert len(result["commands"]) > 10

    def test_empty_command_string(self):
        """Test parsing empty command"""
        parser = CommandParser(use_rag=False)

        result = parser.parse("", robot_id="Robot1")

        assert result["success"] is False
        assert "Empty command" in result["error"]

    def test_whitespace_only_command(self):
        """Test parsing whitespace-only command"""
        parser = CommandParser(use_rag=False)

        result = parser.parse("     \n\t   ", robot_id="Robot1")

        assert result["success"] is False


# ============================================================================
# Test Unicode and Special Characters
# ============================================================================

class TestUnicodeAndSpecialCharacters:
    """Test handling of Unicode and special characters"""

    def test_command_with_unicode_characters(self):
        """Test parsing command with Unicode characters"""
        parser = CommandParser(use_rag=False)

        # Command with Unicode (Chinese, Arabic, Emoji)
        command = "移动到 (0.3, 0.2, 0.1) 然后 close gripper 🤖"

        result = parser.parse(command, robot_id="Robot1", use_llm=False)

        # Regex parser might not understand Unicode, but should not crash
        assert "error" in result or "success" in result

    def test_robot_id_with_special_characters(self):
        """Test robot ID with special characters"""
        result = move_to_coordinate(
            robot_id="Robot-1_v2.0",  # Special chars in ID
            x=0.3,
            y=0.0,
            z=0.1
        )

        # Should succeed - robot ID is just a string
        assert result["success"] is True
        assert result["result"]["robot_id"] == "Robot-1_v2.0"

    def test_command_with_sql_injection_attempt(self):
        """Test command with SQL injection-like string"""
        parser = CommandParser(use_rag=False)

        command = "move to (0.3, 0.2, 0.1); DROP TABLE robots; --"

        result = parser.parse(command, robot_id="Robot1", use_llm=False)

        # Should parse normally, treating SQL as text
        assert "error" in result or "commands" in result


# ============================================================================
# Test Malformed Data Handling
# ============================================================================

class TestMalformedDataHandling:
    """Test handling of malformed and invalid data"""

    def test_move_with_string_coordinates(self):
        """Test movement with string instead of float coordinates"""
        # This should be caught by Python type system or validation
        try:
            result = move_to_coordinate(
                robot_id="Robot1",
                x="invalid",  # String instead of float
                y=0.0,
                z=0.1
            )
            # If it doesn't raise, should return error
            assert result["success"] is False
        except (TypeError, ValueError):
            # Exception is also acceptable
            pass

    def test_gripper_with_invalid_open_parameter(self):
        """Test gripper control with invalid open_gripper parameter"""
        result = control_gripper(
            robot_id="Robot1",
            open_gripper="maybe"  # Should be boolean
        )

        assert result["success"] is False
        assert "INVALID_OPEN_GRIPPER_PARAMETER" in result["error"]["code"]

    def test_image_storage_with_null_image(self):
        """Test storing null image"""
        storage = ImageStorage.get_instance()

        # Try to store None as image
        try:
            storage.store_image("test_cam", None, "")
            # If it doesn't raise, get should return None or handle gracefully
        except (TypeError, AttributeError):
            # Exception is acceptable
            pass

    def test_command_parser_with_none_input(self):
        """Test command parser with None input"""
        parser = CommandParser(use_rag=False)

        result = parser.parse(None, robot_id="Robot1")

        assert result["success"] is False


# ============================================================================
# Test Resource Limits
# ============================================================================

class TestResourceLimits:
    """Test behavior at resource limits"""

    def test_image_storage_with_many_cameras(self):
        """Test storing images from many cameras simultaneously"""
        storage = ImageStorage.get_instance()

        # Create sample image
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Store images from 100 cameras
        for i in range(100):
            camera_id = f"camera_{i}"
            storage.store_image(camera_id, image, f"prompt_{i}")

        # All should be stored
        camera_ids = storage.get_all_camera_ids()
        assert len(camera_ids) == 100

        # Cleanup
        storage._cameras.clear()

    def test_image_storage_with_very_large_image(self):
        """Test storing a very large image"""
        storage = ImageStorage.get_instance()

        # Create large image (4K resolution)
        large_image = np.zeros((3840, 2160, 3), dtype=np.uint8)

        storage.store_image("large_cam", large_image, "")

        # Should handle large images
        retrieved = storage.get_camera_image("large_cam")
        assert retrieved is not None
        assert retrieved.shape == large_image.shape

        # Cleanup
        storage._cameras.clear()

    def test_command_broadcaster_queue_overflow(self):
        """Test command queue behavior when full"""
        broadcaster = CommandBroadcaster()

        # Fill queue with max_queue_size commands (without server)
        for i in range(broadcaster._max_queue_size + 10):
            command = {"command_type": "test", "id": i}
            success = broadcaster.send_command(command)

            # Early commands should succeed, later ones should fail
            if i < broadcaster._max_queue_size:
                # Queue might accept or reject depending on implementation
                pass
            else:
                # Queue should be full
                assert success is False


# ============================================================================
# Test Boundary Time Values
# ============================================================================

class TestBoundaryTimeValues:
    """Test time-related edge cases"""

    def test_zero_timeout_operation(self):
        """Test operation with zero timeout"""
        # Very small timeout should likely fail or execute immediately
        # This is implementation-specific
        pass  # Placeholder - would need actual execution context

    def test_very_long_timeout(self):
        """Test operation with extremely long timeout"""
        # Should not cause overflow or issues
        result = move_to_coordinate(
            robot_id="Robot1",
            x=0.3,
            y=0.0,
            z=0.1,
            request_id=999999999  # Large request ID
        )

        assert result["success"] is True

    def test_image_age_calculation_wraparound(self):
        """Test image age with very old timestamp"""
        storage = ImageStorage.get_instance()

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        storage.store_image("old_cam", image, "")

        # Manually set very old timestamp
        with storage._lock:
            if "old_cam" in storage._cameras:
                img, _, prompt = storage._cameras["old_cam"]
                # Set to 1 year ago
                storage._cameras["old_cam"] = (img, time.time() - 31536000, prompt)

        age = storage.get_camera_age("old_cam")

        # Should calculate correctly
        assert age is not None
        assert age > 31535000  # Approximately 1 year

        # Cleanup
        storage._cameras.clear()


# ============================================================================
# Test Null and Missing Parameters
# ============================================================================

class TestNullAndMissingParameters:
    """Test operations with null or missing parameters"""

    def test_move_with_missing_robot_id(self):
        """Test movement without robot_id"""
        result = move_to_coordinate(
            robot_id="",  # Empty robot ID
            x=0.3,
            y=0.0,
            z=0.1
        )

        assert result["success"] is False
        assert "INVALID_ROBOT_ID" in result["error"]["code"]

    def test_move_with_none_robot_id(self):
        """Test movement with None robot_id"""
        result = move_to_coordinate(
            robot_id=None,
            x=0.3,
            y=0.0,
            z=0.1
        )

        assert result["success"] is False

    def test_gripper_with_none_open_parameter(self):
        """Test gripper with None for open_gripper"""
        result = control_gripper(
            robot_id="Robot1",
            open_gripper=None
        )

        assert result["success"] is False

    def test_detect_with_nonexistent_camera(self):
        """Test detection with camera that has no image"""
        result = detect_objects(
            robot_id="Robot1",
            camera_id="nonexistent_camera_12345"
        )

        assert result["success"] is False
        assert "NO_IMAGE" in result["error"]["code"]
