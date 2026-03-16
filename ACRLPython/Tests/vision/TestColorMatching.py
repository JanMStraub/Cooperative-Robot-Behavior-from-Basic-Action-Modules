#!/usr/bin/env python3
"""
Test color matching logic for VisionOperations
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from operations.VisionOperations import color_matches


def test_color_matches():
    """Test the flexible color matching function"""

    print("Testing color_matches function...")

    # Test exact matches (legacy CubeDetector)
    assert color_matches("blue", "blue"), "Exact match failed: blue == blue"
    assert color_matches("red", "red"), "Exact match failed: red == red"
    assert color_matches("green", "green"), "Exact match failed: green == green"

    # Test partial matches (YOLO detector)
    assert color_matches("blue_cube", "blue"), "Partial match failed: blue in blue_cube"
    assert color_matches("red_cube", "red"), "Partial match failed: red in red_cube"
    assert color_matches(
        "green_cube", "green"
    ), "Partial match failed: green in green_cube"

    # Test case insensitive
    assert color_matches(
        "Blue_Cube", "blue"
    ), "Case insensitive failed: blue matches Blue_Cube"
    assert color_matches(
        "RED_CUBE", "red"
    ), "Case insensitive failed: red matches RED_CUBE"
    assert color_matches("blue", "BLUE"), "Case insensitive failed: BLUE matches blue"

    # Test non-matches
    assert not color_matches("blue_cube", "red"), "Should not match: red != blue_cube"
    assert not color_matches("red", "blue"), "Should not match: blue != red"
    assert not color_matches(None, "blue"), "Should not match: None != blue"
    assert not color_matches("blue", None), "Should not match: blue != None"

    print("✓ All tests passed!")


if __name__ == "__main__":
    test_color_matches()
