#!/usr/bin/env python3
"""
Test visualization using main thread (macOS-compatible)
"""

import sys
import os
import time
import numpy as np
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.VisionProcessor import VisionProcessor
from vision.DetectionDataModels import DetectionObject, DetectionResult


class MockDetector:
    """Mock detector that returns test detections"""

    def detect_objects_stereo(self, imgL, imgR, **kwargs):
        # Create fake detections
        detections = [
            DetectionObject(
                object_id=1,
                color="red_cube",
                bbox=(100, 100, 150, 150),
                confidence=0.95,
                track_id=1,
                depth_m=0.85,
            ),
            DetectionObject(
                object_id=2,
                color="blue_cube",
                bbox=(400, 200, 100, 120),
                confidence=0.88,
                track_id=2,
                depth_m=1.2,
            ),
        ]

        return DetectionResult(
            camera_id="test",
            image_width=1280,
            image_height=960,
            detections=detections,
        )


def create_test_images():
    """Create synthetic stereo images for testing"""
    # Create simple gradient images
    imgL = np.zeros((960, 1280, 3), dtype=np.uint8)
    imgR = np.zeros((960, 1280, 3), dtype=np.uint8)

    # Add some visual content
    imgL[100:250, 100:250] = [0, 0, 255]  # Red square
    imgL[200:320, 400:500] = [255, 0, 0]  # Blue square

    imgR[100:250, 80:230] = [0, 0, 255]  # Red square (shifted)
    imgR[200:320, 380:480] = [255, 0, 0]  # Blue square (shifted)

    return imgL, imgR


def test_visualization_main_thread():
    """Test visualization in main thread (macOS-compatible)"""
    print("Testing VisionProcessor visualization (main thread mode)...")
    print()

    try:
        import cv2
        print("✓ OpenCV available")
    except ImportError:
        print("✗ OpenCV NOT available - cannot test visualization")
        print("  Install: pip install opencv-python")
        return  # Skip test if OpenCV not available

    # Create mock detector
    detector = MockDetector()
    print("✓ Mock detector created")

    # Create processor with visualization enabled AND main_thread=True
    processor = VisionProcessor(
        detector=detector,
        fps=5.0,
        enable_tracking=True,
        enable_visualization=True,
        use_main_thread=True,  # Enable main thread mode for macOS
    )
    print("✓ VisionProcessor created with visualization=True and use_main_thread=True")

    # Inject test images into UnifiedImageStorage
    try:
        from servers.ImageServer import UnifiedImageStorage

        storage = UnifiedImageStorage()
        imgL, imgR = create_test_images()

        # Store test images
        storage.store_stereo_pair(
            "test_camera",
            imgL,
            imgR,
            metadata={"baseline": 0.05, "fov": 60, "timestamp": time.time()},
        )
        print("✓ Test images stored in UnifiedImageStorage")

    except Exception as e:
        print(f"✗ Could not store test images: {e}")
        raise  # Fail test if cannot store images

    print()
    print("=" * 60)
    print("Starting VisionProcessor in main thread...")
    print("Visualization window should appear within 1-2 seconds")
    print("Press 'q' in the window or Ctrl+C to stop")
    print("=" * 60)
    print()

    # Set up a timer to stop after 10 seconds (or user can press Ctrl+C)
    def auto_stop():
        time.sleep(10)
        print("\n[Auto-stop after 10 seconds]")
        processor.stop()

    timer = threading.Thread(target=auto_stop, daemon=True)
    timer.start()

    # Run in main thread (blocking call)
    try:
        processor.run()  # This blocks until processor.stop() is called
    except KeyboardInterrupt:
        print("\n[Interrupted by user]")
        processor.stop()

    print()
    print("✓ Test complete")


if __name__ == "__main__":
    test_visualization_main_thread()
    sys.exit(0)
