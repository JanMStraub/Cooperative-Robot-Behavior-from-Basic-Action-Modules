#!/usr/bin/env python3
"""
Simple test to verify visualization window appears
"""

import sys
import os
import time
import numpy as np

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


def test_visualization():
    """Test that visualization window appears"""
    print("Testing VisionProcessor visualization...")
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

    # Create processor with visualization enabled
    processor = VisionProcessor(
        detector=detector,
        fps=5.0,
        enable_tracking=True,
        enable_visualization=True,  # Enable visualization
    )
    print("✓ VisionProcessor created with visualization=True")

    # Manually inject test images into UnifiedImageStorage
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
        print(f"⚠ Could not store test images: {e}")
        print("  Will test with direct visualization call instead")

        # Direct test of visualization
        imgL, imgR = create_test_images()
        result = detector.detect_objects_stereo(imgL, imgR)

        print(f"✓ Mock detection returned {len(result.detections)} objects")

        # Test drawing function directly
        try:
            vis_image = processor._draw_detections(imgL, result.detections)
            print("✓ _draw_detections() successful")

            # Display manually
            cv2.namedWindow("Test Window", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Test Window", 800, 600)
            cv2.imshow("Test Window", vis_image)
            print()
            print("=" * 60)
            print("Visualization window should be visible now!")
            print("Press any key in the window to close...")
            print("=" * 60)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("✓ Window closed")
            return  # Test passed

        except Exception as e:
            print(f"✗ Error displaying visualization: {e}")
            import traceback

            traceback.print_exc()
            raise  # Fail test on error

    # Start processor
    print()
    print("Starting VisionProcessor...")
    processor.start()
    print("✓ Processor started")

    print()
    print("=" * 60)
    print("Visualization window should appear within 1-2 seconds...")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    # Wait a bit for window to appear
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        pass

    # Stop processor
    print()
    print("Stopping processor...")
    processor.stop()
    print("✓ Processor stopped")


if __name__ == "__main__":
    test_visualization()
    sys.exit(0)
