import numpy as np
from vision.ObjectDetector import CubeDetector

# Create a blue cube image (same as test fixture)
image = np.zeros((480, 640, 3), dtype=np.uint8)
image[200:280, 270:370] = [255, 0, 0]  # BGR for blue

detector = CubeDetector()
result = detector.detect_objects(image, camera_id="test")

print(f"Total detections: {len(result.detections)}")
for det in result.detections:
    print(f"  Color: '{det.color}', Confidence: {det.confidence:.2f}, BBox: ({det.bbox_x}, {det.bbox_y}, {det.bbox_w}, {det.bbox_h})")
