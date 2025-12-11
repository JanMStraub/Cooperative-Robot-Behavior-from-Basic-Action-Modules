import numpy as np
import LLMConfig as cfg
from vision.YOLODetector import YOLODetector

# Create a blue cube image (same as test fixture)
image = np.zeros((480, 640, 3), dtype=np.uint8)
image[200:280, 270:370] = [255, 0, 0]  # BGR for blue

# Create YOLO detector with low confidence to see what it detects
detector = YOLODetector(model_path=cfg.YOLO_MODEL_PATH)
detector.conf_threshold = 0.01  # Very low threshold to see all detections

# Run prediction
results = detector.model(image, conf=detector.conf_threshold, iou=detector.iou_threshold, verbose=False)

print(f"\nRaw YOLO results:")
for r in results:
    if len(r.boxes) > 0:
        print(f"Found {len(r.boxes)} detections")
        for i, box in enumerate(r.boxes):
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = detector.get_class_name(class_id)
            print(f"  Detection {i}: {class_name} (conf={conf:.3f})")
    else:
        print("No detections found")
