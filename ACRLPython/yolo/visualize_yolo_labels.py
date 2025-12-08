#!/usr/bin/env python3
"""
Simple YOLO Label Visualizer
Manually select an image and its label file to visualize bounding boxes.

Usage:
    python visualize_yolo_labels.py
"""

import cv2
import numpy as np
from tkinter import Tk, filedialog
import os


# Color palette for different classes (BGR format for OpenCV)
COLORS = [
    (0, 255, 0),      # Green - class 0
    (255, 0, 0),      # Blue - class 1
    (0, 0, 255),      # Red - class 2
    (0, 255, 255),    # Yellow - class 3
    (255, 0, 255),    # Magenta - class 4
    (255, 255, 0),    # Cyan - class 5
    (128, 0, 128),    # Purple - class 6
    (0, 128, 255),    # Orange - class 7
]


def select_file(title, file_types):
    """Open file dialog to select a file"""
    root = Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front

    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=file_types
    )
    root.destroy()
    return file_path


def parse_yolo_label(label_path):
    """
    Parse YOLO format label file
    Format: class_id x_center y_center width height (normalized 0-1)
    Returns: List of (class_id, x_center, y_center, width, height)
    """
    boxes = []

    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return boxes

    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                boxes.append((class_id, x_center, y_center, width, height))

    return boxes


def draw_bounding_boxes(image, boxes, class_names=None):
    """
    Draw bounding boxes on image

    Args:
        image: OpenCV image (numpy array)
        boxes: List of (class_id, x_center, y_center, width, height) in normalized coords
        class_names: Optional list of class names for labels

    Returns:
        Image with bounding boxes drawn
    """
    img_height, img_width = image.shape[:2]
    result_image = image.copy()

    for class_id, x_center, y_center, width, height in boxes:
        # Convert normalized coordinates to pixel coordinates
        x_center_px = int(x_center * img_width)
        y_center_px = int(y_center * img_height)
        box_width_px = int(width * img_width)
        box_height_px = int(height * img_height)

        # Calculate top-left corner
        x1 = int(x_center_px - box_width_px / 2)
        y1 = int(y_center_px - box_height_px / 2)
        x2 = int(x_center_px + box_width_px / 2)
        y2 = int(y_center_px + box_height_px / 2)

        # Get color for this class
        color = COLORS[class_id % len(COLORS)]

        # Draw bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

        # Draw center point
        cv2.circle(result_image, (x_center_px, y_center_px), 4, color, -1)

        # Create label text
        if class_names and class_id < len(class_names):
            label = f"{class_names[class_id]} (ID:{class_id})"
        else:
            label = f"Class {class_id}"

        # Draw label background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            result_image,
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            color,
            -1
        )

        # Draw label text
        cv2.putText(
            result_image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return result_image


def load_class_names(label_path):
    """
    Try to load class names from classes.txt in the same directory or parent directory
    """
    # Try same directory as label file
    label_dir = os.path.dirname(label_path)
    classes_path = os.path.join(label_dir, "classes.txt")

    if not os.path.exists(classes_path):
        # Try parent directory
        parent_dir = os.path.dirname(label_dir)
        classes_path = os.path.join(parent_dir, "classes.txt")

    if os.path.exists(classes_path):
        with open(classes_path, 'r') as f:
            class_names = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(class_names)} class names from: {classes_path}")
        return class_names

    return None


def main():
    print("=" * 60)
    print("YOLO Label Visualizer")
    print("=" * 60)

    # Select image file
    print("\n1. Select an image file...")
    image_path = select_file(
        "Select Image File",
        [
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("All files", "*.*")
        ]
    )

    if not image_path:
        print("No image selected. Exiting.")
        return

    print(f"Selected image: {image_path}")

    # Select label file
    print("\n2. Select corresponding label file...")
    label_path = select_file(
        "Select YOLO Label File",
        [
            ("Text files", "*.txt"),
            ("All files", "*.*")
        ]
    )

    if not label_path:
        print("No label file selected. Exiting.")
        return

    print(f"Selected label: {label_path}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    print(f"\nImage size: {image.shape[1]}x{image.shape[0]}")

    # Parse labels
    boxes = parse_yolo_label(label_path)
    print(f"Found {len(boxes)} bounding boxes")

    if len(boxes) == 0:
        print("Warning: No bounding boxes found in label file")

    # Try to load class names
    class_names = load_class_names(label_path)

    # Draw bounding boxes
    result_image = draw_bounding_boxes(image, boxes, class_names)

    # Display information
    print("\n" + "=" * 60)
    print("Bounding Box Details:")
    print("=" * 60)
    for i, (class_id, x_center, y_center, width, height) in enumerate(boxes):
        class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"Class {class_id}"
        print(f"Box {i+1}: {class_name}")
        print(f"  - Center: ({x_center:.4f}, {y_center:.4f})")
        print(f"  - Size: {width:.4f} x {height:.4f}")

    # Display result
    print("\n" + "=" * 60)
    print("Controls:")
    print("  - Press 's' to save the annotated image")
    print("  - Press 'q' or ESC to quit")
    print("=" * 60)

    window_name = "YOLO Label Visualization"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, result_image)

    # Wait for key press
    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('s'):  # 's' for save
            # Save annotated image
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(
                os.path.dirname(image_path),
                f"{base_name}_annotated.jpg"
            )
            cv2.imwrite(output_path, result_image)
            print(f"\nSaved annotated image to: {output_path}")

    cv2.destroyAllWindows()
    print("\nDone!")


if __name__ == "__main__":
    main()
