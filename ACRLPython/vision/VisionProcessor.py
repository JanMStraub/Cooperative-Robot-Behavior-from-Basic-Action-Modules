#!/usr/bin/env python3
"""
VisionProcessor.py - Background thread for continuous vision processing

Implements a continuous vision processing pipeline that:
- Polls ImageServer for new stereo frames at configurable FPS
- Processes frames with YOLO detection + depth estimation
- Publishes results to SharedVisionState (optional)
- Runs non-blocking in background thread

This enables streaming vision for multi-robot coordination, providing
low-latency object detection updates without blocking the main thread.

Features:
- Configurable FPS (default: 5 FPS conservative)
- Thread-safe operation
- Graceful shutdown
- Optional object tracking across frames
- Optional shared vision state publishing
- Automatic error recovery

Usage:
    from vision.VisionProcessor import VisionProcessor
    from vision.YOLODetector import YOLODetector

    # Initialize detector
    detector = YOLODetector(model_path="yolo/models/robot_detector.onnx")

    # Create and start processor
    processor = VisionProcessor(detector, fps=5.0)
    processor.start()

    # ... processor runs in background ...

    # Stop when done
    processor.stop()
"""

import platform
import time
import threading
from typing import Optional, Callable, List, Any
import numpy as np

cv2: Any = None
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Import detection components
try:
    from .YOLODetector import YOLODetector
    from .ObjectTracker import ObjectTracker
    from .DetectionDataModels import DetectionObject, DetectionResult
    from .StereoConfig import CameraConfig
except ImportError:
    from vision.YOLODetector import YOLODetector
    from vision.ObjectTracker import ObjectTracker
    from vision.DetectionDataModels import DetectionObject, DetectionResult
    from vision.StereoConfig import CameraConfig

# Import from centralized lazy import system (prevents circular dependencies)
try:
    from ..core.Imports import get_unified_image_storage
except ImportError:
    from core.Imports import get_unified_image_storage


# Helper to get storage instance
def _get_storage():
    """Get UnifiedImageStorage instance using centralized imports"""
    return get_unified_image_storage()


# Import config
try:
    from config.Vision import (
        TRACKING_MAX_AGE,
        TRACKING_MIN_IOU,
        DEFAULT_STEREO_BASELINE,
        DEFAULT_STEREO_FOV,
        DEFAULT_STEREO_CAMERA_POSITION,
        DEFAULT_STEREO_CAMERA_ROTATION,
        YOLO_INPUT_SIZE,
        SCENE_DIFF_THUMB_SIZE,
        SCENE_DIFF_THRESHOLD,
    )
except ImportError:
    from ..config.Vision import (
        TRACKING_MAX_AGE,
        TRACKING_MIN_IOU,
        DEFAULT_STEREO_BASELINE,
        DEFAULT_STEREO_FOV,
        DEFAULT_STEREO_CAMERA_POSITION,
        DEFAULT_STEREO_CAMERA_ROTATION,
        YOLO_INPUT_SIZE,
        SCENE_DIFF_THUMB_SIZE,
        SCENE_DIFF_THRESHOLD,
    )

from core.LoggingSetup import get_logger

logger = get_logger(__name__)


class VisionProcessor:
    """
    Background thread for continuous vision processing.

    Polls ImageServer for new stereo frames, processes with YOLO + depth,
    and optionally publishes to SharedVisionState for multi-robot coordination.

    Attributes:
        detector: YOLODetector instance for object detection
        fps: Target processing rate in frames per second
        enable_tracking: Enable persistent object tracking across frames
        enable_shared_state: Publish results to SharedVisionState
        running: Thread running state
        thread: Background processing thread
        tracker: Optional ObjectTracker for persistent IDs
        on_result_callback: Optional callback for detection results

    Example:
        detector = YOLODetector(model_path="robot_detector.onnx")
        processor = VisionProcessor(detector, fps=5.0, enable_tracking=True)
        processor.start()

        # Set callback for results
        def handle_results(result: DetectionResult):
            print(f"Detected {len(result.detections)} objects")

        processor.on_result_callback = handle_results
    """

    def __init__(
        self,
        detector: Any,
        fps: float = 5.0,
        enable_tracking: bool = False,
        enable_shared_state: bool = False,
        enable_visualization: bool = False,
        use_main_thread: bool = False,
    ):
        """
        Initialize vision processor.

        Args:
            detector: YOLODetector instance
            fps: Target processing rate (default: 5.0 FPS)
            enable_tracking: Enable object tracking (default: False)
            enable_shared_state: Publish to SharedVisionState (default: False)
            enable_visualization: Show live video window with detections (default: False)
            use_main_thread: Run in main thread instead of background (required for macOS GUI) (default: False)
        """
        self.detector = detector
        self.fps = fps
        self.enable_tracking = enable_tracking
        self.enable_shared_state = enable_shared_state
        self.enable_visualization = enable_visualization
        self.use_main_thread = use_main_thread

        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.tracker: Optional[ObjectTracker] = None
        self.on_result_callback: Optional[Callable[[DetectionResult], None]] = None
        self.viz_window_name = "VisionProcessor - Live Detection"
        # Only enable input resize if OpenCV is available
        self._yolo_input_size: Optional[int] = YOLO_INPUT_SIZE if CV2_AVAILABLE else None

        # Initialize tracker if enabled
        if self.enable_tracking:
            self.tracker = ObjectTracker(
                max_age=TRACKING_MAX_AGE, min_iou=TRACKING_MIN_IOU
            )
            logger.info("Object tracking enabled for VisionProcessor")

        # Initialize shared state if enabled
        if self.enable_shared_state:
            try:
                # Use absolute import to avoid "beyond top-level package" error
                from operations.SharedVisionState import get_shared_vision_state

                self.shared_state = get_shared_vision_state()
                logger.info("Shared vision state enabled for VisionProcessor")
            except ImportError as e:
                logger.warning(
                    f"SharedVisionState not available - shared state disabled. ImportError: {e}"
                )
                self.enable_shared_state = False
            except Exception as e:
                logger.warning(
                    f"SharedVisionState initialization failed - shared state disabled. Error: {type(e).__name__}: {e}"
                )
                self.enable_shared_state = False

        # Check if OpenCV is available for visualization
        if self.enable_visualization:
            try:
                # Warn about macOS threading limitations
                if platform.system() == "Darwin" and not use_main_thread:
                    logger.warning(
                        "macOS detected: OpenCV GUI may not work in background threads. "
                        "Consider setting use_main_thread=True and calling run() instead of start()"
                    )
                logger.info("Visualization enabled for VisionProcessor")
            except ImportError:
                logger.warning("OpenCV not available - visualization disabled")
                self.enable_visualization = False

        resize_str = f"{self._yolo_input_size}px" if self._yolo_input_size else "disabled"
        logger.info(
            f"VisionProcessor initialized: fps={fps}, tracking={enable_tracking}, "
            f"shared_state={enable_shared_state}, visualization={enable_visualization}, "
            f"main_thread={use_main_thread}, input_resize={resize_str}"
        )

    def start(self):
        """
        Start background processing thread.

        Note: On macOS, visualization may not work in background threads.
        Use run() instead to run in the main thread.
        """
        if self.running:
            logger.warning("VisionProcessor already running")
            return

        if self.use_main_thread:
            logger.error(
                "Cannot use start() with use_main_thread=True. Use run() instead."
            )
            return

        self.running = True
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()
        logger.info("VisionProcessor started")

    def run(self):
        """
        Run processing loop in the current (main) thread.

        This is a blocking call that runs until stop() is called from another thread
        or KeyboardInterrupt is received. Use this instead of start() on macOS when
        visualization is enabled.

        Example:
            processor = VisionProcessor(detector, enable_visualization=True, use_main_thread=True)
            try:
                processor.run()  # Blocks until Ctrl+C
            except KeyboardInterrupt:
                processor.stop()
        """
        if self.running:
            logger.warning("VisionProcessor already running")
            return

        self.running = True
        logger.info("VisionProcessor running in main thread")
        self._processing_loop()

    def stop(self):
        """
        Stop background processing thread.
        """
        if not self.running:
            logger.warning("VisionProcessor not running")
            return

        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning("VisionProcessor thread did not stop cleanly")
            else:
                logger.info("VisionProcessor stopped")
        self.thread = None

        # Close visualization window if enabled
        if self.enable_visualization:
            try:
                cv2.destroyWindow(self.viz_window_name)
            except:
                pass

    def _processing_loop(self):
        """
        Main processing loop (runs in background thread).

        Process flow:
        1. Cheap timestamp poll — skip frame copy if Unity hasn't sent a new frame
        2. Copy stereo images from storage
        3. Optionally resize to YOLO_INPUT_SIZE before inference
        4. Run YOLO detection (+ depth only when a consumer actually needs it)
        5. Apply object tracking if enabled
        6. Publish to SharedVisionState / invoke callback
        7. Sleep to maintain target FPS
        """
        try:
            storage = _get_storage()
            if storage is None:
                raise RuntimeError("UnifiedImageStorage returned None")
        except Exception as e:
            logger.error(
                f"UnifiedImageStorage not available - cannot run VisionProcessor: {e}"
            )
            self.running = False
            return

        frame_interval = 1.0 / self.fps
        last_processed_timestamp = 0.0
        last_thumb: Optional[Any] = None  # previous frame thumbnail for scene-change comparison

        # Depth/world-position is only useful when something consumes it.
        # If neither shared state nor a callback is registered we run 2D-only,
        # skipping the expensive SGBM disparity step entirely.
        needs_depth = self.enable_shared_state or (self.on_result_callback is not None)

        scene_diff_enabled = bool(SCENE_DIFF_THUMB_SIZE)
        logger.info(
            f"VisionProcessor loop started (target: {self.fps} FPS, "
            f"depth={'enabled' if needs_depth else 'disabled — no consumer'}, "
            f"scene_diff={'enabled (thresh={})'.format(SCENE_DIFF_THRESHOLD) if scene_diff_enabled else 'disabled'})"
        )

        while self.running:
            loop_start_time = time.time()

            try:
                # --- Gate 1: cheap timestamp + thumbnail poll (single lock acquisition) ---
                latest_ts, thumb = storage.get_latest_stereo_poll()

                if latest_ts <= last_processed_timestamp:
                    time.sleep(frame_interval * 0.5)
                    continue

                # --- Gate 2: scene-change check using pre-computed thumbnail ---
                if scene_diff_enabled and thumb is not None and last_thumb is not None:
                    diff = float(np.mean(np.abs(thumb - last_thumb)))
                    if diff < SCENE_DIFF_THRESHOLD:
                        logger.debug(
                            f"VisionProcessor: Scene unchanged (MAD={diff:.2f} < {SCENE_DIFF_THRESHOLD}), skipping"
                        )
                        last_processed_timestamp = latest_ts  # advance so we don't re-check same frame
                        time.sleep(frame_interval * 0.5)
                        continue

                # New, visually distinct frame — pay the full copy cost now
                stereo_data = storage.get_latest_stereo_image()
                if stereo_data is None:
                    time.sleep(frame_interval)
                    continue

                imgL, imgR, prompt, timestamp, metadata = stereo_data

                # Guard against a race between the poll and the copy
                if timestamp <= last_processed_timestamp:
                    time.sleep(frame_interval * 0.5)
                    continue

                last_thumb = thumb  # update reference for next iteration

                logger.debug(
                    f"VisionProcessor: Processing new frame (timestamp: {timestamp:.3f}, "
                    f"delta: {timestamp - last_processed_timestamp:.3f}s)"
                )

                # --- Resize to cap YOLO input resolution ---
                # YOLO internally letterboxes to 640×640 anyway; sending a large image
                # only wastes preprocessing time. Cap the long edge at YOLO_INPUT_SIZE.
                if self._yolo_input_size is not None:
                    h, w = imgL.shape[:2]
                    if max(h, w) > self._yolo_input_size:
                        scale = self._yolo_input_size / max(h, w)
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        imgL = cv2.resize(imgL, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        imgR = cv2.resize(imgR, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                # --- Extract camera config from metadata ---
                camera_config = None
                camera_position = None
                camera_rotation = None
                if metadata:
                    baseline = metadata.get("baseline", DEFAULT_STEREO_BASELINE)
                    fov = metadata.get("fov", DEFAULT_STEREO_FOV)
                    camera_config = CameraConfig(fov=fov, baseline=baseline)
                    camera_position = metadata.get(
                        "camera_position", DEFAULT_STEREO_CAMERA_POSITION
                    )
                    camera_rotation = metadata.get(
                        "camera_rotation", DEFAULT_STEREO_CAMERA_ROTATION
                    )

                # --- Re-evaluate depth need each iteration (callback may be added later) ---
                needs_depth = self.enable_shared_state or (self.on_result_callback is not None)

                if needs_depth:
                    result = self.detector.detect_objects_stereo(
                        imgL,
                        imgR,
                        camera_id="stereo_stream",
                        camera_config=camera_config,
                        camera_position=camera_position,
                        camera_rotation=camera_rotation,
                    )
                else:
                    # 2D-only path: skip SGBM disparity entirely
                    result = self.detector.detect_objects(
                        imgL, camera_id="stereo_stream"
                    )

                # --- Object tracking ---
                if self.tracker and len(result.detections) > 0:
                    tracked_detections = self.tracker.update(result.detections)
                    result = DetectionResult(
                        result.camera_id,
                        result.image_width,
                        result.image_height,
                        tracked_detections,
                    )

                # --- Publish / callback ---
                if self.enable_shared_state and len(result.detections) > 0:
                    try:
                        self.shared_state.update_detections(result.detections)
                        logger.debug(
                            f"Published {len(result.detections)} detections to SharedVisionState"
                        )
                    except Exception as e:
                        logger.error(f"Failed to publish to SharedVisionState: {e}")

                if self.on_result_callback:
                    try:
                        self.on_result_callback(result)
                    except Exception as e:
                        logger.error(f"Error in result callback: {e}")

                # --- Visualization ---
                if self.enable_visualization:
                    try:
                        if (
                            not hasattr(self, "_viz_initialized")
                            or not self._viz_initialized
                        ):
                            if platform.system() == "Darwin":
                                try:
                                    cv2.startWindowThread()
                                except:
                                    pass
                            cv2.namedWindow(self.viz_window_name, cv2.WINDOW_NORMAL)
                            cv2.resizeWindow(self.viz_window_name, 400, 300)
                            self._viz_initialized = True
                            logger.info(
                                f"Visualization window created: {self.viz_window_name}"
                            )

                        vis_image = self._draw_detections(imgL, result.detections)
                        cv2.imshow(self.viz_window_name, vis_image)
                        key = cv2.waitKey(10)
                        if key == ord("q") or key == 27:
                            logger.info("User requested quit via keyboard")
                            self.running = False
                    except Exception as e:
                        logger.error(f"Error displaying visualization: {e}")
                        if not hasattr(self, "_viz_error_count"):
                            self._viz_error_count = 0
                        self._viz_error_count += 1
                        if self._viz_error_count > 5:
                            logger.warning(
                                "Too many visualization errors - disabling visualization"
                            )
                            self.enable_visualization = False

                last_processed_timestamp = timestamp

                processing_time = time.time() - loop_start_time
                logger.debug(
                    f"VisionProcessor: Detected {len(result.detections)} objects "
                    f"in {processing_time*1000:.1f}ms (timestamp: {timestamp:.3f})"
                )

            except Exception as e:
                logger.error(f"Error in VisionProcessor loop: {e}", exc_info=True)
                time.sleep(frame_interval)
                continue

            # Sleep to maintain target FPS
            elapsed = time.time() - loop_start_time
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("VisionProcessor loop ended")

    def _draw_detections(self, image, detections: List[DetectionObject]):
        """
        Draw detection bounding boxes and labels on image.

        Args:
            image: Input image (BGR format)
            detections: List of DetectionObject to draw

        Returns:
            Image with detections drawn
        """
        try:
            # Make a copy to avoid modifying original
            vis_image = image.copy()

            # Color map for different object types
            color_map = {
                "red": (0, 0, 255),  # Red in BGR
                "blue": (255, 0, 0),  # Blue in BGR
                "green": (0, 255, 0),  # Green in BGR
                "yellow": (0, 255, 255),  # Yellow in BGR
                "default": (255, 255, 255),  # White
            }

            for det in detections:
                # Get color (match by prefix)
                bbox_color = color_map["default"]
                for key, value in color_map.items():
                    if key in det.color.lower():
                        bbox_color = value
                        break

                # Draw bounding box
                x, y, w, h = det.bbox_x, det.bbox_y, det.bbox_w, det.bbox_h
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), bbox_color, 2)

                # Build label text
                label_parts = [det.color]
                if det.track_id is not None:
                    label_parts.append(f"ID:{det.track_id}")
                if det.confidence is not None:
                    label_parts.append(f"{det.confidence:.2f}")
                if det.depth_m is not None:
                    label_parts.append(f"{det.depth_m:.2f}m")

                label = " ".join(label_parts)

                # Draw label background
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    vis_image,
                    (x, y - label_h - 10),
                    (x + label_w, y),
                    bbox_color,
                    -1,  # Filled
                )

                # Draw label text
                cv2.putText(
                    vis_image,
                    label,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),  # Black text
                    1,
                    cv2.LINE_AA,
                )

            # Add FPS counter
            fps_text = f"FPS: {self.fps:.1f} | Objects: {len(detections)}"
            cv2.putText(
                vis_image,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),  # Green
                2,
                cv2.LINE_AA,
            )

            return vis_image

        except Exception as e:
            logger.error(f"Error drawing detections: {e}")
            return image

    def get_stats(self) -> dict:
        """
        Get processor statistics.

        Returns:
            Dictionary with processor stats
        """
        stats = {
            "running": self.running,
            "fps": self.fps,
            "tracking_enabled": self.enable_tracking,
            "shared_state_enabled": self.enable_shared_state,
        }

        if self.tracker:
            stats["active_tracks"] = len(self.tracker.get_active_tracks())

        return stats


# ===========================
# Main (for testing)
# ===========================


def main():
    """Test VisionProcessor with mock detector"""
    print("=== VisionProcessor Test ===\n")

    # Mock detector (would normally be YOLODetector)
    class MockDetector:
        def detect_objects_stereo(self, imgL, imgR, **kwargs):
            # Return empty result for testing
            from vision.DetectionDataModels import DetectionResult

            return DetectionResult("test", 1280, 960, [])

    detector = MockDetector()
    processor = VisionProcessor(detector, fps=2.0, enable_tracking=False)

    # Set callback
    def on_result(result: DetectionResult):
        print(f"Callback: {len(result.detections)} detections")

    processor.on_result_callback = on_result

    # Start and run for 5 seconds
    print("Starting processor...")
    processor.start()

    print("Running for 5 seconds...")
    time.sleep(5)

    print("Stopping processor...")
    processor.stop()

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
