#!/usr/bin/env python3
"""
RunDetector.py - Main orchestrator for object detection system

This script:
1. Starts the DetectionServer in a background thread (port 5007)
2. Monitors the ImageStorage singleton for new images
3. Runs CubeDetector on each new image
4. Broadcasts detection results to Unity clients

Can run standalone or alongside RunAnalyzer.py for simultaneous LLM + detection.

Usage:
    python RunDetector.py
    python RunDetector.py --interval 0.5
    python RunDetector.py --camera AR4Left
    python RunDetector.py --debug  # Enable debug images
"""

import logging
import argparse
import sys
import time
import hashlib
from typing import Dict
from pathlib import Path

# Add parent directories to path for direct script execution
_package_dir = Path(__file__).parent.parent
_acrl_root = Path(__file__).parent.parent.parent
if str(_package_dir) not in sys.path:
    sys.path.insert(0, str(_package_dir))
if str(_acrl_root) not in sys.path:
    sys.path.insert(0, str(_acrl_root))

# Import config - support both direct script and module execution
# Try absolute import first (for direct execution), then relative (for module execution)
try:
    from LLMCommunication import LLMConfig as cfg
except ImportError:
    from .. import LLMConfig as cfg

# Import detection components - support both direct script and module execution
try:
    from LLMCommunication.vision.ObjectDetector import CubeDetector
    from LLMCommunication.servers.DetectionServer import (
        DetectionBroadcaster,
        run_detection_server_background,
    )
    from LLMCommunication.core.TCPServerBase import ServerConfig
    from LLMCommunication.servers.StreamingServer import ImageStorage
except ImportError:
    from ..vision.ObjectDetector import CubeDetector
    from ..servers.DetectionServer import (
        DetectionBroadcaster,
        run_detection_server_background,
    )
    from ..core.TCPServerBase import ServerConfig
    from ..servers.StreamingServer import ImageStorage

# Configure logging
logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=cfg.LOG_FORMAT)


def run_detection_loop(args):
    """
    Main detection loop that monitors images and runs detection

    Args:
        args: Command line arguments from argparse
    """
    try:
        # Wait for server to initialize
        logging.info("Waiting for DetectionServer to initialize...")
        time.sleep(cfg.SERVER_INIT_WAIT_TIME)

        # Initialize cube detector
        logging.info("Initializing CubeDetector...")
        detector = CubeDetector()

        # Track processed images to avoid reprocessing
        # Store (timestamp, image_hash) to detect when images actually change
        processed_images: Dict[str, tuple[float, str]] = {}

        # Get processing settings
        min_age = args.min_age if hasattr(args, "min_age") else cfg.MIN_IMAGE_AGE
        max_age = args.max_age if hasattr(args, "max_age") else cfg.MAX_IMAGE_AGE
        check_interval = (
            args.interval if hasattr(args, "interval") else cfg.DETECTION_CHECK_INTERVAL
        )

        # Determine which cameras to monitor
        monitor_cameras = args.camera  # None = all cameras

        logging.info("Starting continuous detection monitoring...")
        logging.info(f"Check interval: {check_interval}s")
        logging.info(f"Image age window: {min_age}s - {max_age}s")
        if monitor_cameras:
            logging.info(f"Monitoring cameras: {', '.join(monitor_cameras)}")
        else:
            logging.info("Monitoring all available cameras")

        # Get the ImageStorage instance
        storage = ImageStorage.get_instance()

        # Main detection loop
        while True:
            try:
                # Get camera IDs to check
                if monitor_cameras:
                    camera_ids = monitor_cameras
                else:
                    camera_ids = storage.get_all_camera_ids()

                # Check each camera for new images
                for cam_id in camera_ids:
                    image = storage.get_camera_image(cam_id)
                    if image is None:
                        continue

                    age = storage.get_camera_age(cam_id)

                    # Skip if age is None (shouldn't happen if image exists)
                    if age is None:
                        continue

                    # Skip if image is too fresh (might still be uploading)
                    if age < min_age:
                        continue

                    # Skip if image is too old
                    if age > max_age:
                        continue

                    # Calculate current image timestamp and hash
                    current_timestamp = time.time() - age

                    # Use MD5 hash to detect if image content changed
                    image_hash = hashlib.md5(image.tobytes()).hexdigest()

                    # Check if we've already processed this exact image recently
                    if cam_id in processed_images:
                        last_timestamp, last_hash = processed_images[cam_id]

                        # Skip if same image AND too close in time
                        if (
                            image_hash == last_hash
                            and (current_timestamp - last_timestamp)
                            < cfg.DUPLICATE_TIME_THRESHOLD
                        ):
                            logging.debug(
                                f"⏭️  Skipping duplicate image from {cam_id} (same content)"
                            )
                            continue  # Same image, skip

                    # Process the NEW image
                    print("\n" + "=" * 80)
                    print(f"🔍 DETECTING CUBES IN IMAGE FROM: {cam_id}")
                    print(f"⏱️  Age: {age:.2f}s")
                    if cam_id in processed_images:
                        print(f"✨ Image content has changed since last detection")
                    else:
                        print(f"🆕 First image from this camera")
                    print("=" * 80)

                    # Run detection
                    start_time = time.time()
                    result = detector.detect_cubes(image, camera_id=cam_id)
                    detection_time = time.time() - start_time

                    # Display detection results
                    print(f"\n📊 DETECTION RESULTS FOR {cam_id}")
                    print("=" * 80)
                    if len(result.detections) > 0:
                        print(f"Found {len(result.detections)} cube(s):")
                        for det in result.detections:
                            print(
                                f"  • {det.color.upper()} cube at ({det.center_x}, {det.center_y}) - "
                                f"bbox: ({det.bbox_w}×{det.bbox_h}px) - confidence: {det.confidence:.2f}"
                            )
                    else:
                        print("No cubes detected")
                    print("=" * 80)
                    print(f"⏱️  Detection time: {detection_time:.3f}s")
                    print("=" * 80 + "\n")

                    # Send result to Unity
                    DetectionBroadcaster.send_result(result.to_dict())
                    logging.info(
                        f"📤 Sent detection result to Unity for camera: {cam_id}"
                    )

                    # Mark as processed
                    processed_images[cam_id] = (current_timestamp, image_hash)

            except Exception as e:
                logging.error(f"Error in detection loop: {e}")
                import traceback

                traceback.print_exc()

            # Wait before next check
            time.sleep(check_interval)

    except KeyboardInterrupt:
        logging.info("\n\nDetection loop shutting down (interrupted by user)")
    except Exception as e:
        logging.error(f"\nDetection loop fatal error: {e}")
        import traceback

        traceback.print_exc()


def main():
    """
    Main entry point - runs server and detection loop together
    """
    parser = argparse.ArgumentParser(
        description="Run DetectionServer and cube detection together",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default settings
  %(prog)s

  # Monitor specific camera only
  %(prog)s --camera AR4Left

  # Adjust check interval
  %(prog)s --interval 0.5

  # Enable debug images
  %(prog)s --debug

Note: This script runs the DetectionServer and monitors ImageStorage for new images.
      Unity should send images to the StreamingServer (port 5005) first.
      Detection results are sent to Unity on port 5007.
        """,
    )

    # Input options
    input_group = parser.add_argument_group("input options")
    input_group.add_argument(
        "--camera",
        "-c",
        nargs="+",
        help="Specific camera ID(s) to monitor (default: all cameras)",
    )

    # Server options
    server_group = parser.add_argument_group("server options")
    server_group.add_argument(
        "--server-host",
        default=cfg.DEFAULT_HOST,
        help=f"DetectionServer host (default: {cfg.DEFAULT_HOST})",
    )
    server_group.add_argument(
        "--server-port",
        type=int,
        default=cfg.DETECTION_SERVER_PORT,
        help=f"DetectionServer port (default: {cfg.DETECTION_SERVER_PORT})",
    )
    server_group.add_argument(
        "--interval",
        "-i",
        type=float,
        default=cfg.DETECTION_CHECK_INTERVAL,
        help=f"Check interval in seconds (default: {cfg.DETECTION_CHECK_INTERVAL})",
    )
    server_group.add_argument(
        "--min-age",
        type=float,
        default=cfg.MIN_IMAGE_AGE,
        help=f"Minimum image age in seconds before processing (default: {cfg.MIN_IMAGE_AGE})",
    )
    server_group.add_argument(
        "--max-age",
        type=float,
        default=cfg.MAX_IMAGE_AGE,
        help=f"Maximum image age in seconds to consider fresh (default: {cfg.MAX_IMAGE_AGE})",
    )

    # Detection options
    detection_group = parser.add_argument_group("detection options")
    detection_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (save annotated images)",
    )

    args = parser.parse_args()

    # Enable debug mode if requested
    if args.debug:
        cfg.ENABLE_DEBUG_IMAGES = True
        debug_dir = Path(cfg.DEBUG_IMAGES_DIR)
        debug_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Debug mode enabled. Images will be saved to: {debug_dir}")

    try:
        # Create server config
        server_config = ServerConfig(host=args.server_host, port=args.server_port)

        # Start DetectionServer in background thread
        logging.info("Starting DetectionServer in background...")
        run_detection_server_background(server_config)

        # Run detection loop in main thread
        run_detection_loop(args)

    except KeyboardInterrupt:
        logging.info("\n\nShutting down (interrupted by user)")
        sys.exit(0)
    except Exception as e:
        logging.error(f"\nFatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
