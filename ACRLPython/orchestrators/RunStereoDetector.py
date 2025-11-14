#!/usr/bin/env python3
"""
RunStereoDetector.py - Orchestrator for stereo object detection pipeline

Runs the complete stereo detection system:
1. StereoDetectionServer - receives stereo image pairs from Unity (port 5006)
2. ResultsServer - sends detection results back to Unity (port 5007)
3. Processing loop - detects objects with 3D positions using stereo disparity

Usage:
    python RunStereoDetector.py --baseline 0.1 --fov 60
"""

import logging
import time
import signal
import sys
from pathlib import Path
import argparse

# Import config - try both import styles
try:
    import LLMConfig as cfg
except ImportError:
    from .. import LLMConfig as cfg
from vision.StereoConfig import CameraConfig

from servers.StereoDetectionServer import (
    StereoDetectionServer,
    StereoImageStorage,
    run_stereo_detection_server_background,
)
from servers.ResultsServer import (
    run_results_server_background,
    ResultsBroadcaster,
)
from vision.ObjectDetector import CubeDetector

logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=cfg.LOG_FORMAT)


class StereoDetectorOrchestrator:
    """
    Orchestrates the stereo detection pipeline.

    Monitors incoming stereo image pairs, runs detection with depth estimation,
    and broadcasts results to Unity.
    """

    def __init__(self, camera_config: CameraConfig, check_interval: float = 0.5):
        """
        Initialize the orchestrator.

        Args:
            camera_config: Camera calibration parameters (baseline, FOV, etc.)
            check_interval: Seconds between checking for new images
        """
        self.camera_config = camera_config
        self.check_interval = check_interval

        # Initialize components
        self.image_storage = StereoImageStorage()
        self.detector = CubeDetector()
        self.results_broadcaster = ResultsBroadcaster()

        # Track processed images
        self.last_processed_time = {}

        # Shutdown flag
        self.shutdown_flag = False

        logging.info("Stereo detector orchestrator initialized")
        logging.info(
            f"Camera config: baseline={camera_config.baseline}m, FOV={camera_config.fov}°"
        )

    def process_loop(self):
        """
        Main processing loop.

        Continuously monitors for new stereo image pairs and processes them.
        """
        logging.info("Starting stereo detection processing loop")

        while not self.shutdown_flag:
            try:
                # Get all available camera pairs
                camera_pair_ids = self.image_storage.get_all_camera_pair_ids()

                # Process each camera pair
                for camera_pair_id in camera_pair_ids:
                    # Check if there's a new stereo pair to process
                    age = self.image_storage.get_pair_age(camera_pair_id)

                    if age is not None:
                        # Check if this is a new image (not processed yet)
                        last_processed = self.last_processed_time.get(camera_pair_id, 0)
                        current_time = time.time()
                        image_time = current_time - age

                        if image_time > last_processed:
                            # New image available - process it
                            self._process_stereo_pair(camera_pair_id)
                            self.last_processed_time[camera_pair_id] = current_time

                # Wait before checking again
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logging.info("Processing loop interrupted")
                break
            except Exception as e:
                logging.error(f"Error in processing loop: {e}")
                time.sleep(self.check_interval)

        logging.info("Processing loop stopped")

    def _process_stereo_pair(self, camera_pair_id: str):
        """
        Process a stereo image pair: detect objects with 3D positions.

        Args:
            camera_pair_id: Identifier for the camera pair
        """
        # Retrieve stereo pair
        stereo_data = self.image_storage.get_stereo_pair(camera_pair_id)
        if stereo_data is None:
            logging.warning(f"No stereo pair found for {camera_pair_id}")
            return

        imgL, imgR, prompt = stereo_data

        # Parse camera parameters from prompt if available
        import json

        camera_config = self.camera_config
        actual_prompt = prompt

        # Camera position and rotation for coordinate transformation
        camera_position = None
        camera_rotation = None

        try:
            # Try to parse as JSON with camera parameters
            prompt_data = json.loads(prompt)
            if isinstance(prompt_data, dict):
                # Extract camera parameters
                baseline = prompt_data.get("baseline", self.camera_config.baseline)
                fov = prompt_data.get("fov", self.camera_config.fov)
                actual_prompt = prompt_data.get("prompt", "")

                # Extract camera position and rotation if provided
                camera_position = prompt_data.get("camera_position")
                camera_rotation = prompt_data.get("camera_rotation")

                # Create config with Unity-provided parameters
                camera_config = CameraConfig(fov=fov, baseline=baseline)
        except json.JSONDecodeError:
            # Prompt is not JSON, use as-is
            pass

        # Print visual separator and processing header
        logging.info("=" * 80)
        logging.info(f"🔍 PROCESSING STEREO PAIR: {camera_pair_id}")
        if actual_prompt:
            logging.info(f"📝 Prompt: '{actual_prompt}'")
        logging.info(f"📷 Camera params: baseline={camera_config.baseline:.4f}m, FOV={camera_config.fov:.1f}°")
        if camera_position:
            logging.info(
                f"📍 Camera position: ({camera_position[0]:.3f}, {camera_position[1]:.3f}, {camera_position[2]:.3f})m"
            )
        if camera_rotation:
            logging.info(
                f"🔄 Camera rotation: ({camera_rotation[0]:.1f}, {camera_rotation[1]:.1f}, {camera_rotation[2]:.1f})°"
            )
        logging.info("=" * 80)

        try:
            # Run stereo detection with depth estimation
            start_time = time.time()
            result = self.detector.detect_cubes_stereo(
                imgL,
                imgR,
                camera_config,
                camera_id=camera_pair_id,
                camera_rotation=camera_rotation,
                camera_position=camera_position,
            )
            duration = time.time() - start_time

            # Prepare result dictionary
            result_dict = result.to_dict()

            # Remove image_width and image_height for Unity DepthResult format
            # (DepthResult doesn't have these fields, only DetectionResult does)
            result_dict.pop("image_width", None)
            result_dict.pop("image_height", None)

            result_dict["metadata"] = {
                "processing_time_seconds": round(duration, 3),
                "prompt": actual_prompt,
                "camera_baseline_m": camera_config.baseline,
                "camera_fov_deg": camera_config.fov,
                "detection_mode": "stereo_3d",
            }

            # Broadcast results to Unity
            self.results_broadcaster.send_result(result_dict)

            # Print visual result summary
            logging.info("")
            logging.info("=" * 80)
            logging.info(f"✓ STEREO DETECTION COMPLETE")
            logging.info("=" * 80)
            logging.info(f"⏱️  Processing time: {duration:.2f}s")
            logging.info(f"🎯 Detected objects: {len(result.detections)}")

            if result.detections:
                logging.info("=" * 80)
                for i, det in enumerate(result.detections, 1):
                    if det.world_position:
                        depth_str = f", depth={det.depth_m:.3f}m" if det.depth_m else ""
                        disp_str = f", disp={det.disparity:.1f}px" if det.disparity else ""
                        logging.info(
                            f"  [{i}] {det.color.upper()} cube:"
                        )
                        logging.info(
                            f"      📍 World position: ({det.world_position[0]:.3f}, {det.world_position[1]:.3f}, {det.world_position[2]:.3f})m"
                        )
                        logging.info(
                            f"      📷 Pixel position: ({det.center_x}, {det.center_y})"
                        )
                        if det.depth_m:
                            logging.info(f"      📏 Depth: {det.depth_m:.3f}m")
                        if det.disparity:
                            logging.info(f"      🔢 Disparity: {det.disparity:.1f}px")
                        logging.info(f"      ✓ Confidence: {det.confidence:.2f}")
                        if i < len(result.detections):
                            logging.info("      " + "-" * 60)

            logging.info("=" * 80)
            logging.info(f"📤 Sent results to Unity")
            logging.info("=" * 80)

        except Exception as e:
            logging.error(f"Failed to process stereo pair: {e}")

            # Send error result
            error_result = {
                "success": False,
                "error": str(e),
                "camera_id": camera_pair_id,
                "detections": [],
            }
            self.results_broadcaster.send_result(error_result)

    def shutdown(self):
        """Shutdown the orchestrator"""
        logging.info("Shutting down stereo detector orchestrator")
        self.shutdown_flag = True


def main():
    """
    Main entry point for the stereo detector system.
    """
    parser = argparse.ArgumentParser(
        description="Stereo Object Detector with 3D Position Estimation"
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=cfg.DEFAULT_STEREO_BASELINE,
        help="Camera baseline distance in meters (default: 0.05)",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=cfg.DEFAULT_STEREO_FOV,
        help="Camera field of view in degrees (default: 60)",
    )
    parser.add_argument(
        "--detection-host",
        type=str,
        default=cfg.DEFAULT_HOST,
        help="Stereo detection server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--detection-port",
        type=int,
        default=cfg.STEREO_DETECTION_PORT,
        help=f"Stereo detection server port (default: {cfg.STEREO_DETECTION_PORT})",
    )
    parser.add_argument(
        "--results-host",
        type=str,
        default=cfg.DEFAULT_HOST,
        help="Results server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--results-port",
        type=int,
        default=cfg.DEPTH_RESULTS_PORT,
        help=f"Results server port for depth detection results (default: {cfg.DEPTH_RESULTS_PORT})",
    )
    parser.add_argument(
        "--check-interval",
        type=float,
        default=cfg.IMAGE_CHECK_INTERVAL,
        help="Seconds between checking for new images (default: 0.5)",
    )
    parser.add_argument(
        "--no-results-server",
        action="store_true",
        help="Don't start ResultsServer (use when RunAnalyzer is already running)",
    )

    args = parser.parse_args()

    # Create camera configuration
    camera_config = CameraConfig(fov=args.fov, baseline=args.baseline)

    logging.info("=" * 60)
    logging.info("Stereo Object Detector with 3D Position Estimation")
    logging.info("=" * 60)
    logging.info(f"Camera baseline: {args.baseline}m")
    logging.info(f"Camera FOV: {args.fov}°")
    logging.info(
        f"Stereo detection server: {args.detection_host}:{args.detection_port}"
    )
    logging.info(f"Results server: {args.results_host}:{args.results_port}")
    logging.info("=" * 60)

    # Start servers
    logging.info("Starting servers...")

    # Start stereo detection server (receives stereo images from Unity)
    stereo_server = run_stereo_detection_server_background(
        args.detection_host, args.detection_port
    )

    # Start results server if requested
    results_server = None
    if not args.no_results_server:
        logging.info(f"Starting ResultsServer on port {args.results_port}")
        from core.TCPServerBase import ServerConfig

        results_config = ServerConfig(host=args.results_host, port=args.results_port)
        results_server = run_results_server_background(results_config)
    else:
        logging.info(
            "Skipping ResultsServer startup (using shared ResultsBroadcaster from RunAnalyzer)"
        )

    # Wait for servers to start
    time.sleep(1.0)

    # Create orchestrator
    orchestrator = StereoDetectorOrchestrator(camera_config, args.check_interval)

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logging.info(f"Received signal {signum}, shutting down...")
        orchestrator.shutdown()
        # Servers are daemon threads and will stop automatically
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start processing loop
    logging.info("Stereo detector ready - waiting for stereo image pairs from Unity")
    logging.info("Press Ctrl+C to stop")

    try:
        orchestrator.process_loop()
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        orchestrator.shutdown()
        # Servers are daemon threads and will stop automatically

    logging.info("Stereo detector stopped")


if __name__ == "__main__":
    main()
