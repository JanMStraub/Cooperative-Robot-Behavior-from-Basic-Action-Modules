#!/usr/bin/env python3
"""
RunAnalyzer.py - Runs both StreamingServer and AnalyzeImage in the same process

This wrapper script starts the StreamingServer in a background thread and runs
the image analyzer in the main thread, allowing them to share the ImageServer
singleton and communicate in-process.

Usage:
    python RunAnalyzer.py
    python RunAnalyzer.py --model llama-3.2-vision
    python RunAnalyzer.py --help
"""

import logging
import argparse
import sys
import hashlib
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
    from LLMCommunication import llm_config as cfg
except ImportError:
    from .. import llm_config as cfg

# Import server components - support both direct script and module execution
try:
    from LLMCommunication.servers.StreamingServer import ImageStorage, run_streaming_server_background
    from LLMCommunication.core.TCPServerBase import ServerConfig
    from LLMCommunication.servers.ResultsServer import ResultsBroadcaster, run_results_server_background
    from LLMCommunication.vision.AnalyzeImage import LMStudioVisionProcessor, save_response
except ImportError:
    from ..servers.StreamingServer import ImageStorage, run_streaming_server_background
    from ..core.TCPServerBase import ServerConfig
    from ..servers.ResultsServer import ResultsBroadcaster, run_results_server_background
    from ..vision.AnalyzeImage import LMStudioVisionProcessor, save_response
from datetime import datetime
import time
from typing import Dict

# Configure logging
logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=cfg.LOG_FORMAT)


def run_analyzer_loop(args):
    """
    Run the image analyzer loop using the ImageStorage singleton.

    Args:
        args: Command line arguments from argparse
    """
    try:
        # Wait for server to initialize
        logging.info("Waiting for StreamingServer to initialize...")
        time.sleep(cfg.SERVER_INIT_WAIT_TIME)

        # Initialize LM Studio processor
        logging.info("Initializing LM Studio vision processor...")
        processor = LMStudioVisionProcessor(model=args.model, base_url=args.base_url)

        # Create output directory if needed
        output_dir = Path(args.output_dir)
        if not args.no_save:
            output_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Saving responses to: {output_dir}")

        # Track processed images to avoid reprocessing
        # Store (timestamp, image_hash) to detect when images actually change
        # Hash is now a string (MD5 hexdigest) for stable cross-process comparison
        processed_images: Dict[str, tuple[float, str]] = {}

        # Get image processing thresholds from config
        min_age = args.min_age if hasattr(args, "min_age") else cfg.MIN_IMAGE_AGE
        max_age = args.max_age if hasattr(args, "max_age") else cfg.MAX_IMAGE_AGE
        check_interval = (
            args.interval if hasattr(args, "interval") else cfg.IMAGE_CHECK_INTERVAL
        )

        # Determine which cameras to monitor
        monitor_cameras = args.camera  # None = all cameras

        logging.info("Starting continuous monitoring mode...")
        logging.info(f"Check interval: {check_interval}s")
        logging.info(f"Image age window: {min_age}s - {max_age}s")
        if monitor_cameras:
            logging.info(f"Monitoring cameras: {', '.join(monitor_cameras)}")
        else:
            logging.info("Monitoring all available cameras")

        # Get the ImageStorage instance (same process, so this works!)
        storage = ImageStorage.get_instance()

        # Main processing loop
        while True:
            try:
                # Get camera IDs to check
                if monitor_cameras:
                    camera_ids = monitor_cameras
                else:
                    camera_ids = storage.get_all_camera_ids()

                # Check each camera for new images with prompts
                for cam_id in camera_ids:
                    image = storage.get_camera_image(cam_id)
                    if image is None:
                        continue

                    prompt = storage.get_camera_prompt(cam_id)
                    age = storage.get_camera_age(cam_id)

                    # Skip if age is None (shouldn't happen if image exists)
                    if age is None:
                        continue

                    # Skip if no prompt provided
                    if not prompt:
                        continue

                    # Skip if image is too fresh (might still be uploading)
                    if age < min_age:
                        continue

                    # Skip if image is too old
                    if age > max_age:
                        continue

                    # Calculate current image timestamp and hash
                    current_timestamp = time.time() - age

                    # Use MD5 hash of image array to detect if content changed
                    # Using hashlib for stable hashing across process restarts
                    # (Python's built-in hash() is randomized per process)
                    image_hash = hashlib.md5(image.tobytes()).hexdigest()

                    # Check if we've already processed this exact image recently
                    if cam_id in processed_images:
                        last_timestamp, last_hash = processed_images[cam_id]

                        # Skip if same image (same hash) AND too close in time
                        # This prevents reprocessing during the Ollama request
                        # but allows reprocessing after a delay (user sending same view again)
                        if (
                            image_hash == last_hash
                            and (current_timestamp - last_timestamp)
                            < cfg.DUPLICATE_TIME_THRESHOLD
                        ):
                            logging.debug(
                                f"⏭️  Skipping duplicate image from {cam_id} (same content, too soon: {current_timestamp - last_timestamp:.1f}s)"
                            )
                            continue  # Same image sent too quickly, skip

                    # Process the NEW image
                    print("\n" + "=" * 80)
                    print(f"🔍 PROCESSING NEW IMAGE FROM: {cam_id}")
                    print(f"📝 Prompt: '{prompt}'")
                    print(f"⏱️  Age: {age:.2f}s")
                    if cam_id in processed_images:
                        print(f"✨ Image content has changed since last screenshot")
                    else:
                        print(f"🆕 First screenshot from this camera")
                    print("=" * 80)

                    result = processor.send_images(
                        images=[image],
                        camera_ids=[cam_id],
                        prompt=prompt,
                        temperature=args.temperature,
                    )

                    # Display response prominently
                    print("\n" + "=" * 80)
                    print(f"🤖 LM STUDIO RESPONSE FOR {cam_id}")
                    print("=" * 80)
                    print(result["response"])
                    print("=" * 80)
                    print(
                        f"⏱️  Processing time: {result['metadata']['duration_seconds']:.2f}s"
                    )
                    print(f"📊 Model: {result['metadata']['model']}")
                    print("=" * 80 + "\n")

                    # Send result to Unity
                    ResultsBroadcaster.send_result(result)
                    logging.info(f"📤 Sent result to Unity for camera: {cam_id}")

                    # Save response
                    if not args.no_save:
                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = output_dir / f"{cam_id}_{timestamp_str}"
                        save_response(result, str(output_path))

                    # Mark as processed (store timestamp and hash)
                    processed_images[cam_id] = (current_timestamp, image_hash)

            except Exception as e:
                logging.error(f"Error in processing loop: {e}")
                import traceback

                traceback.print_exc()

            # Wait before next check
            time.sleep(check_interval)

    except KeyboardInterrupt:
        logging.info("\n\nAnalyzer shutting down (interrupted by user)")
    except Exception as e:
        logging.error(f"\nAnalyzer fatal error: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Main entry point - runs server and analyzer together"""
    parser = argparse.ArgumentParser(
        description="Run StreamingServer and LM Studio image analyzer together",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default model
  %(prog)s

  # Use specific model
  %(prog)s --model llama-3.2-vision

  # Monitor specific camera only
  %(prog)s --camera AR4Left

  # Adjust check interval
  %(prog)s --interval 2.0

Note: This script runs both the StreamingServer and AnalyzeImage in the same process.
      Unity should send images to port 5005 (default).
      LM Studio must be running with the server started.
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

    # LM Studio options
    lmstudio_group = parser.add_argument_group("LM Studio options")
    lmstudio_group.add_argument(
        "--model",
        "-m",
        default=cfg.DEFAULT_LMSTUDIO_MODEL,
        help=f"LM Studio vision model to use (default: {cfg.DEFAULT_LMSTUDIO_MODEL})",
    )
    lmstudio_group.add_argument(
        "--base-url", help=f"LM Studio server base URL (default: {cfg.LMSTUDIO_BASE_URL})"
    )
    lmstudio_group.add_argument(
        "--temperature",
        type=float,
        default=cfg.DEFAULT_TEMPERATURE,
        help=f"Sampling temperature 0.0-2.0 (default: {cfg.DEFAULT_TEMPERATURE})",
    )

    # Server options
    server_group = parser.add_argument_group("server options")
    server_group.add_argument(
        "--server-host",
        default=cfg.DEFAULT_HOST,
        help=f"StreamingServer host (default: {cfg.DEFAULT_HOST})",
    )
    server_group.add_argument(
        "--server-port",
        type=int,
        default=cfg.STREAMING_SERVER_PORT,
        help=f"StreamingServer port (default: {cfg.STREAMING_SERVER_PORT})",
    )
    server_group.add_argument(
        "--results-port",
        type=int,
        default=cfg.RESULTS_SERVER_PORT,
        help=f"ResultsServer port for sending results to Unity (default: {cfg.RESULTS_SERVER_PORT})",
    )
    server_group.add_argument(
        "--interval",
        "-i",
        type=float,
        default=cfg.IMAGE_CHECK_INTERVAL,
        help=f"Check interval in seconds (default: {cfg.IMAGE_CHECK_INTERVAL})",
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

    # Output options
    output_group = parser.add_argument_group("output options")
    output_group.add_argument(
        "--output-dir",
        default=cfg.DEFAULT_OUTPUT_DIR,
        help=f"Directory for saving responses (default: {cfg.DEFAULT_OUTPUT_DIR})",
    )
    output_group.add_argument(
        "--no-save", action="store_true", help="Don't save responses to files"
    )

    args = parser.parse_args()

    try:
        # Create server configs
        streaming_config = ServerConfig(host=args.server_host, port=args.server_port)
        results_config = ServerConfig(host=args.server_host, port=args.results_port)

        # Start StreamingServer in background thread
        logging.info("Starting StreamingServer in background...")
        run_streaming_server_background(streaming_config)

        # Start ResultsServer in background thread
        logging.info("Starting ResultsServer in background...")
        run_results_server_background(results_config)

        # Run analyzer in main thread
        run_analyzer_loop(args)

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
