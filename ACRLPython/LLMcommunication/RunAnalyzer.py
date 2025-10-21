#!/usr/bin/env python3
"""
RunAnalyzer.py - Runs both StreamingServer and AnalyzeImage in the same process

This wrapper script starts the StreamingServer in a background thread and runs
the image analyzer in the main thread, allowing them to share the ImageServer
singleton and communicate in-process.

Usage:
    python RunAnalyzer.py
    python RunAnalyzer.py --model llava:13b
    python RunAnalyzer.py --help
"""

import threading
import logging
import argparse
import sys

# Import server components
from StreamingServer import ImageStorage, run_streaming_server_background
from core.TCPServerBase import ServerConfig
from ResultsServer import ResultsBroadcaster, run_results_server_background

# Import analyzer (but we'll use its functions, not run main directly)
from AnalyzeImage import OllamaVisionProcessor, save_response
from pathlib import Path
from datetime import datetime
import time
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)




def run_analyzer_loop(args):
    """
    Run the image analyzer loop using the ImageStorage singleton.

    Args:
        args: Command line arguments from argparse
    """
    try:
        # Wait for server to initialize
        logging.info("Waiting for StreamingServer to initialize...")
        time.sleep(2)

        # Initialize Ollama processor
        logging.info("Initializing Ollama vision processor...")
        processor = OllamaVisionProcessor(model=args.model, host=args.host)

        # Create output directory if needed
        output_dir = Path(args.output_dir)
        if not args.no_save:
            output_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Saving responses to: {output_dir}")

        # Track processed images to avoid reprocessing
        # Store (timestamp, image_hash) to detect when images actually change
        processed_images: Dict[str, tuple[float, int]] = {}

        # Determine which cameras to monitor
        monitor_cameras = args.camera  # None = all cameras

        logging.info("Starting continuous monitoring mode...")
        logging.info(f"Check interval: {args.interval}s")
        logging.info(f"Image age window: {args.min_age}s - {args.max_age}s")
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
                    if age < args.min_age:
                        continue

                    # Skip if image is too old
                    if age > args.max_age:
                        continue

                    # Calculate current image timestamp and hash
                    current_timestamp = time.time() - age

                    # Use hash of image array to detect if content changed
                    # This is much more efficient than comparing entire arrays
                    image_hash = hash(image.tobytes())

                    # Check if we've already processed this exact image
                    if cam_id in processed_images:
                        last_timestamp, last_hash = processed_images[cam_id]

                        # Skip if same image (same hash) regardless of timestamp
                        if image_hash == last_hash:
                            logging.debug(f"⏭️  Skipping duplicate image from {cam_id} (same content)")
                            continue  # Same image, already processed

                        # Also skip if timestamp hasn't changed much (< 0.1s)
                        # This catches duplicate sends before hash can be computed
                        if abs(current_timestamp - last_timestamp) < 0.1:
                            logging.debug(f"⏭️  Skipping duplicate image from {cam_id} (too close in time)")
                            continue  # Too close in time, likely duplicate

                    # Process the NEW image
                    print("\n" + "="*80)
                    print(f"🔍 PROCESSING NEW IMAGE FROM: {cam_id}")
                    print(f"📝 Prompt: '{prompt}'")
                    print(f"⏱️  Age: {age:.2f}s")
                    if cam_id in processed_images:
                        print(f"✨ Image content has changed since last screenshot")
                    else:
                        print(f"🆕 First screenshot from this camera")
                    print("="*80)

                    result = processor.send_images(
                        images=[image],
                        camera_ids=[cam_id],
                        prompt=prompt,
                        temperature=args.temperature
                    )

                    # Display response prominently
                    print("\n" + "="*80)
                    print(f"🤖 OLLAMA RESPONSE FOR {cam_id}")
                    print("="*80)
                    print(result["response"])
                    print("="*80)
                    print(f"⏱️  Processing time: {result['metadata']['duration_seconds']:.2f}s")
                    print(f"📊 Model: {result['metadata']['model']}")
                    print("="*80 + "\n")

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
            time.sleep(args.interval)

    except KeyboardInterrupt:
        logging.info("\n\nAnalyzer shutting down (interrupted by user)")
    except Exception as e:
        logging.error(f"\nAnalyzer fatal error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point - runs server and analyzer together"""
    parser = argparse.ArgumentParser(
        description="Run StreamingServer and Ollama image analyzer together",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default model (llava)
  %(prog)s

  # Use specific model
  %(prog)s --model llava:13b

  # Monitor specific camera only
  %(prog)s --camera AR4Left

  # Adjust check interval
  %(prog)s --interval 2.0

Note: This script runs both the StreamingServer and AnalyzeImage in the same process.
      Unity should send images to port 5005 (default).
      Ollama must be installed and running locally.
        """
    )

    # Input options
    input_group = parser.add_argument_group("input options")
    input_group.add_argument(
        "--camera", "-c",
        nargs="+",
        help="Specific camera ID(s) to monitor (default: all cameras)"
    )

    # Ollama options
    ollama_group = parser.add_argument_group("Ollama options")
    ollama_group.add_argument(
        "--model", "-m",
        default=OllamaVisionProcessor.DEFAULT_MODEL,
        help=f"Ollama vision model to use (default: {OllamaVisionProcessor.DEFAULT_MODEL})"
    )
    ollama_group.add_argument(
        "--host",
        help="Ollama server host (default: uses Ollama's default localhost)"
    )
    ollama_group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature 0.0-2.0 (default: 0.7)"
    )

    # Server options
    server_group = parser.add_argument_group("server options")
    server_group.add_argument(
        "--server-host",
        default="127.0.0.1",
        help="StreamingServer host (default: 127.0.0.1)"
    )
    server_group.add_argument(
        "--server-port",
        type=int,
        default=5005,
        help="StreamingServer port (default: 5005)"
    )
    server_group.add_argument(
        "--results-port",
        type=int,
        default=5006,
        help="ResultsServer port for sending results to Unity (default: 5006)"
    )
    server_group.add_argument(
        "--interval", "-i",
        type=float,
        default=1.0,
        help="Check interval in seconds (default: 1.0)"
    )
    server_group.add_argument(
        "--min-age",
        type=float,
        default=0.5,
        help="Minimum image age in seconds before processing (default: 0.5)"
    )
    server_group.add_argument(
        "--max-age",
        type=float,
        default=30.0,
        help="Maximum image age in seconds to consider fresh (default: 30.0)"
    )

    # Output options
    output_group = parser.add_argument_group("output options")
    output_group.add_argument(
        "--output-dir",
        default="./llm_responses",
        help="Directory for saving responses (default: ./llm_responses)"
    )
    output_group.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save responses to files"
    )

    args = parser.parse_args()

    try:
        # Create server configs
        streaming_config = ServerConfig(
            host=args.server_host,
            port=args.server_port
        )
        results_config = ServerConfig(
            host=args.server_host,
            port=args.results_port
        )

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
