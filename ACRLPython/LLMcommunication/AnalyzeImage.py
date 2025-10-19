#!/usr/bin/env python3
"""
AnalyzeImage.py - Continuously process Unity robot camera screenshots with Ollama LLM

This script integrates with StreamingServer to get live camera images from Unity
and sends them to Ollama for vision-based analysis. It runs as a daemon, continuously
monitoring for new images with prompts and processing them automatically.

Usage:
    # First, start the StreamingServer in another terminal:
    python StreamingServer.py

    # Then start this script to continuously process images:
    python AnalyzeImage.py
    python AnalyzeImage.py --model llava:13b
    python AnalyzeImage.py --interval 2.0  # Check every 2 seconds

Requirements:
    - ollama Python SDK (pip install ollama)
    - opencv-python and numpy
    - Ollama installed and running locally
    - StreamingServer.py running
"""

import argparse
import json
import sys
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

try:
    import ollama
except ImportError:
    print("Error: 'ollama' package not found. Install with: pip install ollama")
    sys.exit(1)

try:
    import cv2
    import numpy as np
except ImportError:
    print("Error: 'opencv-python' and 'numpy' required. Install with: pip install opencv-python numpy")
    sys.exit(1)

# Import ImageServer from StreamingServer
try:
    from StreamingServer import ImageServer
except ImportError:
    print("Error: StreamingServer.py not found in the same directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


class OllamaVisionProcessor:
    """Handles sending screenshots to Ollama for vision-based LLM processing"""

    # Popular Ollama vision models
    VISION_MODELS = [
        "llava",
        "llava:13b",
        "llava:34b",
        "llama3.2-vision",
        "llama3.2-vision:11b",
        "llama3.2-vision:90b",
        "bakllava",
    ]

    DEFAULT_MODEL = "gemma3"

    def __init__(self, model: str = DEFAULT_MODEL, host: Optional[str] = None):
        """
        Initialize the Ollama vision processor

        Args:
            model: Ollama vision model to use
            host: Ollama server host (default: uses Ollama's default)
        """
        self._model = model
        self._client = ollama.Client(host=host) if host else ollama.Client()

        # Test connection
        try:
            self._client.list()
            logging.info(f"Connected to Ollama, using model: {self._model}")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Ollama: {e}. Make sure Ollama is running.")

    def encode_image_to_bytes(self, image: np.ndarray) -> bytes:
        """
        Encode a numpy image array to PNG bytes

        Args:
            image: OpenCV/numpy image array (BGR format)

        Returns:
            PNG encoded bytes
        """
        success, buffer = cv2.imencode('.png', image)
        if not success:
            raise ValueError("Failed to encode image as PNG")
        return buffer.tobytes()

    def send_images(
        self,
        images: List[np.ndarray],
        camera_ids: List[str],
        prompt: str,
        temperature: float = 0.7
    ) -> Dict:
        """
        Send images to Ollama for vision-based analysis

        Args:
            images: List of numpy image arrays
            camera_ids: List of camera IDs corresponding to images
            prompt: The prompt/question to ask the LLM about the images
            temperature: Sampling temperature (0.0-2.0)

        Returns:
            Dictionary containing response and metadata
        """
        if not images:
            raise ValueError("No images provided")

        logging.info(f"Processing {len(images)} image(s) with Ollama...")

        # Prepare images as bytes for Ollama
        image_bytes_list = []
        for i, (image, cam_id) in enumerate(zip(images, camera_ids)):
            try:
                img_bytes = self.encode_image_to_bytes(image)
                image_bytes_list.append(img_bytes)
                logging.info(f"  [{i+1}/{len(images)}] Encoded: {cam_id} ({image.shape[1]}x{image.shape[0]})")
            except Exception as e:
                logging.error(f"  Error encoding image from {cam_id}: {e}")
                raise

        # Build prompt with camera context if multiple cameras
        if len(camera_ids) > 1:
            camera_context = "Images from cameras: " + ", ".join(camera_ids) + ". "
            full_prompt = camera_context + prompt
        else:
            full_prompt = prompt

        # Make request to Ollama
        logging.info(f"Sending request to Ollama ({self._model})...")
        start_time = datetime.now()

        try:
            response = self._client.chat(
                model=self._model,
                messages=[{
                    'role': 'user',
                    'content': full_prompt,
                    'images': image_bytes_list
                }],
                options={
                    'temperature': temperature
                }
            )
        except Exception as e:
            logging.error(f"Ollama error: {e}")
            raise

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Extract response text
        response_text = response['message']['content']

        # Build result
        result = {
            "success": True,
            "response": response_text,
            "metadata": {
                "model": self._model,
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": duration,
                "image_count": len(images),
                "camera_ids": camera_ids,
                "prompt": prompt,
                "full_prompt": full_prompt
            }
        }

        logging.info(f"✓ Response received in {duration:.2f}s")

        return result


def get_images_from_server(camera_ids: Optional[List[str]] = None) -> tuple[List[np.ndarray], List[str], List[str]]:
    """
    Get images from the StreamingServer

    Args:
        camera_ids: List of specific camera IDs to fetch, or None for all cameras

    Returns:
        Tuple of (images list, camera_ids list, prompts list)
    """
    server = ImageServer.get_instance()

    # Get all available cameras if not specified
    if camera_ids is None:
        camera_ids = server.get_all_camera_ids()

    if not camera_ids:
        raise ValueError("No cameras available. Is the StreamingServer running and receiving images?")

    images = []
    valid_camera_ids = []
    prompts = []

    for cam_id in camera_ids:
        image = server.get_camera_image(cam_id)
        if image is not None:
            age = server.get_camera_age(cam_id)
            prompt = server.get_camera_prompt(cam_id) or ""
            prompt_info = f", prompt: '{prompt}'" if prompt else ""
            print(f"  ✓ {cam_id}: {image.shape[1]}x{image.shape[0]}, {age:.1f}s ago{prompt_info}")
            images.append(image)
            valid_camera_ids.append(cam_id)
            prompts.append(prompt)
        else:
            print(f"  ✗ {cam_id}: No image available")

    if not images:
        raise ValueError(f"No images available from cameras: {camera_ids}")

    return images, valid_camera_ids, prompts


def save_response(result: Dict, output_path: Optional[str] = None):
    """
    Save LLM's response to files

    Args:
        result: Result dictionary from send_images
        output_path: Optional custom output path (without extension)
    """
    if output_path:
        from pathlib import Path
        base_path = Path(output_path)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        from pathlib import Path
        base_path = Path(f"llm_response_{timestamp}")

    # Save full JSON result
    json_path = base_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved full response to: {json_path}")


def main():
    """Main entry point - runs as a daemon monitoring for images with prompts"""
    parser = argparse.ArgumentParser(
        description="Continuously process Unity robot camera images with Ollama vision LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start processing with default model (llava)
  %(prog)s

  # Use specific model
  %(prog)s --model llava:13b

  # Monitor specific camera only
  %(prog)s --camera AR4Left

  # Adjust check interval
  %(prog)s --interval 2.0

  # List available cameras
  %(prog)s --list-cameras

Note: StreamingServer.py must be running for this script to work.
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
    input_group.add_argument(
        "--list-cameras", "-l",
        action="store_true",
        help="List available cameras and exit"
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
        # List cameras mode
        if args.list_cameras:
            server = ImageServer.get_instance()
            camera_ids = server.get_all_camera_ids()
            if camera_ids:
                print("Available cameras:")
                for cam_id in camera_ids:
                    age = server.get_camera_age(cam_id)
                    prompt = server.get_camera_prompt(cam_id) or "(no prompt)"
                    print(f"  • {cam_id} - age: {age:.1f}s, prompt: {prompt}")
            else:
                print("No cameras available. Is StreamingServer running?")
            sys.exit(0)

        # Initialize Ollama processor
        logging.info("Initializing Ollama vision processor...")
        processor = OllamaVisionProcessor(model=args.model, host=args.host)

        # Create output directory if needed
        output_dir = Path(args.output_dir)
        if not args.no_save:
            output_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Saving responses to: {output_dir}")

        # Track processed images to avoid reprocessing
        processed_timestamps: Dict[str, float] = {}

        # Determine which cameras to monitor
        monitor_cameras = args.camera  # None = all cameras

        logging.info("Starting continuous monitoring mode...")
        logging.info(f"Check interval: {args.interval}s")
        logging.info(f"Image age window: {args.min_age}s - {args.max_age}s")
        if monitor_cameras:
            logging.info(f"Monitoring cameras: {', '.join(monitor_cameras)}")
        else:
            logging.info("Monitoring all available cameras")

        # Main processing loop
        while True:
            try:
                server = ImageServer.get_instance()

                # Get camera IDs to check
                if monitor_cameras:
                    camera_ids = monitor_cameras
                else:
                    camera_ids = server.get_all_camera_ids()

                # Check each camera for new images with prompts
                for cam_id in camera_ids:
                    image = server.get_camera_image(cam_id)
                    if image is None:
                        continue

                    prompt = server.get_camera_prompt(cam_id)
                    age = server.get_camera_age(cam_id)

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

                    # Check if we've already processed this exact image
                    last_processed = processed_timestamps.get(cam_id, 0)
                    current_timestamp = time.time() - age

                    if current_timestamp <= last_processed:
                        continue  # Already processed
                    else:
                        # Process the image
                        logging.info(f"\n{'='*80}")
                        logging.info(f"Processing image from: {cam_id}")
                        logging.info(f"Age: {age:.2f}s, Prompt: '{prompt}'")

                        result = processor.send_images(
                            images=[image],
                            camera_ids=[cam_id],
                            prompt=prompt,
                            temperature=args.temperature
                        )

                    # Display response
                    print("\n" + "="*80)
                    print(f"RESPONSE FOR {cam_id}:")
                    print("="*80)
                    print(result["response"])
                    print("="*80 + "\n")

                    # Save response
                    if not args.no_save:
                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = output_dir / f"{cam_id}_{timestamp_str}"
                        save_response(result, str(output_path))

                    # Mark as processed
                    processed_timestamps[cam_id] = current_timestamp

            except Exception as e:
                logging.error(f"Error in processing loop: {e}")
                import traceback
                traceback.print_exc()

            # Wait before next check
            time.sleep(args.interval)

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
