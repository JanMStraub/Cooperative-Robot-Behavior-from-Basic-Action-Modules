#!/usr/bin/env python3
"""
AnalyzeImage.py - Continuously process Unity robot camera screenshots with LM Studio LLM

This script integrates with StreamingServer to get live camera images from Unity
and sends them to LM Studio for vision-based analysis. It runs as a daemon, continuously
monitoring for new images with prompts and processing them automatically.

Usage:
    # First, start the StreamingServer in another terminal:
    python StreamingServer.py

    # Then start this script to continuously process images:
    python AnalyzeImage.py
    python AnalyzeImage.py --model llama-3.2-vision
    python AnalyzeImage.py --interval 2.0  # Check every 2 seconds

Requirements:
    - openai Python SDK (pip install openai)
    - opencv-python and numpy
    - LM Studio running locally with server started
    - StreamingServer.py running
"""

import argparse
import json
import sys
import time
import logging
import base64
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

try:
    import openai
    from openai import OpenAI
except ImportError:
    logging.info("Error: 'openai' package not found. Install with: pip install openai")
    sys.exit(1)

try:
    import cv2
    import numpy as np
except ImportError:
    logging.info(
        "Error: 'opencv-python' and 'numpy' required. Install with: pip install opencv-python numpy"
    )
    sys.exit(1)

# Add LLMCommunication package directory to path
_package_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_package_dir))

# Import config - support both direct script and module execution
try:
    from .. import LLMConfig as cfg
except ImportError:
    import LLMCommunication.LLMConfig as cfg

# Import ImageStorage - support both direct script and module execution
try:
    from ..servers.StreamingServer import ImageStorage
except ImportError:
    try:
        from servers.StreamingServer import ImageStorage
    except ImportError:
        logging.info(
            "Error: StreamingServer not found. Make sure servers package is available."
        )
        sys.exit(1)

# Configure logging
logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=cfg.LOG_FORMAT)


class LMStudioVisionProcessor:
    """Handles sending screenshots to LM Studio for vision-based LLM processing"""

    # Popular LM Studio vision models (from config)
    VISION_MODELS = cfg.VISION_MODELS

    DEFAULT_MODEL = cfg.DEFAULT_LMSTUDIO_MODEL

    def __init__(self, model: str = DEFAULT_MODEL, base_url: Optional[str] = None):
        """
        Initialize the LM Studio vision processor

        Args:
            model: LM Studio vision model to use
            base_url: LM Studio server base URL (default: http://127.0.0.1:1234/v1)
        """
        self._model = model
        self._base_url = base_url if base_url else cfg.LMSTUDIO_BASE_URL
        self._client = OpenAI(base_url=self._base_url, api_key="not-needed")

        # Test connection
        try:
            self._client.models.list()
            logging.info(
                f"Connected to LM Studio at {self._base_url}, using model: {self._model}"
            )
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to LM Studio: {e}. Make sure LM Studio server is running."
            )

    def encode_image_to_bytes(self, image: np.ndarray) -> bytes:
        """
        Encode a numpy image array to PNG bytes

        Args:
            image: OpenCV/numpy image array (BGR format)

        Returns:
            PNG encoded bytes
        """
        success, buffer = cv2.imencode(".png", image)
        if not success:
            raise ValueError("Failed to encode image as PNG")
        return buffer.tobytes()

    def send_images(
        self,
        images: List[np.ndarray],
        camera_ids: List[str],
        prompt: str,
        temperature: Optional[float] = None,
    ) -> Dict:
        """
        Send images to LM Studio for vision-based analysis

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

        # Use config default if not specified
        if temperature is None:
            temperature = cfg.DEFAULT_TEMPERATURE

        logging.info(f"Processing {len(images)} image(s) with LM Studio...")

        # Prepare images as base64 encoded data URLs for OpenAI API
        image_content = []
        for i, (image, cam_id) in enumerate(zip(images, camera_ids)):
            try:
                img_bytes = self.encode_image_to_bytes(image)
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                image_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    }
                )
                logging.info(
                    f"  [{i+1}/{len(images)}] Encoded: {cam_id} ({image.shape[1]}x{image.shape[0]})"
                )
            except Exception as e:
                logging.error(f"  Error encoding image from {cam_id}: {e}")
                raise

        # Build prompt with camera context if multiple cameras
        if len(camera_ids) > 1:
            camera_context = "Images from cameras: " + ", ".join(camera_ids) + ". "
            full_prompt = camera_context + prompt
        else:
            full_prompt = prompt

        # Build message content with text and images
        # Combine text prompt with image content for vision API
        message_content = [{"type": "text", "text": full_prompt}] + image_content

        # Make request to LM Studio
        logging.info(f"Sending request to LM Studio ({self._model})...")
        start_time = datetime.now()

        try:
            # OpenAI vision API format with properly typed message
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": message_content,  # type: ignore - LM Studio accepts this format
                    }
                ],
                temperature=temperature,
                max_tokens=1000,
            )
        except Exception as e:
            logging.error(f"LM Studio error: {e}")
            raise

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Extract response text
        response_text = response.choices[0].message.content

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
                "full_prompt": full_prompt,
            },
        }

        logging.info(f"✓ Response received in {duration:.2f}s")

        return result


def get_images_from_server(
    camera_ids: Optional[List[str]] = None,
) -> tuple[List[np.ndarray], List[str], List[str]]:
    """
    Get images from the StreamingServer

    Args:
        camera_ids: List of specific camera IDs to fetch, or None for all cameras

    Returns:
        Tuple of (images list, camera_ids list, prompts list)
    """
    storage = ImageStorage.get_instance()

    # Get all available cameras if not specified
    if camera_ids is None:
        camera_ids = storage.get_all_camera_ids()

    if not camera_ids:
        raise ValueError(
            "No cameras available. Is the StreamingServer running and receiving images?"
        )

    images = []
    valid_camera_ids = []
    prompts = []

    for cam_id in camera_ids:
        image = storage.get_camera_image(cam_id)
        if image is not None:
            age = storage.get_camera_age(cam_id)
            prompt = storage.get_camera_prompt(cam_id) or ""
            prompt_info = f", prompt: '{prompt}'" if prompt else ""
            logging.info(
                f"  ✓ {cam_id}: {image.shape[1]}x{image.shape[0]}, {age:.1f}s ago{prompt_info}"
            )
            images.append(image)
            valid_camera_ids.append(cam_id)
            prompts.append(prompt)
        else:
            logging.info(f"  ✗ {cam_id}: No image available")

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
    logging.info(f"\n✓ Saved full response to: {json_path}")


def main():
    """Main entry point - runs as a daemon monitoring for images with prompts"""
    parser = argparse.ArgumentParser(
        description="Continuously process Unity robot camera images with LM Studio vision LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start processing with default model
  %(prog)s

  # Use specific model
  %(prog)s --model llama-3.2-vision

  # Monitor specific camera only
  %(prog)s --camera AR4Left

  # Adjust check interval
  %(prog)s --interval 2.0

  # List available cameras
  %(prog)s --list-cameras

Note: StreamingServer.py must be running for this script to work.
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
    input_group.add_argument(
        "--list-cameras",
        "-l",
        action="store_true",
        help="List available cameras and exit",
    )

    # LM Studio options
    lmstudio_group = parser.add_argument_group("LM Studio options")
    lmstudio_group.add_argument(
        "--model",
        "-m",
        default=LMStudioVisionProcessor.DEFAULT_MODEL,
        help=f"LM Studio vision model to use (default: {LMStudioVisionProcessor.DEFAULT_MODEL})",
    )
    lmstudio_group.add_argument(
        "--base-url",
        help=f"LM Studio server base URL (default: {cfg.LMSTUDIO_BASE_URL})",
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
        # List cameras mode
        if args.list_cameras:
            storage = ImageStorage.get_instance()
            camera_ids = storage.get_all_camera_ids()
            if camera_ids:
                logging.info("Available cameras:")
                for cam_id in camera_ids:
                    age = storage.get_camera_age(cam_id)
                    prompt = storage.get_camera_prompt(cam_id) or "(no prompt)"
                    logging.info(f"  • {cam_id} - age: {age:.1f}s, prompt: {prompt}")
            else:
                logging.info("No cameras available. Is StreamingServer running?")
            sys.exit(0)

        # Initialize LM Studio processor
        logging.info("Initializing LM Studio vision processor...")
        processor = LMStudioVisionProcessor(model=args.model, base_url=args.base_url)

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
                storage = ImageStorage.get_instance()

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
                            temperature=args.temperature,
                        )

                    # Display response
                    logging.info("\n" + "=" * 80)
                    logging.info(f"RESPONSE FOR {cam_id}:")
                    logging.info("=" * 80)
                    logging.info(result["response"])
                    logging.info("=" * 80 + "\n")

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
