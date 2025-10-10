#!/usr/bin/env python3
"""
SendScreenshots.py - Send Unity robot camera screenshots to Claude API for analysis

This script integrates with StreamingServer to get live camera images from Unity
and sends them to the Claude API for vision-based analysis.

Usage:
    # First, start the StreamingServer in another terminal:
    python StreamingServer.py

    # Then send images to Claude:
    python SendScreenshots.py --camera AR4Left --prompt "Describe the scene"
    python SendScreenshots.py --camera AR4Left AR4Right --prompt "Compare both views"
    python SendScreenshots.py --all-cameras --prompt "Analyze all robot perspectives"

Requirements:
    - anthropic Python SDK
    - python-dotenv (for API key management)
    - Valid ANTHROPIC_API_KEY in environment or .env file
    - StreamingServer.py running
"""

import argparse
import base64
import json
import sys
from datetime import datetime
from typing import List, Dict, Optional

try:
    import anthropic
except ImportError:
    print("Error: 'anthropic' package not found. Install with: pip install anthropic")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: 'python-dotenv' not found. Using environment variables only.")

try:
    import cv2
    import numpy as np
except ImportError:
    print("Error: 'opencv-python' and 'numpy' required. Install with: pip install opencv-python numpy")
    sys.exit(1)

import os

# Import ImageServer from StreamingServer
try:
    from StreamingServer import ImageServer
except ImportError:
    print("Error: StreamingServer.py not found in the same directory")
    sys.exit(1)


class ScreenshotSender:
    """Handles sending screenshots to Claude API and processing responses"""

    # Claude API models and their costs per million tokens
    MODELS = {
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    }

    DEFAULT_MODEL = "claude-3-5-haiku-20241022"

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        """
        Initialize the screenshot sender

        Args:
            api_key: Anthropic API key (if None, reads from ANTHROPIC_API_KEY env var)
            model: Claude model to use
        """
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set it as an environment variable "
                "or create a .env file with ANTHROPIC_API_KEY=your_key_here"
            )

        self._model = model
        self._client = anthropic.Anthropic(api_key=self._api_key)

    def encode_image_array(self, image: np.ndarray) -> str:
        """
        Encode a numpy image array to base64 PNG

        Args:
            image: OpenCV/numpy image array (BGR format)

        Returns:
            Base64 encoded PNG string
        """
        # Encode image as PNG
        success, buffer = cv2.imencode('.png', image)
        if not success:
            raise ValueError("Failed to encode image as PNG")

        # Convert to base64
        image_data = base64.standard_b64encode(buffer.tobytes()).decode("utf-8")
        return image_data

    def send_images(
        self,
        images: List[np.ndarray],
        camera_ids: List[str],
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 1.0
    ) -> Dict:
        """
        Send images to Claude API for analysis

        Args:
            images: List of numpy image arrays
            camera_ids: List of camera IDs corresponding to images
            prompt: The prompt/question to ask Claude about the images
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Dictionary containing response and metadata
        """
        if not images:
            raise ValueError("No images provided")

        print(f"Encoding {len(images)} image(s)...")

        # Build message content with images
        content = []

        # Add all images with labels
        for i, (image, cam_id) in enumerate(zip(images, camera_ids)):
            try:
                image_data = self.encode_image_array(image)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_data,
                    }
                })
                print(f"  [{i+1}/{len(images)}] Encoded: {cam_id} ({image.shape[1]}x{image.shape[0]})")
            except Exception as e:
                print(f"  Error encoding image from {cam_id}: {e}")
                raise

        # Add the text prompt
        content.append({
            "type": "text",
            "text": prompt
        })

        # Make API request
        print(f"\nSending request to Claude ({self._model})...")
        start_time = datetime.now()

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{
                    "role": "user",
                    "content": content
                }]
            )
        except anthropic.APIError as e:
            print(f"\nAPI Error: {e}")
            raise

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Extract response text
        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text += block.text

        # Calculate cost estimate
        cost = self._estimate_cost(response.usage.input_tokens, response.usage.output_tokens)

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
                "tokens": {
                    "input": response.usage.input_tokens,
                    "output": response.usage.output_tokens,
                    "total": response.usage.input_tokens + response.usage.output_tokens
                },
                "estimated_cost_usd": cost,
                "stop_reason": response.stop_reason
            }
        }

        print(f"\n✓ Response received in {duration:.2f}s")
        print(f"  Tokens: {result['metadata']['tokens']['input']} input, "
              f"{result['metadata']['tokens']['output']} output")
        print(f"  Estimated cost: ${cost:.4f}")

        return result

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate the cost of an API call

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        if self._model not in self.MODELS:
            return 0.0

        rates = self.MODELS[self._model]
        input_cost = (input_tokens / 1_000_000) * rates["input"]
        output_cost = (output_tokens / 1_000_000) * rates["output"]

        return input_cost + output_cost


def get_images_from_server(camera_ids: Optional[List[str]] = None) -> tuple[List[np.ndarray], List[str]]:
    """
    Get images from the StreamingServer

    Args:
        camera_ids: List of specific camera IDs to fetch, or None for all cameras

    Returns:
        Tuple of (images list, camera_ids list)
    """
    server = ImageServer.get_instance()

    # Get all available cameras if not specified
    if camera_ids is None:
        camera_ids = server.get_all_camera_ids()

    if not camera_ids:
        raise ValueError("No cameras available. Is the StreamingServer running and receiving images?")

    images = []
    valid_camera_ids = []

    for cam_id in camera_ids:
        image = server.get_camera_image(cam_id)
        if image is not None:
            age = server.get_camera_age(cam_id)
            print(f"  ✓ {cam_id}: {image.shape[1]}x{image.shape[0]}, {age:.1f}s ago")
            images.append(image)
            valid_camera_ids.append(cam_id)
        else:
            print(f"  ✗ {cam_id}: No image available")

    if not images:
        raise ValueError(f"No images available from cameras: {camera_ids}")

    return images, valid_camera_ids


def save_response(result: Dict, output_path: Optional[str] = None):
    """
    Save Claude's response to files

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
        base_path = Path(f"claude_response_{timestamp}")

    # Save full JSON result
    json_path = base_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved full response to: {json_path}")

    # Save text response only
    txt_path = base_path.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["response"])
    print(f"✓ Saved response text to: {txt_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Send Unity robot camera images to Claude API for analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze latest image from AR4Left camera
  %(prog)s --camera AR4Left --prompt "Describe what the robot sees"

  # Analyze both cameras
  %(prog)s --camera AR4Left AR4Right --prompt "Compare both views"

  # Analyze all available cameras
  %(prog)s --all-cameras --prompt "Describe the overall scene"

  # List available cameras
  %(prog)s --list-cameras

Note: StreamingServer.py must be running for this script to work.
        """
    )

    # Input options
    input_group = parser.add_argument_group("input options")
    input_group.add_argument(
        "--camera", "-c",
        nargs="+",
        help="Specific camera ID(s) to use (e.g., AR4Left AR4Right)"
    )
    input_group.add_argument(
        "--all-cameras", "-a",
        action="store_true",
        help="Use all available cameras"
    )
    input_group.add_argument(
        "--list-cameras", "-l",
        action="store_true",
        help="List available cameras and exit"
    )

    # API options
    api_group = parser.add_argument_group("API options")
    api_group.add_argument(
        "--prompt", "-p",
        help="The prompt/question to ask Claude about the image(s)"
    )
    api_group.add_argument(
        "--model", "-m",
        default=ScreenshotSender.DEFAULT_MODEL,
        choices=list(ScreenshotSender.MODELS.keys()),
        help=f"Claude model to use (default: {ScreenshotSender.DEFAULT_MODEL})"
    )
    api_group.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens in response (default: 4096)"
    )
    api_group.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature 0.0-1.0 (default: 1.0)"
    )
    api_group.add_argument(
        "--api-key",
        help="Anthropic API key (default: reads from ANTHROPIC_API_KEY env var)"
    )

    # Output options
    output_group = parser.add_argument_group("output options")
    output_group.add_argument(
        "--output", "-o",
        help="Output path for saving response (without extension)"
    )
    output_group.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save response to files"
    )
    output_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimize output (only show Claude's response)"
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
                    print(f"  • {cam_id} (last updated {age:.1f}s ago)")
            else:
                print("No cameras available. Is StreamingServer running?")
            sys.exit(0)

        # Require prompt for analysis
        if not args.prompt:
            parser.error("--prompt is required (unless using --list-cameras)")

        # Determine which cameras to use
        camera_ids = None
        if args.camera:
            camera_ids = args.camera
        elif not args.all_cameras:
            # Default to all cameras if nothing specified
            args.all_cameras = True

        # Get images from streaming server
        if not args.quiet:
            print("Fetching images from StreamingServer...")

        images, valid_camera_ids = get_images_from_server(camera_ids)

        if not args.quiet:
            print(f"\nFound {len(images)} image(s) from {len(valid_camera_ids)} camera(s)")

        # Send to Claude API
        sender = ScreenshotSender(api_key=args.api_key, model=args.model)
        result = sender.send_images(
            images=images,
            camera_ids=valid_camera_ids,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )

        # Display response
        print("\n" + "="*80)
        print("CLAUDE'S RESPONSE:")
        print("="*80)
        print(result["response"])
        print("="*80 + "\n")

        # Save response
        if not args.no_save:
            save_response(result, args.output)

        # Exit with success
        sys.exit(0)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
