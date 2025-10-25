#!/usr/bin/env python3
"""
StreamingServer.py - Receives camera images from Unity
"""

import socket
import cv2
import numpy as np
import threading
import time
import logging
from typing import Optional, Tuple, List, Dict
import signal
import sys
from pathlib import Path

# Add LLMCommunication package directory to path
_package_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_package_dir))

# Import config
import config as cfg

# Import base classes
from core.TCPServerBase import TCPServerBase, ServerConfig
from core.UnityProtocol import UnityProtocol

logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=cfg.LOG_FORMAT)


class ImageStorage:
    """
    Singleton for storing and accessing camera images.

    Renamed from 'ImageServer' for clarity - this stores images, not a server.
    Thread-safe storage of images received from Unity cameras.
    """

    _instance = None
    _cameras: Dict[str, Tuple[np.ndarray, float, str]] = {}
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "ImageStorage":
        """Get the singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def store_image(self, camera_id: str, image: np.ndarray, prompt: str = ""):
        """Store an image for a camera"""
        timestamp = time.time()
        with self._lock:
            self._cameras[camera_id] = (image, timestamp, prompt)

    def get_camera_image(self, camera_id: str) -> Optional[np.ndarray]:
        """Get the latest image from a camera"""
        with self._lock:
            if camera_id in self._cameras:
                image, _, _ = self._cameras[camera_id]
                return image.copy()
            return None

    def get_camera_prompt(self, camera_id: str) -> Optional[str]:
        """Get the prompt associated with latest image"""
        with self._lock:
            if camera_id in self._cameras:
                _, _, prompt = self._cameras[camera_id]
                return prompt
            return None

    def get_camera_age(self, camera_id: str) -> Optional[float]:
        """Get the age (in seconds) of the latest image"""
        with self._lock:
            if camera_id in self._cameras:
                _, timestamp, _ = self._cameras[camera_id]
                return time.time() - timestamp
            return None

    def get_all_camera_ids(self) -> List[str]:
        """Get list of all active camera IDs"""
        with self._lock:
            return list(self._cameras.keys())


class StreamingServer(TCPServerBase):
    """
    TCP server that receives camera images from Unity.

    Inherits connection management from TCPServerBase (~150 lines saved).
    Handles image-specific protocol decoding and storage.
    """

    def __init__(self, server_config: ServerConfig):
        if server_config is None:
            server_config = cfg.get_streaming_config()

        super().__init__(server_config)
        self._storage = ImageStorage.get_instance()
        logging.info("StreamingServer initialized")

    def handle_client_connection(self, client: socket.socket, address: tuple):
        """
        Handle a Unity client connection - receives images continuously.

        This method is called by TCPServerBase in a separate thread per client.
        """
        logging.info(f"Unity camera client connected from {address}")

        try:
            while self.is_running():
                # Receive one complete image message
                camera_id, prompt, image_bytes = self._receive_image_message(client)

                if image_bytes is None:
                    # Connection closed or error
                    break

                # Decode PNG/JPG image
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if image is None:
                    logging.warning(f"Failed to decode image from {camera_id}")
                    continue

                # Store in singleton
                camera_id = camera_id or "unknown_camera"
                prompt = prompt or ""
                self._storage.store_image(camera_id, image, prompt)

                # Log receipt
                prompt_info = f" with prompt: '{prompt}'" if prompt else ""
                logging.info(
                    f"Received image from {camera_id}: "
                    f"{image.shape[1]}x{image.shape[0]}{prompt_info}"
                )

        except Exception as e:
            logging.error(f"Error handling client {address}: {e}")

    def _receive_image_message(
        self, client: socket.socket
    ) -> Tuple[Optional[str], Optional[str], Optional[bytes]]:
        """
        Receive one complete image message from Unity.

        Uses UnityProtocol format:
        [camera_id_len][camera_id][prompt_len][prompt][image_len][image_data]

        Returns:
            (camera_id, prompt, image_bytes) or (None, None, None) on failure
        """
        import struct

        try:
            # Set socket timeout to prevent indefinite hangs on partial data
            client.settimeout(cfg.SOCKET_RECEIVE_TIMEOUT)

            # Read camera ID length and data
            id_length_bytes = self._recv_exactly(client, UnityProtocol.INT_SIZE)
            if not id_length_bytes:
                return None, None, None
            id_length = struct.unpack(UnityProtocol.INT_FORMAT, id_length_bytes)[0]

            if id_length > UnityProtocol.MAX_STRING_LENGTH:
                logging.error(f"Camera ID length {id_length} exceeds maximum")
                return None, None, None

            camera_id_bytes = self._recv_exactly(client, id_length)
            if not camera_id_bytes:
                return None, None, None
            camera_id = camera_id_bytes.decode("utf-8")

            # Read prompt length and data
            prompt_length_bytes = self._recv_exactly(client, UnityProtocol.INT_SIZE)
            if not prompt_length_bytes:
                return None, None, None
            prompt_length = struct.unpack(
                UnityProtocol.INT_FORMAT, prompt_length_bytes
            )[0]

            if prompt_length > UnityProtocol.MAX_STRING_LENGTH:
                logging.error(f"Prompt length {prompt_length} exceeds maximum")
                return None, None, None

            prompt = ""
            if prompt_length > 0:
                prompt_bytes = self._recv_exactly(client, prompt_length)
                if not prompt_bytes:
                    return None, None, None
                prompt = prompt_bytes.decode("utf-8")

            # Read image length and data
            image_length_bytes = self._recv_exactly(client, UnityProtocol.INT_SIZE)
            if not image_length_bytes:
                return None, None, None
            image_length = struct.unpack(UnityProtocol.INT_FORMAT, image_length_bytes)[
                0
            ]

            if image_length > UnityProtocol.MAX_IMAGE_SIZE:
                logging.error(f"Image size {image_length} exceeds maximum")
                return None, None, None

            image_bytes = self._recv_exactly(client, image_length)
            if not image_bytes:
                return None, None, None

            return camera_id, prompt, image_bytes

        except Exception as e:
            logging.error(f"Error receiving image message: {e}")
            return None, None, None

    def _recv_exactly(self, sock: socket.socket, num_bytes: int) -> Optional[bytes]:
        """
        Receive exactly num_bytes from socket with timeout protection

        Args:
            sock: Socket to receive from (should have timeout set)
            num_bytes: Exact number of bytes to receive

        Returns:
            Bytes received or None if connection closed/timeout
        """
        data = b""
        while len(data) < num_bytes:
            try:
                chunk = sock.recv(num_bytes - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                logging.warning(
                    f"Socket timeout while receiving data (got {len(data)}/{num_bytes} bytes)"
                )
                return None
        return data


def run_server(server_config: ServerConfig, setup_signals: bool = True):
    """
    Start the StreamingServer (blocking)

    Args:
        server_config: Server configuration
        setup_signals: If True, setup signal handlers (only valid in main thread)
    """
    server = StreamingServer(server_config)

    # Setup signal handlers (only if in main thread)
    if setup_signals:

        def signal_handler(_sig, _frame):
            logging.info("Shutdown signal received")
            server.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    try:
        server.start()

        # Status monitoring loop
        storage = ImageStorage.get_instance()
        while server.is_running():
            time.sleep(cfg.STREAMING_SERVER_MONITOR)

            camera_ids = storage.get_all_camera_ids()
            if camera_ids:
                logging.info(f"Active cameras: {camera_ids}")
                for cam_id in camera_ids:
                    age = storage.get_camera_age(cam_id)
                    prompt = storage.get_camera_prompt(cam_id)
                    prompt_info = f", prompt: '{prompt}'" if prompt else ""
                    logging.info(f"  {cam_id}: age={age:.1f}s{prompt_info}")

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        server.stop()


def run_streaming_server_background(server_config: ServerConfig):
    """Start the StreamingServer in a background thread"""
    server_config = server_config or cfg.get_streaming_config()

    thread = threading.Thread(
        target=run_server,
        args=(server_config, False),  # setup_signals=False in background thread
        daemon=True,
    )
    thread.start()
    logging.info("StreamingServer started in background thread")
    return thread


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unity camera image streaming server")
    parser.add_argument("--host", default=cfg.DEFAULT_HOST, help="Host to bind to")
    parser.add_argument(
        "--port", type=int, default=cfg.STREAMING_SERVER_PORT, help="Port to bind to"
    )

    args = parser.parse_args()

    server_config = ServerConfig(host=args.host, port=args.port)
    run_server(server_config)
