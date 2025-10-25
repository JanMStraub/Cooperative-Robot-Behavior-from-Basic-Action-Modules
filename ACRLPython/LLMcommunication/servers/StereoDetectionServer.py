#!/usr/bin/env python3
"""
StereoDetectionServer.py - TCP server for stereo object detection with depth

Receives stereo image pairs from Unity, performs object detection with depth
estimation, and sends results back with 3D world coordinates.

Port: 5009 (receives stereo images)
Results sent via ResultsServer (port 5006)
"""

import logging
import socket
import struct
import threading
import time
import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2

# Add LLMCommunication package directory to path
_package_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_package_dir))

# Import config - support both direct script and module execution
try:
    from .. import config as cfg
except ImportError:
    import config as cfg

# Import core infrastructure - support both direct script and module execution
try:
    from ..core.TCPServerBase import TCPServerBase
    from ..vision.ObjectDetector import CubeDetector
except ImportError:
    from core.TCPServerBase import TCPServerBase
    from vision.ObjectDetector import CubeDetector

# Configure logging (safe for testing with mocked config)
try:
    log_level = getattr(logging, cfg.LOG_LEVEL) if isinstance(cfg.LOG_LEVEL, str) else logging.INFO
    log_format = cfg.LOG_FORMAT if isinstance(cfg.LOG_FORMAT, str) else '%(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format)
except (AttributeError, TypeError):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


class StereoImageStorage:
    """
    Thread-safe storage for received stereo image pairs.

    Stores the most recent stereo pair from each camera pair.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_storage()
        return cls._instance

    def _init_storage(self):
        """Initialize storage structures"""
        self._images = {}  # camera_pair_id -> (imgL, imgR, prompt, timestamp)
        self._data_lock = threading.Lock()

    def store_stereo_pair(
        self, camera_pair_id: str, imgL: np.ndarray, imgR: np.ndarray, prompt: str
    ):
        """
        Store a stereo image pair.

        Args:
            camera_pair_id: Identifier for the camera pair (e.g., "AR4_Stereo")
            imgL: Left camera image
            imgR: Right camera image
            prompt: Associated prompt/metadata
        """
        with self._data_lock:
            self._images[camera_pair_id] = (imgL, imgR, prompt, time.time())
            logging.info(
                f"Stored stereo pair for '{camera_pair_id}' (L: {imgL.shape}, R: {imgR.shape})"
            )

    def get_stereo_pair(
        self, camera_pair_id: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
        """
        Retrieve the most recent stereo pair.

        Args:
            camera_pair_id: Identifier for the camera pair

        Returns:
            Tuple of (imgL, imgR, prompt) or None if not available
        """
        with self._data_lock:
            if camera_pair_id in self._images:
                imgL, imgR, prompt, _ = self._images[camera_pair_id]
                return imgL, imgR, prompt
            return None

    def get_pair_age(self, camera_pair_id: str) -> Optional[float]:
        """
        Get age of stored stereo pair in seconds.

        Args:
            camera_pair_id: Identifier for the camera pair

        Returns:
            Age in seconds or None if not available
        """
        with self._data_lock:
            if camera_pair_id in self._images:
                _, _, _, timestamp = self._images[camera_pair_id]
                return time.time() - timestamp
            return None


class StereoDetectionServer(TCPServerBase):
    """
    TCP server that receives stereo image pairs and stores them for processing.

    Protocol: [cam_pair_id_len][cam_pair_id][camera_L_id_len][camera_L_id]
              [camera_R_id_len][camera_R_id][prompt_len][prompt]
              [image_L_len][image_L_data][image_R_len][image_R_data]
    """

    def __init__(self, server_config=None):
        """
        Initialize the stereo detection server.

        Args:
            server_config: Server configuration (host, port, etc.)
        """
        if server_config is None:
            from core.TCPServerBase import ServerConfig
            server_config = ServerConfig()
            server_config.host = "127.0.0.1"
            server_config.port = 5009

        super().__init__(server_config)
        self.image_storage = StereoImageStorage()
        logging.info("StereoDetectionServer initialized")

    def handle_client_connection(self, client: socket.socket, address: tuple):
        """
        Handle a single client connection.

        Receives stereo image pairs and stores them for processing.

        Args:
            client: Client socket
            address: Client address tuple
        """
        logging.info(f"Stereo detection client connected from {address}")

        try:
            client.settimeout(5.0)

            while not self.should_shutdown():
                # Read camera pair ID length
                cam_pair_id_len_data = self._receive_exactly(client, 4)
                if cam_pair_id_len_data is None:
                    break

                cam_pair_id_len = struct.unpack("I", cam_pair_id_len_data)[0]
                if cam_pair_id_len > cfg.MAX_STRING_LENGTH:
                    logging.error(f"Camera pair ID length {cam_pair_id_len} exceeds maximum")
                    break

                # Read camera pair ID
                cam_pair_id_data = self._receive_exactly(client, cam_pair_id_len)
                if cam_pair_id_data is None:
                    break
                cam_pair_id = cam_pair_id_data.decode("utf-8")

                # Read camera L ID length
                cam_L_id_len_data = self._receive_exactly(client, 4)
                if cam_L_id_len_data is None:
                    break

                cam_L_id_len = struct.unpack("I", cam_L_id_len_data)[0]
                if cam_L_id_len > cfg.MAX_STRING_LENGTH:
                    logging.error(f"Camera L ID length {cam_L_id_len} exceeds maximum")
                    break

                # Read camera L ID
                cam_L_id_data = self._receive_exactly(client, cam_L_id_len)
                if cam_L_id_data is None:
                    break
                cam_L_id = cam_L_id_data.decode("utf-8")

                # Read camera R ID length
                cam_R_id_len_data = self._receive_exactly(client, 4)
                if cam_R_id_len_data is None:
                    break

                cam_R_id_len = struct.unpack("I", cam_R_id_len_data)[0]
                if cam_R_id_len > cfg.MAX_STRING_LENGTH:
                    logging.error(f"Camera R ID length {cam_R_id_len} exceeds maximum")
                    break

                # Read camera R ID
                cam_R_id_data = self._receive_exactly(client, cam_R_id_len)
                if cam_R_id_data is None:
                    break
                cam_R_id = cam_R_id_data.decode("utf-8")

                # Read prompt length
                prompt_len_data = self._receive_exactly(client, 4)
                if prompt_len_data is None:
                    break

                prompt_len = struct.unpack("I", prompt_len_data)[0]
                if prompt_len > cfg.MAX_STRING_LENGTH:
                    logging.error(f"Prompt length {prompt_len} exceeds maximum")
                    break

                # Read prompt
                prompt_data = self._receive_exactly(client, prompt_len)
                if prompt_data is None:
                    break
                prompt = prompt_data.decode("utf-8") if prompt_len > 0 else ""

                # Read left image length
                img_L_len_data = self._receive_exactly(client, 4)
                if img_L_len_data is None:
                    break

                img_L_len = struct.unpack("I", img_L_len_data)[0]
                if img_L_len > cfg.MAX_IMAGE_SIZE:
                    logging.error(f"Left image size {img_L_len} exceeds maximum")
                    break

                # Read left image data
                img_L_data = self._receive_exactly(client, img_L_len)
                if img_L_data is None:
                    break

                # Read right image length
                img_R_len_data = self._receive_exactly(client, 4)
                if img_R_len_data is None:
                    break

                img_R_len = struct.unpack("I", img_R_len_data)[0]
                if img_R_len > cfg.MAX_IMAGE_SIZE:
                    logging.error(f"Right image size {img_R_len} exceeds maximum")
                    break

                # Read right image data
                img_R_data = self._receive_exactly(client, img_R_len)
                if img_R_data is None:
                    break

                # Decode images
                imgL = cv2.imdecode(np.frombuffer(img_L_data, np.uint8), cv2.IMREAD_COLOR)
                imgR = cv2.imdecode(np.frombuffer(img_R_data, np.uint8), cv2.IMREAD_COLOR)

                if imgL is None or imgR is None:
                    logging.error("Failed to decode stereo images")
                    continue

                # Store stereo pair
                self.image_storage.store_stereo_pair(cam_pair_id, imgL, imgR, prompt)

                logging.info(
                    f"Received stereo pair '{cam_pair_id}' (L: {cam_L_id}, R: {cam_R_id}, "
                    f"sizes: {img_L_len}B + {img_R_len}B)"
                )

        except socket.timeout:
            logging.debug("Client socket timeout")
        except Exception as e:
            if not self.should_shutdown():
                logging.error(f"Error handling stereo detection client: {e}")
        finally:
            client.close()
            logging.info(f"Stereo detection client disconnected from {address}")

    def _receive_exactly(self, sock: socket.socket, num_bytes: int) -> Optional[bytes]:
        """
        Receive exactly num_bytes from socket.

        Args:
            sock: Socket to receive from
            num_bytes: Number of bytes to receive

        Returns:
            Received bytes or None if connection closed
        """
        data = b""
        timeout_count = 0
        max_timeouts = 3  # Limit timeout retries

        while len(data) < num_bytes:
            try:
                packet = sock.recv(num_bytes - len(data))
                if not packet:
                    return None
                data += packet
                timeout_count = 0  # Reset on successful recv
            except socket.timeout:
                timeout_count += 1
                if timeout_count >= max_timeouts:
                    logging.warning(f"Socket receive timed out after {max_timeouts} retries")
                    return None
                continue
            except Exception as e:
                logging.error(f"Socket receive error: {e}")
                return None
        return data


def run_stereo_detection_server_background(host: str = "127.0.0.1", port: int = 5009) -> StereoDetectionServer:
    """
    Run stereo detection server in background thread.

    Args:
        host: Server host
        port: Server port

    Returns:
        Server instance
    """
    from core.TCPServerBase import ServerConfig
    config = ServerConfig()
    config.host = host
    config.port = port

    server = StereoDetectionServer(config)
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    logging.info(f"Stereo detection server started on {host}:{port} (background)")
    return server


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stereo Detection Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=5009, help="Server port")

    args = parser.parse_args()

    logging.info(f"Starting stereo detection server on {args.host}:{args.port}")

    from core.TCPServerBase import ServerConfig
    config = ServerConfig()
    config.host = args.host
    config.port = args.port

    server = StereoDetectionServer(config)
    server.start()
