#!/usr/bin/env python3
"""
ImageServer.py - Unified image receiving server

Consolidates StreamingServer (port 5005) and StereoDetectionServer (port 5006)
into a single server with unified storage.

Ports:
    5005 - Single camera images
    5006 - Stereo image pairs
"""

import socket
import struct
import threading
import time
import logging
from typing import Optional, Tuple, List, Dict
import numpy as np
import cv2

# Import config
try:
    import LLMConfig as cfg
except ImportError:
    from .. import LLMConfig as cfg

# Import base classes
try:
    from core.TCPServerBase import TCPServerBase, ServerConfig, ConnectionState
    from core.UnityProtocol import UnityProtocol, MessageType
except ImportError:
    from ..core.TCPServerBase import TCPServerBase, ServerConfig, ConnectionState
    from ..core.UnityProtocol import UnityProtocol, MessageType

logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=cfg.LOG_FORMAT)
logger = logging.getLogger(__name__)


class UnifiedImageStorage:
    """
    Thread-safe storage for all image types (single and stereo).

    Provides unified access for detection and analysis operations.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_storage()
        return cls._instance

    def _init_storage(self):
        """Initialize storage structures."""
        self._single_images: Dict[str, Tuple[np.ndarray, float, str]] = {}
        # Stereo: (imgL, imgR, prompt, timestamp, metadata)
        self._stereo_images: Dict[
            str, Tuple[np.ndarray, np.ndarray, str, float, dict]
        ] = {}
        self._data_lock = threading.Lock()

    # Single camera methods
    def store_single_image(self, camera_id: str, image: np.ndarray, prompt: str = ""):
        """Store a single camera image."""
        with self._data_lock:
            self._single_images[camera_id] = (image, time.time(), prompt)

    def get_single_image(self, camera_id: str) -> Optional[np.ndarray]:
        """Get the latest single camera image."""
        with self._data_lock:
            if camera_id in self._single_images:
                return self._single_images[camera_id][0].copy()
            return None

    def get_single_prompt(self, camera_id: str) -> Optional[str]:
        """Get the prompt for a single camera image."""
        with self._data_lock:
            if camera_id in self._single_images:
                return self._single_images[camera_id][2]
            return None

    def get_single_age(self, camera_id: str) -> Optional[float]:
        """Get age of single camera image in seconds."""
        with self._data_lock:
            if camera_id in self._single_images:
                return time.time() - self._single_images[camera_id][1]
            return None

    def get_latest_single(self) -> Optional[Tuple[str, np.ndarray, str]]:
        """Get the most recently stored single image."""
        with self._data_lock:
            if not self._single_images:
                return None
            latest_id = max(
                self._single_images.keys(), key=lambda k: self._single_images[k][1]
            )
            img, _, prompt = self._single_images[latest_id]
            return latest_id, img.copy(), prompt

    # Stereo camera methods
    def store_stereo_pair(
        self,
        camera_pair_id: str,
        imgL: np.ndarray,
        imgR: np.ndarray,
        prompt: str = "",
        metadata: Optional[dict] = None,
    ):
        """Store a stereo image pair with optional metadata."""
        with self._data_lock:
            self._stereo_images[camera_pair_id] = (
                imgL,
                imgR,
                prompt,
                time.time(),
                metadata or {},
            )
            logger.info(
                f"Stored stereo pair '{camera_pair_id}' "
                f"(L: {imgL.shape}, R: {imgR.shape})"
            )

    def get_stereo_pair(
        self, camera_pair_id: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
        """Get a stereo image pair."""
        with self._data_lock:
            if camera_pair_id in self._stereo_images:
                imgL, imgR, prompt, _, _ = self._stereo_images[camera_pair_id]
                return imgL.copy(), imgR.copy(), prompt
            return None

    def get_stereo_metadata(self, camera_pair_id: str) -> Optional[dict]:
        """Get metadata for a stereo pair (baseline, fov, camera_position, camera_rotation)."""
        with self._data_lock:
            if camera_pair_id in self._stereo_images:
                return self._stereo_images[camera_pair_id][4]
            return None

    def get_stereo_age(self, camera_pair_id: str) -> Optional[float]:
        """Get age of stereo pair in seconds."""
        with self._data_lock:
            if camera_pair_id in self._stereo_images:
                return time.time() - self._stereo_images[camera_pair_id][3]
            return None

    def get_stereo_timestamp(self, camera_pair_id: str) -> Optional[float]:
        """Get the timestamp when stereo pair was received."""
        with self._data_lock:
            if camera_pair_id in self._stereo_images:
                return self._stereo_images[camera_pair_id][3]
            return None

    def get_latest_stereo(self) -> Optional[Tuple[str, np.ndarray, np.ndarray, str]]:
        """Get the most recently stored stereo pair."""
        with self._data_lock:
            if not self._stereo_images:
                return None
            latest_id = max(
                self._stereo_images.keys(), key=lambda k: self._stereo_images[k][3]
            )
            imgL, imgR, prompt, _, _ = self._stereo_images[latest_id]
            return latest_id, imgL.copy(), imgR.copy(), prompt

    def get_latest_stereo_image(
        self,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, str, float, dict]]:
        """
        Get the most recently stored stereo pair with full metadata.

        Returns:
            Tuple of (imgL, imgR, prompt, timestamp, metadata) or None if no stereo images
        """
        with self._data_lock:
            if not self._stereo_images:
                return None
            latest_id = max(
                self._stereo_images.keys(), key=lambda k: self._stereo_images[k][3]
            )
            imgL, imgR, prompt, timestamp, metadata = self._stereo_images[latest_id]
            return imgL.copy(), imgR.copy(), prompt, timestamp, metadata

    def get_all_stereo_ids(self) -> List[str]:
        """Get all stereo camera pair IDs."""
        with self._data_lock:
            return list(self._stereo_images.keys())

    # General methods
    def get_all_camera_ids(self) -> List[str]:
        """Get all camera IDs (single and stereo)."""
        with self._data_lock:
            single = list(self._single_images.keys())
            stereo = [f"{k} (stereo)" for k in self._stereo_images.keys()]
            return single + stereo

    def cleanup_old_images(self, max_age_seconds: float = 300.0):
        """Remove images older than max_age_seconds."""
        with self._data_lock:
            current = time.time()

            # Clean single images
            to_remove = [
                k
                for k, v in self._single_images.items()
                if current - v[1] > max_age_seconds
            ]
            for k in to_remove:
                del self._single_images[k]

            # Clean stereo images
            to_remove = [
                k
                for k, v in self._stereo_images.items()
                if current - v[3] > max_age_seconds
            ]
            for k in to_remove:
                del self._stereo_images[k]


class SingleImageServer(TCPServerBase):
    """
    TCP server for receiving single camera images (port 5005).
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        if config is None:
            config = ServerConfig(host=cfg.DEFAULT_HOST, port=cfg.STREAMING_SERVER_PORT)
        super().__init__(config)
        self._storage = UnifiedImageStorage()

    def handle_client_connection(self, client: socket.socket, address: tuple):
        """Handle single camera image reception."""
        logger.info(f"Single camera client connected from {address}")
        client.settimeout(None)

        try:
            while self.is_running():
                self._update_client_state(client, ConnectionState.IDLE)

                # Read Protocol V2 header
                header = self._recv_exactly(client, UnityProtocol.HEADER_SIZE)
                if not header:
                    break

                msg_type = header[0]
                request_id = struct.unpack(UnityProtocol.INT_FORMAT, header[1:5])[0]

                if msg_type != MessageType.IMAGE:
                    logger.error(f"Expected IMAGE, got {msg_type}")
                    break

                # Read camera_id
                id_len = self._read_int(client)
                if id_len is None or id_len > cfg.MAX_STRING_LENGTH:
                    break
                camera_id_bytes = self._recv_exactly(client, id_len)
                if camera_id_bytes is None:
                    break
                camera_id = camera_id_bytes.decode("utf-8")

                # Read prompt
                prompt_len = self._read_int(client)
                if prompt_len is None or prompt_len > cfg.MAX_STRING_LENGTH:
                    break
                if prompt_len > 0:
                    prompt_bytes = self._recv_exactly(client, prompt_len)
                    if prompt_bytes is None:
                        break
                    prompt = prompt_bytes.decode("utf-8")
                else:
                    prompt = ""

                # Read image
                img_len = self._read_int(client)
                if img_len is None or img_len > cfg.MAX_IMAGE_SIZE:
                    break
                img_data = self._recv_exactly(client, img_len)
                if not img_data:
                    break

                # Decode and store
                image = cv2.imdecode(
                    np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR
                )
                if image is not None:
                    self._storage.store_single_image(camera_id, image, prompt)
                    logger.info(
                        f"[req={request_id}] Received {camera_id}: {image.shape[1]}x{image.shape[0]}"
                    )

        except Exception as e:
            logger.error(f"Error handling client {address}: {e}")

    def _recv_exactly(self, sock: socket.socket, num_bytes: int) -> Optional[bytes]:
        """Receive exactly num_bytes."""
        self._update_client_state(sock, ConnectionState.RECEIVING)
        data = b""
        while len(data) < num_bytes:
            try:
                chunk = sock.recv(num_bytes - len(data))
                if not chunk:
                    return None
                data += chunk
                self._record_bytes_received(sock, len(chunk))
            except Exception:
                return None
        return data

    def _read_int(self, sock: socket.socket) -> Optional[int]:
        """Read a 4-byte integer."""
        data = self._recv_exactly(sock, 4)
        if data:
            return struct.unpack(UnityProtocol.INT_FORMAT, data)[0]
        return None


class StereoImageServer(TCPServerBase):
    """
    TCP server for receiving stereo image pairs (port 5006).
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        if config is None:
            config = ServerConfig(host=cfg.DEFAULT_HOST, port=cfg.STEREO_DETECTION_PORT)
        super().__init__(config)
        self._storage = UnifiedImageStorage()

    def handle_client_connection(self, client: socket.socket, address: tuple):
        """Handle stereo image pair reception."""
        logger.info(f"Stereo camera client connected from {address}")
        client.settimeout(None)

        try:
            while self.is_running():
                self._update_client_state(client, ConnectionState.IDLE)

                # Read Protocol V2 header
                header = self._recv_exactly(client, UnityProtocol.HEADER_SIZE)
                if not header:
                    break

                msg_type = header[0]
                request_id = struct.unpack(UnityProtocol.INT_FORMAT, header[1:5])[0]

                if msg_type != MessageType.STEREO_IMAGE:
                    logger.error(f"Expected STEREO_IMAGE, got {msg_type}")
                    break

                # Read camera_pair_id
                pair_id_len = self._read_int(client)
                if pair_id_len is None or pair_id_len > cfg.MAX_STRING_LENGTH:
                    break
                pair_id_bytes = self._recv_exactly(client, pair_id_len)
                if pair_id_bytes is None:
                    break
                camera_pair_id = pair_id_bytes.decode("utf-8")

                # Read camera_L_id (not used but part of protocol)
                cam_L_len = self._read_int(client)
                if cam_L_len is None:
                    break
                self._recv_exactly(client, cam_L_len)

                # Read camera_R_id (not used but part of protocol)
                cam_R_len = self._read_int(client)
                if cam_R_len is None:
                    break
                self._recv_exactly(client, cam_R_len)

                # Read prompt
                prompt_len = self._read_int(client)
                if prompt_len is None or prompt_len > cfg.MAX_STRING_LENGTH:
                    break
                if prompt_len > 0:
                    prompt_bytes = self._recv_exactly(client, prompt_len)
                    if prompt_bytes is None:
                        break
                    prompt = prompt_bytes.decode("utf-8")
                else:
                    prompt = ""

                # Read left image
                img_L_len = self._read_int(client)
                if img_L_len is None or img_L_len > cfg.MAX_IMAGE_SIZE:
                    break
                img_L_data = self._recv_exactly(client, img_L_len)
                if not img_L_data:
                    break

                # Read right image
                img_R_len = self._read_int(client)
                if img_R_len is None or img_R_len > cfg.MAX_IMAGE_SIZE:
                    break
                img_R_data = self._recv_exactly(client, img_R_len)
                if not img_R_data:
                    break

                # Read metadata (if available)
                metadata = {}
                try:
                    meta_len = self._read_int(client)
                    if (
                        meta_len is not None
                        and meta_len > 0
                        and meta_len < cfg.MAX_STRING_LENGTH * 10
                    ):
                        meta_data = self._recv_exactly(client, meta_len)
                        if meta_data:
                            import json

                            metadata = json.loads(meta_data.decode("utf-8"))
                            logger.debug(f"Received metadata: {metadata}")
                except Exception as e:
                    logger.debug(f"No metadata received (legacy client): {e}")

                # Decode images
                imgL = cv2.imdecode(
                    np.frombuffer(img_L_data, np.uint8), cv2.IMREAD_COLOR
                )
                imgR = cv2.imdecode(
                    np.frombuffer(img_R_data, np.uint8), cv2.IMREAD_COLOR
                )

                if imgL is not None and imgR is not None:
                    self._storage.store_stereo_pair(
                        camera_pair_id, imgL, imgR, prompt, metadata
                    )
                    logger.info(
                        f"[req={request_id}] Received stereo '{camera_pair_id}' "
                        f"(L: {img_L_len/1024:.1f}KB, R: {img_R_len/1024:.1f}KB)"
                    )

        except Exception as e:
            logger.error(f"Error handling stereo client {address}: {e}")

    def _recv_exactly(self, sock: socket.socket, num_bytes: int) -> Optional[bytes]:
        """Receive exactly num_bytes."""
        self._update_client_state(sock, ConnectionState.RECEIVING)
        data = b""
        while len(data) < num_bytes:
            try:
                chunk = sock.recv(num_bytes - len(data))
                if not chunk:
                    return None
                data += chunk
                self._record_bytes_received(sock, len(chunk))
            except Exception:
                return None
        return data

    def _read_int(self, sock: socket.socket) -> Optional[int]:
        """Read a 4-byte integer."""
        data = self._recv_exactly(sock, 4)
        if data:
            return struct.unpack(UnityProtocol.INT_FORMAT, data)[0]
        return None


class ImageServer:
    """
    Unified image server that manages both single and stereo image reception.

    Usage:
        server = ImageServer()
        server.start()

        # Access images via storage
        storage = server.get_storage()
        latest = storage.get_latest_stereo()
    """

    def __init__(
        self,
        single_port: int = cfg.STREAMING_SERVER_PORT,
        stereo_port: int = cfg.STEREO_DETECTION_PORT,
        host: str = cfg.DEFAULT_HOST,
    ):
        """
        Initialize the unified image server.

        Args:
            single_port: Port for single camera images
            stereo_port: Port for stereo image pairs
            host: Host to bind to
        """
        self._single_config = ServerConfig(host=host, port=single_port)
        self._stereo_config = ServerConfig(host=host, port=stereo_port)

        self._single_server = SingleImageServer(self._single_config)
        self._stereo_server = StereoImageServer(self._stereo_config)
        self._storage = UnifiedImageStorage()

    def start(self):
        """Start both image servers."""
        logger.info(
            f"Starting ImageServer (single: {self._single_config.port}, "
            f"stereo: {self._stereo_config.port})"
        )
        self._single_server.start()
        self._stereo_server.start()

    def stop(self):
        """Stop both image servers."""
        self._single_server.stop()
        self._stereo_server.stop()
        logger.info("ImageServer stopped")

    def is_running(self) -> bool:
        """Check if servers are running."""
        return self._single_server.is_running() or self._stereo_server.is_running()

    def get_storage(self) -> UnifiedImageStorage:
        """Get the unified image storage."""
        return self._storage


def run_image_server_background(
    single_port: int = cfg.STREAMING_SERVER_PORT,
    stereo_port: int = cfg.STEREO_DETECTION_PORT,
    host: str = cfg.DEFAULT_HOST,
) -> ImageServer:
    """
    Start the ImageServer in background threads.

    Args:
        single_port: Port for single camera images
        stereo_port: Port for stereo image pairs
        host: Host to bind to

    Returns:
        ImageServer instance
    """
    server = ImageServer(single_port, stereo_port, host)
    server.start()
    return server


if __name__ == "__main__":
    import argparse
    import signal

    parser = argparse.ArgumentParser(description="Unified Image Server")
    parser.add_argument("--host", default=cfg.DEFAULT_HOST)
    parser.add_argument("--single-port", type=int, default=cfg.STREAMING_SERVER_PORT)
    parser.add_argument("--stereo-port", type=int, default=cfg.STEREO_DETECTION_PORT)
    args = parser.parse_args()

    server = ImageServer(args.single_port, args.stereo_port, args.host)

    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        server.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    server.start()

    try:
        while server.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
