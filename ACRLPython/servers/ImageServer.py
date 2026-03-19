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
import time
from typing import Optional
import numpy as np
import cv2

# Import config
try:
    from config.Servers import (
        DEFAULT_HOST,
        STREAMING_SERVER_PORT,
        STEREO_DETECTION_PORT,
        MAX_STRING_LENGTH,
        MAX_IMAGE_SIZE,
    )
    from config.Vision import ENABLE_VISION_STREAMING
    from core.LoggingSetup import get_logger
except ImportError:
    from ..config.Servers import (
        DEFAULT_HOST,
        STREAMING_SERVER_PORT,
        STEREO_DETECTION_PORT,
        MAX_STRING_LENGTH,
        MAX_IMAGE_SIZE,
    )
    from ..config.Vision import ENABLE_VISION_STREAMING
    from ..core.LoggingSetup import get_logger

# Import base classes
try:
    from core.TCPServerBase import TCPServerBase, ServerConfig, ConnectionState
    from core.UnityProtocol import UnityProtocol, MessageType
except ImportError:
    from ..core.TCPServerBase import TCPServerBase, ServerConfig, ConnectionState
    from ..core.UnityProtocol import UnityProtocol, MessageType

# Import storage singleton from core module (no circular dependency)
try:
    from .ImageStorageCore import UnifiedImageStorage
except ImportError:
    from servers.ImageStorageCore import UnifiedImageStorage

logger = get_logger(__name__)


class SingleImageServer(TCPServerBase):
    """
    TCP server for receiving single camera images (port 5005).
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        if config is None:
            config = ServerConfig(host=DEFAULT_HOST, port=STREAMING_SERVER_PORT)
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
                if id_len is None or id_len > MAX_STRING_LENGTH:
                    break
                camera_id_bytes = self._recv_exactly(client, id_len)
                if camera_id_bytes is None:
                    break
                camera_id = camera_id_bytes.decode("utf-8")

                # Read prompt
                prompt_len = self._read_int(client)
                if prompt_len is None or prompt_len > MAX_STRING_LENGTH:
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
                if img_len is None or img_len > MAX_IMAGE_SIZE:
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


class StereoImageServer(TCPServerBase):
    """
    TCP server for receiving stereo image pairs (port 5006).
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        if config is None:
            config = ServerConfig(host=DEFAULT_HOST, port=STEREO_DETECTION_PORT)
        super().__init__(config)
        self._storage = UnifiedImageStorage()

    def handle_client_connection(self, client: socket.socket, address: tuple):
        """Handle stereo image pair reception."""
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
                if pair_id_len is None or pair_id_len > MAX_STRING_LENGTH:
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
                if prompt_len is None or prompt_len > MAX_STRING_LENGTH:
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
                if img_L_len is None or img_L_len > MAX_IMAGE_SIZE:
                    break
                img_L_data = self._recv_exactly(client, img_L_len)
                if not img_L_data:
                    break

                # Read right image
                img_R_len = self._read_int(client)
                if img_R_len is None or img_R_len > MAX_IMAGE_SIZE:
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
                        and meta_len < MAX_STRING_LENGTH * 10
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

                    if not ENABLE_VISION_STREAMING:
                        logger.debug(
                            f"[req={request_id}] Received stereo '{camera_pair_id}' "
                            f"(L: {img_L_len/1024:.1f}KB, R: {img_R_len/1024:.1f}KB)"
                        )

        except Exception as e:
            logger.error(f"Error handling stereo client {address}: {e}")


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
        single_port: int = STREAMING_SERVER_PORT,
        stereo_port: int = STEREO_DETECTION_PORT,
        host: str = DEFAULT_HOST,
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
    single_port: int = STREAMING_SERVER_PORT,
    stereo_port: int = STEREO_DETECTION_PORT,
    host: str = DEFAULT_HOST,
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
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--single-port", type=int, default=STREAMING_SERVER_PORT)
    parser.add_argument("--stereo-port", type=int, default=STEREO_DETECTION_PORT)
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
