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

# Import config
# Import config - try both import styles
try:
    import LLMConfig as cfg
except ImportError:
    from .. import LLMConfig as cfg

# Import base classes - try both import styles
try:
    from core.TCPServerBase import TCPServerBase, ServerConfig, ConnectionState
    from core.UnityProtocol import UnityProtocol
except ImportError:
    from ..core.TCPServerBase import TCPServerBase, ServerConfig, ConnectionState
    from ..core.UnityProtocol import UnityProtocol

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
    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "ImageStorage":
        """Get the singleton instance with thread-safe double-check locking"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:  # Double-check after acquiring lock
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

    def cleanup_old_images(self, max_age_seconds: float = 300.0):
        """Remove images older than max_age_seconds"""
        with self._lock:
            current_time = time.time()
            to_remove = [
                cam_id
                for cam_id, (_, timestamp, _) in self._cameras.items()
                if current_time - timestamp > max_age_seconds
            ]
            for cam_id in to_remove:
                del self._cameras[cam_id]
                logging.debug(f"Cleaned up old image from {cam_id}")


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

    def handle_client_connection(self, client: socket.socket, address: tuple):
        """
        Handle a Unity client connection - receives images continuously.

        This method is called by TCPServerBase in a separate thread per client.
        """
        logging.info(f"Unity camera client connected from {address}")

        try:
            # Set socket to blocking mode (no timeout for persistent connections)
            client.settimeout(None)

            while self.is_running():
                # Update state to IDLE before receiving
                self._update_client_state(client, ConnectionState.IDLE)

                # Receive one complete image message (Protocol V2)
                request_id, camera_id, prompt, image_bytes = self._receive_image_message(client)

                if image_bytes is None:
                    # Connection closed or error - check client state
                    client_info = self.get_client_info(client)
                    if client_info and client_info.state == ConnectionState.ERROR:
                        # Fatal error - exit gracefully
                        break
                    # Otherwise, connection closed cleanly
                    break

                # Decode PNG/JPG image
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if image is None:
                    logging.warning(f"Failed to decode image from {camera_id} (request_id={request_id})")
                    continue

                # Store in singleton
                camera_id = camera_id or "unknown_camera"
                prompt = prompt or ""
                self._storage.store_image(camera_id, image, prompt)

                # Log receipt with request_id for tracing
                prompt_info = f" with prompt: '{prompt}'" if prompt else ""
                logging.info(
                    f"[req={request_id}] Received image from {camera_id}: "
                    f"{image.shape[1]}x{image.shape[0]}{prompt_info}"
                )

        except Exception as e:
            logging.error(f"Error handling client {address}: {e}")

    def _receive_image_message(
        self, client: socket.socket
    ) -> Tuple[Optional[int], Optional[str], Optional[str], Optional[bytes]]:
        """
        Receive one complete image message from Unity (Protocol V2).

        Uses UnityProtocol format:
        [type:1][request_id:4][camera_id_len:4][camera_id:N][prompt_len:4][prompt:N][image_len:4][image_data:N]

        Returns:
            (request_id, camera_id, prompt, image_bytes) or (None, None, None, None) on failure
        """
        import struct

        try:
            # No timeout - allow persistent connections to stay open indefinitely
            # Unity will reconnect automatically if connection drops
            # client.settimeout(None)  # None = blocking, no timeout (default)

            # Read header (type + request_id) - Protocol V2
            header_bytes = self._recv_exactly(client, UnityProtocol.HEADER_SIZE)
            if not header_bytes:
                return None, None, None, None

            # Decode header
            msg_type = header_bytes[0]  # First byte is message type
            request_id = struct.unpack(UnityProtocol.INT_FORMAT, header_bytes[1:5])[0]

            # Validate message type
            from core.UnityProtocol import MessageType
            if msg_type != MessageType.IMAGE:
                logging.error(f"Expected IMAGE message, got type {msg_type}")
                return None, None, None, None

            # Read camera ID length and data
            id_length_bytes = self._recv_exactly(client, UnityProtocol.INT_SIZE)
            if not id_length_bytes:
                return None, None, None, None
            id_length = struct.unpack(UnityProtocol.INT_FORMAT, id_length_bytes)[0]

            if id_length > UnityProtocol.MAX_STRING_LENGTH:
                logging.error(f"Camera ID length {id_length} exceeds maximum")
                return None, None, None, None

            camera_id_bytes = self._recv_exactly(client, id_length)
            if not camera_id_bytes:
                return None, None, None, None
            camera_id = camera_id_bytes.decode("utf-8")

            # Read prompt length and data
            prompt_length_bytes = self._recv_exactly(client, UnityProtocol.INT_SIZE)
            if not prompt_length_bytes:
                return None, None, None, None
            prompt_length = struct.unpack(
                UnityProtocol.INT_FORMAT, prompt_length_bytes
            )[0]

            if prompt_length > UnityProtocol.MAX_STRING_LENGTH:
                logging.error(f"Prompt length {prompt_length} exceeds maximum")
                return None, None, None, None

            prompt = ""
            if prompt_length > 0:
                prompt_bytes = self._recv_exactly(client, prompt_length)
                if not prompt_bytes:
                    return None, None, None, None
                prompt = prompt_bytes.decode("utf-8")

            # Read image length and data
            image_length_bytes = self._recv_exactly(client, UnityProtocol.INT_SIZE)
            if not image_length_bytes:
                return None, None, None, None
            image_length = struct.unpack(UnityProtocol.INT_FORMAT, image_length_bytes)[
                0
            ]

            if image_length > UnityProtocol.MAX_IMAGE_SIZE:
                logging.error(f"Image size {image_length} exceeds maximum")
                return None, None, None, None

            image_bytes = self._recv_exactly(client, image_length)
            if not image_bytes:
                return None, None, None, None

            return request_id, camera_id, prompt, image_bytes

        except Exception as e:
            logging.error(f"Error receiving image message: {e}")
            return None, None, None, None

    def _recv_exactly(self, sock: socket.socket, num_bytes: int) -> Optional[bytes]:
        """
        Receive exactly num_bytes from socket with proper error handling.

        Args:
            sock: Socket to receive from
            num_bytes: Exact number of bytes to receive

        Returns:
            Bytes received or None if connection closed or error
        """
        # Update state to receiving
        self._update_client_state(sock, ConnectionState.RECEIVING)

        data = b""
        while len(data) < num_bytes:
            try:
                chunk = sock.recv(num_bytes - len(data))
                if not chunk:
                    # Connection closed cleanly
                    logging.debug(f"Client connection closed cleanly")
                    self._update_client_state(sock, ConnectionState.DISCONNECTED)
                    return None
                data += chunk
                self._record_bytes_received(sock, len(chunk))

            except Exception as e:
                # Determine if error is fatal
                is_fatal, error_desc = self._is_connection_error_fatal(e)

                if is_fatal:
                    # Fatal error - connection lost
                    logging.debug(f"Client disconnected: {error_desc}")
                    self._record_client_error(sock)
                else:
                    # Non-fatal error - just log at debug level
                    logging.debug(f"Socket idle: {error_desc}")

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
                
                storage.cleanup_old_images(max_age_seconds=300.0)  # 5 minutes

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
