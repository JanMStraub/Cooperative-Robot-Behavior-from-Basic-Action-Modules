import socket
import struct
import cv2
import numpy as np
import threading
import time
import logging
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


@dataclass
class ServerConfig:
    """Server configuration settings."""

    host: str = "127.0.0.1"
    port: int = 5005
    max_id_length: int = 256
    max_image_size: int = 10 * 1024 * 1024  # 10MB max


class GracefulShutdown:
    """Handles graceful shutdown signal."""

    shutdown_requested = False

    @classmethod
    def request_shutdown(cls, _signum=None, _frame=None):
        """Request graceful shutdown."""
        logging.info("Shutdown signal received...")
        cls.shutdown_requested = True


class ImageServer:
    """Singleton to access camera images from the streaming server."""

    _instance = None
    _cameras_dict: Dict[str, Tuple[np.ndarray, float]] = {}
    _cameras_lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def set_storage(cls, cameras_dict: Dict, cameras_lock: threading.Lock):
        """Set the shared storage references (called by server)."""
        cls._cameras_dict = cameras_dict
        cls._cameras_lock = cameras_lock

    def get_camera_image(self, camera_id: str) -> Optional[np.ndarray]:
        """
        Get the latest image from a specific camera.

        Args:
            camera_id: Camera identifier (e.g., "AR4Left", "AR4Right")

        Returns:
            numpy array of the image, or None if not available
        """
        with self._cameras_lock:
            if camera_id in self._cameras_dict:
                image, _ = self._cameras_dict[camera_id]
                return image.copy()  # Return a copy to prevent race conditions
            return None

    def get_all_camera_ids(self) -> List[str]:
        """
        Get list of all active camera IDs.

        Returns:
            List of camera IDs
        """
        with self._cameras_lock:
            return list(self._cameras_dict.keys())

    def get_camera_age(self, camera_id: str) -> Optional[float]:
        """
        Get the age (in seconds) of the latest image from a camera.

        Args:
            camera_id: Camera identifier

        Returns:
            Age in seconds, or None if camera not found
        """
        with self._cameras_lock:
            if camera_id in self._cameras_dict:
                _, timestamp = self._cameras_dict[camera_id]
                return time.time() - timestamp
            return None


def receive_exactly(sock: socket.socket, num_bytes: int) -> Optional[bytes]:
    """
    Receives exactly num_bytes from the socket.

    Args:
        sock: Socket to receive from
        num_bytes: Exact number of bytes to receive

    Returns:
        Bytes received or None if connection closes
    """
    data = b""
    while len(data) < num_bytes:
        packet = sock.recv(num_bytes - len(data))
        if not packet:
            return None  # Connection closed
        data += packet
    return data


def receive_image(
    sock: socket.socket, config: ServerConfig
) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """
    Receives one image from the socket.

    Args:
        sock: Socket to receive from
        config: Server configuration for validation

    Returns:
        Tuple of (cam_id, image) or (None, None) on failure
    """
    try:
        # Receive camera ID length (4 bytes)
        id_length_data = receive_exactly(sock, 4)
        if id_length_data is None:
            return None, None
        id_length = struct.unpack("I", id_length_data)[0]

        # Validate camera ID length
        if id_length == 0 or id_length > config.max_id_length:
            logging.error(f"Invalid camera ID length: {id_length}")
            return None, None

        # Receive camera ID string
        id_data = receive_exactly(sock, id_length)
        if id_data is None:
            return None, None
        cam_id = id_data.decode("utf-8")

        # Receive image size (4 bytes)
        size_info = receive_exactly(sock, 4)
        if size_info is None:
            return None, None
        image_size = struct.unpack("I", size_info)[0]

        # Validate image size
        if image_size == 0 or image_size > config.max_image_size:
            logging.error(f"Invalid image size: {image_size} bytes")
            return None, None

        # Receive image data
        image_data = receive_exactly(sock, image_size)
        if image_data is None:
            return None, None

        # Decode the PNG/JPG image
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            logging.warning(f"Failed to decode image from camera {cam_id}")
            return None, None

        return cam_id, image

    except UnicodeDecodeError as e:
        logging.error(f"Failed to decode camera ID: {e}")
        return None, None
    except Exception as e:
        logging.error(f"Error receiving image: {e}")
        return None, None


def handle_client(
    conn: socket.socket,
    cameras_dict: Dict[str, Tuple[np.ndarray, float]],
    cameras_lock: threading.Lock,
    config: ServerConfig,
) -> None:
    """
    Handles a client connection by receiving images and storing them by camera ID.

    Args:
        conn: Client socket connection
        cameras_dict: Dictionary mapping camera ID to (image, timestamp)
        cameras_lock: Lock for thread-safe access to cameras_dict
        config: Server configuration
    """
    try:
        while not GracefulShutdown.shutdown_requested:
            cam_id, image = receive_image(conn, config)
            if image is None:
                logging.warning("Failed to receive image from client")
                break

            timestamp = time.time()
            logging.info(
                f"Received image from camera: {cam_id} ({image.shape[1]}x{image.shape[0]})"
            )

            # Store image by camera ID with timestamp (thread-safe)
            with cameras_lock:
                cameras_dict[cam_id] = (image, timestamp)

    except Exception as e:
        logging.error(f"Client error: {e}")
    finally:
        conn.close()
        logging.info("Client disconnected")


def accept_clients(
    server_socket: socket.socket,
    cameras_dict: Dict[str, Tuple[np.ndarray, float]],
    cameras_lock: threading.Lock,
    config: ServerConfig,
) -> None:
    """
    Continuously accepts incoming client connections.

    Args:
        server_socket: Server socket for accepting connections
        cameras_dict: Dictionary for storing images by camera ID
        cameras_lock: Lock for thread-safe access
        config: Server configuration
    """
    server_socket.settimeout(1.0)  # Allow periodic checks for shutdown

    while not GracefulShutdown.shutdown_requested:
        try:
            conn, addr = server_socket.accept()
            logging.info(f"Client connected from {addr}")
            threading.Thread(
                target=handle_client,
                args=(conn, cameras_dict, cameras_lock, config),
                daemon=True,
            ).start()
        except socket.timeout:
            continue
        except Exception as e:
            if not GracefulShutdown.shutdown_requested:
                logging.error(f"Error accepting client: {e}")


def run_server(config: ServerConfig = ServerConfig()) -> None:
    """
    Main server function that receives and stores images from Unity cameras.

    Args:
        config: Server configuration settings
    """
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, GracefulShutdown.request_shutdown)
    signal.signal(signal.SIGTERM, GracefulShutdown.request_shutdown)

    server_socket = None

    try:
        # Create server socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((config.host, config.port))
        server_socket.listen(5)
        logging.info(f"Server listening on {config.host}:{config.port}...")
        logging.info("Ready to receive images from Unity cameras")

        # Shared data structures
        cameras_dict: Dict[str, Tuple[np.ndarray, float]] = {}
        cameras_lock = threading.Lock()

        # Make images available via ImageServer singleton
        ImageServer.set_storage(cameras_dict, cameras_lock)
        logging.info("ImageServer singleton is ready for external access")

        # Start background thread for accepting clients
        accept_thread = threading.Thread(
            target=accept_clients,
            args=(server_socket, cameras_dict, cameras_lock, config),
            daemon=True,
        )
        accept_thread.start()

        # Main loop - display camera statistics
        while not GracefulShutdown.shutdown_requested:
            time.sleep(5.0)  # Update every 5 seconds

            with cameras_lock:
                if cameras_dict:
                    logging.info(f"Active cameras: {list(cameras_dict.keys())}")
                    for cam_id, (img, timestamp) in cameras_dict.items():
                        age = time.time() - timestamp
                        logging.info(
                            f"  {cam_id}: {img.shape[1]}x{img.shape[0]}, {age:.1f}s ago"
                        )

    except Exception as e:
        logging.error(f"Server error: {e}")

    finally:
        logging.info("Shutting down server...")

        if server_socket is not None:
            try:
                server_socket.close()
            except Exception as e:
                logging.error(f"Error closing server socket: {e}")

        logging.info("Server shutdown complete")


if __name__ == "__main__":
    config = ServerConfig()
    run_server(config)
