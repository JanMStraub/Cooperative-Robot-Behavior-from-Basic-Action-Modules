"""
Streaming server for real-time stereo reconstruction.

This module provides a TCP server that receives stereo image pairs from Unity,
performs 3D reconstruction, and visualizes the point cloud using Open3D.
"""

import logging
import queue
import signal
import socket
import struct
import sys
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import open3d as o3d

from .StereoConfig import (
    CameraConfig,
    ReconstructionConfig,
    ServerConfig,
    DEFAULT_CAMERA_CONFIG,
    DEFAULT_RECONSTRUCTION_CONFIG,
    DEFAULT_SERVER_CONFIG,
)
from .Reconstruct import stereo_reconstruct_stream

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class StereoStreamingServer:
    """
    TCP server for receiving and processing stereo image streams.

    This server receives stereo image pairs, reconstructs 3D point clouds,
    and provides real-time visualization using Open3D.
    """

    def __init__(
        self,
        camera_config: Optional[CameraConfig] = None,
        recon_config: Optional[ReconstructionConfig] = None,
        server_config: Optional[ServerConfig] = None,
    ):
        """
        Initialize the streaming server.

        Args:
            camera_config: Camera configuration (uses defaults if None)
            recon_config: Reconstruction configuration (uses defaults if None)
            server_config: Server configuration (uses defaults if None)
        """
        self.camera_config = camera_config or DEFAULT_CAMERA_CONFIG
        self.recon_config = recon_config or DEFAULT_RECONSTRUCTION_CONFIG
        self.server_config = server_config or DEFAULT_SERVER_CONFIG

        self.image_queue: queue.Queue = queue.Queue()  # type: ignore[type-arg]
        self.shutdown_event = threading.Event()
        self.server_socket: Optional[socket.socket] = None
        self.visualizer: Optional[o3d.visualization.Visualizer] = None  # type: ignore[attr-defined]
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud_lock = (
            threading.Lock()
        )  # Protect point cloud access from multiple threads

    def receive_exactly(self, sock: socket.socket, num_bytes: int) -> Optional[bytes]:
        """
        Receive exactly num_bytes from the socket.

        Args:
            sock: Socket to receive from
            num_bytes: Number of bytes to receive

        Returns:
            Received bytes or None if connection closed
        """
        data = b""
        while len(data) < num_bytes:
            try:
                packet = sock.recv(num_bytes - len(data))
                if not packet:
                    return None  # Connection closed
                data += packet
            except socket.timeout:
                continue
            except Exception as e:
                logging.error(f"Socket receive error: {e}")
                return None
        return data

    def receive_image(
        self, sock: socket.socket
    ) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """
        Receive one image from the socket.

        Args:
            sock: Socket to receive from

        Returns:
            Tuple of (camera_id, image) or (None, None) on failure
        """
        # Receive camera ID (1 byte ASCII character)
        header = self.receive_exactly(sock, 1)
        if header is None:
            return None, None
        cam_id = header.decode("ascii")

        # Receive image size (4 bytes unsigned int)
        size_info = self.receive_exactly(sock, 4)
        if size_info is None:
            return None, None
        image_size = struct.unpack("I", size_info)[0]

        # Receive PNG image data
        image_data = self.receive_exactly(sock, image_size)
        if image_data is None:
            return None, None

        # Decode PNG image
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            logging.warning(f"Failed to decode image from camera {cam_id}")

        return cam_id, image

    def handle_client(self, conn: socket.socket, addr: tuple) -> None:
        """
        Handle a single client connection.

        Args:
            conn: Client socket connection
            addr: Client address
        """
        logging.info(f"Client connected from {addr}")

        try:
            conn.settimeout(self.server_config.timeout)

            while not self.shutdown_event.is_set():
                # Receive left image
                cam_id, left_img = self.receive_image(conn)
                if left_img is None or cam_id != "L":
                    if not self.shutdown_event.is_set():
                        logging.warning(
                            "Failed to receive left image or incorrect camera ID"
                        )
                    break

                # Receive right image
                cam_id, right_img = self.receive_image(conn)
                if right_img is None or cam_id != "R":
                    if not self.shutdown_event.is_set():
                        logging.warning(
                            "Failed to receive right image or incorrect camera ID"
                        )
                    break

                # Queue images for processing
                try:
                    self.image_queue.put_nowait((left_img, right_img))
                except queue.Full:
                    logging.warning("Image queue full, dropping frame")

        except Exception as e:
            if not self.shutdown_event.is_set():
                logging.error(f"Client error: {e}")
        finally:
            conn.close()
            logging.info(f"Client disconnected from {addr}")

    def process_image_queue(self) -> None:
        """
        Process images from the queue and update the point cloud.

        This method runs in a separate thread and continuously processes
        queued stereo image pairs.
        """
        while not self.shutdown_event.is_set():
            try:
                left_img, right_img = self.image_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                logging.info("Processing stereo images for point cloud reconstruction")

                # Perform stereo reconstruction
                point_cloud_data = stereo_reconstruct_stream(
                    left_img, right_img, self.camera_config, self.recon_config
                )

                if point_cloud_data is None:
                    logging.error("Reconstruction returned None")
                    continue

                # Extract points and colors
                pts = point_cloud_data.verts.reshape(-1, 3)
                colors = point_cloud_data.colors.reshape(-1, 3) / 255.0

                if len(pts) == 0:
                    logging.warning("Reconstruction returned empty point cloud")
                    continue

                logging.info(f"Generated {len(pts)} points from stereo reconstruction")

                # Create Open3D point cloud
                new_pcd = o3d.geometry.PointCloud()
                new_pcd.points = o3d.utility.Vector3dVector(pts)
                new_pcd.colors = o3d.utility.Vector3dVector(colors)

                # Downsample for smoother visualization
                new_pcd = new_pcd.voxel_down_sample(
                    voxel_size=self.server_config.voxel_downsample_size
                )

                # Update shared point cloud (thread-safe)
                with self.point_cloud_lock:
                    self.point_cloud.points = new_pcd.points
                    self.point_cloud.colors = new_pcd.colors

                logging.info("Updated point cloud")

            except Exception as e:
                logging.error(f"Error processing images: {e}")

    def accept_clients(self) -> None:
        """
        Continuously accept incoming client connections.

        This method runs in a separate thread and spawns new threads
        for each connected client.
        """
        while not self.shutdown_event.is_set():
            try:
                assert self.server_socket is not None
                self.server_socket.settimeout(1.0)
                try:
                    conn, addr = self.server_socket.accept()
                    thread = threading.Thread(
                        target=self.handle_client, args=(conn, addr), daemon=True
                    )
                    thread.start()
                except socket.timeout:
                    continue
            except Exception as e:
                if not self.shutdown_event.is_set():
                    logging.error(f"Error accepting client: {e}")

    def run(self) -> None:
        """
        Start the streaming server with visualization.

        This method starts the TCP server, processing threads, and
        Open3D visualization window.
        """
        # Create server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.server_socket.bind((self.server_config.host, self.server_config.port))
            self.server_socket.listen(self.server_config.max_connections)
            logging.info(
                f"Server listening on {self.server_config.host}:{self.server_config.port}"
            )

            # Create Open3D visualizer
            visualizer = o3d.visualization.Visualizer()  # type: ignore[attr-defined]
            visualizer.create_window(
                window_name="Stereo Point Cloud Viewer",
                width=self.server_config.window_width,
                height=self.server_config.window_height,
            )
            visualizer.add_geometry(self.point_cloud)
            self.visualizer = visualizer

            # Start background threads
            processor_thread = threading.Thread(
                target=self.process_image_queue, daemon=True
            )
            processor_thread.start()

            acceptor_thread = threading.Thread(target=self.accept_clients, daemon=True)
            acceptor_thread.start()

            # Main visualization loop
            logging.info("Visualization started. Press 'Q' to quit.")
            try:
                while not self.shutdown_event.is_set():
                    # Update visualization (thread-safe)
                    assert self.visualizer is not None
                    with self.point_cloud_lock:
                        self.visualizer.update_geometry(self.point_cloud)
                    if not self.visualizer.poll_events():
                        break
                    self.visualizer.update_renderer()
                    time.sleep(self.server_config.update_rate)
            except KeyboardInterrupt:
                logging.info("Interrupted by user")

        except Exception as e:
            logging.error(f"Server error: {e}")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """
        Gracefully shutdown the server.

        This method signals all threads to stop and cleans up resources.
        """
        logging.info("Shutting down server...")
        self.shutdown_event.set()

        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception as e:
                logging.warning(f"Error closing server socket: {e}")

        # Close visualizer
        if self.visualizer:
            try:
                self.visualizer.destroy_window()
            except Exception as e:
                logging.warning(f"Error closing visualizer: {e}")

        logging.info("Server shutdown complete")


def main():
    """
    Main entry point for the streaming server.

    This function sets up signal handlers and starts the server.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Stereo streaming server with point cloud visualization"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=5005, help="Server port (default: 5005)"
    )
    parser.add_argument(
        "--fov", type=float, default=60.0, help="Camera field of view in degrees"
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=0.1,
        help="Distance between stereo cameras in meters",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.02,
        help="Voxel size for downsampling point cloud",
    )

    args = parser.parse_args()

    # Create configurations
    camera_config = CameraConfig(fov=args.fov, baseline=args.baseline)
    recon_config = ReconstructionConfig()
    server_config = ServerConfig(
        host=args.host, port=args.port, voxel_downsample_size=args.voxel_size
    )

    # Create and start server
    server = StereoStreamingServer(camera_config, recon_config, server_config)

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logging.info(f"Received signal {signum}")
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run server
    server.run()


if __name__ == "__main__":
    main()
