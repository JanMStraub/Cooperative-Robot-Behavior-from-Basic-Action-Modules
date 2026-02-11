"""
ROS Bridge - Python Backend Client for ROS Motion Server
=========================================================

Connects to the ROSMotionClient TCP server (port 5020) running in Docker.
Provides a clean API for the Python operations to use without requiring rclpy.

Usage:
    from ros2.ROSBridge import ROSBridge

    bridge = ROSBridge()
    if bridge.connect():
        result = bridge.plan_and_execute(
            position={"x": 0.3, "y": 0.15, "z": 0.1},
            orientation={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
        )
"""

import json
import socket
import logging
import threading

logger = logging.getLogger(__name__)


class ROSBridge:
    """Client bridge to communicate with ROS Motion Server via TCP."""

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        """Get or create singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self, host=None, port=None):
        """
        Initialize the ROS bridge client.

        Args:
            host: ROS bridge server host (default from config)
            port: ROS bridge server port (default from config)
        """
        try:
            from config.ROS import ROS_BRIDGE_HOST, ROS_BRIDGE_PORT
            self._host = host or ROS_BRIDGE_HOST
            self._port = port or ROS_BRIDGE_PORT
        except ImportError:
            self._host = host or "127.0.0.1"
            self._port = port or 5020

        self._socket = None
        self._connected = False
        self._lock = threading.Lock()

    @property
    def is_connected(self):
        """Whether the bridge is connected to the ROS motion server."""
        return self._connected

    def connect(self, timeout=5.0):
        """
        Connect to the ROS motion server.

        Args:
            timeout: Connection timeout in seconds.

        Returns:
            True if connected successfully.
        """
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(timeout)
            self._socket.connect((self._host, self._port))
            self._connected = True
            logger.info(f"Connected to ROS bridge at {self._host}:{self._port}")

            # Verify connection with ping
            result = self._send_command({"command": "ping"})
            if result and result.get("success"):
                return True
            else:
                self.disconnect()
                return False

        except Exception as e:
            logger.error(f"Failed to connect to ROS bridge: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """Disconnect from the ROS motion server."""
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
        self._socket = None
        self._connected = False
        logger.info("Disconnected from ROS bridge")

    def _send_command(self, command, timeout=30.0):
        """
        Send a command to the ROS motion server and wait for response.

        Args:
            command: Dict command to send.
            timeout: Response timeout in seconds.

        Returns:
            Response dict or None on error.
        """
        if not self._connected or not self._socket:
            logger.error("Not connected to ROS bridge")
            return None

        with self._lock:
            try:
                # Send command
                msg = json.dumps(command) + "\n"
                self._socket.sendall(msg.encode("utf-8"))

                # Receive response
                self._socket.settimeout(timeout)
                buffer = ""
                while "\n" not in buffer:
                    data = self._socket.recv(4096)
                    if not data:
                        logger.error("Connection closed by server")
                        self._connected = False
                        return None
                    buffer += data.decode("utf-8")

                line = buffer.split("\n")[0]
                return json.loads(line)

            except socket.timeout:
                logger.error(f"Timeout waiting for response to {command.get('command')}")
                return None
            except Exception as e:
                logger.error(f"Error sending command: {e}")
                self._connected = False
                return None

    def plan_and_execute(self, position, orientation=None, planning_time=5.0, robot_id="Robot1"):
        """
        Plan and execute a motion to target pose for a specific robot.

        Args:
            position: Dict with x, y, z coordinates.
            orientation: Dict with x, y, z, w quaternion (default: identity).
            planning_time: Max planning time in seconds.
            robot_id: Robot namespace (e.g., "Robot1", "Robot2").

        Returns:
            Dict with success status and details.
        """
        if orientation is None:
            orientation = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}

        return self._send_command({
            "command": "plan_and_execute",
            "robot_id": robot_id,
            "position": position,
            "orientation": orientation,
            "planning_time": planning_time,
        })

    def plan_to_pose(self, position, orientation=None, planning_time=5.0, robot_id="Robot1"):
        """
        Plan (but don't execute) a trajectory to target pose for a specific robot.

        Args:
            position: Dict with x, y, z coordinates.
            orientation: Dict with x, y, z, w quaternion.
            planning_time: Max planning time in seconds.
            robot_id: Robot namespace (e.g., "Robot1", "Robot2").

        Returns:
            Dict with success status and trajectory info.
        """
        if orientation is None:
            orientation = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}

        return self._send_command({
            "command": "plan_to_pose",
            "robot_id": robot_id,
            "position": position,
            "orientation": orientation,
            "planning_time": planning_time,
        })

    def get_current_pose(self, robot_id="Robot1"):
        """
        Get current end-effector pose for a specific robot.

        Args:
            robot_id: Robot namespace (e.g., "Robot1", "Robot2").

        Returns:
            Dict with joint positions and names.
        """
        return self._send_command({
            "command": "get_current_pose",
            "robot_id": robot_id,
        })

    def control_gripper(self, position, robot_id="Robot1"):
        """
        Send gripper command via ROS for a specific robot.

        Args:
            position: Gripper position (0=closed, 0.014=open).
            robot_id: Robot namespace (e.g., "Robot1", "Robot2").

        Returns:
            Dict with success status.
        """
        return self._send_command({
            "command": "control_gripper",
            "robot_id": robot_id,
            "position": position,
        })

    def ping(self):
        """
        Check if the ROS bridge is responsive.

        Returns:
            True if bridge responds.
        """
        result = self._send_command({"command": "ping"}, timeout=5.0)
        return result is not None and result.get("success", False)
