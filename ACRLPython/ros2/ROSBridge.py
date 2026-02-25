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
            from config.ROS import ROS_BRIDGE_HOST, ROS_BRIDGE_PORT, ROS_TIMEOUT_BASE, ROS_TIMEOUT_PER_CANDIDATE
            self._host = host or ROS_BRIDGE_HOST
            self._port = port or ROS_BRIDGE_PORT
            self._timeout_base = ROS_TIMEOUT_BASE
            self._timeout_per_candidate = ROS_TIMEOUT_PER_CANDIDATE
        except ImportError:
            self._host = host or "127.0.0.1"
            self._port = port or 5020
            self._timeout_base = 5.0
            self._timeout_per_candidate = 0.5

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

    def validate_grasp_candidates(self, candidates, robot_id="Robot1", timeout=10.0):
        """
        Validate grasp candidates using MoveIt's IK service.

        Tests each candidate's grasp pose for IK reachability.

        Args:
            candidates: List of candidate dicts with 'position' and 'rotation' keys
            robot_id: Robot namespace (e.g., "Robot1", "Robot2")
            timeout: Command timeout in seconds (default 10s for batch processing)

        Returns:
            Dict with success status and validation results:
            {
                "success": True,
                "results": [(is_valid, quality_score), ...],
                "candidates_validated": N
            }
        """
        cmd = {
            "command": "validate_grasp_candidates",
            "robot_id": robot_id,
            "candidates": candidates,
        }

        # Adjust timeout based on number of candidates
        adjusted_timeout = max(timeout, self._timeout_base + len(candidates) * self._timeout_per_candidate)

        return self._send_command(cmd, timeout=adjusted_timeout)

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

    def plan_and_execute(
        self,
        position,
        orientation=None,
        planning_time=5.0,
        robot_id="Robot1",
        max_velocity_scaling=0.0,
        max_acceleration_scaling=0.0,
    ):
        """
        Plan and execute a motion to target pose for a specific robot.

        Args:
            position: Dict with x, y, z coordinates (Unity world space).
            orientation: Dict with x, y, z, w quaternion. If None, MoveIt plans
                         to the position with any feasible orientation.
            planning_time: Max planning time in seconds.
            robot_id: Robot namespace (e.g., "Robot1", "Robot2").
            max_velocity_scaling: MoveIt velocity scaling factor (0.0 = default = 1.0).
                Use values < 1.0 for slow, smooth descent motions (e.g. 0.3 for grasp approach).
            max_acceleration_scaling: MoveIt acceleration scaling factor (0.0 = default = 1.0).

        Returns:
            Dict with success status and details.
        """
        cmd = {
            "command": "plan_and_execute",
            "robot_id": robot_id,
            "position": position,
            "planning_time": planning_time,
        }
        if orientation is not None:
            cmd["orientation"] = orientation
        if max_velocity_scaling > 0.0:
            cmd["max_velocity_scaling"] = max_velocity_scaling
        if max_acceleration_scaling > 0.0:
            cmd["max_acceleration_scaling"] = max_acceleration_scaling

        return self._send_command(cmd)

    def plan_cartesian_descent(
        self,
        position,
        orientation=None,
        robot_id="Robot1",
        max_velocity_scaling=0.3,
        max_acceleration_scaling=0.3,
    ):
        """
        Plan and execute a straight-line Cartesian descent to a target position.

        Uses MoveIt's GetCartesianPath service to constrain the end-effector to
        follow a straight line, preventing wrist joints from rotating to an
        alternate IK solution and causing lateral gripper offset.

        Falls back automatically to free-space planning if the Cartesian path
        is blocked (e.g. collision avoidance needed).

        Args:
            position: Dict with x, y, z in Unity world coordinates.
            orientation: Dict with x, y, z, w quaternion. Should be the same
                         orientation used for the preceding pre-grasp move.
            robot_id: Robot namespace (e.g., "Robot1", "Robot2").
            max_velocity_scaling: Velocity scaling (default 0.3 for slow descent).
            max_acceleration_scaling: Acceleration scaling (default 0.3).

        Returns:
            Dict with success status and details.
        """
        cmd = {
            "command": "plan_cartesian_descent",
            "robot_id": robot_id,
            "position": position,
            "max_velocity_scaling": max_velocity_scaling,
            "max_acceleration_scaling": max_acceleration_scaling,
        }
        if orientation is not None:
            cmd["orientation"] = orientation

        return self._send_command(cmd)

    def plan_to_pose(self, position, orientation=None, planning_time=5.0, robot_id="Robot1"):
        """
        Plan (but don't execute) a trajectory to target pose for a specific robot.

        Args:
            position: Dict with x, y, z coordinates.
            orientation: Dict with x, y, z, w quaternion. If None, MoveIt plans
                         to the position with any feasible orientation.
            planning_time: Max planning time in seconds.
            robot_id: Robot namespace (e.g., "Robot1", "Robot2").

        Returns:
            Dict with success status and trajectory info.
        """
        cmd = {
            "command": "plan_to_pose",
            "robot_id": robot_id,
            "position": position,
            "planning_time": planning_time,
        }
        if orientation is not None:
            cmd["orientation"] = orientation

        return self._send_command(cmd)

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

    def plan_multi_waypoint(self, waypoints, robot_id="Robot1", planning_time=10.0):
        """
        Plan and execute a multi-waypoint trajectory.

        Sends each waypoint as a sequential plan_and_execute to the ROS motion
        server. The server handles trajectory concatenation.

        Args:
            waypoints: List of position dicts with x, y, z keys.
            robot_id: Robot namespace (e.g., "Robot1", "Robot2").
            planning_time: Max planning time per waypoint in seconds.

        Returns:
            Dict with success status and details.
        """
        cmd = {
            "command": "plan_multi_waypoint",
            "robot_id": robot_id,
            "waypoints": waypoints,
            "planning_time": planning_time,
        }
        return self._send_command(cmd, timeout=planning_time * len(waypoints) + 10)

    def plan_orientation_change(self, orientation, robot_id="Robot1", planning_time=5.0):
        """
        Plan and execute an orientation change while maintaining position.

        Args:
            orientation: Dict with roll, pitch, yaw in degrees (converted to
                         quaternion by the ROS motion server).
            robot_id: Robot namespace (e.g., "Robot1", "Robot2").
            planning_time: Max planning time in seconds.

        Returns:
            Dict with success status and details.
        """
        cmd = {
            "command": "plan_orientation_change",
            "robot_id": robot_id,
            "orientation": orientation,
            "planning_time": planning_time,
        }
        return self._send_command(cmd)

    def plan_grasp(self, object_id, robot_id="Robot1", approach="auto", planning_time=10.0):
        """
        Plan and execute a grasp on an object via MoveIt.

        The ROS motion server queries the object pose from the planning scene
        and plans a grasp approach, grasp, and retreat sequence.

        Args:
            object_id: ID of the object to grasp.
            robot_id: Robot namespace (e.g., "Robot1", "Robot2").
            approach: Approach direction ("auto", "top", "front", "side").
            planning_time: Max planning time in seconds.

        Returns:
            Dict with success status and details.
        """
        cmd = {
            "command": "plan_grasp",
            "robot_id": robot_id,
            "object_id": object_id,
            "approach": approach,
            "planning_time": planning_time,
        }
        return self._send_command(cmd, timeout=planning_time + 10)

    def plan_return_to_start(self, robot_id="Robot1", planning_time=5.0):
        """
        Plan and execute a return to the robot's start/home configuration.

        Args:
            robot_id: Robot namespace (e.g., "Robot1", "Robot2").
            planning_time: Max planning time in seconds.

        Returns:
            Dict with success status and details.
        """
        cmd = {
            "command": "plan_return_to_start",
            "robot_id": robot_id,
            "planning_time": planning_time,
        }
        return self._send_command(cmd)

    def ping(self):
        """
        Check if the ROS bridge is responsive.

        Returns:
            True if bridge responds.
        """
        result = self._send_command({"command": "ping"}, timeout=5.0)
        return result is not None and result.get("success", False)
