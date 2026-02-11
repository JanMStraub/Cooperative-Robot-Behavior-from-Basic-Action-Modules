"""
ROS Motion Client - ROS 2 Node for MoveIt Plan-Only Integration
================================================================

This script runs INSIDE the Docker container as a ROS 2 node.
It connects to MoveIt 2 move_group for PLANNING ONLY, then publishes
the planned trajectory directly to Unity via a ROS topic.

Architecture (Plan-Only / "Bypass" approach):
    MoveIt's move_group expects a FollowJointTrajectory action server
    for execution (normally provided by ros2_control). Since Unity is
    our physics executor (not a real robot), we skip MoveIt's execution
    entirely:

    1. Python backend requests a plan via TCP:5020
    2. This node asks MoveIt to plan (planning_only=True)
    3. MoveIt returns a RobotTrajectory
    4. This node extracts the JointTrajectory and publishes it
       to /arm_controller/joint_trajectory
    5. Unity's ROSTrajectorySubscriber receives and executes it

    This avoids needing ros2_control or a fake hardware interface.

Protocol:
    Request:  JSON terminated by newline
    Response: JSON terminated by newline

Supported commands:
    plan_to_pose:     Plan trajectory to target pose (plan only, returns trajectory)
    plan_and_execute: Plan + publish trajectory to Unity topic
    get_current_pose: Get current end-effector pose from /joint_states
    control_gripper:  Publish gripper command to /gripper/command topic
    ping:             Health check
"""

import json
import socket
import threading
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ROSMotionClient] %(message)s")
logger = logging.getLogger(__name__)

# ROS 2 imports - only available inside Docker
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from moveit_msgs.action import MoveGroup
    from moveit_msgs.msg import (
        MotionPlanRequest,
        Constraints,
        PositionConstraint,
        OrientationConstraint,
        BoundingVolume,
        RobotState,
    )
    from geometry_msgs.msg import PoseStamped, Point, Quaternion, Vector3
    from shape_msgs.msg import SolidPrimitive
    from trajectory_msgs.msg import JointTrajectory
    from sensor_msgs.msg import JointState

    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    logger.warning("ROS 2 packages not available. Running in stub mode.")


class ROSMotionServer:
    """TCP server that accepts motion planning requests from Python backend.

    Uses MoveIt for planning only. Execution is done by publishing the
    planned trajectory to a ROS topic that Unity subscribes to.

    Supports multiple robots by maintaining separate MoveIt action clients
    and publishers for each robot namespace.
    """

    def __init__(self, host="0.0.0.0", port=5020):
        """Initialize the ROS motion server with multi-robot support."""
        self.host = host
        self.port = port
        self._running = False
        self._node = None

        # Multi-robot support: dict of robot_id -> clients/publishers
        self._move_group_clients = {}  # robot_id -> ActionClient
        self._trajectory_pubs = {}     # robot_id -> Publisher
        self._gripper_pubs = {}        # robot_id -> Publisher
        self._joint_state_subs = {}    # robot_id -> Subscription
        self._current_joint_states = {} # robot_id -> JointState msg
        self._last_planned_trajectories = {} # robot_id -> JointTrajectory

        if HAS_ROS:
            rclpy.init()
            self._node = rclpy.create_node("acrl_motion_client")

            # Initialize clients for both Robot1 and Robot2
            for robot_id in ["Robot1", "Robot2"]:
                self._initialize_robot(robot_id)

            logger.info("ROS 2 node initialized (plan-only mode, multi-robot support)")

    def _initialize_robot(self, robot_id: str):
        """Initialize MoveIt clients and publishers for a specific robot.

        Args:
            robot_id: Robot namespace (e.g., "Robot1", "Robot2")
        """
        logger.info(f"Initializing ROS clients for {robot_id}")

        # MoveIt action client (namespaced)
        # Action name is relative to the robot namespace
        self._move_group_clients[robot_id] = ActionClient(
            self._node, MoveGroup, f"/{robot_id}/move_action"
        )

        # NOTE: Do not block here waiting for the action server. MoveIt may take
        # 20-30s to fully initialize, and blocking in __init__ causes timeouts when
        # MoveIt is still starting. Instead, we check server availability at request
        # time in _call_move_group_plan with a generous timeout.
        logger.info(f"Registered move_action client for {robot_id} (non-blocking)")

        # Publisher: planned trajectories -> Unity's ROSTrajectorySubscriber
        self._trajectory_pubs[robot_id] = self._node.create_publisher(
            JointTrajectory, f"/{robot_id}/arm_controller/joint_trajectory", 10
        )

        # Publisher: gripper commands -> Unity's ROSGripperSubscriber
        self._gripper_pubs[robot_id] = self._node.create_publisher(
            JointState, f"/{robot_id}/gripper/command", 10
        )

        # Subscriber: joint states from Unity's ROSJointStatePublisher
        self._current_joint_states[robot_id] = None
        self._joint_state_subs[robot_id] = self._node.create_subscription(
            JointState,
            f"/{robot_id}/joint_states",
            lambda msg, rid=robot_id: self._joint_state_callback(rid, msg),
            10
        )

        # Cache the last planned trajectory for inspection
        self._last_planned_trajectories[robot_id] = None

        logger.info(f"Successfully initialized {robot_id}")

    def _joint_state_callback(self, robot_id: str, msg):
        """Cache latest joint state from Unity for a specific robot.

        Args:
            robot_id: Robot namespace
            msg: JointState message
        """
        # First time receiving joint states for this robot
        if self._current_joint_states[robot_id] is None:
            logger.info(
                f"Received first joint state from {robot_id}: "
                f"{len(msg.name)} joints ({', '.join(msg.name[:6])}...)"
            )
        self._current_joint_states[robot_id] = msg

    def start(self):
        """Start the TCP server and ROS spin thread."""
        self._running = True

        if HAS_ROS:
            self._ros_thread = threading.Thread(target=self._ros_spin, daemon=True)
            self._ros_thread.start()

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(5)
        server.settimeout(1.0)

        logger.info(f"Motion server listening on {self.host}:{self.port}")

        while self._running:
            try:
                client, addr = server.accept()
                logger.info(f"Client connected from {addr}")
                handler = threading.Thread(
                    target=self._handle_client, args=(client,), daemon=True
                )
                handler.start()
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Server error: {e}")

        server.close()

    def _ros_spin(self):
        """Spin ROS 2 node in background."""
        while self._running and rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.1)

    def _handle_client(self, client_socket):
        """Handle a single client connection."""
        buffer = ""
        try:
            while self._running:
                data = client_socket.recv(4096)
                if not data:
                    break

                buffer += data.decode("utf-8")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        request = json.loads(line)
                        response = self._process_request(request)
                    except json.JSONDecodeError as e:
                        response = {"success": False, "error": f"Invalid JSON: {e}"}

                    response_json = json.dumps(response) + "\n"
                    client_socket.sendall(response_json.encode("utf-8"))

        except Exception as e:
            logger.error(f"Client handler error: {e}")
        finally:
            client_socket.close()

    def _process_request(self, request):
        """Process a motion planning request.

        Request must include robot_id to route to correct MoveIt instance.
        """
        command = request.get("command", "")
        robot_id = request.get("robot_id", "Robot1")  # Default to Robot1 for backward compatibility

        logger.info(f"Processing command: {command} for {robot_id}")

        if not HAS_ROS:
            return {
                "success": False,
                "error": "ROS 2 not available (stub mode)",
            }

        # Validate robot_id
        if robot_id not in self._move_group_clients:
            return {
                "success": False,
                "error": f"Unknown robot_id: {robot_id}. Available: {list(self._move_group_clients.keys())}",
            }

        try:
            if command == "plan_to_pose":
                return self._plan_to_pose(request, robot_id)
            elif command == "plan_and_execute":
                return self._plan_and_publish(request, robot_id)
            elif command == "get_current_pose":
                return self._get_current_pose(robot_id)
            elif command == "control_gripper":
                return self._control_gripper(request, robot_id)
            elif command == "ping":
                return {"success": True, "message": "pong", "timestamp": time.time()}
            else:
                return {"success": False, "error": f"Unknown command: {command}"}

        except Exception as e:
            logger.error(f"Error processing {command} for {robot_id}: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _build_move_group_goal(self, request, robot_id):
        """Build a MoveGroup goal from request parameters.

        Sets planning_only=True so MoveIt plans but does NOT try to execute
        (which would require a FollowJointTrajectory action server).

        Args:
            request: Request dict with position, orientation, planning_time
            robot_id: Robot namespace to populate start_state from cached joint states
        """
        position = request.get("position", {})
        orientation = request.get("orientation", {})
        planning_time = request.get("planning_time", 5.0)

        goal = MoveGroup.Goal()
        goal.request = MotionPlanRequest()
        goal.request.group_name = "arm"
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = planning_time

        # CRITICAL: plan only, do NOT attempt execution
        # Without this, MoveIt tries to send to a FollowJointTrajectory
        # action server that doesn't exist (no ros2_control), and hangs.
        goal.planning_options.plan_only = True

        # Set explicit start_state from cached joint states to avoid race
        # condition where MoveIt hasn't processed the latest /joint_states yet.
        # IMPORTANT: Filter to only arm joints (joint_1..joint_6). Unity publishes
        # 8 joints including gripper_jaw1/jaw2, but MoveIt's "arm" planning group
        # only knows about the 6 arm joints. Including gripper joints makes the
        # start state "invalid" and causes OMPL to reject it.
        joint_state = self._current_joint_states.get(robot_id)
        if joint_state is not None:
            arm_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
            filtered_js = JointState()
            filtered_js.header = joint_state.header
            for name in arm_joint_names:
                if name in joint_state.name:
                    idx = list(joint_state.name).index(name)
                    filtered_js.name.append(name)
                    filtered_js.position.append(joint_state.position[idx])
                    if joint_state.velocity:
                        filtered_js.velocity.append(joint_state.velocity[idx])

            start_state = RobotState()
            start_state.joint_state = filtered_js
            goal.request.start_state = start_state

        # Set target pose constraint
        pose_goal = PoseStamped()
        pose_goal.header.frame_id = "base_link"
        pose_goal.pose.position = Point(
            x=position.get("x", 0.0),
            y=position.get("y", 0.0),
            z=position.get("z", 0.0),
        )
        pose_goal.pose.orientation = Quaternion(
            x=orientation.get("x", 0.0),
            y=orientation.get("y", 0.0),
            z=orientation.get("z", 0.0),
            w=orientation.get("w", 1.0),
        )

        # Position constraint
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = "base_link"
        pos_constraint.link_name = "ee_link"
        pos_constraint.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)

        bounding_vol = BoundingVolume()
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [0.01]  # 1cm tolerance
        bounding_vol.primitives.append(primitive)
        bounding_vol.primitive_poses.append(pose_goal.pose)
        pos_constraint.constraint_region = bounding_vol
        pos_constraint.weight = 1.0

        # Orientation constraint
        orient_constraint = OrientationConstraint()
        orient_constraint.header.frame_id = "base_link"
        orient_constraint.link_name = "ee_link"
        orient_constraint.orientation = pose_goal.pose.orientation
        orient_constraint.absolute_x_axis_tolerance = 0.1
        orient_constraint.absolute_y_axis_tolerance = 0.1
        orient_constraint.absolute_z_axis_tolerance = 0.1
        orient_constraint.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pos_constraint)
        constraints.orientation_constraints.append(orient_constraint)
        goal.request.goal_constraints.append(constraints)

        return goal

    def _wait_for_joint_states(self, robot_id: str, timeout: float = 5.0) -> bool:
        """Wait for initial joint states from Unity.

        Args:
            robot_id: Robot namespace
            timeout: Max time to wait in seconds

        Returns:
            True if joint states received, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._current_joint_states.get(robot_id) is not None:
                return True
            time.sleep(0.1)
        return False

    def _call_move_group_plan(self, goal, robot_id, planning_time=5.0):
        """Send a planning-only goal to MoveGroup and return the result.

        Args:
            goal: MoveGroup.Goal message
            robot_id: Robot namespace to route to correct MoveIt instance
            planning_time: Max planning time in seconds

        Returns:
            (trajectory, planning_time, error_msg) tuple.
            trajectory is a JointTrajectory or None on failure.
        """
        move_group_client = self._move_group_clients[robot_id]

        # Wait for joint states before planning
        if not self._wait_for_joint_states(robot_id, timeout=10.0):
            logger.warning(f"No joint states received for {robot_id} yet, planning may fail")

        if not move_group_client.wait_for_server(timeout_sec=15.0):
            return None, 0, f"MoveGroup action server not available for {robot_id}"

        # Send goal asynchronously and poll for completion.
        # NOTE: We must NOT call rclpy.spin_until_future_complete() here because
        # the _ros_spin thread is already spinning the node. Two threads spinning
        # the same node is not thread-safe in rclpy and causes deadlocks.
        # Instead, we poll the future and let the spin thread handle callbacks.
        future = move_group_client.send_goal_async(goal)
        deadline = time.time() + 10.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)

        if not future.done() or future.result() is None:
            return None, 0, "Goal send timed out"

        goal_handle = future.result()
        if not goal_handle.accepted:
            return None, 0, "Goal rejected by MoveGroup"

        result_future = goal_handle.get_result_async()
        result_deadline = time.time() + planning_time + 10.0
        while not result_future.done() and time.time() < result_deadline:
            time.sleep(0.05)

        if not result_future.done() or result_future.result() is None:
            return None, 0, "Planning timed out"

        result = result_future.result().result

        # MoveIt error codes (from moveit_msgs/msg/MoveItErrorCodes.msg)
        ERROR_CODES = {
            1: "SUCCESS",
            -1: "FAILURE",
            -2: "PLANNING_FAILED",
            -3: "INVALID_MOTION_PLAN",
            -4: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE",
            -5: "CONTROL_FAILED",
            -6: "UNABLE_TO_AQUIRE_SENSOR_DATA",
            -7: "TIMED_OUT",
            -10: "PREEMPTED",
            -11: "START_STATE_IN_COLLISION",
            -12: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
            -13: "GOAL_IN_COLLISION",
            -14: "GOAL_VIOLATES_PATH_CONSTRAINTS",
            -15: "GOAL_CONSTRAINTS_VIOLATED",
            -16: "INVALID_GROUP_NAME",
            -17: "INVALID_GOAL_CONSTRAINTS",
            -18: "INVALID_ROBOT_STATE",
            -19: "INVALID_LINK_NAME",
            -20: "INVALID_OBJECT_NAME",
            -21: "FRAME_TRANSFORM_FAILURE",
            -22: "COLLISION_CHECKING_UNAVAILABLE",
            -23: "ROBOT_STATE_STALE",
            -24: "SENSOR_INFO_STALE",
            -25: "COMMUNICATION_FAILURE",
            -26: "CRASH",
            -27: "ABORT",
            -28: "UNABLE_TO_SET_ROBOT_STATE",
            -29: "UNABLE_TO_SET_JOINT_GOAL",
            -31: "NO_IK_SOLUTION",
        }

        error_code = result.error_code.val
        if error_code == 1:
            trajectory = result.planned_trajectory.joint_trajectory
            self._last_planned_trajectories[robot_id] = trajectory
            logger.info(f"Planning succeeded for {robot_id}: {len(trajectory.points)} waypoints")
            return trajectory, result.planning_time, None
        else:
            error_name = ERROR_CODES.get(error_code, f"UNKNOWN_ERROR_{error_code}")
            error_msg = f"Planning failed: {error_name} (code {error_code})"
            logger.error(f"{robot_id}: {error_msg}")
            return None, 0, error_msg

    def _trajectory_to_dict(self, trajectory):
        """Convert a JointTrajectory message to a JSON-serializable dict."""
        points = []
        for pt in trajectory.points:
            point_dict = {
                "positions": list(pt.positions),
                "velocities": list(pt.velocities) if pt.velocities else [],
                "accelerations": list(pt.accelerations) if pt.accelerations else [],
                "time_from_start": pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9,
            }
            points.append(point_dict)

        return {
            "joint_names": list(trajectory.joint_names),
            "points": points,
        }

    def _plan_to_pose(self, request, robot_id):
        """Plan a trajectory to a target pose (plan only, no execution).

        Args:
            request: Request dict with position, orientation, planning_time
            robot_id: Robot namespace to route to correct MoveIt instance

        Returns:
            Dict with success status and trajectory info.
        """
        planning_time = request.get("planning_time", 5.0)
        goal = self._build_move_group_goal(request, robot_id)

        trajectory, plan_time, error = self._call_move_group_plan(goal, robot_id, planning_time)

        if trajectory is None:
            return {"success": False, "error": error, "robot_id": robot_id}

        return {
            "success": True,
            "robot_id": robot_id,
            "trajectory": self._trajectory_to_dict(trajectory),
            "trajectory_points": len(trajectory.points),
            "planning_time": plan_time,
        }

    def _plan_and_publish(self, request, robot_id):
        """Plan a trajectory, then publish it to the ROS topic for Unity.

        This is the main execution path:
        1. MoveIt plans (planning_only=True) for specific robot
        2. We extract the JointTrajectory from the result
        3. We publish it to /{robot_id}/arm_controller/joint_trajectory
        4. Unity's ROSTrajectorySubscriber picks it up and executes

        Args:
            request: Request dict with position, orientation, planning_time
            robot_id: Robot namespace to route to correct MoveIt instance

        Returns:
            Dict with success status and execution info.
        """
        planning_time = request.get("planning_time", 5.0)
        goal = self._build_move_group_goal(request, robot_id)

        trajectory, plan_time, error = self._call_move_group_plan(goal, robot_id, planning_time)

        if trajectory is None:
            return {"success": False, "error": error, "robot_id": robot_id}

        # Publish the planned trajectory to Unity (namespaced topic)
        trajectory_pub = self._trajectory_pubs[robot_id]
        trajectory_pub.publish(trajectory)

        logger.info(
            f"Published {len(trajectory.points)}-point trajectory to {robot_id} "
            f"(planned in {plan_time:.2f}s)"
        )

        return {
            "success": True,
            "robot_id": robot_id,
            "trajectory_points": len(trajectory.points),
            "planning_time": plan_time,
            "status": "published_to_unity",
        }

    def _get_current_pose(self, robot_id):
        """Get the current joint state from Unity via namespaced /joint_states topic.

        Args:
            robot_id: Robot namespace

        Returns:
            Dict with joint positions and names.
        """
        joint_state = self._current_joint_states.get(robot_id)

        if joint_state is None:
            return {
                "success": False,
                "error": f"No joint state received yet from Unity for {robot_id}",
                "robot_id": robot_id,
            }

        return {
            "success": True,
            "robot_id": robot_id,
            "joint_positions": list(joint_state.position),
            "joint_names": list(joint_state.name),
            "timestamp": time.time(),
        }

    def _control_gripper(self, request, robot_id):
        """Publish gripper command directly to namespaced /gripper/command topic.

        Publishes a JointState message where position[0] is the normalized
        gripper opening (0.0=closed, 1.0=open), matching what Unity's
        ROSGripperSubscriber expects.

        Args:
            request: Request dict with position
            robot_id: Robot namespace to route to correct gripper

        Returns:
            Dict with success status.
        """
        position = request.get("position", 0.0)

        msg = JointState()
        msg.name = ["gripper_jaw1_joint"]
        # ROSGripperSubscriber expects normalized 0-1, but also accepts
        # raw URDF values. Convert: 0.014m = fully open = 1.0 normalized
        normalized = min(position / 0.014, 1.0) if position > 1.0 else position
        msg.position = [normalized]

        gripper_pub = self._gripper_pubs[robot_id]
        gripper_pub.publish(msg)

        logger.info(f"Published gripper command to {robot_id}: position={normalized:.3f}")

        return {
            "success": True,
            "robot_id": robot_id,
            "gripper_position": normalized,
            "status": "published_to_unity",
        }

    def shutdown(self):
        """Clean shutdown."""
        self._running = False
        if HAS_ROS and self._node:
            self._node.destroy_node()
            rclpy.shutdown()


def main():
    """Entry point when running inside Docker."""
    server = ROSMotionServer(host="0.0.0.0", port=5020)
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
