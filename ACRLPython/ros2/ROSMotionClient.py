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
import signal
import socket
import threading
import time

try:
    from core.LoggingSetup import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

try:
    from config.ROS import MOVEIT_PLANNING_TIME, MOVEIT_PLANNING_ATTEMPTS, MOVEIT_GOAL_TOLERANCE
except ImportError:
    MOVEIT_PLANNING_TIME = 5.0
    MOVEIT_PLANNING_ATTEMPTS = 10
    MOVEIT_GOAL_TOLERANCE = 0.01

try:
    from config.Robot import ROBOT_BASE_POSITIONS
except ImportError:
    ROBOT_BASE_POSITIONS = {
        "Robot1": (-0.475, 0.0, 0.0),
        "Robot2": (0.475, 0.0, 0.0),
    }

# ROS 2 imports - only available inside Docker
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from moveit_msgs.action import MoveGroup
    from moveit_msgs.msg import (
        MotionPlanRequest,
        Constraints,
        JointConstraint,
        PositionConstraint,
        OrientationConstraint,
        BoundingVolume,
        RobotState,
        PlanningScene,
        CollisionObject,
    )
    from moveit_msgs.srv import GetPositionIK, GetCartesianPath
    from geometry_msgs.msg import PoseStamped, Point, Quaternion, Vector3, Pose
    from shape_msgs.msg import SolidPrimitive
    from trajectory_msgs.msg import JointTrajectory
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String

    HAS_ROS = True
    # GetCartesianPath gained velocity/acceleration scaling fields in moveit_msgs ~2.3.
    # Probe once at import time so _plan_cartesian_descent can set them unconditionally.
    _CARTESIAN_HAS_SCALING = hasattr(GetCartesianPath.Request(), "max_velocity_scaling_factor")
except ImportError:
    HAS_ROS = False
    _CARTESIAN_HAS_SCALING = False
    logger.warning("ROS 2 packages not available. Running in stub mode.")
    rclpy = None  # type: ignore[assignment]
    Node = None  # type: ignore[assignment,misc]
    ActionClient = None  # type: ignore[assignment,misc]
    MoveGroup = None  # type: ignore[assignment,misc]
    MotionPlanRequest = None  # type: ignore[assignment,misc]
    Constraints = None  # type: ignore[assignment,misc]
    PositionConstraint = None  # type: ignore[assignment,misc]
    OrientationConstraint = None  # type: ignore[assignment,misc]
    BoundingVolume = None  # type: ignore[assignment,misc]
    RobotState = None  # type: ignore[assignment,misc]
    PlanningScene = None  # type: ignore[assignment,misc]
    CollisionObject = None  # type: ignore[assignment,misc]
    JointConstraint = None  # type: ignore[assignment,misc]
    GetPositionIK = None  # type: ignore[assignment,misc]
    GetCartesianPath = None  # type: ignore[assignment,misc]
    PoseStamped = None  # type: ignore[assignment,misc]
    Point = None  # type: ignore[assignment,misc]
    Quaternion = None  # type: ignore[assignment,misc]
    Vector3 = None  # type: ignore[assignment,misc]
    Pose = None  # type: ignore[assignment,misc]
    SolidPrimitive = None  # type: ignore[assignment,misc]
    JointTrajectory = None  # type: ignore[assignment,misc]
    JointState = None  # type: ignore[assignment,misc]
    String = None  # type: ignore[assignment,misc]


class ROSMotionServer:
    """TCP server that accepts motion planning requests from Python backend.

    Uses MoveIt for planning only. Execution is done by publishing the
    planned trajectory to a ROS topic that Unity subscribes to.

    Supports multiple robots by maintaining separate MoveIt action clients
    and publishers for each robot namespace.
    """

    # Robot base positions and rotations in Unity world coordinates.
    # Positions sourced from ROBOT_BASE_POSITIONS in config/Robot.py.
    # Robot1: Unity rotation (0, 0, 0, -1) = 360° = 0° effective rotation - facing forward (+Z in Unity)
    # Robot2: Unity rotation (0, 1, 0, 0) = 180° around Y - facing backward (-Z in Unity)
    ROBOT_BASE_TRANSFORMS = {
        robot_id: {"position": pos, "y_rotation": 0.0 if robot_id == "Robot1" else 180.0}
        for robot_id, pos in ROBOT_BASE_POSITIONS.items()
    }

    def __init__(self, host="0.0.0.0", port=5020):
        """Initialize the ROS motion server with multi-robot support."""
        self.host = host
        self.port = port
        self._running = False
        self._node = None

        # Multi-robot support: dict of robot_id -> clients/publishers
        self._move_group_clients = {}  # robot_id -> ActionClient
        self._ik_service_clients = {}  # robot_id -> Service Client for IK
        self._cartesian_path_clients = {}  # robot_id -> Service Client for Cartesian paths
        self._trajectory_pubs = {}  # robot_id -> Publisher
        self._gripper_pubs = {}  # robot_id -> Publisher
        self._joint_state_subs = {}  # robot_id -> Subscription
        self._current_joint_states = {}  # robot_id -> JointState msg
        self._last_planned_trajectories = {}  # robot_id -> JointTrajectory
        self._trajectory_feedback = {}  # robot_id -> last feedback status string
        self._trajectory_feedback_event = {}  # robot_id -> threading.Event

        if HAS_ROS:
            rclpy.init()
            self._node = rclpy.create_node("acrl_motion_client")

            # Per-robot planning scene publishers (namespaced, one per robot)
            # move_group runs under /{robot_id}/ namespace, so its planning scene
            # topic is /{robot_id}/planning_scene.
            self._planning_scene_pubs = {}

            # Initialize clients for both Robot1 and Robot2
            for robot_id in ["Robot1", "Robot2"]:
                self._initialize_robot(robot_id)

            logger.info("ROS 2 node initialized (plan-only mode, multi-robot support)")

            # Publish ground plane to each robot's planning scene so MoveIt
            # won't plan trajectories that pass through the table surface.
            # Brief sleep so the publisher has time to connect to move_group's
            # planning scene subscriber before we send the first message.
            time.sleep(0.5)
            for robot_id in ["Robot1", "Robot2"]:
                self._publish_ground_plane(robot_id)

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

        # MoveIt IK service client (for grasp candidate validation)
        self._ik_service_clients[robot_id] = self._node.create_client(
            GetPositionIK, f"/{robot_id}/compute_ik"
        )

        # MoveIt Cartesian path service client (for straight-line descents)
        self._cartesian_path_clients[robot_id] = self._node.create_client(
            GetCartesianPath, f"/{robot_id}/compute_cartesian_path"
        )

        # NOTE: Do not block here waiting for the action server. MoveIt may take
        # 20-30s to fully initialize, and blocking in __init__ causes timeouts when
        # MoveIt is still starting. Instead, we check server availability at request
        # time in _call_move_group_plan with a generous timeout.
        logger.info(f"Registered move_action client and IK service for {robot_id} (non-blocking)")

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
            10,
        )

        # Cache the last planned trajectory for inspection
        self._last_planned_trajectories[robot_id] = None

        # Execution feedback from Unity's ROSTrajectorySubscriber
        self._trajectory_feedback[robot_id] = None
        self._trajectory_feedback_event[robot_id] = threading.Event()
        self._node.create_subscription(
            String,
            f"/{robot_id}/arm_controller/feedback",
            lambda msg, rid=robot_id: self._feedback_callback(rid, msg),
            10,
        )

        logger.info(f"Successfully initialized {robot_id}")

    def _publish_ground_plane(self, robot_id: str):
        """Publish a ground/table collision object to a robot's MoveIt planning scene.

        The table surface in Unity is at Y=0 (same as robot base_link origin).
        In ROS base_link frame (Z-up), Z=0 is the floor. We add a large flat box
        at Z=-0.01 (just below the surface) so MoveIt treats the table surface as
        a collision boundary and plans paths that stay above it.

        Published to /{robot_id}/planning_scene once at startup. MoveIt's planning
        scene monitor picks it up and applies it to all subsequent planning requests.

        Args:
            robot_id: Robot namespace (e.g., "Robot1", "Robot2")
        """
        if not HAS_ROS:
            return

        # Create publisher if not already created for this robot
        if robot_id not in self._planning_scene_pubs:
            self._planning_scene_pubs[robot_id] = self._node.create_publisher(
                PlanningScene, f"/{robot_id}/planning_scene", 10
            )

        scene = PlanningScene()
        scene.is_diff = True  # Incremental update, not a full scene replacement

        # Ground plane: 2m x 2m x 10cm slab in base_link frame.
        # Top face at Z=-0.10 (10cm below table surface).
        # Prevents trajectories from going far below the table while giving
        # enough room for the arm links (link_2, link_3) to sweep low when
        # reaching laterally at grasp height. At full extension (Y≈-0.27),
        # the elbow can dip to Z≈-0.07 — a ground plane at Z=-0.05 caused
        # "Unable to sample any valid states for goal tree" because the arm
        # geometry collided with the collision box at every IK solution.
        ground = CollisionObject()
        ground.header.frame_id = "base_link"
        ground.id = "ground_plane"
        ground.operation = CollisionObject.ADD

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [2.0, 2.0, 0.10]  # x, y, z in metres

        box_pose = Pose()
        box_pose.position.x = 0.0
        box_pose.position.y = 0.0
        box_pose.position.z = -0.15  # Top face at Z=-0.10, 10cm below table surface
        box_pose.orientation.w = 1.0

        ground.primitives = [box]
        ground.primitive_poses = [box_pose]

        scene.world.collision_objects = [ground]
        self._planning_scene_pubs[robot_id].publish(scene)
        logger.info(f"Published ground plane collision object to {robot_id} planning scene")

    def _transform_world_to_local(self, world_position: dict, robot_id: str) -> dict:
        """Transform Unity world coordinates to ROS base_link coordinates.

        Performs three transformations:
        1. Translation: Unity world → robot-centered Unity coordinates
        2. Rotation: Apply robot's Y-axis rotation (for Robot2's 180° flip)
        3. Axis conversion: Unity (Y-up) → ROS (Z-up)

        Unity coordinate system: X=right, Y=up, Z=forward (left-handed)
        ROS coordinate system: X=forward, Y=left, Z=up (right-handed)

        Example for Robot2 at Unity world (0.475, 0, 0):
        - User sends Unity world: (0.2, 0.15, 0)
        - Translate: (0.2-0.475, 0.15, 0) = (-0.275, 0.15, 0) in Unity local
        - Rotate 180°: (0.275, 0.15, 0) in Unity local
        - Convert to ROS: (Z, -X, Y) = (0, -0.275, 0.15) in ROS base_link

        Args:
            world_position: Dict with x, y, z in Unity world coordinates (Y-up)
            robot_id: Robot namespace (e.g., "Robot1", "Robot2")

        Returns:
            Dict with x, y, z in ROS base_link coordinates (Z-up)
        """
        import math

        if robot_id not in self.ROBOT_BASE_TRANSFORMS:
            logger.warning(
                f"Unknown robot_id '{robot_id}' for coordinate transform. "
                f"Using coordinates as-is. Known robots: {list(self.ROBOT_BASE_TRANSFORMS.keys())}"
            )
            return world_position

        transform = self.ROBOT_BASE_TRANSFORMS[robot_id]
        base_x, base_y, base_z = transform["position"]
        y_rotation_deg = transform["y_rotation"]

        # Step 1: Translate to robot-centered Unity coordinates
        unity_local_x = world_position.get("x", 0.0) - base_x
        unity_local_y = world_position.get("y", 0.0) - base_y
        unity_local_z = world_position.get("z", 0.0) - base_z

        # Step 2: Apply robot's Y-rotation in Unity space (if any)
        y_rotation_rad = math.radians(y_rotation_deg)
        cos_theta = math.cos(y_rotation_rad)
        sin_theta = math.sin(y_rotation_rad)

        rotated_x = cos_theta * unity_local_x + sin_theta * unity_local_z
        rotated_y = unity_local_y
        rotated_z = -sin_theta * unity_local_x + cos_theta * unity_local_z

        # Step 3: Convert from Unity (Y-up, left-handed) to ROS (Z-up, right-handed)
        # Unity axes: X=right, Y=up, Z=forward
        # ROS axes:   X=forward, Y=left, Z=up
        # Conversion: Unity (X, Y, Z) → ROS (Z, -X, Y)
        ros_x = rotated_z  # Unity Z (forward) → ROS X (forward)
        ros_y = -rotated_x  # Unity X (right) → ROS -Y (since ROS Y is left)
        ros_z = rotated_y  # Unity Y (up) → ROS Z (up)

        local_position = {
            "x": ros_x,
            "y": ros_y,
            "z": ros_z,
        }

        return local_position

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

    def _feedback_callback(self, robot_id: str, msg):
        """Handle trajectory execution feedback from Unity's ROSTrajectorySubscriber.

        Args:
            robot_id: Robot namespace
            msg: String message containing JSON feedback with 'status' field
        """
        try:
            data = json.loads(msg.data)
            status = data.get("status", "")
            self._trajectory_feedback[robot_id] = status
            if status in ("completed", "aborted", "rejected"):
                self._trajectory_feedback_event[robot_id].set()
        except (json.JSONDecodeError, Exception):
            pass

    def _wait_for_trajectory_completion(self, robot_id: str, timeout: float = 30.0) -> bool:
        """Wait until Unity reports the current trajectory as completed or aborted.

        Args:
            robot_id: Robot namespace
            timeout: Maximum seconds to wait

        Returns:
            True if completed successfully, False if aborted/rejected/timed out.
        """
        event = self._trajectory_feedback_event[robot_id]
        event.clear()
        signalled = event.wait(timeout=timeout)
        if not signalled:
            logger.warning(f"{robot_id}: Timed out waiting for trajectory completion")
            return False
        status = self._trajectory_feedback.get(robot_id, "")
        return status == "completed"

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
        """Handle a single TCP client connection for the lifetime of that connection.

        Reads newline-delimited JSON requests, routes each through
        _process_request(), and sends the JSON response back. Runs in its own
        daemon thread so multiple clients can be served concurrently.

        Args:
            client_socket: The accepted socket returned by server.accept().
        """
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
        robot_id = request.get(
            "robot_id", "Robot1"
        )  # Default to Robot1 for backward compatibility

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
            elif command == "plan_multi_waypoint":
                return self._plan_multi_waypoint(request, robot_id)
            elif command == "plan_orientation_change":
                return self._plan_orientation_change(request, robot_id)
            elif command == "plan_return_to_start":
                return self._plan_return_to_start(request, robot_id)
            elif command == "validate_grasp_candidates":
                return self._validate_grasp_candidates(request, robot_id)
            elif command == "plan_cartesian_descent":
                return self._plan_cartesian_descent(request, robot_id)
            elif command == "ping":
                return {"success": True, "message": "pong", "timestamp": time.time()}
            else:
                return {"success": False, "error": f"Unknown command: {command}"}

        except Exception as e:
            logger.error(
                f"Error processing {command} for {robot_id}: {e}", exc_info=True
            )
            return {"success": False, "error": str(e)}

    def _build_move_group_goal(self, request, robot_id):
        """Build a MoveGroup goal from request parameters.

        Sets planning_only=True so MoveIt plans but does NOT try to execute
        (which would require a FollowJointTrajectory action server).

        When orientation is not provided, only a position constraint is added,
        allowing MoveIt to reach the target with any feasible orientation.

        Args:
            request: Request dict with position, orientation, planning_time
            robot_id: Robot namespace to populate start_state from cached joint states
        """
        position = request.get("position", {})
        orientation = request.get("orientation")
        planning_time = request.get("planning_time", MOVEIT_PLANNING_TIME)

        # Log incoming world position for debugging coordinate transform issues
        logger.info(
            f"[GRASP_DEBUG] {robot_id} raw Unity world position: "
            f"x={position.get('x',0):.3f}, y={position.get('y',0):.3f}, z={position.get('z',0):.3f}"
        )

        # Transform world coordinates to robot-local base_link coordinates
        position = self._transform_world_to_local(position, robot_id)

        logger.info(
            f"[GRASP_DEBUG] {robot_id} ROS base_link position: "
            f"x={position.get('x',0):.3f}, y={position.get('y',0):.3f}, z={position.get('z',0):.3f}"
        )

        goal = MoveGroup.Goal()
        goal.request = MotionPlanRequest()
        goal.request.group_name = "arm"
        goal.request.num_planning_attempts = MOVEIT_PLANNING_ATTEMPTS
        goal.request.allowed_planning_time = planning_time

        # Optional velocity/acceleration scaling (0.0 = MoveIt default = no scaling).
        # Use values < 1.0 for the descent phase to produce slow, smooth approach trajectories.
        vel_scaling = request.get("max_velocity_scaling", 0.0)
        acc_scaling = request.get("max_acceleration_scaling", 0.0)
        if vel_scaling > 0.0:
            goal.request.max_velocity_scaling_factor = vel_scaling
        if acc_scaling > 0.0:
            goal.request.max_acceleration_scaling_factor = acc_scaling

        # Set workspace bounds so OMPL knows the planning volume
        goal.request.workspace_parameters.header.frame_id = "base_link"
        goal.request.workspace_parameters.min_corner = Vector3(x=-1.0, y=-1.0, z=-1.0)
        goal.request.workspace_parameters.max_corner = Vector3(x=1.0, y=1.0, z=1.0)

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
        #
        # Also clamp each joint to its URDF bounds before submitting. Unity's
        # ArticulationBody physics can overshoot by a small amount at the end of
        # a trajectory (especially after settle timeout), causing OMPL to reject
        # the start state entirely with "invalid bounds" even for sub-milliradian
        # violations — which aborts planning with error code 99999.
        _ARM_JOINT_LIMITS = {
            "joint_1": (-2.9670597283903604, 2.9670597283903604),
            "joint_2": (-0.7330382858376184, 1.5707963267948966),
            "joint_3": (-1.5533430342749532, 0.9075712110370514),
            "joint_4": (-3.141592653589793, 3.141592653589793),
            "joint_5": (-1.8325957145940461, 1.8325957145940461),
            "joint_6": (-3.141592653589793, 3.141592653589793),
        }
        joint_state = self._current_joint_states.get(robot_id)
        if joint_state is not None:
            filtered_js = JointState()
            filtered_js.header = joint_state.header
            for name, (lower, upper) in _ARM_JOINT_LIMITS.items():
                if name in joint_state.name:
                    idx = list(joint_state.name).index(name)
                    raw = joint_state.position[idx]
                    clamped = max(lower, min(upper, raw))
                    if abs(clamped - raw) > 1e-6:
                        logger.warning(
                            f"{robot_id} {name} position {raw:.6f} rad out of bounds "
                            f"[{lower:.4f}, {upper:.4f}] — clamped to {clamped:.6f} for MoveIt start state"
                        )
                    filtered_js.name.append(name)
                    filtered_js.position.append(clamped)
                    if joint_state.velocity:
                        filtered_js.velocity.append(0.0)  # zero velocity at start

            start_state = RobotState()
            start_state.is_diff = True  # Differential update: only override listed joints,
            # keep remaining joints from MoveIt's current state.  Without is_diff=True,
            # unlisted joints default to 0 which can put the robot in collision with
            # itself or the ground plane — triggering UNKNOWN_ERROR_99999.
            start_state.joint_state = filtered_js
            goal.request.start_state = start_state

        # Build pose for constraints
        pose_goal = PoseStamped()
        pose_goal.header.frame_id = "base_link"
        pose_goal.pose.position = Point(
            x=position.get("x", 0.0),
            y=position.get("y", 0.0),
            z=position.get("z", 0.0),
        )

        # Use provided orientation or default for pose
        if orientation:
            pose_goal.pose.orientation = Quaternion(
                x=orientation.get("x", 0.0),
                y=orientation.get("y", 0.0),
                z=orientation.get("z", 0.0),
                w=orientation.get("w", 1.0),
            )
        else:
            pose_goal.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        # Position constraint
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = "base_link"
        pos_constraint.link_name = "ee_link"
        pos_constraint.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)

        bounding_vol = BoundingVolume()
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [MOVEIT_GOAL_TOLERANCE]  # position goal tolerance (from config.ROS)
        bounding_vol.primitives.append(primitive)
        bounding_vol.primitive_poses.append(pose_goal.pose)
        pos_constraint.constraint_region = bounding_vol
        pos_constraint.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pos_constraint)

        # Only add orientation constraint when explicitly requested
        if orientation:
            orient_constraint = OrientationConstraint()
            orient_constraint.header.frame_id = "base_link"
            orient_constraint.link_name = "ee_link"
            orient_constraint.orientation = pose_goal.pose.orientation
            orient_constraint.absolute_x_axis_tolerance = 0.1
            orient_constraint.absolute_y_axis_tolerance = 0.1
            orient_constraint.absolute_z_axis_tolerance = 0.1
            orient_constraint.weight = 1.0
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

        # Re-publish ground plane before each planning call.
        # MoveIt may reset its planning scene on reconnect or after a long idle,
        # so we ensure the table collision object is always present.
        self._publish_ground_plane(robot_id)

        # Wait for joint states before planning
        if not self._wait_for_joint_states(robot_id, timeout=10.0):
            logger.warning(
                f"No joint states received for {robot_id} yet, planning may fail"
            )

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

            logger.info(
                f"Planning succeeded for {robot_id}: {len(trajectory.points)} waypoints"
            )

            return trajectory, result.planning_time, None
        else:
            error_name = ERROR_CODES.get(error_code, f"UNKNOWN_ERROR_{error_code}")
            error_msg = f"Planning failed: {error_name} (code {error_code})"
            logger.error(f"{robot_id}: {error_msg}")
            return None, 0, error_msg

    def _trajectory_to_dict(self, trajectory):
        """Convert a JointTrajectory ROS message to a JSON-serializable dict.

        Extracts positions, velocities, accelerations, and time stamps from
        each trajectory point so the result can be sent over the TCP wire to
        the Python backend or logged.

        Args:
            trajectory: JointTrajectory message (from MoveIt plan result or
                        GetCartesianPath response).

        Returns:
            Dict with keys:
                - joint_names (List[str]): Ordered list of joint name strings.
                - points (List[Dict]): One dict per waypoint with keys
                  'positions', 'velocities', 'accelerations' (all List[float])
                  and 'time_from_start' (float, seconds).
        """
        points = []
        for pt in trajectory.points:
            point_dict = {
                "positions": list(pt.positions),
                "velocities": list(pt.velocities) if pt.velocities else [],
                "accelerations": list(pt.accelerations) if pt.accelerations else [],
                "time_from_start": pt.time_from_start.sec
                + pt.time_from_start.nanosec * 1e-9,
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
        planning_time = request.get("planning_time", MOVEIT_PLANNING_TIME)
        goal = self._build_move_group_goal(request, robot_id)

        trajectory, plan_time, error = self._call_move_group_plan(
            goal, robot_id, planning_time
        )

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
        planning_time = request.get("planning_time", MOVEIT_PLANNING_TIME)
        goal = self._build_move_group_goal(request, robot_id)

        trajectory, plan_time, error = self._call_move_group_plan(
            goal, robot_id, planning_time
        )

        if trajectory is None:
            return {"success": False, "error": error, "robot_id": robot_id}

        # Publish the planned trajectory to Unity (namespaced topic)
        trajectory_pub = self._trajectory_pubs[robot_id]

        # Clear any stale feedback before publishing so the wait below starts fresh
        self._trajectory_feedback_event[robot_id].clear()
        self._trajectory_feedback[robot_id] = None

        trajectory_pub.publish(trajectory)

        logger.info(
            f"Published {len(trajectory.points)}-point trajectory to {robot_id} "
            f"(planned in {plan_time:.2f}s)"
        )

        # Estimate execution timeout: trajectory duration * speed_scale_factor + buffer.
        # Unity's ROSTrajectorySubscriber runs at 0.5x speed scaling by default,
        # so actual wall time ≈ 2x the trajectory time_from_start of the last point.
        # Add 10s fixed buffer to cover:
        #   - Unity physics settle wait (up to 1.5s per ROSTrajectorySubscriber)
        #   - ROS topic round-trip latency for feedback message
        #   - Short MoveIt plans (near-zero traj_duration) that still take several
        #     seconds to execute due to speed scaling and settle
        # Minimum 15s ensures even instantaneous-plan trajectories have enough time.
        if trajectory.points:
            last_pt = trajectory.points[-1]
            traj_duration = last_pt.time_from_start.sec + last_pt.time_from_start.nanosec * 1e-9
        else:
            traj_duration = 5.0
        execution_timeout = max(15.0, traj_duration * 2.5 + 10.0)

        completed = self._wait_for_trajectory_completion(robot_id, timeout=execution_timeout)
        if not completed:
            logger.warning(
                f"{robot_id}: Trajectory completion not confirmed within {execution_timeout:.1f}s "
                "(may have timed out or been aborted)"
            )

        return {
            "success": completed,
            "robot_id": robot_id,
            "trajectory_points": len(trajectory.points),
            "planning_time": plan_time,
            "status": "completed" if completed else "timeout",
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

        logger.info(
            f"Published gripper command to {robot_id}: position={normalized:.3f}"
        )

        return {
            "success": True,
            "robot_id": robot_id,
            "gripper_position": normalized,
            "status": "published_to_unity",
        }

    def _plan_multi_waypoint(self, request, robot_id):
        """Plan and publish trajectories for multiple waypoints sequentially.

        Each waypoint is planned as a separate plan_and_execute call.
        The robot moves through each waypoint in order.

        Args:
            request: Request dict with waypoints list and planning_time.
            robot_id: Robot namespace.

        Returns:
            Dict with success status and total planning time.
        """
        waypoints = request.get("waypoints", [])
        planning_time = request.get("planning_time", MOVEIT_PLANNING_TIME)

        if len(waypoints) < 1:
            return {"success": False, "error": "At least one waypoint required", "robot_id": robot_id}

        total_planning_time = 0.0
        total_points = 0

        for i, wp in enumerate(waypoints):
            wp_request = {
                "position": wp,
                "planning_time": planning_time,
            }
            result = self._plan_and_publish(wp_request, robot_id)
            if not result.get("success"):
                return {
                    "success": False,
                    "error": f"Planning failed at waypoint {i}: {result.get('error', 'Unknown')}",
                    "robot_id": robot_id,
                    "waypoints_completed": i,
                }
            total_planning_time += result.get("planning_time", 0)
            total_points += result.get("trajectory_points", 0)

            # _plan_and_publish already waits for trajectory completion, so each
            # waypoint is fully executed before the next one is published.

        return {
            "success": True,
            "robot_id": robot_id,
            "waypoints_completed": len(waypoints),
            "total_trajectory_points": total_points,
            "planning_time": total_planning_time,
            "status": "published_to_unity",
        }

    def _plan_orientation_change(self, request, robot_id):
        """Plan and publish a trajectory for orientation change at current position.

        Gets current end-effector position, then plans to same position with
        new orientation converted from RPY to quaternion.

        Args:
            request: Request dict with orientation (roll, pitch, yaw in degrees).
            robot_id: Robot namespace.

        Returns:
            Dict with success status and planning time.
        """
        import math

        orientation = request.get("orientation", {})
        planning_time = request.get("planning_time", MOVEIT_PLANNING_TIME)

        # Convert RPY degrees to quaternion
        roll = math.radians(orientation.get("roll", 0.0))
        pitch = math.radians(orientation.get("pitch", 0.0))
        yaw = math.radians(orientation.get("yaw", 0.0))

        # RPY to quaternion conversion
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        # Get current position from joint states to maintain it
        current_pose = self._get_current_pose(robot_id)
        if not current_pose.get("success"):
            return {"success": False, "error": "Cannot get current pose for orientation change", "robot_id": robot_id}

        # Use plan_and_execute with current position and new orientation
        orient_request = {
            "position": current_pose.get("position", {"x": 0.0, "y": 0.0, "z": 0.3}),
            "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
            "planning_time": planning_time,
        }
        return self._plan_and_publish(orient_request, robot_id)

    def _plan_cartesian_descent(self, request, robot_id):
        """Plan and publish a straight-line Cartesian descent to a target position.

        Uses MoveIt's GetCartesianPath service to constrain the end-effector to
        follow a straight line from the current pose to the target. This prevents
        IK redundancy from causing joint 4 (or any wrist joint) to rotate to a
        different solution mid-descent, which would offset the gripper laterally.

        Intended for the final grasp descent: after arriving at the pre-grasp
        hover, call this instead of plan_and_execute so the arm descends straight
        down to the object without any lateral drift.

        Args:
            request: Dict with keys:
                position: Target position dict {x, y, z} in Unity world coords.
                orientation: Target orientation dict {x, y, z, w} (optional).
                max_velocity_scaling: Velocity scaling factor (default 0.3).
                max_acceleration_scaling: Acceleration scaling factor (default 0.3).
            robot_id: Robot namespace (e.g. "Robot1").

        Returns:
            Dict with success status and trajectory info.
        """
        cartesian_client = self._cartesian_path_clients.get(robot_id)
        if cartesian_client is None:
            logger.error(f"No Cartesian path client for {robot_id}")
            return {"success": False, "error": f"No Cartesian path client for {robot_id}"}

        if not cartesian_client.wait_for_service(timeout_sec=10.0):
            logger.error(f"Cartesian path service not available for {robot_id}")
            return {"success": False, "error": "compute_cartesian_path service unavailable"}

        position = request.get("position", {})
        orientation = request.get("orientation")
        vel_scaling = request.get("max_velocity_scaling", 0.3)
        acc_scaling = request.get("max_acceleration_scaling", 0.3)

        # Transform Unity world coords → ROS base_link frame
        local_position = self._transform_world_to_local(position, robot_id)

        logger.info(
            f"[CARTESIAN] {robot_id} descent to base_link: "
            f"x={local_position['x']:.3f}, y={local_position['y']:.3f}, z={local_position['z']:.3f}"
        )

        # Build target waypoint pose
        target_pose = PoseStamped()
        target_pose.header.frame_id = "base_link"
        target_pose.pose.position = Point(
            x=local_position.get("x", 0.0),
            y=local_position.get("y", 0.0),
            z=local_position.get("z", 0.0),
        )
        if orientation:
            target_pose.pose.orientation = Quaternion(
                x=orientation.get("x", 0.0),
                y=orientation.get("y", 0.0),
                z=orientation.get("z", 0.0),
                w=orientation.get("w", 1.0),
            )
        else:
            target_pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        req = GetCartesianPath.Request()
        req.header.frame_id = "base_link"
        req.group_name = "arm"
        req.link_name = "ee_link"
        req.waypoints = [target_pose.pose]
        req.max_step = 0.01          # 1cm maximum interpolation step along the path
        req.jump_threshold = 0.0     # Disable jump detection (causes false failures)
        req.avoid_collisions = True
        # max_velocity/acceleration_scaling_factor were added to GetCartesianPath in
        # moveit_msgs ~2.3. _CARTESIAN_HAS_SCALING is set once at import time.
        if _CARTESIAN_HAS_SCALING:
            req.max_velocity_scaling_factor = vel_scaling
            req.max_acceleration_scaling_factor = acc_scaling
        else:
            logger.warning(
                "moveit_msgs lacks velocity/acceleration scaling fields on GetCartesianPath — "
                "running Cartesian descent at default MoveIt speed"
            )

        # Set current joint state as start (clamped to URDF bounds, is_diff=True)
        joint_state = self._current_joint_states.get(robot_id)
        if joint_state is not None:
            _ARM_JOINT_LIMITS_CART = {
                "joint_1": (-2.9670597283903604, 2.9670597283903604),
                "joint_2": (-0.7330382858376184, 1.5707963267948966),
                "joint_3": (-1.5533430342749532, 0.9075712110370514),
                "joint_4": (-3.141592653589793, 3.141592653589793),
                "joint_5": (-1.8325957145940461, 1.8325957145940461),
                "joint_6": (-3.141592653589793, 3.141592653589793),
            }
            filtered_js = JointState()
            filtered_js.header = joint_state.header
            for name, (lower, upper) in _ARM_JOINT_LIMITS_CART.items():
                if name in joint_state.name:
                    idx = list(joint_state.name).index(name)
                    raw = joint_state.position[idx]
                    clamped = max(lower, min(upper, raw))
                    if abs(clamped - raw) > 1e-6:
                        logger.warning(
                            f"{robot_id} {name} position {raw:.6f} rad out of bounds "
                            f"[{lower:.4f}, {upper:.4f}] — clamped for Cartesian start state"
                        )
                    filtered_js.name.append(name)
                    filtered_js.position.append(clamped)
            start_state = RobotState()
            start_state.is_diff = True  # Differential update: only override listed joints
            start_state.joint_state = filtered_js
            req.start_state = start_state

        future = cartesian_client.call_async(req)

        # Poll for completion — do NOT call spin_until_future_complete (ros_spin thread is spinning)
        deadline = time.time() + 15.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)

        if not future.done() or future.result() is None:
            return {"success": False, "error": "Cartesian path planning timed out", "robot_id": robot_id}

        response = future.result()
        fraction = response.fraction

        if fraction < 0.95:
            if fraction < 0.50:
                # Very low completion means the goal itself is likely in collision
                # or kinematically unreachable — don't waste time retrying with
                # free-space planning (OMPL will also fail with "Unable to sample
                # any valid states for goal tree").
                logger.error(
                    f"{robot_id}: Cartesian path only {fraction*100:.0f}% complete — "
                    "goal likely unreachable or in collision, skipping free-space fallback"
                )
                return {"success": False, "error": f"Cartesian descent failed ({fraction*100:.0f}% complete) — goal unreachable", "robot_id": robot_id}

            logger.warning(
                f"{robot_id}: Cartesian path only {fraction*100:.0f}% complete — "
                "falling back to free-space plan"
            )
            # Fall back to free-space planning if Cartesian path is mostly blocked
            return self._plan_and_publish(request, robot_id)

        trajectory = response.solution.joint_trajectory
        if not trajectory.points:
            return {"success": False, "error": "Cartesian path returned empty trajectory", "robot_id": robot_id}

        logger.info(
            f"{robot_id}: Cartesian path {fraction*100:.0f}% complete, "
            f"{len(trajectory.points)} points"
        )

        # Publish and wait for completion (same as _plan_and_publish)
        trajectory_pub = self._trajectory_pubs[robot_id]
        self._trajectory_feedback_event[robot_id].clear()
        self._trajectory_feedback[robot_id] = None
        trajectory_pub.publish(trajectory)

        if trajectory.points:
            last_pt = trajectory.points[-1]
            traj_duration = last_pt.time_from_start.sec + last_pt.time_from_start.nanosec * 1e-9
        else:
            traj_duration = 5.0
        execution_timeout = max(15.0, traj_duration * 2.5 + 10.0)

        completed = self._wait_for_trajectory_completion(robot_id, timeout=execution_timeout)
        if not completed:
            logger.warning(f"{robot_id}: Cartesian trajectory completion not confirmed within {execution_timeout:.1f}s")

        return {
            "success": completed,
            "robot_id": robot_id,
            "trajectory_points": len(trajectory.points),
            "cartesian_fraction": fraction,
            "status": "completed" if completed else "timeout",
        }

    def _plan_return_to_start(self, request, robot_id):
        """Plan and publish a trajectory to return to the start/home configuration.

        Uses joint-space planning with JointConstraints targeting all 6 arm
        joints at 0 rad (the URDF home pose), which is exact and avoids the
        IK ambiguity of a Cartesian position approximation.

        Args:
            request: Request dict with planning_time.
            robot_id: Robot namespace.

        Returns:
            Dict with success status and planning time.
        """
        planning_time = request.get("planning_time", MOVEIT_PLANNING_TIME)

        goal = MoveGroup.Goal()
        goal.request = MotionPlanRequest()
        goal.request.group_name = "arm"
        goal.request.num_planning_attempts = MOVEIT_PLANNING_ATTEMPTS
        goal.request.allowed_planning_time = planning_time
        goal.planning_options.plan_only = True

        # Set workspace bounds
        goal.request.workspace_parameters.header.frame_id = "base_link"
        goal.request.workspace_parameters.min_corner = Vector3(x=-1.0, y=-1.0, z=-1.0)
        goal.request.workspace_parameters.max_corner = Vector3(x=1.0, y=1.0, z=1.0)

        # Set start state from cached joint states (same clamping logic as _build_move_group_goal)
        _ARM_JOINT_LIMITS = {
            "joint_1": (-2.9670597283903604, 2.9670597283903604),
            "joint_2": (-0.7330382858376184, 1.5707963267948966),
            "joint_3": (-1.5533430342749532, 0.9075712110370514),
            "joint_4": (-3.141592653589793, 3.141592653589793),
            "joint_5": (-1.8325957145940461, 1.8325957145940461),
            "joint_6": (-3.141592653589793, 3.141592653589793),
        }
        joint_state = self._current_joint_states.get(robot_id)
        if joint_state is not None:
            filtered_js = JointState()
            filtered_js.header = joint_state.header
            for name, (lower, upper) in _ARM_JOINT_LIMITS.items():
                if name in joint_state.name:
                    idx = list(joint_state.name).index(name)
                    raw = joint_state.position[idx]
                    clamped = max(lower, min(upper, raw))
                    filtered_js.name.append(name)
                    filtered_js.position.append(clamped)
                    if joint_state.velocity:
                        filtered_js.velocity.append(0.0)
            start_state = RobotState()
            start_state.is_diff = True
            start_state.joint_state = filtered_js
            goal.request.start_state = start_state

        # Joint-space goal: all 6 arm joints at 0 rad (URDF home pose)
        constraints = Constraints()
        for joint_name in _ARM_JOINT_LIMITS:
            jc = JointConstraint()
            jc.joint_name = joint_name
            jc.position = 0.0
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        goal.request.goal_constraints.append(constraints)

        logger.info(f"Planning return-to-start for {robot_id} (joint-space, all joints→0)")
        trajectory, plan_time, error = self._call_move_group_plan(goal, robot_id, planning_time)

        if trajectory is None:
            return {"success": False, "error": error, "robot_id": robot_id}

        # Publish trajectory to Unity and wait for completion (same as _plan_and_publish)
        trajectory_pub = self._trajectory_pubs[robot_id]
        self._trajectory_feedback_event[robot_id].clear()
        self._trajectory_feedback[robot_id] = None
        trajectory_pub.publish(trajectory)

        if trajectory.points:
            last_pt = trajectory.points[-1]
            traj_duration = last_pt.time_from_start.sec + last_pt.time_from_start.nanosec * 1e-9
        else:
            traj_duration = 5.0
        execution_timeout = max(15.0, traj_duration * 2.5 + 10.0)

        completed = self._wait_for_trajectory_completion(robot_id, timeout=execution_timeout)
        return {
            "success": completed,
            "robot_id": robot_id,
            "trajectory_points": len(trajectory.points),
            "planning_time": plan_time,
            "status": "completed" if completed else "timeout",
        }

    def _validate_grasp_candidates(self, request, robot_id):
        """
        Validate grasp candidates using MoveIt's IK service.

        Tests each candidate's grasp and pre-grasp poses for IK reachability.
        Returns validation results with IK quality scores.

        Args:
            request: Request dict with 'candidates' list
            robot_id: Robot namespace

        Returns:
            Dict with success status and validation results per candidate
        """
        candidates = request.get("candidates", [])

        if not candidates:
            return {
                "success": False,
                "error": "No candidates provided",
            }

        logger.info(f"Validating {len(candidates)} grasp candidates for {robot_id}")

        # Validate IK for each candidate
        results = self._validate_ik_batch(candidates, robot_id)

        return {
            "success": True,
            "results": results,
            "candidates_validated": len(results),
        }

    def _validate_ik_batch(self, candidates, robot_id):
        """
        Validate IK for a batch of grasp candidates.

        Uses MoveIt's compute_ik service to check if each candidate's
        grasp pose can be reached by the robot.

        Args:
            candidates: List of candidate dicts with position/orientation
            robot_id: Robot namespace

        Returns:
            List of (is_valid, quality_score) tuples for each candidate
        """
        ik_client = self._ik_service_clients.get(robot_id)

        if not ik_client:
            logger.error(f"No IK service client for {robot_id}")
            return [(False, 0.0)] * len(candidates)

        # Wait for IK service to be available
        if not ik_client.wait_for_service(timeout_sec=5.0):
            logger.error(f"IK service not available for {robot_id}")
            return [(False, 0.0)] * len(candidates)

        # --- Loop 1: fire all IK requests without waiting ---
        # Because the _ros_spin thread continuously spins the node, all
        # outstanding futures are serviced concurrently.  Firing every request
        # before we start polling converts O(N × 500ms) serial latency into
        # O(batch_timeout) parallel latency.
        # NOTE: Do NOT call rclpy.spin_once() in the polling loop below — the
        # _ros_spin thread already owns the node's executor (see _call_move_group_plan).
        futures = []
        for i, candidate in enumerate(candidates):
            try:
                # Extract grasp position and rotation
                position = candidate.get("position", {})
                rotation = candidate.get("rotation", {})

                # Transform world coordinates to robot-local frame
                local_position = self._transform_world_to_local(position, robot_id)

                # Build IK request
                ik_request = GetPositionIK.Request()
                ik_request.ik_request.group_name = "arm"
                ik_request.ik_request.avoid_collisions = False  # Fast validation only
                ik_request.ik_request.timeout.sec = 0
                ik_request.ik_request.timeout.nanosec = 100_000_000  # 100ms per candidate

                # Set target pose
                ik_request.ik_request.pose_stamped.header.frame_id = "base_link"
                ik_request.ik_request.pose_stamped.pose.position = Point(
                    x=local_position.get("x", 0.0),
                    y=local_position.get("y", 0.0),
                    z=local_position.get("z", 0.0),
                )
                ik_request.ik_request.pose_stamped.pose.orientation = Quaternion(
                    x=rotation.get("x", 0.0),
                    y=rotation.get("y", 0.0),
                    z=rotation.get("z", 0.0),
                    w=rotation.get("w", 1.0),
                )

                # Set current robot state as starting point
                joint_state = self._current_joint_states.get(robot_id)
                if joint_state is not None:
                    # Filter to arm joints only
                    arm_joint_names = [
                        "joint_1", "joint_2", "joint_3",
                        "joint_4", "joint_5", "joint_6"
                    ]
                    filtered_js = JointState()
                    filtered_js.header = joint_state.header
                    for name in arm_joint_names:
                        if name in joint_state.name:
                            idx = list(joint_state.name).index(name)
                            filtered_js.name.append(name)
                            filtered_js.position.append(joint_state.position[idx])

                    ik_request.ik_request.robot_state.joint_state = filtered_js

                futures.append((i, ik_client.call_async(ik_request)))

            except Exception as e:
                logger.error(f"Error building IK request for candidate {i}: {e}")
                futures.append((i, None))

        # --- Loop 2: collect results with a shared batch-wide deadline ---
        # Each future gets up to 500ms, but the whole batch shares a single
        # generous deadline so a few slow candidates don't starve the rest.
        batch_timeout = 0.5 + len(candidates) * 0.1
        batch_deadline = time.time() + batch_timeout
        results = [None] * len(candidates)

        for i, future in futures:
            if future is None:
                results[i] = (False, 0.0)
                continue

            per_candidate_timeout = 0.5
            start_time = time.time()
            while not future.done():
                time.sleep(0.01)
                elapsed = time.time() - start_time
                remaining = batch_deadline - time.time()
                if elapsed > per_candidate_timeout or remaining <= 0:
                    logger.warning(f"IK validation timeout for candidate {i}")
                    results[i] = (False, 0.0)
                    break
            else:
                try:
                    response = future.result()
                    if response.error_code.val == 1:  # SUCCESS
                        results[i] = (True, 1.0)
                        logger.debug(f"Candidate {i}: IK valid")
                    else:
                        results[i] = (False, 0.0)
                        logger.debug(f"Candidate {i}: IK failed (error={response.error_code.val})")
                except Exception as e:
                    logger.error(f"Error reading IK result for candidate {i}: {e}")
                    results[i] = (False, 0.0)

        # Fill any slots that were never written (shouldn't happen, but defensive)
        results = [(False, 0.0) if r is None else r for r in results]

        valid_count = sum(1 for is_valid, _ in results if is_valid)
        logger.info(
            f"IK validation complete: {valid_count}/{len(candidates)} candidates valid"
        )

        return results

    def shutdown(self):
        """Cleanly stop the TCP server and shut down the ROS 2 node.

        Sets the running flag to False (stops the accept loop and the ROS spin
        thread), waits for the spin thread to exit, then destroys the ROS node
        and calls rclpy.shutdown() so all publishers/subscribers/action clients
        are released before the process exits.
        """
        self._running = False

        # Wait for the ROS spin thread to exit before destroying the node.
        # Without this join, the daemon thread may still be inside spin_once()
        # when destroy_node() is called, leaving stale ROS graph registrations.
        if hasattr(self, "_ros_thread") and self._ros_thread is not None:
            self._ros_thread.join(timeout=5.0)
            if self._ros_thread.is_alive():
                logger.warning("ROS spin thread did not exit within 5s")

        if HAS_ROS and self._node:
            self._node.destroy_node()
            rclpy.shutdown()


def main():
    """Entry point when running inside Docker."""
    server = ROSMotionServer(host="0.0.0.0", port=5020)

    def _signal_handler(sig, frame):
        """Handle SIGINT and SIGTERM for graceful shutdown inside Docker."""
        logger.info(f"Received signal {sig}, shutting down...")
        server.shutdown()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
