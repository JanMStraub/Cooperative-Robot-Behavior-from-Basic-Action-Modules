namespace Core
{
    /// <summary>
    /// Centralized system-level constants for the ACRL project.
    /// Robot-specific parameters have been moved to ScriptableObject configs for runtime tuning.
    /// This file contains only infrastructure constants that should not change during runtime.
    /// </summary>
    public static class RobotConstants
    {
        // Inverse Kinematics
        /// <summary>
        /// Number of dimensions for the Jacobian matrix (3 position + 3 orientation)
        /// </summary>
        public const int JACOBIAN_DIMENSIONS = 6;

        /// <summary>
        /// Maximum joint angular velocity in radians per second (IKSolver damping clamp).
        /// </summary>
        public const float MAX_JOINT_VELOCITY_RAD_PER_SEC = 5.0f;

        /// <summary>
        /// IK position gain (Kp) used in RobotController ArticulationBody drive targets.
        /// </summary>
        public const float IK_POSITION_GAIN = 3.5f;

        /// <summary>
        /// IK velocity/damping gain (Kd) used in RobotController ArticulationBody drive targets.
        /// </summary>
        public const float IK_VELOCITY_GAIN = 0.5f;

        /// <summary>
        /// Maximum joint step in degrees per physics frame under normal (non-stalled) conditions.
        /// </summary>
        public const float MAX_JOINT_DEGREES_PER_FRAME = 5.0f;

        /// <summary>
        /// Maximum joint step in degrees per physics frame when the IK solver is stalled.
        /// </summary>
        public const float MAX_JOINT_DEGREES_PER_FRAME_STALLED = 8.0f;

        /// <summary>
        /// Squared velocity magnitude threshold below which an ArticulationBody is considered settled.
        /// </summary>
        public const float VELOCITY_SETTLE_THRESHOLD_SQR = 0.005f;

        /// <summary>
        /// Movement detection threshold (meters)
        /// Robot is considered moving if distance to target exceeds this value (1cm)
        /// </summary>
        public const float MOVEMENT_THRESHOLD = 0.01f;

        /// <summary>
        /// Rotation convergence threshold in degrees.
        /// IK declares "target reached" once orientation error is below this value.
        /// Tighter values improve accuracy; looser values reduce convergence time.
        /// </summary>
        public const float ROTATION_CONVERGENCE_THRESHOLD_DEG = 3.0f;

        /// <summary>
        /// Rotation acceptance threshold when stalled (degrees).
        /// If the robot is settled, position is reached, and orientation error is below
        /// this value, target is declared reached rather than stalling indefinitely.
        /// Looser than ROTATION_CONVERGENCE_THRESHOLD_DEG to handle ArticulationBody
        /// friction preventing full convergence.
        /// </summary>
        public const float ROTATION_STALL_ACCEPTANCE_DEG = 8.0f;

        /// <summary>
        /// Consecutive stalled physics frames before the controller treats the stall as
        /// sustained (not transient friction) and engages orientation relaxation / position
        /// stall-acceptance. ~0.6 s at the 50 Hz FixedUpdate rate.
        /// </summary>
        public const int STALL_FRAMES_BEFORE_RELAX = 30;

        /// <summary>
        /// Orientation weight applied to the IK solve once a position stall is sustained.
        /// Lowered from the default 1.0 so the arm prioritises reaching the target position
        /// instead of an orientation it cannot satisfy near a wrist singularity (the cause of
        /// B16 unity-mode move_to_coordinate stalling a few cm short and timing out).
        /// </summary>
        public const float STALL_ORIENTATION_RELAX_WEIGHT = 0.3f;

        /// <summary>
        /// Position acceptance threshold (meters) for a sustained stall. Mirrors
        /// ROTATION_STALL_ACCEPTANCE_DEG for position: if the arm is settled within this band
        /// after orientation relaxation, declare the target reached rather than fighting
        /// ArticulationBody friction until the operation times out.
        /// </summary>
        public const float POSITION_STALL_ACCEPTANCE_M = 0.05f;

        // GameObject Naming
        /// <summary>
        /// Suffix for temporary grasp target GameObjects
        /// </summary>
        public const string GRASP_TARGET_SUFFIX = "_GraspTarget";

        /// <summary>
        /// Suffix for temporary coordinate target GameObjects
        /// </summary>
        public const string TEMP_TARGET_SUFFIX = "_TempTarget";
    }

    public static class SceneConstants
    {
        /// <summary>
        /// Minimum object size threshold for registration (magnitude)
        /// Objects smaller than this are considered too small to track
        /// </summary>
        public const float SMALL_OBJECT_SIZE_THRESHOLD = 0.01f;

        /// <summary>
        /// Maximum object size threshold for graspable detection (magnitude)
        /// Objects larger than this are considered too large to grasp
        /// </summary>
        public const float GRASPABLE_OBJECT_SIZE_THRESHOLD = 0.5f;
    }

    public static class CameraConstants
    {
        /// <summary>
        /// Distance threshold for detecting target movement (meters)
        /// If target moves more than this distance, it's considered a new target
        /// </summary>
        public const float TARGET_DISTANCE_THRESHOLD = 0.01f;

        /// <summary>
        /// Position reached threshold for robot target detection (meters)
        /// Robot is considered at target position if within this distance
        /// </summary>
        public const float POSITION_REACHED_THRESHOLD = 0.1f;
    }

    public static class LoggingConstants
    {
        /// <summary>
        /// Default environment sampling rate (seconds between snapshots)
        /// </summary>
        public const float DEFAULT_ENVIRONMENT_SAMPLE_RATE = 2.0f;

        /// <summary>
        /// Default trajectory sampling rate (seconds between trajectory points)
        /// </summary>
        public const float DEFAULT_TRAJECTORY_SAMPLE_RATE = 0.2f;
    }

    public static class CollisionConstants
    {
        /// <summary>
        /// Default collision cooldown period (seconds)
        /// Prevents duplicate collision events within this time window
        /// </summary>
        public const float DEFAULT_COLLISION_COOLDOWN = 0.5f;

        /// <summary>
        /// Default target reward value for goal collisions
        /// </summary>
        public const float DEFAULT_TARGET_REWARD = 1.0f;
    }

    public static class CommunicationConstants
    {
        /// <summary>
        /// Hostname for server (default: 127.0.0.1)
        /// </summary>
        public const string SERVER_HOST = "127.0.0.1";

        /// <summary>
        /// Stereo detection server port (receives stereo image pairs)
        /// </summary>
        public const int STEREO_DETECTION_PORT = 5006;

        /// <summary>
        /// CommandServer port - bidirectional commands and results
        /// </summary>
        public const int COMMAND_SERVER_PORT = 5007;

        /// <summary>
        /// Sequence server port (multi-command sequence execution) - primary communication port
        /// </summary>
        public const int SEQUENCE_SERVER_PORT = 5008;

        /// <summary>
        /// AutoRT server port (autonomous task generation)
        /// </summary>
        public const int AUTORT_SERVER_PORT = 5010;

        /// <summary>
        /// World state streaming port (one-way broadcast of robot/object states)
        /// </summary>
        public const int WORLD_STATE_PORT = 5009;

        /// <summary>
        /// Maximum JSON message size (10MB)
        /// </summary>
        public const int MAX_JSON_LENGTH = 10 * 1024 * 1024;

        /// <summary>
        /// Auto-reconnect interval after connection loss (seconds)
        /// </summary>
        public const float RECONNECT_INTERVAL = 2f;

        /// <summary>
        /// ROS TCP endpoint port (ros_tcp_endpoint bridge between Unity and ROS 2)
        /// </summary>
        public const int ROS_TCP_ENDPOINT_PORT = 10000;
    }

    /// <summary>
    /// Constants for inter-robot proximity safety.
    /// Thresholds must match Python's CoordinationVerifier.STATIC_COLLISION_RADIUS.
    /// </summary>
    public static class ProximityConstants
    {
        /// <summary>EE-to-EE distance below which motion is halted (meters).</summary>
        public const float EE_STOP_THRESHOLD = 0.25f;

        /// <summary>EE-to-EE distance above which a frozen robot resumes (meters). Hysteresis band prevents chattering.</summary>
        public const float EE_RESUME_THRESHOLD = 0.35f;

        /// <summary>Link-to-link stop threshold (meters). Tighter than EE since link positions are less precise.</summary>
        public const float LINK_STOP_THRESHOLD = 0.15f;

        /// <summary>Joint indices (0-based) checked for link-to-link proximity. Joints 2-5 are most collision-prone on AR4.</summary>
        public static readonly int[] MONITORED_LINK_INDICES = { 2, 3, 4, 5 };
    }
}
