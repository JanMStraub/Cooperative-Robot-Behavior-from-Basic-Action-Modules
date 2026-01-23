using UnityEngine;

namespace Core
{
    /// <summary>
    /// Centralized constants for the ACRL project.
    /// Eliminates magic numbers and provides single source of truth for configuration values.
    /// </summary>
    public static class RobotConstants
    {
        // Inverse Kinematics
        /// <summary>
        /// Number of dimensions for the Jacobian matrix (3 position + 3 orientation)
        /// </summary>
        public const int JACOBIAN_DIMENSIONS = 6;

        /// <summary>
        /// Default damping factor for pseudo-inverse stability in IK calculations
        /// Moderate damping for stable IK without over-regularization
        /// </summary>
        public const float DEFAULT_DAMPING_FACTOR = 0.2f;

        /// <summary>
        /// Default convergence threshold for IK target reached detection (meters)
        /// Set to 0.015m (15mm) for reliable physics-based grasping
        /// Phase 2: Grasp Reliability - relaxed from 2mm to 15mm to accommodate ArticulationBody
        /// physics constraints and prevent Zeno's paradox with over-damping
        /// 15mm (1.5cm) is sufficient precision for grasping with gripper fingers
        /// </summary>
        public const float DEFAULT_CONVERGENCE_THRESHOLD = 0.015f;

        /// <summary>
        /// Default maximum joint step size per iteration (radians)
        /// Moderate steps for smooth, controlled motion (balanced between 0.02 and 0.2)
        /// </summary>
        public const float DEFAULT_MAX_JOINT_STEP_RAD = 0.05f;

        /// <summary>
        /// Minimum step speed when very close to target
        /// Slow final approach for precision (balanced for stability)
        /// </summary>
        public const float MIN_STEP_SPEED_NEAR_TARGET = 0.3f;

        /// <summary>
        /// Maximum step speed for IK adjustments
        /// Increased for faster motion while maintaining stability (2x original)
        /// </summary>
        public const float MAX_STEP_SPEED = 1.0f;

        /// <summary>
        /// Movement detection threshold (meters)
        /// Robot is considered moving if distance to target exceeds this value (1cm)
        /// </summary>
        public const float MOVEMENT_THRESHOLD = 0.01f;

        /// <summary>
        /// Default orientation convergence threshold (degrees)
        /// Target orientation is considered reached if angle error is below this value
        /// </summary>
        public const float DEFAULT_ORIENTATION_THRESHOLD_DEGREES = 10f;

        /// <summary>
        /// Distance at which orientation ramping starts (meters)
        /// Robot begins tracking target orientation when closer than this distance
        /// Uses smooth quadratic ramping with position-error scaling to prevent oscillation
        /// </summary>
        public const float ORIENTATION_RAMP_START_DISTANCE = 0.30f;

        /// <summary>
        /// Default timeout for grasp operations (seconds)
        /// Prevents infinite waiting when grasp target is unreachable
        /// Increased to 30s to accommodate slow IK convergence for multi-waypoint grasps
        /// </summary>
        public const float DEFAULT_GRASP_TIMEOUT_SECONDS = 30f;

        /// <summary>
        /// Default timeout for movement operations (seconds)
        /// Prevents infinite waiting when movement target is unreachable
        /// </summary>
        public const float DEFAULT_MOVEMENT_TIMEOUT_SECONDS = 15f;

        /// <summary>
        /// Grasp convergence multiplier (relative to DEFAULT_CONVERGENCE_THRESHOLD)
        /// No relaxation for grasp precision (1.0 * 0.002m = 2mm)
        /// Phase 2: Grasp Reliability - tight tolerance required for >95% success rate
        /// The grasp point calculation in GraspCandidateGenerator already accounts for
        /// finger depth, so the IK target position must be precise
        /// </summary>
        public const float GRASP_CONVERGENCE_MULTIPLIER = 1.0f;

        /// <summary>
        /// Pre-grasp convergence multiplier (relative to DEFAULT_CONVERGENCE_THRESHOLD)
        /// Relaxed threshold for pre-grasp waypoints (5.0 * 0.002m = 10mm)
        /// Pre-grasp is just a safe approach position before the final grasp
        /// The actual precision happens at the grasp waypoint
        /// 10mm tolerance is acceptable for waypoint positioning
        /// </summary>
        public const float PREGRASP_CONVERGENCE_MULTIPLIER = 5.0f;

        // Target Finding
        /// <summary>
        /// Radius for searching for real objects when setting target by coordinate (meters)
        /// </summary>
        public const float OBJECT_FINDING_RADIUS = 0.15f;

        /// <summary>
        /// Distance threshold for matching coordinate to real object (meters, 10cm)
        /// If found object is within this distance, use it instead of creating temp target
        /// </summary>
        public const float OBJECT_DISTANCE_THRESHOLD = 0.1f;

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

    /// <summary>
    /// Constants for object detection and scene analysis
    /// </summary>
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

    /// <summary>
    /// Constants for camera and vision systems
    /// </summary>
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

    /// <summary>
    /// Constants for logging and data collection
    /// </summary>
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

    /// <summary>
    /// Constants for collision detection
    /// </summary>
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

    /// <summary>
    /// Constants for Python communication and network
    /// </summary>
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
        /// CommandServer port - receives all results (LLM and depth) and bidirectional commands
        /// </summary>
        public const int LLM_RESULTS_PORT = 5010;

        /// <summary>
        /// RAG server port (semantic search for robot operations)
        /// </summary>
        public const int RAG_SERVER_PORT = 5011;

        /// <summary>
        /// Status server port (bidirectional robot status queries)
        /// </summary>
        public const int STATUS_SERVER_PORT = 5012;

        /// <summary>
        /// Sequence server port (multi-command sequence execution) - primary communication port
        /// </summary>
        public const int SEQUENCE_SERVER_PORT = 5013;

        /// <summary>
        /// Default timeout for Python processes (seconds)
        /// </summary>
        public const int DEFAULT_PYTHON_TIMEOUT = 300;

        /// <summary>
        /// Maximum concurrent Python processes
        /// </summary>
        public const int MAX_CONCURRENT_PROCESSES = 3;

        /// <summary>
        /// Output buffer size for process streaming (bytes)
        /// </summary>
        public const int OUTPUT_BUFFER_SIZE = 4096;

        /// <summary>
        /// Thread join timeout when stopping receive threads (milliseconds)
        /// </summary>
        public const int THREAD_JOIN_TIMEOUT_MS = 1000;

        /// <summary>
        /// Maximum JSON message size (10MB)
        /// </summary>
        public const int MAX_JSON_LENGTH = 10 * 1024 * 1024;

        /// <summary>
        /// Auto-reconnect interval after connection loss (seconds)
        /// </summary>
        public const float RECONNECT_INTERVAL = 2f;
    }

    /// <summary>
    /// Constants for gripper control
    /// </summary>
    public static class GripperConstants
    {
        /// <summary>
        /// Default gripper smooth time for SmoothDamp (seconds)
        /// </summary>
        public const float DEFAULT_SMOOTH_TIME = 0.5f;
    }

    /// <summary>
    /// Constants for grasp planning and IK validation
    /// </summary>
    public static class GraspPlanningConstants
    {
        /// <summary>
        /// Enable debug logging for grasp IK filter (disable in production for performance)
        /// </summary>
        public const bool ENABLE_GRASP_IK_DEBUG_LOGGING = false;
    }
}
