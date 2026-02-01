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
        /// Movement detection threshold (meters)
        /// Robot is considered moving if distance to target exceeds this value (1cm)
        /// </summary>
        public const float MOVEMENT_THRESHOLD = 0.01f;

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
        /// Sequence server port (multi-command sequence execution) - primary communication port
        /// </summary>
        public const int SEQUENCE_SERVER_PORT = 5013;

        /// <summary>
        /// World state streaming port (one-way broadcast of robot/object states)
        /// </summary>
        public const int WORLD_STATE_PORT = 5014;

        /// <summary>
        /// Maximum JSON message size (10MB)
        /// </summary>
        public const int MAX_JSON_LENGTH = 10 * 1024 * 1024;

        /// <summary>
        /// Auto-reconnect interval after connection loss (seconds)
        /// </summary>
        public const float RECONNECT_INTERVAL = 2f;

        /// <summary>
        /// Thread join timeout when stopping receive threads (milliseconds)
        /// </summary>
        public const int THREAD_JOIN_TIMEOUT_MS = 1000;
    }
}
