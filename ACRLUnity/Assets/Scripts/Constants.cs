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
        /// </summary>
        public const float DEFAULT_DAMPING_FACTOR = 0.1f;

        /// <summary>
        /// Default convergence threshold for IK target reached detection (meters)
        /// </summary>
        public const float DEFAULT_CONVERGENCE_THRESHOLD = 0.1f;

        /// <summary>
        /// Default maximum joint step size per iteration (radians)
        /// </summary>
        public const float DEFAULT_MAX_JOINT_STEP_RAD = 0.1f;

        /// <summary>
        /// Minimum step speed when very close to target
        /// </summary>
        public const float MIN_STEP_SPEED_NEAR_TARGET = 0.1f;

        /// <summary>
        /// Maximum step speed for IK adjustments
        /// </summary>
        public const float MAX_STEP_SPEED = 0.5f;
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
        /// Default streaming server port (receives images from Unity)
        /// </summary>
        public const int STREAMING_SERVER_PORT = 5005;

        /// <summary>
        /// Stereo detection server port (receives stereo image pairs)
        /// </summary>
        public const int STEREO_DETECTION_SERVER_PORT = 5006;

        /// <summary>
        /// LLM results server port (sends LLM analysis results to Unity from RunAnalyzer)
        /// </summary>
        public const int LLM_RESULTS_PORT = 5010;

        /// <summary>
        /// Depth results server port (sends depth detection results with 3D coordinates from RunStereoDetector)
        /// </summary>
        public const int DEPTH_RESULTS_PORT = 5007;

        /// <summary>
        /// Legacy: Default results server port (alias for LLM_RESULTS_PORT)
        /// </summary>
        public const int RESULTS_SERVER_PORT = LLM_RESULTS_PORT;

        /// <summary>
        /// Legacy: Detection server port (alias for DEPTH_RESULTS_PORT)
        /// </summary>
        public const int DETECTION_SERVER_PORT = DEPTH_RESULTS_PORT;

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
}
