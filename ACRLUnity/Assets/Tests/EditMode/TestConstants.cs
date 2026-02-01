using UnityEngine;

namespace Tests.EditMode
{
    /// <summary>
    /// Centralized test constants that reference production constants.
    /// NOTE: Many constants have been moved to ScriptableObject configs (IKConfig, GripperConfig, TrajectoryConfig).
    /// Test constants now use hardcoded default values matching the config defaults.
    ///
    /// Usage:
    /// - Import production constants from Core.RobotConstants (for infrastructure constants)
    /// - Use hardcoded defaults for robot-specific parameters (now in configs)
    /// - Add test-specific values (iterations, tolerance factors, timeouts)
    /// </summary>
    public static class TestConstants
    {
        #region Production Constants (from Core.RobotConstants - Infrastructure Only)

        // Communication Ports
        public const int STEREO_DETECTION_PORT = Core.CommunicationConstants.STEREO_DETECTION_PORT; // 5006
        public const int LLM_RESULTS_PORT = Core.CommunicationConstants.LLM_RESULTS_PORT; // 5010
        public const int SEQUENCE_SERVER_PORT = Core.CommunicationConstants.SEQUENCE_SERVER_PORT; // 5013

        #endregion

        #region Default Values (from ScriptableObject Configs)

        // NOTE: These match the defaults in IKConfig, GripperConfig, TrajectoryConfig
        // If configs change, update these test constants accordingly

        // Inverse Kinematics (from IKConfig)
        public const float DEFAULT_CONVERGENCE_THRESHOLD = 0.02f; // 2cm (20mm)
        public const float DEFAULT_DAMPING_FACTOR = 0.2f;
        public const float DEFAULT_MAX_JOINT_STEP_RAD = 0.2f; // radians
        public const float DEFAULT_ORIENTATION_THRESHOLD_DEGREES = 10f; // degrees
        public const float ORIENTATION_RAMP_START_DISTANCE = 0.30f; // 30cm

        // Timeouts (from IKConfig)
        public const float DEFAULT_GRASP_TIMEOUT_SECONDS = 30f; // seconds
        public const float DEFAULT_MOVEMENT_TIMEOUT_SECONDS = 15f; // seconds

        // Grasp Planning (from IKConfig)
        public const float GRASP_CONVERGENCE_MULTIPLIER = 0.33f;
        public const float OBJECT_FINDING_RADIUS = 0.15f; // 15cm
        public const float OBJECT_DISTANCE_THRESHOLD = 0.1f; // 10cm

        // Gripper (from GripperConfig)
        public const float DEFAULT_GRIPPER_SMOOTH_TIME = 0.5f; // seconds

        #endregion

        #region Test-Specific Constants

        // Test Tolerances
        /// <summary>
        /// Standard epsilon for floating point comparisons in tests (1mm)
        /// </summary>
        public const float EPSILON = 0.001f;

        /// <summary>
        /// Loose epsilon for physics tests where Unity physics introduces noise (5mm)
        /// </summary>
        public const float PHYSICS_EPSILON = 0.005f;

        /// <summary>
        /// Angular epsilon for rotation comparisons (0.1 degrees)
        /// </summary>
        public const float ANGULAR_EPSILON = 0.1f;

        // Test Iterations
        /// <summary>
        /// Number of iterations for stress tests (memory leaks, performance)
        /// </summary>
        public const int STRESS_TEST_ITERATIONS = 100;

        /// <summary>
        /// Number of physics frames to wait for stabilization
        /// </summary>
        public const int PHYSICS_STABILIZATION_FRAMES = 10;

        // Test Timeouts
        /// <summary>
        /// Maximum wait time for async test operations (seconds)
        /// </summary>
        public const float TEST_TIMEOUT_SECONDS = 5f;

        /// <summary>
        /// Short wait time for quick convergence tests (seconds)
        /// </summary>
        public const float SHORT_WAIT_SECONDS = 0.5f;

        /// <summary>
        /// Wait time slightly above minimum contact duration for GripperContactSensor tests (seconds)
        /// </summary>
        public const float CONTACT_DURATION_WAIT = 0.15f; // > 0.1s minimum

        // Phase 1: Motion Control Constants (from RobotControlRedesign.md)
        /// <summary>
        /// PD control position gain (from TrajectoryController)
        /// </summary>
        public const float PD_CONTROL_KP = 10.0f;

        /// <summary>
        /// PD control velocity gain (damping term)
        /// </summary>
        public const float PD_CONTROL_KD = 2.0f;

        /// <summary>
        /// Maximum joint velocity clamp (rad/sec) to prevent singularity spikes
        /// </summary>
        public const float MAX_JOINT_VELOCITY_RAD_PER_SEC = 5.0f;

        /// <summary>
        /// End effector velocity convergence threshold (m/s)
        /// </summary>
        public const float VELOCITY_CONVERGENCE_THRESHOLD = 0.05f; // 5 cm/s

        /// <summary>
        /// ArticulationBody stiffness (unified across all joints)
        /// </summary>
        public const float ARTICULATION_STIFFNESS = 2000f;

        /// <summary>
        /// IK solver max step size per iteration (radians)
        /// </summary>
        public const float IK_MAX_JOINT_STEP = 0.05f;

        /// <summary>
        /// IK solver damping factor (lambda) for pseudo-inverse regularization
        /// </summary>
        public const float IK_DAMPING_LAMBDA = 0.5f;

        // Phase 2: Grasp Reliability Constants
        /// <summary>
        /// Minimum contact duration for stable grasp detection (seconds)
        /// </summary>
        public const float MIN_CONTACT_DURATION = 0.1f; // 100ms

        /// <summary>
        /// Force estimation moving average window size (frames)
        /// </summary>
        public const int FORCE_WINDOW_SIZE = 5;

        /// <summary>
        /// Minimum force threshold for grasp stability (Newtons)
        /// </summary>
        public const float MIN_GRASP_FORCE = 5f;

        /// <summary>
        /// Maximum force clamp to prevent infinity spikes (Newtons)
        /// </summary>
        public const float MAX_FORCE_CLAMP = 1000f;

        /// <summary>
        /// Finger depth for grasp point calculation (meters)
        /// Prevents hover bug by ensuring contact
        /// </summary>
        public const float FINGER_DEPTH = 0.02f; // 2cm

        /// <summary>
        /// IK validation threshold for grasp planning (meters)
        /// Tightened from 10mm to 2mm for 4% precision on 5cm objects
        /// </summary>
        public const float GRASP_IK_VALIDATION_THRESHOLD = 0.002f; // 2mm

        // Phase 3: Coordination Constants
        /// <summary>
        /// Minimum safe separation distance between robots (meters)
        /// </summary>
        public const float MIN_SAFE_SEPARATION = 0.2f; // 20cm

        /// <summary>
        /// Coordination check interval (seconds)
        /// CollaborativeStrategy checks for conflicts every 500ms
        /// </summary>
        public const float COORDINATION_CHECK_INTERVAL = 0.5f;

        /// <summary>
        /// Wait time for coordination tests (slightly above check interval)
        /// </summary>
        public const float COORDINATION_WAIT = 0.6f;

        /// <summary>
        /// Waypoint collision avoidance offset (meters)
        /// Vertical/lateral offset when replanning to avoid collisions
        /// </summary>
        public const float WAYPOINT_AVOIDANCE_OFFSET = 0.1f; // 10cm

        // Test Object Sizes
        /// <summary>
        /// Standard test cube size (meters)
        /// </summary>
        public const float TEST_CUBE_SIZE = 0.05f; // 5cm

        /// <summary>
        /// Standard test target distance from origin (meters)
        /// </summary>
        public const float TEST_TARGET_DISTANCE = 0.5f; // 50cm

        /// <summary>
        /// Far distance for "no contact" tests (meters)
        /// </summary>
        public const float FAR_DISTANCE = 1.0f; // 1 meter

        // Test Robot Configuration
        /// <summary>
        /// Number of joints in AR4 robot
        /// </summary>
        public const int AR4_JOINT_COUNT = 6;

        /// <summary>
        /// Simplified joint count for unit tests
        /// </summary>
        public const int SIMPLE_JOINT_COUNT = 2;

        /// <summary>
        /// Joint spacing for test robot chains (meters)
        /// </summary>
        public const float TEST_JOINT_SPACING = 0.1f; // 10cm

        #endregion

        #region Test Timeouts (Added for refactoring plan)

        /// <summary>
        /// Short timeout for quick operations (1 second)
        /// </summary>
        public const float SHORT_TIMEOUT = 1.0f;

        /// <summary>
        /// Medium timeout for moderate operations (5 seconds)
        /// </summary>
        public const float MEDIUM_TIMEOUT = 5.0f;

        /// <summary>
        /// Long timeout for complex operations (15 seconds)
        /// </summary>
        public const float LONG_TIMEOUT = 15.0f;

        #endregion

        #region Test Ports (Offset from production ports for isolated testing)

        /// <summary>
        /// Test port for CommandServer (offset from production 5010)
        /// </summary>
        public const int TEST_COMMAND_SERVER_PORT = 6010;

        /// <summary>
        /// Test port for SequenceServer (offset from production 5013)
        /// </summary>
        public const int TEST_SEQUENCE_SERVER_PORT = 6013;

        /// <summary>
        /// Test port for ImageServer single camera (offset from production 5005)
        /// </summary>
        public const int TEST_IMAGE_SERVER_PORT = 6005;

        /// <summary>
        /// Test port for ImageServer stereo (offset from production 5006)
        /// </summary>
        public const int TEST_STEREO_IMAGE_PORT = 6006;

        #endregion

        #region Test Robot Configurations

        /// <summary>
        /// Default test robot starting position
        /// </summary>
        public static readonly Vector3 TEST_ROBOT_START_POSITION = Vector3.zero;

        /// <summary>
        /// Test robot IDs for multi-robot scenarios
        /// </summary>
        public static readonly string[] TEST_ROBOT_IDS = { "TestRobot1", "TestRobot2", "TestRobot3" };

        /// <summary>
        /// Default test robot ID (single robot scenarios)
        /// </summary>
        public const string DEFAULT_TEST_ROBOT_ID = "TestRobot1";

        #endregion

        #region Test Object Configurations

        /// <summary>
        /// Standard test object size (5cm cube)
        /// </summary>
        public const float TEST_OBJECT_SIZE = 0.05f;

        /// <summary>
        /// Test object spawn positions (above table surface)
        /// </summary>
        public static readonly Vector3[] TEST_OBJECT_POSITIONS = new Vector3[]
        {
            new Vector3(0.3f, 0.1f, 0.2f),
            new Vector3(0.4f, 0.1f, 0.3f),
            new Vector3(0.2f, 0.1f, 0.1f)
        };

        /// <summary>
        /// Test object colors (for detection tests)
        /// </summary>
        public static readonly Color[] TEST_OBJECT_COLORS = new Color[]
        {
            Color.red,
            Color.blue,
            Color.green
        };

        #endregion

        #region Helper Methods

        /// <summary>
        /// Calculate expected grasp convergence threshold
        /// (DEFAULT_CONVERGENCE_THRESHOLD * GRASP_CONVERGENCE_MULTIPLIER)
        /// </summary>
        public static float GetGraspConvergenceThreshold()
        {
            return DEFAULT_CONVERGENCE_THRESHOLD * GRASP_CONVERGENCE_MULTIPLIER; // ~0.005m (5mm)
        }

        /// <summary>
        /// Calculate critical damping coefficient for ArticulationBody
        /// Formula: 2 * sqrt(stiffness * inertiaTensor)
        /// </summary>
        public static float CalculateCriticalDamping(float stiffness, float inertiaTensor)
        {
            return 2f * Mathf.Sqrt(stiffness * inertiaTensor);
        }

        /// <summary>
        /// Convert degrees to radians (for test assertions)
        /// </summary>
        public static float DegreesToRadians(float degrees)
        {
            return degrees * Mathf.Deg2Rad;
        }

        /// <summary>
        /// Convert radians to degrees (for test assertions)
        /// </summary>
        public static float RadiansToDegrees(float radians)
        {
            return radians * Mathf.Rad2Deg;
        }

        /// <summary>
        /// Get wait time for Unity physics to settle (WaitForSeconds)
        /// </summary>
        public static WaitForSeconds GetPhysicsSettleWait()
        {
            return new WaitForSeconds(0.1f);
        }

        /// <summary>
        /// Get a random test robot ID from the predefined list.
        /// </summary>
        public static string GetRandomTestRobotId()
        {
            return TEST_ROBOT_IDS[Random.Range(0, TEST_ROBOT_IDS.Length)];
        }

        /// <summary>
        /// Get a test object position by index (wraps around if out of range).
        /// </summary>
        public static Vector3 GetTestObjectPosition(int index)
        {
            return TEST_OBJECT_POSITIONS[index % TEST_OBJECT_POSITIONS.Length];
        }

        /// <summary>
        /// Get a test object color by index (wraps around if out of range).
        /// </summary>
        public static Color GetTestObjectColor(int index)
        {
            return TEST_OBJECT_COLORS[index % TEST_OBJECT_COLORS.Length];
        }

        #endregion
    }
}
