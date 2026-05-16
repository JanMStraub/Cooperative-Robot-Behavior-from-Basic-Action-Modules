using Configuration;
using UnityEngine;

namespace Tests.EditMode
{
    /// <summary>
    /// Centralized test constants that reference production constants.
    /// NOTE: Many constants have been moved to ScriptableObject configs (IKConfig, GripperConfig, TrajectoryConfig).
    /// Config-derived defaults are read live from ScriptableObject.CreateInstance so that test values
    /// stay in sync with config field initializers automatically. Pure test-infrastructure constants
    /// (EPSILON, STRESS_TEST_ITERATIONS, etc.) remain as const.
    ///
    /// Usage:
    /// - Import production constants from Core.RobotConstants (for infrastructure constants)
    /// - Call GetIKConfigDefault() / GetGripperConfigDefault() helpers for config-derived values
    /// - Add test-specific values (iterations, tolerance factors, timeouts)
    /// </summary>
    public static class TestConstants
    {
        /// <summary>
        /// Read the IK convergence threshold from IKConfig field initializers.
        /// Creates and immediately destroys a transient ScriptableObject instance.
        /// </summary>
        public static float GetDefaultConvergenceThreshold()
        {
            var cfg = ScriptableObject.CreateInstance<IKConfig>();
            float value = cfg.convergenceThreshold;
            Object.DestroyImmediate(cfg);
            return value;
        }

        public static float GetDefaultDampingFactor()
        {
            var cfg = ScriptableObject.CreateInstance<IKConfig>();
            float value = cfg.dampingFactor;
            Object.DestroyImmediate(cfg);
            return value;
        }

        public static float GetGraspConvergenceMultiplier()
        {
            var cfg = ScriptableObject.CreateInstance<IKConfig>();
            float value = cfg.graspConvergenceMultiplier;
            Object.DestroyImmediate(cfg);
            return value;
        }

        // Communication Ports
        public const int STEREO_DETECTION_PORT = Core.CommunicationConstants.STEREO_DETECTION_PORT; // 5006
        public const int COMMAND_SERVER_PORT = Core.CommunicationConstants.COMMAND_SERVER_PORT; // 5007
        public const int SEQUENCE_SERVER_PORT = Core.CommunicationConstants.SEQUENCE_SERVER_PORT; // 5008

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

        // Test Tolerances
        /// <summary>
        /// Standard epsilon for floating point comparisons in tests (1mm)
        /// </summary>
        public const float EPSILON = 0.001f;

        /// <summary>
        /// Loose epsilon for physics tests where Unity physics introduces noise (5mm)
        /// </summary>
        public const float PHYSICS_EPSILON = 0.005f;

        public const float ANGULAR_EPSILON = 0.1f;

        // Test Iterations
        public const int STRESS_TEST_ITERATIONS = 100;
        public const int PHYSICS_STABILIZATION_FRAMES = 10;

        // Test Timeouts
        public const float TEST_TIMEOUT_SECONDS = 5f;
        public const float SHORT_WAIT_SECONDS = 0.5f;

        /// <summary>
        /// Wait time slightly above minimum contact duration for GripperContactSensor tests (seconds)
        /// </summary>
        public const float CONTACT_DURATION_WAIT = 0.15f; // > 0.1s minimum

        // Phase 1: Motion Control Constants (from RobotControlRedesign.md)
        public const float PD_CONTROL_KP = 10.0f;
        public const float PD_CONTROL_KD = 2.0f;

        /// <summary>
        /// Maximum joint velocity clamp (rad/sec) to prevent singularity spikes
        /// </summary>
        public const float MAX_JOINT_VELOCITY_RAD_PER_SEC = 5.0f;

        public const float VELOCITY_CONVERGENCE_THRESHOLD = 0.05f; // 5 cm/s
        public const float ARTICULATION_STIFFNESS = 2000f;
        public const float IK_MAX_JOINT_STEP = 0.05f;

        /// <summary>
        /// IK solver damping factor (lambda) for pseudo-inverse regularization
        /// </summary>
        public const float IK_DAMPING_LAMBDA = 0.5f;

        // Phase 2: Grasp Reliability Constants
        public const float MIN_CONTACT_DURATION = 0.1f; // 100ms
        public const int FORCE_WINDOW_SIZE = 5;
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
        public const float MIN_SAFE_SEPARATION = 0.2f; // 20cm

        /// <summary>
        /// Waypoint collision avoidance offset (meters)
        /// Vertical/lateral offset when replanning to avoid collisions
        /// </summary>
        public const float WAYPOINT_AVOIDANCE_OFFSET = 0.1f; // 10cm

        // Test Object Sizes
        public const float TEST_CUBE_SIZE = 0.05f; // 5cm
        public const float TEST_TARGET_DISTANCE = 0.5f; // 50cm

        /// <summary>
        /// Far distance for "no contact" tests (meters)
        /// </summary>
        public const float FAR_DISTANCE = 1.0f; // 1 meter

        // Test Robot Configuration
        public const int AR4_JOINT_COUNT = 6;
        public const int SIMPLE_JOINT_COUNT = 2;
        public const float TEST_JOINT_SPACING = 0.1f; // 10cm

        public const float SHORT_TIMEOUT = 1.0f;
        public const float MEDIUM_TIMEOUT = 5.0f;
        public const float LONG_TIMEOUT = 15.0f;

        /// <summary>
        /// Test port for CommandServer (offset from production 5007)
        /// </summary>
        public const int TEST_COMMAND_SERVER_PORT = 6007;

        /// <summary>
        /// Test port for SequenceServer (offset from production 5008)
        /// </summary>
        public const int TEST_SEQUENCE_SERVER_PORT = 6011;

        /// <summary>
        /// Test port for ImageServer single camera (offset from production 5005)
        /// </summary>
        public const int TEST_IMAGE_SERVER_PORT = 6005;

        /// <summary>
        /// Test port for ImageServer stereo (offset from production 5006)
        /// </summary>
        public const int TEST_STEREO_IMAGE_PORT = 6006;

        public static readonly Vector3 TEST_ROBOT_START_POSITION = Vector3.zero;
        public static readonly string[] TEST_ROBOT_IDS =
        {
            "TestRobot1",
            "TestRobot2",
            "TestRobot3",
        };
        public const string DEFAULT_TEST_ROBOT_ID = "TestRobot1";

        public const float TEST_OBJECT_SIZE = 0.05f;

        /// <summary>
        /// Test object spawn positions (above table surface)
        /// </summary>
        public static readonly Vector3[] TEST_OBJECT_POSITIONS = new Vector3[]
        {
            new Vector3(0.3f, 0.1f, 0.2f),
            new Vector3(0.4f, 0.1f, 0.3f),
            new Vector3(0.2f, 0.1f, 0.1f),
        };

        public static readonly Color[] TEST_OBJECT_COLORS = new Color[]
        {
            Color.red,
            Color.blue,
            Color.green,
        };

        /// <summary>
        /// Calculate expected grasp convergence threshold using live config defaults.
        /// (IKConfig.convergenceThreshold * IKConfig.graspConvergenceMultiplier)
        /// </summary>
        public static float GetGraspConvergenceThreshold()
        {
            return GetDefaultConvergenceThreshold() * GetGraspConvergenceMultiplier();
        }

        /// <summary>
        /// Calculate critical damping coefficient for ArticulationBody
        /// Formula: 2 * sqrt(stiffness * inertiaTensor)
        /// </summary>
        public static float CalculateCriticalDamping(float stiffness, float inertiaTensor)
        {
            return 2f * Mathf.Sqrt(stiffness * inertiaTensor);
        }

        public static float DegreesToRadians(float degrees)
        {
            return degrees * Mathf.Deg2Rad;
        }

        public static float RadiansToDegrees(float radians)
        {
            return radians * Mathf.Rad2Deg;
        }

        public static WaitForSeconds GetPhysicsSettleWait()
        {
            return new WaitForSeconds(0.1f);
        }

        /// <summary>
        /// Get a test robot ID by index. Use [TestCase] attributes in tests instead of
        /// calling this randomly to ensure deterministic, reproducible coverage.
        /// Index wraps around if out of range.
        /// </summary>
        public static string GetTestRobotId(int index)
        {
            return TEST_ROBOT_IDS[index % TEST_ROBOT_IDS.Length];
        }

        public static Vector3 GetTestObjectPosition(int index)
        {
            return TEST_OBJECT_POSITIONS[index % TEST_OBJECT_POSITIONS.Length];
        }

        public static Color GetTestObjectColor(int index)
        {
            return TEST_OBJECT_COLORS[index % TEST_OBJECT_COLORS.Length];
        }
    }
}
