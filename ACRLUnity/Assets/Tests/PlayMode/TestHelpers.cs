using UnityEngine;
using Robotics;
using Simulation;
using Configuration;
using PythonCommunication;
using System.Net.Sockets;
using System;
using System.Collections;
using Tests.EditMode;

namespace Tests.PlayMode
{
    /// <summary>
    /// Common test utilities and helpers for Unity tests.
    /// Provides factory methods for creating test GameObjects and mock data.
    /// Enhanced with config factories, Python backend helpers, and improved assertions.
    /// </summary>
    public static class TestHelpers
    {
        #region GameObject Creation

        /// <summary>
        /// Creates a test robot GameObject with RobotController component.
        /// </summary>
        /// <param name="name">Name for the GameObject</param>
        /// <returns>Tuple of (GameObject, RobotController)</returns>
        public static (GameObject gameObject, RobotController controller) CreateTestRobot(string name = "TestRobot")
        {
            var robotObject = new GameObject(name);
            var controller = robotObject.AddComponent<RobotController>();
            return (robotObject, controller);
        }

        /// <summary>
        /// Creates a test target GameObject.
        /// </summary>
        /// <param name="position">World position for the target</param>
        /// <param name="name">Name for the GameObject</param>
        /// <returns>Target GameObject</returns>
        public static GameObject CreateTestTarget(Vector3 position, string name = "TestTarget")
        {
            var target = new GameObject(name);
            target.transform.position = position;
            return target;
        }

        /// <summary>
        /// Creates a test cube with collider.
        /// </summary>
        /// <param name="position">World position</param>
        /// <param name="name">Name for the GameObject</param>
        /// <returns>Cube GameObject</returns>
        public static GameObject CreateTestCube(Vector3 position, string name = "TestCube")
        {
            var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cube.name = name;
            cube.transform.position = position;
            cube.transform.localScale = Vector3.one * 0.1f;
            return cube;
        }

        #endregion

        #region Manager Creation

        /// <summary>
        /// Creates a fresh SimulationManager instance, destroying any existing one.
        /// </summary>
        /// <returns>Tuple of (GameObject, SimulationManager)</returns>
        public static (GameObject gameObject, SimulationManager manager) CreateSimulationManager()
        {
            if (SimulationManager.Instance != null)
            {
                UnityEngine.Object.DestroyImmediate(SimulationManager.Instance.gameObject);
            }

            var obj = new GameObject("TestSimulationManager");
            var manager = obj.AddComponent<SimulationManager>();
            return (obj, manager);
        }

        /// <summary>
        /// Creates a fresh RobotManager instance, destroying any existing one.
        /// </summary>
        /// <returns>Tuple of (GameObject, RobotManager)</returns>
        public static (GameObject gameObject, RobotManager manager) CreateRobotManager()
        {
            if (RobotManager.Instance != null)
            {
                UnityEngine.Object.DestroyImmediate(RobotManager.Instance.gameObject);
            }

            var obj = new GameObject("TestRobotManager");
            var manager = obj.AddComponent<RobotManager>();
            return (obj, manager);
        }

        #endregion 

        #region Cleanup

        /// <summary>
        /// Destroys all test objects safely.
        /// </summary>
        /// <param name="objects">GameObjects to destroy</param>
        public static void DestroyAll(params GameObject[] objects)
        {
            foreach (var obj in objects)
            {
                if (obj != null)
                {
                    UnityEngine.Object.DestroyImmediate(obj);
                }
            }
        }

        /// <summary>
        /// Cleans up all singleton instances for a fresh test.
        /// </summary>
        public static void CleanupAllSingletons()
        {
            if (SimulationManager.Instance != null)
            {
                UnityEngine.Object.DestroyImmediate(SimulationManager.Instance.gameObject);
            }

            if (RobotManager.Instance != null)
            {
                UnityEngine.Object.DestroyImmediate(RobotManager.Instance.gameObject);
            }

            // Clean up all RobotController instances from previous tests
            var allRobots = UnityEngine.Object.FindObjectsByType<Robotics.RobotController>(
                UnityEngine.FindObjectsSortMode.None
            );
            foreach (var robot in allRobots)
            {
                if (robot != null && robot.gameObject != null)
                {
                    UnityEngine.Object.DestroyImmediate(robot.gameObject);
                }
            }
        }

        #endregion

        #region Config Factories

        /// <summary>
        /// Creates a test RobotConfig with default AR4 profile.
        /// </summary>
        /// <returns>RobotConfig instance for testing</returns>
        public static RobotConfig CreateTestRobotConfig()
        {
            var config = ScriptableObject.CreateInstance<RobotConfig>();
            config.InitializeDefaultAR4Profile();
            return config;
        }

        /// <summary>
        /// Creates a test SimulationConfig.
        /// </summary>
        /// <param name="mode">Coordination mode to use</param>
        /// <returns>SimulationConfig instance for testing</returns>
        public static SimulationConfig CreateTestSimulationConfig(RobotCoordinationMode mode = RobotCoordinationMode.Independent)
        {
            var config = ScriptableObject.CreateInstance<SimulationConfig>();
            config.coordinationMode = mode;
            config.autoStart = false;
            config.resetOnError = true;
            config.timeScale = 1f;
            return config;
        }

        /// <summary>
        /// Creates a test IKConfig with default settings.
        /// </summary>
        /// <returns>IKConfig instance for testing</returns>
        public static IKConfig CreateTestIKConfig()
        {
            var config = ScriptableObject.CreateInstance<IKConfig>();
            // Uses default values set in IKConfig class
            return config;
        }

        /// <summary>
        /// Creates a test GripperConfig with default settings.
        /// </summary>
        /// <returns>GripperConfig instance for testing</returns>
        public static GripperConfig CreateTestGripperConfig()
        {
            var config = ScriptableObject.CreateInstance<GripperConfig>();
            // Uses default values set in GripperConfig class
            return config;
        }

        /// <summary>
        /// Creates a test TrajectoryConfig with default PD gains.
        /// </summary>
        /// <returns>TrajectoryConfig instance for testing</returns>
        public static TrajectoryConfig CreateTestTrajectoryConfig()
        {
            var config = ScriptableObject.CreateInstance<TrajectoryConfig>();
            // Uses default values set in TrajectoryConfig class
            return config;
        }

        /// <summary>
        /// Creates a test CoordinationConfig with default settings.
        /// </summary>
        /// <param name="mode">Verification mode to use</param>
        /// <returns>CoordinationConfig instance for testing</returns>
        public static CoordinationConfig CreateTestCoordinationConfig(VerificationMode mode = VerificationMode.UnityOnly)
        {
            var config = ScriptableObject.CreateInstance<CoordinationConfig>();
            config.verificationMode = mode;
            config.minSafeSeparation = 0.2f;
            config.enablePathReplanning = true;
            return config;
        }

        #endregion

        #region Scene Setup Helpers

        /// <summary>
        /// Sets up a minimal ArticulationBody chain for testing.
        /// Creates a simplified 2-joint chain for unit tests.
        /// </summary>
        /// <param name="controller">RobotController to add ArticulationBody chain to</param>
        public static void SetupMinimalArticulationChain(RobotController controller)
        {
            var rootObject = controller.gameObject;

            // Create root ArticulationBody
            var rootBody = rootObject.AddComponent<ArticulationBody>();
            rootBody.immovable = true;
            rootBody.useGravity = false;

            // Create first joint
            var joint1Object = new GameObject("Joint1");
            joint1Object.transform.SetParent(rootObject.transform);
            joint1Object.transform.localPosition = new Vector3(0, 0.1f, 0);

            var joint1Body = joint1Object.AddComponent<ArticulationBody>();
            joint1Body.jointType = ArticulationJointType.RevoluteJoint;
            joint1Body.useGravity = false;

            // Create second joint (end effector)
            var joint2Object = new GameObject("Joint2_EndEffector");
            joint2Object.transform.SetParent(joint1Object.transform);
            joint2Object.transform.localPosition = new Vector3(0, 0.1f, 0);

            var joint2Body = joint2Object.AddComponent<ArticulationBody>();
            joint2Body.jointType = ArticulationJointType.RevoluteJoint;
            joint2Body.useGravity = false;

            // Tag as end effector
            joint2Object.tag = "EndEffector";
        }

        #endregion

        #region Python Backend Helpers

        /// <summary>
        /// Checks if Python backend is available by attempting to connect to SequenceServer.
        /// </summary>
        /// <returns>True if Python backend is listening on port 5013</returns>
        public static bool IsPythonBackendAvailable()
        {
            try
            {
                using (var client = new TcpClient())
                {
                    var result = client.BeginConnect("127.0.0.1", TestConstants.SEQUENCE_SERVER_PORT, null, null);
                    var success = result.AsyncWaitHandle.WaitOne(TimeSpan.FromSeconds(1));

                    if (success)
                    {
                        client.EndConnect(result);
                        return true;
                    }
                }
            }
            catch (Exception)
            {
                // Connection failed
            }

            return false;
        }

        /// <summary>
        /// Skips the current test if Python backend is not available.
        /// Call this at the start of integration tests that require Python.
        /// </summary>
        public static void SkipIfPythonUnavailable()
        {
            if (!IsPythonBackendAvailable())
            {
                NUnit.Framework.Assert.Ignore("Python backend not available - skipping integration test");
            }
        }

        /// <summary>
        /// Creates a mock SequenceClient for unit testing without Python backend.
        /// </summary>
        /// <returns>Mock SequenceClient instance</returns>
        public static SequenceClient CreateMockSequenceClient()
        {
            var clientObject = new GameObject("MockSequenceClient");
            var client = clientObject.AddComponent<SequenceClient>();
            return client;
        }

        #endregion

        #region Coroutine Helpers

        /// <summary>
        /// Waits until a condition is true, or fails the test after the given timeout.
        /// Drop-in replacement for new WaitUntil(condition, timeout) which requires additional
        /// parameters in Unity 6's test framework.
        /// </summary>
        /// <param name="condition">Predicate to poll each frame</param>
        /// <param name="timeoutSeconds">Maximum time to wait before failing</param>
        /// <param name="failureMessage">Message shown when the timeout is exceeded</param>
        public static IEnumerator WaitUntil(Func<bool> condition, float timeoutSeconds, string failureMessage = "WaitUntil timed out")
        {
            float deadline = UnityEngine.Time.time + timeoutSeconds;
            while (!condition())
            {
                if (UnityEngine.Time.time > deadline)
                {
                    NUnit.Framework.Assert.Fail(failureMessage);
                    yield break;
                }
                yield return null;
            }
        }

        #endregion

        #region Assertions

        /// <summary>
        /// Asserts that a Vector3 is approximately equal to expected value.
        /// </summary>
        /// <param name="expected">Expected Vector3</param>
        /// <param name="actual">Actual Vector3</param>
        /// <param name="tolerance">Tolerance for comparison</param>
        /// <param name="message">Optional message for assertion failure</param>
        public static void AssertVector3Approximately(Vector3 expected, Vector3 actual, float tolerance = 0.001f, string message = "")
        {
            string prefix = string.IsNullOrEmpty(message) ? "" : message + " - ";
            NUnit.Framework.Assert.AreEqual(expected.x, actual.x, tolerance, $"{prefix}X component mismatch");
            NUnit.Framework.Assert.AreEqual(expected.y, actual.y, tolerance, $"{prefix}Y component mismatch");
            NUnit.Framework.Assert.AreEqual(expected.z, actual.z, tolerance, $"{prefix}Z component mismatch");
        }

        /// <summary>
        /// Asserts that a Quaternion is approximately equal to expected value.
        /// </summary>
        /// <param name="expected">Expected Quaternion</param>
        /// <param name="actual">Actual Quaternion</param>
        /// <param name="tolerance">Tolerance for comparison</param>
        /// <param name="message">Optional message for assertion failure</param>
        public static void AssertQuaternionApproximately(Quaternion expected, Quaternion actual, float tolerance = 0.001f, string message = "")
        {
            float dot = Quaternion.Dot(expected, actual);
            string prefix = string.IsNullOrEmpty(message) ? "" : message + " - ";
            NUnit.Framework.Assert.Greater(Mathf.Abs(dot), 1f - tolerance, $"{prefix}Quaternion mismatch");
        }

        /// <summary>
        /// Asserts that two floats are approximately equal.
        /// </summary>
        /// <param name="expected">Expected value</param>
        /// <param name="actual">Actual value</param>
        /// <param name="tolerance">Tolerance for comparison</param>
        /// <param name="message">Optional message for assertion failure</param>
        public static void AssertApproximately(float expected, float actual, float tolerance = 0.001f, string message = "")
        {
            NUnit.Framework.Assert.AreEqual(expected, actual, tolerance, message);
        }

        #endregion
    }
}
