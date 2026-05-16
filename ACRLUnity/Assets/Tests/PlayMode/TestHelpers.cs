using System;
using System.Collections;
using System.Net.Sockets;
using Configuration;
using PythonCommunication;
using Robotics;
using Simulation;
using Tests.EditMode;
using UnityEngine;

namespace Tests.PlayMode
{
    /// <summary>
    /// Common test utilities and helpers for Unity tests.
    /// Provides factory methods for creating test GameObjects and mock data.
    /// Enhanced with config factories, Python backend helpers, and improved assertions.
    /// </summary>
    public static class TestHelpers
    {
        public static (GameObject gameObject, RobotController controller) CreateTestRobot(
            string name = "TestRobot"
        )
        {
            var robotObject = new GameObject(name);
            var controller = robotObject.AddComponent<RobotController>();
            return (robotObject, controller);
        }

        public static GameObject CreateTestTarget(Vector3 position, string name = "TestTarget")
        {
            var target = new GameObject(name);
            target.transform.position = position;
            return target;
        }

        public static GameObject CreateTestCube(Vector3 position, string name = "TestCube")
        {
            var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cube.name = name;
            cube.transform.position = position;
            cube.transform.localScale = Vector3.one * 0.1f;
            return cube;
        }

        /// <summary>
        /// Creates a fresh SimulationManager instance, destroying any existing one.
        /// </summary>
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

        public static RobotConfig CreateTestRobotConfig()
        {
            var config = ScriptableObject.CreateInstance<RobotConfig>();
            config.InitializeDefaultAR4Profile();
            return config;
        }

        /// <summary>
        /// Creates a test SimulationConfig with default settings.
        /// All coordination is Python-driven via signal/wait operations.
        /// </summary>
        public static SimulationConfig CreateTestSimulationConfig()
        {
            var config = ScriptableObject.CreateInstance<SimulationConfig>();
            config.autoStart = false;
            config.resetOnError = true;
            config.timeScale = 1f;
            return config;
        }

        public static IKConfig CreateTestIKConfig()
        {
            var config = ScriptableObject.CreateInstance<IKConfig>();
            // Uses default values set in IKConfig class
            return config;
        }

        public static GripperConfig CreateTestGripperConfig()
        {
            var config = ScriptableObject.CreateInstance<GripperConfig>();
            // Uses default values set in GripperConfig class
            return config;
        }

        public static TrajectoryConfig CreateTestTrajectoryConfig()
        {
            var config = ScriptableObject.CreateInstance<TrajectoryConfig>();
            // Uses default values set in TrajectoryConfig class
            return config;
        }

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

        public static bool IsPythonBackendAvailable()
        {
            try
            {
                using (var client = new TcpClient())
                {
                    var result = client.BeginConnect(
                        "127.0.0.1",
                        TestConstants.SEQUENCE_SERVER_PORT,
                        null,
                        null
                    );
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
                NUnit.Framework.Assert.Ignore(
                    "Python backend not available - skipping integration test"
                );
            }
        }

        public static SequenceClient CreateMockSequenceClient()
        {
            var clientObject = new GameObject("MockSequenceClient");
            var client = clientObject.AddComponent<SequenceClient>();
            return client;
        }

        /// <summary>
        /// Waits until a condition is true, or fails the test after the given timeout.
        /// Drop-in replacement for new WaitUntil(condition, timeout) which requires additional
        /// parameters in Unity 6's test framework.
        /// </summary>
        public static IEnumerator WaitUntil(
            Func<bool> condition,
            float timeoutSeconds,
            string failureMessage = "WaitUntil timed out"
        )
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

        public static void AssertVector3Approximately(
            Vector3 expected,
            Vector3 actual,
            float tolerance = 0.001f,
            string message = ""
        )
        {
            string prefix = string.IsNullOrEmpty(message) ? "" : message + " - ";
            NUnit.Framework.Assert.AreEqual(
                expected.x,
                actual.x,
                tolerance,
                $"{prefix}X component mismatch"
            );
            NUnit.Framework.Assert.AreEqual(
                expected.y,
                actual.y,
                tolerance,
                $"{prefix}Y component mismatch"
            );
            NUnit.Framework.Assert.AreEqual(
                expected.z,
                actual.z,
                tolerance,
                $"{prefix}Z component mismatch"
            );
        }

        public static void AssertQuaternionApproximately(
            Quaternion expected,
            Quaternion actual,
            float tolerance = 0.001f,
            string message = ""
        )
        {
            float dot = Quaternion.Dot(expected, actual);
            string prefix = string.IsNullOrEmpty(message) ? "" : message + " - ";
            NUnit.Framework.Assert.Greater(
                Mathf.Abs(dot),
                1f - tolerance,
                $"{prefix}Quaternion mismatch"
            );
        }

        public static void AssertApproximately(
            float expected,
            float actual,
            float tolerance = 0.001f,
            string message = ""
        )
        {
            NUnit.Framework.Assert.AreEqual(expected, actual, tolerance, message);
        }
    }
}
