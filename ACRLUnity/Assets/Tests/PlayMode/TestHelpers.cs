using UnityEngine;
using Robotics;
using Simulation;

namespace Tests.PlayMode
{
    /// <summary>
    /// Common test utilities and helpers for Unity tests.
    /// Provides factory methods for creating test GameObjects and mock data.
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
        }

        #endregion

        #region Assertions

        /// <summary>
        /// Asserts that a Vector3 is approximately equal to expected value.
        /// </summary>
        /// <param name="expected">Expected Vector3</param>
        /// <param name="actual">Actual Vector3</param>
        /// <param name="tolerance">Tolerance for comparison</param>
        public static void AssertVector3Approximately(Vector3 expected, Vector3 actual, float tolerance = 0.001f)
        {
            NUnit.Framework.Assert.AreEqual(expected.x, actual.x, tolerance, $"X component mismatch");
            NUnit.Framework.Assert.AreEqual(expected.y, actual.y, tolerance, $"Y component mismatch");
            NUnit.Framework.Assert.AreEqual(expected.z, actual.z, tolerance, $"Z component mismatch");
        }

        /// <summary>
        /// Asserts that a Quaternion is approximately equal to expected value.
        /// </summary>
        /// <param name="expected">Expected Quaternion</param>
        /// <param name="actual">Actual Quaternion</param>
        /// <param name="tolerance">Tolerance for comparison</param>
        public static void AssertQuaternionApproximately(Quaternion expected, Quaternion actual, float tolerance = 0.001f)
        {
            float dot = Quaternion.Dot(expected, actual);
            NUnit.Framework.Assert.Greater(Mathf.Abs(dot), 1f - tolerance, "Quaternion mismatch");
        }

        #endregion
    }
}
