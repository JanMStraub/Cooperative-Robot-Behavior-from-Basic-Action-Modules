using UnityEngine;
using Robotics;
using Simulation;
using Logging;

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

        /// <summary>
        /// Creates a fresh MainLogger instance, destroying any existing one.
        /// </summary>
        /// <param name="enableLogging">Whether to enable logging</param>
        /// <returns>Tuple of (GameObject, MainLogger)</returns>
        public static (GameObject gameObject, MainLogger logger) CreateMainLogger(bool enableLogging = false)
        {
            if (MainLogger.Instance != null)
            {
                UnityEngine.Object.DestroyImmediate(MainLogger.Instance.gameObject);
            }

            var obj = new GameObject("TestMainLogger");
            var logger = obj.AddComponent<MainLogger>();
            logger.enableLogging = enableLogging;
            return (obj, logger);
        }

        #endregion

        #region Data Creation

        /// <summary>
        /// Creates a test RobotAction with common fields populated.
        /// </summary>
        /// <param name="actionName">Name for the action</param>
        /// <param name="type">Action type</param>
        /// <returns>RobotAction instance</returns>
        public static RobotAction CreateTestAction(string actionName = "test_action", ActionType type = ActionType.Movement)
        {
            return new RobotAction
            {
                actionId = System.Guid.NewGuid().ToString(),
                actionName = actionName,
                description = "Test action description",
                type = type,
                status = ActionStatus.Started,
                robotIds = new[] { "Robot1" },
                objectIds = new string[0],
                timestamp = System.DateTime.UtcNow.ToString("o"),
                gameTime = Time.time,
                startPosition = Vector3.zero,
                targetPosition = Vector3.one,
                success = false,
                qualityScore = 0f
            };
        }

        /// <summary>
        /// Creates a test SceneSnapshot with common fields populated.
        /// </summary>
        /// <returns>SceneSnapshot instance</returns>
        public static SceneSnapshot CreateTestSnapshot()
        {
            return new SceneSnapshot
            {
                snapshotId = System.Guid.NewGuid().ToString(),
                timestamp = System.DateTime.UtcNow.ToString("o"),
                gameTime = Time.time,
                totalObjects = 0,
                graspableObjects = 0,
                sceneDescription = "Test scene",
                objects = new Logging.Object[0],
                robots = new RobotState[0]
            };
        }

        /// <summary>
        /// Creates a test RobotState with common fields populated.
        /// </summary>
        /// <param name="robotId">Robot identifier</param>
        /// <returns>RobotState instance</returns>
        public static RobotState CreateTestRobotState(string robotId = "Robot1")
        {
            return new RobotState
            {
                robotId = robotId,
                position = Vector3.zero,
                rotation = Quaternion.identity,
                jointAngles = new float[] { 0, 0, 0, 0, 0, 0 },
                targetPosition = Vector3.one,
                distanceToTarget = 1.73f,
                isMoving = false,
                currentAction = "idle"
            };
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

            if (MainLogger.Instance != null)
            {
                UnityEngine.Object.DestroyImmediate(MainLogger.Instance.gameObject);
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
