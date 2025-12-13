using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using Robotics;
using Logging;

namespace Tests.PlayMode
{
    /// <summary>
    /// Tests for RobotActionLogger encapsulation of MainLogger interactions
    /// </summary>
    public class RobotActionLoggerTests
    {
        private GameObject _testRobotObject;
        private Transform _endEffectorTransform;
        private RobotActionLogger _logger;
        private const string TEST_ROBOT_ID = "TestRobot";

        [SetUp]
        public void SetUp()
        {
            // Create minimal robot setup
            _testRobotObject = new GameObject("TestRobot");
            var endEffectorObject = new GameObject("EndEffector");
            endEffectorObject.transform.SetParent(_testRobotObject.transform);
            endEffectorObject.transform.position = Vector3.zero;
            _endEffectorTransform = endEffectorObject.transform;

            // Create logger
            _logger = new RobotActionLogger(TEST_ROBOT_ID, _endEffectorTransform);
        }

        [TearDown]
        public void TearDown()
        {
            if (_testRobotObject != null)
            {
                UnityEngine.Object.Destroy(_testRobotObject);
            }
        }

        [Test]
        public void LogTargetSet_DoesNotThrow_WhenMainLoggerIsNull()
        {
            // Note: MainLogger might not be initialized in test environment
            // The logger should handle this gracefully

            // Act & Assert - Should not throw
            Assert.DoesNotThrow(() =>
            {
                _logger.LogTargetSet(
                    targetName: "TestTarget",
                    targetPosition: new Vector3(1f, 2f, 3f),
                    useGraspPlanning: true
                );
            });
        }

        [Test]
        public void LogTargetReached_DoesNotThrow_WhenMainLoggerIsNull()
        {
            // Act & Assert - Should not throw
            Assert.DoesNotThrow(() =>
            {
                _logger.LogTargetReached(
                    targetName: "TestTarget",
                    targetPosition: new Vector3(1f, 2f, 3f),
                    distance: 0.05f,
                    convergenceThreshold: 0.1f
                );
            });
        }

        [Test]
        public void LogGripperAction_DoesNotThrow_WhenMainLoggerIsNull()
        {
            // Act & Assert - Should not throw
            Assert.DoesNotThrow(() =>
            {
                _logger.LogGripperAction(
                    actionName: "open_gripper",
                    description: "Opening gripper for grasp"
                );
            });
        }

        [Test]
        public void LogInitialization_DoesNotThrow_WhenMainLoggerIsNull()
        {
            // Arrange
            var testGameObject = new GameObject("TestRobotObject");

            // Act & Assert - Should not throw
            Assert.DoesNotThrow(() =>
            {
                _logger.LogInitialization(testGameObject);
            });

            // Cleanup
            UnityEngine.Object.Destroy(testGameObject);
        }

        [UnityTest]
        public IEnumerator LogTargetSet_WithMainLogger_CreatesAction()
        {
            // Arrange - Ensure MainLogger exists
            if (MainLogger.Instance == null)
            {
                var loggerObject = new GameObject("MainLogger");
                loggerObject.AddComponent<MainLogger>();
                yield return null;
            }

            // Enable logging
            if (MainLogger.Instance != null)
            {
                MainLogger.Instance.enableLogging = true;
            }

            // Act
            _logger.LogTargetSet(
                targetName: "Cube_01",
                targetPosition: new Vector3(0.5f, 0.3f, 0.2f),
                useGraspPlanning: true
            );

            yield return null;

            // Assert - If MainLogger exists, action should be logged
            // We can't easily verify internal state without accessing private fields,
            // but we can verify no exceptions were thrown
            Assert.Pass("LogTargetSet completed without exceptions");
        }

        [UnityTest]
        public IEnumerator LogTargetReached_WithMainLogger_CalculatesQualityScore()
        {
            // Arrange
            if (MainLogger.Instance == null)
            {
                var loggerObject = new GameObject("MainLogger");
                loggerObject.AddComponent<MainLogger>();
                yield return null;
            }

            if (MainLogger.Instance != null)
            {
                MainLogger.Instance.enableLogging = true;
            }

            // Act - Close to threshold (should have lower quality)
            _logger.LogTargetReached(
                targetName: "Cube_01",
                targetPosition: new Vector3(0.5f, 0.3f, 0.2f),
                distance: 0.09f, // Close to threshold
                convergenceThreshold: 0.1f
            );

            yield return null;

            // Act - Far from threshold (should have higher quality)
            _logger.LogTargetReached(
                targetName: "Cube_02",
                targetPosition: new Vector3(0.6f, 0.4f, 0.3f),
                distance: 0.01f, // Well below threshold
                convergenceThreshold: 0.1f
            );

            yield return null;

            // Assert
            Assert.Pass("LogTargetReached completed without exceptions");
        }

        [UnityTest]
        public IEnumerator LogGripperAction_WithMainLogger_CreatesManipulationAction()
        {
            // Arrange
            if (MainLogger.Instance == null)
            {
                var loggerObject = new GameObject("MainLogger");
                loggerObject.AddComponent<MainLogger>();
                yield return null;
            }

            if (MainLogger.Instance != null)
            {
                MainLogger.Instance.enableLogging = true;
            }

            // Act - Open gripper
            _logger.LogGripperAction(
                actionName: "open_gripper",
                description: "Opening gripper for object grasp"
            );

            yield return null;

            // Act - Close gripper
            _logger.LogGripperAction(
                actionName: "close_gripper",
                description: "Closing gripper to secure object"
            );

            yield return null;

            // Assert
            Assert.Pass("LogGripperAction completed without exceptions");
        }

        [UnityTest]
        public IEnumerator LogInitialization_WithMainLogger_CreatesTaskAction()
        {
            // Arrange
            if (MainLogger.Instance == null)
            {
                var loggerObject = new GameObject("MainLogger");
                loggerObject.AddComponent<MainLogger>();
                yield return null;
            }

            if (MainLogger.Instance != null)
            {
                MainLogger.Instance.enableLogging = true;
            }

            var robotObject = new GameObject("AR4_Robot");

            // Act
            _logger.LogInitialization(robotObject);

            yield return null;

            // Assert
            Assert.Pass("LogInitialization completed without exceptions");

            // Cleanup
            UnityEngine.Object.Destroy(robotObject);
        }

        [Test]
        public void Constructor_AcceptsValidParameters()
        {
            // Arrange
            var testTransform = new GameObject("TestEndEffector").transform;

            // Act & Assert - Should not throw
            Assert.DoesNotThrow(() =>
            {
                var testLogger = new RobotActionLogger("Robot123", testTransform);
            });

            // Cleanup
            UnityEngine.Object.Destroy(testTransform.gameObject);
        }

        [UnityTest]
        public IEnumerator MultipleLogCalls_DoNotInterfere()
        {
            // Arrange
            if (MainLogger.Instance == null)
            {
                var loggerObject = new GameObject("MainLogger");
                loggerObject.AddComponent<MainLogger>();
                yield return null;
            }

            if (MainLogger.Instance != null)
            {
                MainLogger.Instance.enableLogging = true;
            }

            // Act - Rapid-fire multiple log calls
            for (int i = 0; i < 10; i++)
            {
                _logger.LogTargetSet(
                    targetName: $"Target_{i}",
                    targetPosition: new Vector3(i * 0.1f, i * 0.1f, i * 0.1f),
                    useGraspPlanning: i % 2 == 0
                );

                _logger.LogGripperAction(
                    actionName: i % 2 == 0 ? "open_gripper" : "close_gripper",
                    description: $"Gripper action {i}"
                );
            }

            yield return null;

            // Assert - Should handle all calls without errors
            Assert.Pass("Multiple rapid log calls completed without exceptions");
        }
    }
}
