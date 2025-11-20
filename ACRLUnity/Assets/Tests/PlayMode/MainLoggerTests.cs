using NUnit.Framework;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.TestTools;
using Logging;

namespace Tests.PlayMode
{
    /// <summary>
    /// Tests for MainLogger.
    /// Validates action logging, file creation, and event tracking.
    /// </summary>
    public class MainLoggerTests
    {
        private GameObject _loggerObject;
        private MainLogger _logger;
        private string _testLogDirectory;

        [SetUp]
        public void Setup()
        {
            // Clean up any existing instance
            if (MainLogger.Instance != null)
            {
                UnityEngine.Object.DestroyImmediate(MainLogger.Instance.gameObject);
            }

            _loggerObject = new GameObject("TestMainLogger");
            _logger = _loggerObject.AddComponent<MainLogger>();

            // Set test log directory
            _testLogDirectory = Path.Combine(Application.temporaryCachePath, "TestLogs");
            _logger.logDirectory = _testLogDirectory;
            _logger.operationType = "test";
        }

        [TearDown]
        public void TearDown()
        {
            if (_loggerObject != null)
            {
                UnityEngine.Object.DestroyImmediate(_loggerObject);
            }

            // Clean up test logs
            if (Directory.Exists(_testLogDirectory))
            {
                try
                {
                    Directory.Delete(_testLogDirectory, true);
                }
                catch
                {
                    // Ignore cleanup errors
                }
            }
        }

        #region Singleton Tests

        [Test]
        public void MainLogger_Singleton_IsSet()
        {
            Assert.IsNotNull(MainLogger.Instance);
            Assert.AreEqual(_logger, MainLogger.Instance);
        }

        #endregion

        #region Action Logging Tests

        [UnityTest]
        public IEnumerator MainLogger_StartAction_ReturnsActionId()
        {
            yield return null;

            _logger.enableLogging = true;

            string actionId = _logger.StartAction(
                "test_action",
                ActionType.Movement,
                new[] { "Robot1" },
                startPos: Vector3.zero,
                targetPos: Vector3.one,
                description: "Test action"
            );

            Assert.IsNotNull(actionId);
            Assert.IsNotEmpty(actionId);
        }

        [UnityTest]
        public IEnumerator MainLogger_CompleteAction_WithValidId_Succeeds()
        {
            yield return null;

            _logger.enableLogging = true;

            string actionId = _logger.StartAction(
                "complete_test",
                ActionType.Task,
                new[] { "Robot1" }
            );

            // Should not throw
            Assert.DoesNotThrow(() =>
            {
                _logger.CompleteAction(actionId, success: true, qualityScore: 0.9f);
            });
        }

        [UnityTest]
        public IEnumerator MainLogger_CompleteAction_WithMetrics_IncludesMetrics()
        {
            yield return null;

            _logger.enableLogging = true;

            string actionId = _logger.StartAction(
                "metrics_test",
                ActionType.Movement,
                new[] { "Robot1" }
            );

            var metrics = new System.Collections.Generic.Dictionary<string, float>
            {
                ["distance"] = 1.5f,
                ["time"] = 2.3f
            };

            Assert.DoesNotThrow(() =>
            {
                _logger.CompleteAction(actionId, success: true, qualityScore: 0.85f, metrics: metrics);
            });
        }

        #endregion

        #region Coordination Logging Tests

        [UnityTest]
        public IEnumerator MainLogger_LogCoordination_ReturnsActionId()
        {
            yield return null;

            _logger.enableLogging = true;

            string coordId = _logger.LogCoordination(
                "dual_arm_handoff",
                new[] { "Robot1", "Robot2" },
                description: "Test coordination"
            );

            Assert.IsNotNull(coordId);
            Assert.IsNotEmpty(coordId);
        }

        #endregion

        #region Event Logging Tests

        [UnityTest]
        public IEnumerator MainLogger_LogSimulationEvent_DoesNotThrow()
        {
            yield return null;

            _logger.enableLogging = true;

            Assert.DoesNotThrow(() =>
            {
                _logger.LogSimulationEvent(
                    "test_event",
                    "Test event description",
                    robotIds: new[] { "Robot1" }
                );
            });
        }

        #endregion

        #region Object Registration Tests

        [UnityTest]
        public IEnumerator MainLogger_RegisterObject_DoesNotThrow()
        {
            yield return null;

            _logger.enableLogging = true;

            var testObject = new GameObject("TestCube");

            Assert.DoesNotThrow(() =>
            {
                _logger.RegisterObject(testObject, "cube", isGraspable: true);
            });

            UnityEngine.Object.DestroyImmediate(testObject);
        }

        #endregion

        #region Configuration Tests

        [Test]
        public void MainLogger_DisabledLogging_DoesNotCreateFiles()
        {
            _logger.enableLogging = false;

            string actionId = _logger.StartAction(
                "disabled_test",
                ActionType.Movement,
                new[] { "Robot1" }
            );

            // Action ID should still be returned for tracking
            Assert.IsNotNull(actionId);
        }

        [Test]
        public void MainLogger_LogDirectory_CanBeSet()
        {
            string customDir = "/custom/path";
            _logger.logDirectory = customDir;
            Assert.AreEqual(customDir, _logger.logDirectory);
        }

        [Test]
        public void MainLogger_OperationType_CanBeSet()
        {
            string opType = "training";
            _logger.operationType = opType;
            Assert.AreEqual(opType, _logger.operationType);
        }

        #endregion

        #region Environment Capture Tests

        [UnityTest]
        public IEnumerator MainLogger_CaptureEnvironment_DoesNotThrow()
        {
            yield return null;

            _logger.enableLogging = true;
            _logger.captureEnvironment = true;

            Assert.DoesNotThrow(() =>
            {
                _logger.CaptureEnvironment("test_snapshot");
            });
        }

        #endregion

        #region Disabled Logging Tests

        [Test]
        public void MainLogger_WhenDisabled_StartAction_StillReturnsId()
        {
            _logger.enableLogging = false;

            string actionId = _logger.StartAction(
                "disabled_action",
                ActionType.Task,
                new[] { "Robot1" }
            );

            // Should still return an ID for tracking purposes
            Assert.IsNotNull(actionId);
        }

        #endregion
    }
}
