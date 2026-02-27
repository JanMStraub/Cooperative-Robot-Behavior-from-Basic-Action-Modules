using NUnit.Framework;
using ConfigScripts;
using PythonCommunication.DataModels;
using UnityEngine;

namespace Tests.EditMode
{
    /// <summary>
    /// Unit tests for AutoRTConfig ScriptableObject.
    /// Tests configuration validation and default values.
    /// </summary>
    public class TestAutoRTConfig
    {
        private AutoRTConfig _config;

        [SetUp]
        public void SetUp()
        {
            // Create a new config instance for each test
            _config = ScriptableObject.CreateInstance<AutoRTConfig>();
        }

        [TearDown]
        public void TearDown()
        {
            // Clean up
            Object.DestroyImmediate(_config);
        }

        #region Default Values Tests

        [Test]
        public void TestDefaultValues()
        {
            // Assert - Verify default values match plan specification
            Assert.AreEqual(5, _config.maxTaskCandidates, "Default max task candidates should be 5");
            Assert.AreEqual(TaskSelectionStrategy.Balanced, _config.strategy, "Default strategy should be Balanced");
            Assert.AreEqual(false, _config.enableContinuousLoop, "Continuous loop should be disabled by default");
            Assert.AreEqual(5f, _config.loopDelaySeconds, "Default loop delay should be 5 seconds");
            Assert.AreEqual(10, _config.maxDisplayTasks, "Default max display tasks should be 10");
            Assert.AreEqual(true, _config.autoRefresh, "Auto-refresh should be enabled by default");
            Assert.AreEqual(0.5f, _config.uiRefreshRate, "Default UI refresh rate should be 0.5 seconds");
        }

        [Test]
        public void TestDefaultRobotIds()
        {
            // Assert
            Assert.IsNotNull(_config.robotIds, "Robot IDs should not be null");
            Assert.AreEqual(2, _config.robotIds.Length, "Should have 2 default robots");
            Assert.AreEqual("Robot1", _config.robotIds[0], "First robot should be Robot1");
            Assert.AreEqual("Robot2", _config.robotIds[1], "Second robot should be Robot2");
        }

        [Test]
        public void TestDefaultCollaborativeTasks()
        {
            // Assert
            Assert.True(_config.enableCollaborativeTasks, "Collaborative tasks should be enabled by default");
        }

        #endregion

        #region Validation Tests

        [Test]
        public void TestOnValidate_EmptyRobotIds()
        {
            // Arrange - Set robot IDs to null
            _config.robotIds = null;

            // Act
            _config.OnValidate();

            // Assert - Should add default Robot1
            Assert.IsNotNull(_config.robotIds, "Robot IDs should not be null after validation");
            Assert.AreEqual(1, _config.robotIds.Length, "Should have 1 default robot");
            Assert.AreEqual("Robot1", _config.robotIds[0], "Default robot should be Robot1");
        }

        [Test]
        public void TestOnValidate_EmptyArray()
        {
            // Arrange - Set robot IDs to empty array
            _config.robotIds = new string[0];

            // Act
            _config.OnValidate();

            // Assert - Should add default Robot1
            Assert.AreEqual(1, _config.robotIds.Length, "Should have 1 default robot");
            Assert.AreEqual("Robot1", _config.robotIds[0]);
        }

        [Test]
        public void TestOnValidate_LoopDelayTooShort()
        {
            // Arrange - Set loop delay below minimum
            _config.loopDelaySeconds = 0.5f;

            // Act
            _config.OnValidate();

            // Assert - Should be clamped to minimum
            Assert.GreaterOrEqual(_config.loopDelaySeconds, 1f, "Loop delay should be at least 1 second");
        }

        [Test]
        public void TestOnValidate_ValidConfiguration()
        {
            // Arrange - Set valid configuration
            _config.robotIds = new string[] { "Robot1", "Robot2", "Robot3" };
            _config.loopDelaySeconds = 5f;

            // Act
            _config.OnValidate();

            // Assert - Should remain unchanged
            Assert.AreEqual(3, _config.robotIds.Length, "Should keep valid robot IDs");
            Assert.AreEqual(5f, _config.loopDelaySeconds, "Should keep valid loop delay");
        }

        #endregion

        #region Helper Method Tests

        [Test]
        public void TestGetRobotIdsString_MultipleRobots()
        {
            // Arrange
            _config.robotIds = new string[] { "Robot1", "Robot2", "Robot3" };

            // Act
            string result = _config.GetRobotIdsString();

            // Assert
            Assert.AreEqual("Robot1, Robot2, Robot3", result);
        }

        [Test]
        public void TestGetRobotIdsString_SingleRobot()
        {
            // Arrange
            _config.robotIds = new string[] { "Robot1" };

            // Act
            string result = _config.GetRobotIdsString();

            // Assert
            Assert.AreEqual("Robot1", result);
        }

        [Test]
        public void TestGetRobotIdsString_NullArray()
        {
            // Arrange
            _config.robotIds = null;

            // Act
            string result = _config.GetRobotIdsString();

            // Assert
            Assert.AreEqual("None", result);
        }

        [Test]
        public void TestGetRobotIdsString_EmptyArray()
        {
            // Arrange
            _config.robotIds = new string[0];

            // Act
            string result = _config.GetRobotIdsString();

            // Assert
            Assert.AreEqual("", result);
        }

        #endregion

        #region Range Tests

        [Test]
        public void TestMaxTaskCandidates_ValidRange()
        {
            // The [Range(1, 10)] attribute should enforce this in inspector
            // But we can test valid values programmatically

            // Arrange & Act
            _config.maxTaskCandidates = 1;
            Assert.AreEqual(1, _config.maxTaskCandidates);

            _config.maxTaskCandidates = 5;
            Assert.AreEqual(5, _config.maxTaskCandidates);

            _config.maxTaskCandidates = 10;
            Assert.AreEqual(10, _config.maxTaskCandidates);
        }

        [Test]
        public void TestLoopDelaySeconds_ValidRange()
        {
            // Test valid range values (1-60 seconds)

            // Arrange & Act
            _config.loopDelaySeconds = 1f;
            Assert.AreEqual(1f, _config.loopDelaySeconds);

            _config.loopDelaySeconds = 30f;
            Assert.AreEqual(30f, _config.loopDelaySeconds);

            _config.loopDelaySeconds = 60f;
            Assert.AreEqual(60f, _config.loopDelaySeconds);
        }

        [Test]
        public void TestMaxDisplayTasks_ValidRange()
        {
            // Test valid range values (5-20)

            // Arrange & Act
            _config.maxDisplayTasks = 5;
            Assert.AreEqual(5, _config.maxDisplayTasks);

            _config.maxDisplayTasks = 15;
            Assert.AreEqual(15, _config.maxDisplayTasks);

            _config.maxDisplayTasks = 20;
            Assert.AreEqual(20, _config.maxDisplayTasks);
        }

        [Test]
        public void TestUIRefreshRate_ValidRange()
        {
            // Test valid range values (0.1-2 seconds)

            // Arrange & Act
            _config.uiRefreshRate = 0.1f;
            Assert.AreEqual(0.1f, _config.uiRefreshRate);

            _config.uiRefreshRate = 1f;
            Assert.AreEqual(1f, _config.uiRefreshRate);

            _config.uiRefreshRate = 2f;
            Assert.AreEqual(2f, _config.uiRefreshRate);
        }

        #endregion

        #region Strategy Tests

        [Test]
        public void TestAllStrategies()
        {
            // Test that all strategies can be set

            // Arrange & Act & Assert
            _config.strategy = TaskSelectionStrategy.Balanced;
            Assert.AreEqual(TaskSelectionStrategy.Balanced, _config.strategy);

            _config.strategy = TaskSelectionStrategy.Explore;
            Assert.AreEqual(TaskSelectionStrategy.Explore, _config.strategy);

            _config.strategy = TaskSelectionStrategy.Exploit;
            Assert.AreEqual(TaskSelectionStrategy.Exploit, _config.strategy);

            _config.strategy = TaskSelectionStrategy.Random;
            Assert.AreEqual(TaskSelectionStrategy.Random, _config.strategy);
        }

        #endregion

        #region Integration Tests

        [Test]
        public void TestCompleteConfiguration()
        {
            // Test a complete, realistic configuration

            // Arrange
            _config.maxTaskCandidates = 7;
            _config.strategy = TaskSelectionStrategy.Explore;
            _config.enableContinuousLoop = true;
            _config.loopDelaySeconds = 10f;
            _config.robotIds = new string[] { "RobotA", "RobotB", "RobotC" };
            _config.enableCollaborativeTasks = true;
            _config.maxDisplayTasks = 15;
            _config.autoRefresh = true;
            _config.uiRefreshRate = 1f;

            // Act - Validate
            _config.OnValidate();

            // Assert - All values should be valid and unchanged
            Assert.AreEqual(7, _config.maxTaskCandidates);
            Assert.AreEqual(TaskSelectionStrategy.Explore, _config.strategy);
            Assert.True(_config.enableContinuousLoop);
            Assert.AreEqual(10f, _config.loopDelaySeconds);
            Assert.AreEqual(3, _config.robotIds.Length);
            Assert.True(_config.enableCollaborativeTasks);
            Assert.AreEqual(15, _config.maxDisplayTasks);
            Assert.True(_config.autoRefresh);
            Assert.AreEqual(1f, _config.uiRefreshRate);
        }

        [Test]
        public void TestMinimalConfiguration()
        {
            // Test minimal valid configuration

            // Arrange
            _config.maxTaskCandidates = 1;
            _config.strategy = TaskSelectionStrategy.Balanced;
            _config.enableContinuousLoop = false;
            _config.loopDelaySeconds = 1f;
            _config.robotIds = new string[] { "Robot1" };
            _config.enableCollaborativeTasks = false;
            _config.maxDisplayTasks = 5;
            _config.autoRefresh = false;
            _config.uiRefreshRate = 0.1f;

            // Act - Validate
            _config.OnValidate();

            // Assert - All values should be valid
            Assert.AreEqual(1, _config.maxTaskCandidates);
            Assert.AreEqual(1, _config.robotIds.Length);
            Assert.GreaterOrEqual(_config.loopDelaySeconds, 1f);
        }

        #endregion

        #region Edge Cases

        [Test]
        public void TestRobotIds_SpecialCharacters()
        {
            // Arrange - Robot IDs with special characters
            _config.robotIds = new string[] { "Robot-1", "Robot_A", "Robot.Test" };

            // Act
            string result = _config.GetRobotIdsString();

            // Assert - Should handle special characters
            Assert.True(result.Contains("Robot-1"));
            Assert.True(result.Contains("Robot_A"));
            Assert.True(result.Contains("Robot.Test"));
        }

        [Test]
        public void TestRobotIds_EmptyStrings()
        {
            // Arrange - Array with empty strings
            _config.robotIds = new string[] { "", "Robot1", "" };

            // Act
            string result = _config.GetRobotIdsString();

            // Assert - Should include empty strings in output
            Assert.AreEqual(", Robot1, ", result);
        }

        [Test]
        public void TestLoopDelay_PrecisionHandling()
        {
            // Test that float precision is handled correctly

            // Arrange & Act
            _config.loopDelaySeconds = 5.123456f;

            // Assert - Should maintain reasonable precision
            Assert.AreEqual(5.123456f, _config.loopDelaySeconds, 0.0001f);
        }

        #endregion
    }
}
