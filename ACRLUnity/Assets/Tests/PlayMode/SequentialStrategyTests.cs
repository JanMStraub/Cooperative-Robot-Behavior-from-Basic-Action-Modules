using System.Collections;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using Robotics;
using Simulation.CoordinationStrategies;

namespace Tests.PlayMode
{
    /// <summary>
    /// Tests for SequentialStrategy coordination improvements.
    /// Tests robot ID matching, timeout behavior, and dictionary handling.
    /// </summary>
    public class SequentialStrategyTests
    {
        private GameObject _testRobot1;
        private GameObject _testRobot2;
        private RobotController _controller1;
        private RobotController _controller2;

        [UnitySetUp]
        public IEnumerator SetUp()
        {
            // Create test robot 1
            _testRobot1 = new GameObject("TestRobot1");
            _controller1 = _testRobot1.AddComponent<RobotController>();
            _controller1.robotId = "Robot1";

            // Create test robot 2
            _testRobot2 = new GameObject("TestRobot2");
            _controller2 = _testRobot2.AddComponent<RobotController>();
            _controller2.robotId = "Robot2";

            // Expect initialization warnings from both controllers during Start()
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] No GripperController found in children of Robot1");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] No GripperController found in children of Robot2");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");

            yield return null; // Allow Start() to fire and consume the expected logs
        }

        [TearDown]
        public void TearDown()
        {
            Object.DestroyImmediate(_testRobot1);
            Object.DestroyImmediate(_testRobot2);
        }

        [Test]
        public void RobotIdMatching_UsesControllerRobotId_NotGameObjectName()
        {
            // Arrange
            var strategy = new SequentialStrategy();
            var controllers = new[] { _controller1, _controller2 };
            var targetReached = new Dictionary<string, bool>();

            // Act - First update sets up initial state
            strategy.Update(controllers, targetReached);

            // Assert - Should use controller.robotId, not gameObject.name
            Assert.IsTrue(strategy.IsRobotActive("Robot1"), "Robot1 should be active");
            Assert.IsFalse(strategy.IsRobotActive("Robot2"), "Robot2 should not be active");
            Assert.AreEqual("Robot1", strategy.GetActiveRobotId(), "Active robot ID should be Robot1");
        }

        [Test]
        public void MissingDictionaryEntry_DefaultsToFalse_NotTrue()
        {
            // Arrange
            var strategy = new SequentialStrategy();
            var controllers = new[] { _controller1, _controller2 };
            var targetReached = new Dictionary<string, bool>();
            // Note: Not adding Robot1 to dictionary (simulates missing entry)

            // Act
            strategy.Update(controllers, targetReached);
            strategy.Update(controllers, targetReached); // Second update should not switch

            // Assert - Missing entry should default to false, robot should not switch
            Assert.AreEqual("Robot1", strategy.GetActiveRobotId(),
                "Robot should not switch when entry is missing (defaults to false)");
        }

        [UnityTest]
        public IEnumerator Timeout_SwitchesToNextRobot_After30Seconds()
        {
            // Arrange
            var strategy = new SequentialStrategy(robotTimeout: 2f); // 2 second timeout for faster test
            var controllers = new[] { _controller1, _controller2 };
            var targetReached = new Dictionary<string, bool> { { "Robot1", false } };

            // Act - Initial update
            strategy.Update(controllers, targetReached);
            Assert.AreEqual("Robot1", strategy.GetActiveRobotId(), "Initial robot should be Robot1");

            // Wait for timeout + a buffer frame by polling until the strategy switches
            yield return TestHelpers.WaitUntil(() =>
            {
                strategy.Update(controllers, targetReached);
                return strategy.GetActiveRobotId() == "Robot2";
            }, 3.0f);

            // Assert - Should have switched to Robot2 due to timeout
            Assert.AreEqual("Robot2", strategy.GetActiveRobotId(),
                "Should switch to Robot2 after timeout");
        }

        [Test]
        public void TargetReached_SwitchesToNextRobot_Immediately()
        {
            // Arrange
            var strategy = new SequentialStrategy();
            var controllers = new[] { _controller1, _controller2 };
            var targetReached = new Dictionary<string, bool> { { "Robot1", false } };

            // Act - Initial update
            strategy.Update(controllers, targetReached);
            Assert.AreEqual("Robot1", strategy.GetActiveRobotId());

            // Mark Robot1 as reached
            targetReached["Robot1"] = true;
            strategy.Update(controllers, targetReached);

            // Assert - Should switch to Robot2
            Assert.AreEqual("Robot2", strategy.GetActiveRobotId(),
                "Should switch to Robot2 when target reached");
        }

        [Test]
        public void Reset_ResetsToFirstRobot()
        {
            // Arrange
            var strategy = new SequentialStrategy();
            var controllers = new[] { _controller1, _controller2 };
            var targetReached = new Dictionary<string, bool> { { "Robot1", true } };

            // Act - Switch to Robot2
            strategy.Update(controllers, targetReached);
            strategy.Update(controllers, targetReached);
            Assert.AreEqual("Robot2", strategy.GetActiveRobotId());

            // Reset
            strategy.Reset();
            targetReached["Robot1"] = false; // Reset Robot1's target reached status
            strategy.Update(controllers, targetReached);

            // Assert - Should be back to Robot1
            Assert.AreEqual("Robot1", strategy.GetActiveRobotId(),
                "Should reset to Robot1 after Reset()");
        }

        [Test]
        public void SequentialOrder_CyclesThroughAllRobots()
        {
            // Arrange
            var strategy = new SequentialStrategy();
            var controllers = new[] { _controller1, _controller2 };
            var targetReached = new Dictionary<string, bool>();

            // Act & Assert - Cycle through robots
            strategy.Update(controllers, targetReached);
            Assert.AreEqual("Robot1", strategy.GetActiveRobotId());

            targetReached["Robot1"] = true;
            strategy.Update(controllers, targetReached);
            Assert.AreEqual("Robot2", strategy.GetActiveRobotId());

            targetReached["Robot2"] = true;
            strategy.Update(controllers, targetReached);
            // Should cycle back to Robot1
            Assert.AreEqual("Robot1", strategy.GetActiveRobotId(),
                "Should cycle back to Robot1");
        }
    }
}
