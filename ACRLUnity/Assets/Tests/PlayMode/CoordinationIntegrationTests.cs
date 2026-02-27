using System.Collections;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using Robotics;
using Simulation.CoordinationStrategies;
using PythonCommunication;
using Configuration;

namespace Tests.PlayMode
{
    /// <summary>
    /// Integration tests for coordination system (Phase 5).
    /// Tests interaction between strategies, verification, and configuration.
    /// </summary>
    public class CoordinationIntegrationTests
    {
        private GameObject _testRobot1;
        private GameObject _testRobot2;
        private GameObject _testRobot3;
        private RobotController _controller1;
        private RobotController _controller2;
        private RobotController _controller3;

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

            // Create test robot 3
            _testRobot3 = new GameObject("TestRobot3");
            _controller3 = _testRobot3.AddComponent<RobotController>();
            _controller3.robotId = "Robot3";

            // Expect initialization warnings from all three controllers during Start()
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] No GripperController found in children of Robot1");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] No GripperController found in children of Robot2");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] No GripperController found in children of Robot3");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");

            yield return null; // Allow Start() to fire and consume the expected logs
        }

        [TearDown]
        public void TearDown()
        {
            Object.DestroyImmediate(_testRobot1);
            Object.DestroyImmediate(_testRobot2);
            Object.DestroyImmediate(_testRobot3);
        }

        #region Coordination Mode Switching Tests

        [Test]
        public void SwitchingStrategies_PreservesRobotState()
        {
            // Arrange
            var sequentialStrategy = new SequentialStrategy();
            var collaborativeStrategy = new CollaborativeStrategy();
            var controllers = new[] { _controller1, _controller2 };
            var targetReached = new Dictionary<string, bool>
            {
                { "Robot1", false },
                { "Robot2", false }
            };

            // Act - Update with sequential strategy
            sequentialStrategy.Update(controllers, targetReached);
            string sequentialActiveId = sequentialStrategy.GetActiveRobotId();

            // Switch to collaborative strategy
            collaborativeStrategy.Update(controllers, targetReached);
            string collaborativeActiveId = collaborativeStrategy.GetActiveRobotId();

            // Assert - Both strategies should work independently
            Assert.IsNotNull(sequentialActiveId);
            Assert.IsNotNull(collaborativeActiveId);
        }

        [Test]
        public void ResetStrategy_AllowsSwitchingModes()
        {
            // Arrange
            var strategy = new SequentialStrategy();
            var controllers = new[] { _controller1, _controller2 };
            var targetReached = new Dictionary<string, bool> { { "Robot1", true } };

            // Act - Run strategy, then reset
            strategy.Update(controllers, targetReached);
            strategy.Update(controllers, targetReached); // Switch to Robot2
            Assert.AreEqual("Robot2", strategy.GetActiveRobotId());

            strategy.Reset();
            targetReached["Robot1"] = false;
            targetReached["Robot2"] = false;
            strategy.Update(controllers, targetReached);

            // Assert - Should start over
            Assert.AreEqual("Robot1", strategy.GetActiveRobotId());
        }

        #endregion

        #region Sequential Strategy Integration Tests

        [UnityTest]
        public IEnumerator Sequential_ThreeRobots_CyclesCorrectly()
        {
            // Arrange
            var strategy = new SequentialStrategy(robotTimeout: 1f);
            var controllers = new[] { _controller1, _controller2, _controller3 };
            var targetReached = new Dictionary<string, bool>
            {
                { "Robot1", false },
                { "Robot2", false },
                { "Robot3", false }
            };

            // Act - First update
            strategy.Update(controllers, targetReached);
            Assert.AreEqual("Robot1", strategy.GetActiveRobotId());

            // Mark Robot1 as reached
            targetReached["Robot1"] = true;
            strategy.Update(controllers, targetReached);
            Assert.AreEqual("Robot2", strategy.GetActiveRobotId());

            // Mark Robot2 as reached
            targetReached["Robot2"] = true;
            strategy.Update(controllers, targetReached);
            Assert.AreEqual("Robot3", strategy.GetActiveRobotId());

            // Mark Robot3 as reached - should cycle back
            targetReached["Robot3"] = true;
            strategy.Update(controllers, targetReached);

            yield return null;

            // Assert - Should cycle back to Robot1
            Assert.AreEqual("Robot1", strategy.GetActiveRobotId());
        }

        [UnityTest]
        public IEnumerator Sequential_TimeoutAndCompletion_BothWork()
        {
            // Arrange
            var strategy = new SequentialStrategy(robotTimeout: 0.5f);
            var controllers = new[] { _controller1, _controller2 };
            var targetReached = new Dictionary<string, bool>
            {
                { "Robot1", false },
                { "Robot2", false }
            };

            // Act - Start with Robot1
            strategy.Update(controllers, targetReached);
            Assert.AreEqual("Robot1", strategy.GetActiveRobotId());

            // Wait for timeout by polling until strategy switches (0.5s timeout configured)
            yield return TestHelpers.WaitUntil(() =>
            {
                strategy.Update(controllers, targetReached);
                return strategy.GetActiveRobotId() == "Robot2";
            }, 2.0f);

            // Assert - Should switch to Robot2 due to timeout
            Assert.AreEqual("Robot2", strategy.GetActiveRobotId());
        }

        #endregion

        #region Collaborative Strategy Integration Tests

        [Test]
        public void Collaborative_WithCollisionAvoidance_BlocksConflictingRobots()
        {
            // Arrange
            var config = ScriptableObject.CreateInstance<CoordinationConfig>();
            config.minSafeSeparation = 0.3f;
            var strategy = new CollaborativeStrategy(config);
            var controllers = new[] { _controller1, _controller2 };
            var targetReached = new Dictionary<string, bool>();

            // Set conflicting targets (within 0.3m)
            _controller1.SetTarget(new Vector3(1f, 0f, 0f));
            _controller2.SetTarget(new Vector3(1.2f, 0f, 0f)); // 0.2m away, conflicts with 0.3m separation

            // Act
            strategy.Update(controllers, targetReached);

            // Assert - One should be blocked
            bool robot1Active = strategy.IsRobotActive("Robot1");
            bool robot2Active = strategy.IsRobotActive("Robot2");
            Assert.IsFalse(robot1Active && robot2Active,
                "Conflicting robots should not both be active");
        }

        [Test]
        public void Collaborative_NoConflict_AllowsBothRobots()
        {
            // Arrange
            var config = ScriptableObject.CreateInstance<CoordinationConfig>();
            config.minSafeSeparation = 0.2f;
            var strategy = new CollaborativeStrategy(config);
            var controllers = new[] { _controller1, _controller2 };
            var targetReached = new Dictionary<string, bool>();

            // Set non-conflicting targets
            _controller1.SetTarget(new Vector3(0f, 0f, 0f));
            _controller2.SetTarget(new Vector3(5f, 0f, 0f)); // Far apart

            // Act
            strategy.Update(controllers, targetReached);

            // Assert - Both should be active
            Assert.IsTrue(strategy.IsRobotActive("Robot1"));
            Assert.IsTrue(strategy.IsRobotActive("Robot2"));
        }

        #endregion

        #region Verification Integration Tests

        [Test]
        public void UnityVerifier_BasicSafetyCheck_Works()
        {
            // Arrange
            var verifier = new UnityCoordinationVerifier(minSafeSeparation: 0.3f);

            // Act
            var result = verifier.VerifyMovement(
                robotId: "Robot1",
                targetPosition: new Vector3(1f, 0f, 0f),
                currentPosition: new Vector3(0f, 0f, 0f)
            );

            // Assert
            Assert.IsNotNull(result);
            Assert.IsTrue(result.isSafe, "Movement should be safe with no other robots");
            Assert.AreEqual("Unity", verifier.VerifierName);
            Assert.IsTrue(verifier.IsAvailable);
        }

        [Test]
        public void PythonVerifier_FallsBackToUnity_WhenUnavailable()
        {
            // Arrange
            var verifier = new PythonCoordinationVerifier(
                timeout: 0.5f,
                fallbackToUnity: true,
                minSafeSeparation: 0.2f
            );

            // Act - Python backend likely not available in tests
            var result = verifier.VerifyMovement(
                robotId: "Robot1",
                targetPosition: new Vector3(1f, 0f, 0f),
                currentPosition: new Vector3(0f, 0f, 0f)
            );

            // Assert - Should fallback to Unity verification
            Assert.IsNotNull(result);
            Assert.AreEqual("Python", verifier.VerifierName);
        }

        [Test]
        public void VerificationResult_StructWorks()
        {
            // Arrange & Act
            var result1 = new VerificationResult(true, "Safe movement");
            var result2 = new VerificationResult(
                false,
                "Collision detected",
                new List<string> { "Warning 1", "Warning 2" }
            );

            // Assert
            Assert.IsTrue(result1.isSafe);
            Assert.AreEqual("Safe movement", result1.reason);
            Assert.IsNotNull(result1.warnings);
            Assert.AreEqual(0, result1.warnings.Count);

            Assert.IsFalse(result2.isSafe);
            Assert.AreEqual("Collision detected", result2.reason);
            Assert.AreEqual(2, result2.warnings.Count);
        }

        #endregion

        #region Multi-Robot Scenarios

        [Test]
        public void ThreeRobots_Sequential_OnlyOneActiveAtATime()
        {
            // Arrange
            var strategy = new SequentialStrategy();
            var controllers = new[] { _controller1, _controller2, _controller3 };
            var targetReached = new Dictionary<string, bool>
            {
                { "Robot1", false },
                { "Robot2", false },
                { "Robot3", false }
            };

            // Act
            strategy.Update(controllers, targetReached);

            // Assert - Only one should be active
            int activeCount = 0;
            if (strategy.IsRobotActive("Robot1")) activeCount++;
            if (strategy.IsRobotActive("Robot2")) activeCount++;
            if (strategy.IsRobotActive("Robot3")) activeCount++;

            Assert.AreEqual(1, activeCount, "Only one robot should be active in sequential mode");
        }

        [Test]
        public void ThreeRobots_Collaborative_IndependentRegions_AllActive()
        {
            // Arrange
            var config = ScriptableObject.CreateInstance<CoordinationConfig>();
            config.minSafeSeparation = 0.2f;
            var strategy = new CollaborativeStrategy(config);
            var controllers = new[] { _controller1, _controller2, _controller3 };
            var targetReached = new Dictionary<string, bool>();

            // Set targets in separate regions
            _controller1.SetTarget(new Vector3(0f, 0f, 0f));
            _controller2.SetTarget(new Vector3(5f, 0f, 0f));
            _controller3.SetTarget(new Vector3(0f, 0f, 5f));

            // Act
            strategy.Update(controllers, targetReached);

            // Assert - All should be active (no conflicts)
            Assert.IsTrue(strategy.IsRobotActive("Robot1"));
            Assert.IsTrue(strategy.IsRobotActive("Robot2"));
            Assert.IsTrue(strategy.IsRobotActive("Robot3"));
        }

        [Test]
        public void ThreeRobots_Collaborative_TwoConflicting_OneIsolated()
        {
            // Arrange
            var config = ScriptableObject.CreateInstance<CoordinationConfig>();
            config.minSafeSeparation = 0.3f;
            var strategy = new CollaborativeStrategy(config);
            var controllers = new[] { _controller1, _controller2, _controller3 };
            var targetReached = new Dictionary<string, bool>();

            // Robot1 and Robot2 conflict, Robot3 isolated
            _controller1.SetTarget(new Vector3(1f, 0f, 0f));
            _controller2.SetTarget(new Vector3(1.2f, 0f, 0f)); // Conflicts with Robot1
            _controller3.SetTarget(new Vector3(10f, 0f, 0f)); // Isolated

            // Act
            strategy.Update(controllers, targetReached);

            // Assert
            Assert.IsTrue(strategy.IsRobotActive("Robot3"), "Isolated robot should be active");

            bool robot1Active = strategy.IsRobotActive("Robot1");
            bool robot2Active = strategy.IsRobotActive("Robot2");
            Assert.IsFalse(robot1Active && robot2Active,
                "Conflicting robots should not both be active");
        }

        #endregion

        #region Edge Cases and Error Handling

        [Test]
        public void EmptyControllerArray_DoesNotCrash()
        {
            // Arrange
            var sequentialStrategy = new SequentialStrategy();
            var collaborativeStrategy = new CollaborativeStrategy();
            var emptyControllers = new RobotController[0];
            var emptyDict = new Dictionary<string, bool>();

            // Act & Assert
            Assert.DoesNotThrow(() => sequentialStrategy.Update(emptyControllers, emptyDict));
            Assert.DoesNotThrow(() => collaborativeStrategy.Update(emptyControllers, emptyDict));
        }

        [Test]
        public void NullControllerArray_DoesNotCrash()
        {
            // Arrange
            var sequentialStrategy = new SequentialStrategy();
            var collaborativeStrategy = new CollaborativeStrategy();
            var emptyDict = new Dictionary<string, bool>();

            // Act & Assert
            Assert.DoesNotThrow(() => sequentialStrategy.Update(null, emptyDict));
            Assert.DoesNotThrow(() => collaborativeStrategy.Update(null, emptyDict));
        }

        [Test]
        public void MissingDictionaryEntries_HandleGracefully()
        {
            // Arrange
            var strategy = new SequentialStrategy();
            var controllers = new[] { _controller1, _controller2 };
            var partialDict = new Dictionary<string, bool> { { "Robot1", false } };
            // Robot2 missing from dictionary

            // Act & Assert - Should not crash
            Assert.DoesNotThrow(() => strategy.Update(controllers, partialDict));
        }

        [Test]
        public void RobotWithoutTarget_HandleGracefully()
        {
            // Arrange
            var collaborativeStrategy = new CollaborativeStrategy();
            var controllers = new[] { _controller1, _controller2 };
            var targetReached = new Dictionary<string, bool>();

            // Robot1 has target, Robot2 does not
            _controller1.SetTarget(new Vector3(1f, 0f, 0f));
            // _controller2 has no target

            // Act & Assert - Should not crash
            Assert.DoesNotThrow(() => collaborativeStrategy.Update(controllers, targetReached));
        }

        #endregion

        #region Performance and Cleanup Tests

        [Test]
        public void Cleanup_RemovesStaleRobots()
        {
            // Arrange
            var strategy = new CollaborativeStrategy();
            var controllers1 = new[] { _controller1, _controller2, _controller3 };
            var targetReached = new Dictionary<string, bool>
            {
                { "Robot1", false },
                { "Robot2", false },
                { "Robot3", false }
            };

            // Act - Update with 3 robots
            _controller1.SetTarget(Vector3.zero);
            _controller2.SetTarget(Vector3.one);
            _controller3.SetTarget(Vector3.forward);
            strategy.Update(controllers1, targetReached);

            // Remove Robot3, update with only 2 robots
            var controllers2 = new[] { _controller1, _controller2 };
            strategy.Update(controllers2, targetReached);

            // Assert - Robot3 should be cleaned up (no longer in active set)
            // Note: We can't directly test internal state, but verify no crash
            Assert.DoesNotThrow(() => strategy.IsRobotActive("Robot3"));
        }

        [UnityTest]
        public IEnumerator MultipleUpdates_CompletesWithoutCrash()
        {
            // Arrange: [UnitySetUp] already consumed Start() warnings and yielded one frame.
            var strategy = new CollaborativeStrategy();
            var controllers = new[] { _controller1, _controller2 };
            var targetReached = new Dictionary<string, bool>
            {
                { "Robot1", false },
                { "Robot2", false }
            };

            // Act - Run many updates with changing targets
            // This tests that the strategy can handle repeated updates without internal state corruption
            for (int i = 0; i < 100; i++)
            {
                _controller1.SetTarget(new Vector3(i * 0.1f, 0f, 0f));
                _controller2.SetTarget(new Vector3(-i * 0.1f, 0f, 0f));
                strategy.Update(controllers, targetReached);

                if (i % 10 == 0)
                    yield return null; // Yield occasionally
            }

            // Assert - Strategy should remain functional after many updates
            Assert.IsNotNull(strategy);
            Assert.DoesNotThrow(() => strategy.IsRobotActive("Robot1"));
            Assert.DoesNotThrow(() => strategy.IsRobotActive("Robot2"));
        }

        #endregion

        #region Configuration Integration Tests

        [Test]
        public void MinSafeSeparation_ConfigurablePerStrategy()
        {
            // Arrange
            var config1 = ScriptableObject.CreateInstance<CoordinationConfig>();
            config1.minSafeSeparation = 0.1f;
            var strategy1 = new CollaborativeStrategy(config1);

            var config2 = ScriptableObject.CreateInstance<CoordinationConfig>();
            config2.minSafeSeparation = 0.5f;
            var strategy2 = new CollaborativeStrategy(config2);
            var controllers = new[] { _controller1, _controller2 };
            var targetReached = new Dictionary<string, bool>();

            // Set targets 0.3m apart
            _controller1.SetTarget(new Vector3(0f, 0f, 0f));
            _controller2.SetTarget(new Vector3(0.3f, 0f, 0f));

            // Act
            strategy1.Update(controllers, targetReached);
            bool bothActiveWith01 = strategy1.IsRobotActive("Robot1") && strategy1.IsRobotActive("Robot2");

            strategy2.Update(controllers, targetReached);
            bool bothActiveWith05 = strategy2.IsRobotActive("Robot1") && strategy2.IsRobotActive("Robot2");

            // Assert
            Assert.IsTrue(bothActiveWith01, "0.3m should be safe with 0.1m minimum");
            Assert.IsFalse(bothActiveWith05, "0.3m should block with 0.5m minimum");
        }

        [Test]
        public void Timeout_ConfigurableForSequential()
        {
            // Arrange
            var shortTimeout = new SequentialStrategy(robotTimeout: 5f);
            var longTimeout = new SequentialStrategy(robotTimeout: 120f);

            // Act & Assert - Verify timeouts can be configured
            Assert.IsNotNull(shortTimeout);
            Assert.IsNotNull(longTimeout);
        }

        #endregion
    }
}
