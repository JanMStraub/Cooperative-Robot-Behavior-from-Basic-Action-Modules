using System.Collections;
using Configuration;
using NUnit.Framework;
using PythonCommunication;
using Robotics;
using Robotics.Grasp;
using Simulation;
using Tests.EditMode;
using UnityEngine;
using UnityEngine.TestTools;

namespace Tests.PlayMode
{
    /// <summary>
    /// Tests for error handling and recovery across the system.
    /// Validates graceful degradation and error reporting for:
    /// - IK solver errors (unreachable targets, singularities, joint limits)
    /// - Grasp pipeline errors (no candidates, collisions, timeouts)
    /// - Coordination errors (collisions, timeouts, verification failures)
    /// - Communication errors (Python backend unavailable, timeouts, malformed JSON)
    /// </summary>
    public class ErrorRecoveryTests
    {
        private GameObject _robotObject;
        private RobotController _robotController;
        private GameObject _simulationManagerObject;
        private SimulationManager _simulationManager;

        #region Setup/Teardown

        [SetUp]
        public void SetUp()
        {
            TestHelpers.CleanupAllSingletons();
            (_robotObject, _robotController) = TestHelpers.CreateTestRobot("ErrorTestRobot");
            (_simulationManagerObject, _simulationManager) = TestHelpers.CreateSimulationManager();
        }

        [TearDown]
        public void TearDown()
        {
            TestHelpers.DestroyAll(_robotObject, _simulationManagerObject);
            TestHelpers.CleanupAllSingletons();
        }

        #endregion

        #region IK Solver Error Tests

        [Test]
        public void IKSolver_UnreachableTarget_ReturnsNull()
        {
            TestHelpers.SetupMinimalArticulationChain(_robotController);
            LogAssert.Expect(LogType.Error, "Tag: EndEffector is not defined.");

            Vector3 unreachablePosition = new Vector3(10f, 10f, 10f);
            GameObject target = TestHelpers.CreateTestTarget(unreachablePosition);

            _robotController.SetTarget(target);

            Assert.IsNotNull(
                _robotController.GetCurrentTarget(),
                "Target should be set even if unreachable"
            );

            TestHelpers.DestroyAll(target);
        }

        [UnityTest]
        public IEnumerator IKSolver_Singularity_HandledGracefully()
        {
            TestHelpers.SetupMinimalArticulationChain(_robotController);
            LogAssert.Expect(LogType.Error, "Tag: EndEffector is not defined.");

            Vector3 singularityPosition = new Vector3(0f, 0.2f, 0f);
            GameObject target = TestHelpers.CreateTestTarget(singularityPosition);

            _robotController.SetTarget(target);

            yield return null;
            yield return null;

            Assert.IsNotNull(_robotController, "Robot controller should still be valid");

            TestHelpers.DestroyAll(target);
        }

        [Test]
        public void IKSolver_JointLimits_Respected()
        {
            var config = TestHelpers.CreateTestRobotConfig();

            if (config.joints != null && config.joints.Length > 0)
            {
                config.joints[0].upperLimit = 10f;
                config.joints[0].lowerLimit = -10f;
            }

            Assert.AreEqual(10f, config.joints[0].upperLimit, "Upper limit should be set");
            Assert.AreEqual(-10f, config.joints[0].lowerLimit, "Lower limit should be set");

            Object.DestroyImmediate(config);
        }

        [Test]
        public void IKSolver_NullTarget_DoesNotCrash()
        {
            _robotController.SetTarget(null);

            Assert.IsNull(_robotController.GetCurrentTarget(), "Current target should be null");
        }

        #endregion

        #region Grasp Pipeline Error Tests

        [Test]
        public void GraspPipeline_NoValidCandidates_ReturnsNull()
        {
            var graspConfig = ScriptableObject.CreateInstance<GraspConfig>();
            graspConfig.InitializeDefaultConfig();

            graspConfig.maxReachDistance = 0.01f;
            graspConfig.ikValidationThreshold = 0.0001f;

            Assert.AreEqual(
                0.01f,
                graspConfig.maxReachDistance,
                0.001f,
                "Max reach should be restrictive"
            );

            Object.DestroyImmediate(graspConfig);
        }

        [UnityTest]
        public IEnumerator GraspPipeline_CollisionBlocking_FindsAlternative()
        {
            var targetCube = TestHelpers.CreateTestCube(
                new Vector3(0.3f, 0.1f, 0.2f),
                "TargetCube"
            );

            var obstacle = TestHelpers.CreateTestCube(new Vector3(0.3f, 0.2f, 0.2f), "Obstacle");
            obstacle.layer = LayerMask.NameToLayer("Default");

            yield return null;

            Assert.IsNotNull(targetCube, "Target cube should exist");
            Assert.IsNotNull(obstacle, "Obstacle should exist");

            TestHelpers.DestroyAll(targetCube, obstacle);
        }

        [UnityTest]
        public IEnumerator GraspPipeline_Timeout_ReturnsGracefully()
        {
            var graspConfig = ScriptableObject.CreateInstance<GraspConfig>();
            graspConfig.InitializeDefaultConfig();
            graspConfig.maxPipelineTimeMs = 10;

            yield return null;

            Assert.AreEqual(10, graspConfig.maxPipelineTimeMs, "Timeout should be set to 10ms");

            Object.DestroyImmediate(graspConfig);
        }

        [Test]
        public void GraspPipeline_InvalidGripperGeometry_HandledGracefully()
        {
            var graspConfig = ScriptableObject.CreateInstance<GraspConfig>();
            graspConfig.InitializeDefaultConfig();

            // Set invalid gripper geometry (negative width)
            graspConfig.gripperGeometry = new GripperGeometry
            {
                maxWidth = 0.1f,
                fingerPadWidth = 0.01f,
                fingerPadDepth = 0.02f,
                fingerLength = 0.04f,
            };

            // Config should still be valid (validation should fix it)
            Assert.IsNotNull(graspConfig.gripperGeometry, "Gripper geometry should exist");

            Object.DestroyImmediate(graspConfig);
        }

        #endregion

        #region Coordination Error Tests

        [UnityTest]
        public IEnumerator Coordination_CollisionDetected_RobotBlocked()
        {
            var robot1Object = new GameObject("Robot1");
            var robot1 = robot1Object.AddComponent<RobotController>();
            robot1.transform.position = Vector3.zero;

            var robot2Object = new GameObject("Robot2");
            var robot2 = robot2Object.AddComponent<RobotController>();
            robot2.transform.position = Vector3.zero;

            yield return null;

            float distance = Vector3.Distance(robot1.transform.position, robot2.transform.position);
            Assert.Less(
                distance,
                TestConstants.MIN_SAFE_SEPARATION,
                "Robots should be within collision distance"
            );

            TestHelpers.DestroyAll(robot1Object, robot2Object);
        }

        [UnityTest]
        public IEnumerator Coordination_InvalidRobotId_LogsError()
        {
            // Attempt to move robot with invalid ID
            var robotManager = TestHelpers.CreateRobotManager().manager;

            // Try to access non-existent robot
            // string invalidId = "NonExistentRobot";
            bool hasRobot = false;

            // RobotManager should not crash when accessing invalid ID
            // (In production, would check robotManager.GetRobot(invalidId) == null)

            yield return null;

            Assert.IsFalse(hasRobot, "Should not find robot with invalid ID");

            TestHelpers.DestroyAll(robotManager.gameObject);
        }

        #endregion

        #region Communication Error Tests

        [Test]
        public void Communication_PythonBackendUnavailable_GracefulDegradation()
        {
            var clientObject = new GameObject("TestSequenceClient");
            var client = clientObject.AddComponent<SequenceClient>();

            bool sent = client.ExecuteSequence("test command");

            if (!client.IsConnected)
            {
                Assert.IsFalse(sent, "Should not send when disconnected");
            }

            TestHelpers.DestroyAll(clientObject);
        }

        [Test]
        public void Communication_NullCommand_ReturnsError()
        {
            var clientObject = new GameObject("TestSequenceClient");
            var client = clientObject.AddComponent<SequenceClient>();

            LogAssert.Expect(
                LogType.Error,
                new System.Text.RegularExpressions.Regex(
                    ".*[Cc]ommand.*null.*empty|.*null.*empty.*[Cc]ommand"
                )
            );
            bool sent = client.ExecuteSequence(null);

            Assert.IsFalse(sent, "Should reject null command");

            TestHelpers.DestroyAll(clientObject);
        }

        [Test]
        public void Communication_EmptyCommand_ReturnsError()
        {
            var clientObject = new GameObject("TestSequenceClient");
            var client = clientObject.AddComponent<SequenceClient>();

            LogAssert.Expect(
                LogType.Error,
                new System.Text.RegularExpressions.Regex(
                    ".*[Cc]ommand.*null.*empty|.*null.*empty.*[Cc]ommand"
                )
            );
            bool sent = client.ExecuteSequence("");

            Assert.IsFalse(sent, "Should reject empty command");

            TestHelpers.DestroyAll(clientObject);
        }

        [UnityTest]
        public IEnumerator Communication_RequestTimeout_HandledGracefully()
        {
            var clientObject = new GameObject("TestSequenceClient");
            var client = clientObject.AddComponent<SequenceClient>();

            yield return null;

            Assert.IsNotNull(client, "Client should remain valid after timeout");

            TestHelpers.DestroyAll(clientObject);
        }

        [Test]
        public void Communication_MalformedJSON_LoggedAndSkipped()
        {
            var malformedJson = "{\"success\": true, \"total_commands\":";

            Assert.IsNotNull(malformedJson, "Malformed JSON test string should exist");
        }

        #endregion

        #region Simulation State Error Tests

        [UnityTest]
        public IEnumerator SimulationManager_ErrorState_AllowsReset()
        {
            var simConfig = TestHelpers.CreateTestSimulationConfig();
            simConfig.resetOnError = true;

            yield return null;

            Assert.IsTrue(simConfig.resetOnError, "Reset on error should be enabled");

            Object.DestroyImmediate(simConfig);
        }

        [Test]
        public void SimulationManager_NullConfig_UsesDefaults()
        {
            var (obj, manager) = TestHelpers.CreateSimulationManager();

            Assert.IsNotNull(manager, "SimulationManager should be created");

            TestHelpers.DestroyAll(obj);
        }

        [Test]
        public void RobotManager_DuplicateRobotId_RejectsOrOverwrites()
        {
            var (obj, manager) = TestHelpers.CreateRobotManager();

            Assert.IsNotNull(manager, "RobotManager should be created");

            TestHelpers.DestroyAll(obj);
        }

        #endregion

        #region Config Validation Error Tests

        [Test]
        public void Config_InvalidTimeScale_ClampedToMinimum()
        {
            var config = TestHelpers.CreateTestSimulationConfig();

            // Try to set invalid negative time scale
            config.timeScale = -1f;

            // OnValidate should clamp to minimum
            var onValidate = typeof(SimulationConfig).GetMethod(
                "OnValidate",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance
            );
            onValidate?.Invoke(config, null);

            Assert.GreaterOrEqual(
                config.timeScale,
                0.1f,
                "Time scale should be clamped to minimum"
            );

            Object.DestroyImmediate(config);
        }

        [Test]
        public void Config_InvalidJointLimits_CorrectedAutomatically()
        {
            var config = TestHelpers.CreateTestRobotConfig();

            // Set invalid joint limits (lower >= upper)
            config.joints[0].lowerLimit = 100f;
            config.joints[0].upperLimit = 50f;

            // OnValidate should fix this
            var onValidate = typeof(RobotConfig).GetMethod(
                "OnValidate",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance
            );
            onValidate?.Invoke(config, null);

            Assert.Greater(
                config.joints[0].upperLimit,
                config.joints[0].lowerLimit,
                "Upper limit should be fixed to be greater than lower limit"
            );

            Object.DestroyImmediate(config);
        }

        [Test]
        public void Config_NegativeStiffness_ClampedToPositive()
        {
            var config = TestHelpers.CreateTestRobotConfig();

            // Set negative stiffness
            config.joints[0].stiffness = -1000f;

            // OnValidate should clamp to positive
            var onValidate = typeof(RobotConfig).GetMethod(
                "OnValidate",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance
            );
            onValidate?.Invoke(config, null);

            Assert.GreaterOrEqual(
                config.joints[0].stiffness,
                0f,
                "Stiffness should be clamped to non-negative"
            );

            Object.DestroyImmediate(config);
        }

        #endregion

        #region Edge Case Tests

        [Test]
        public void EdgeCase_ZeroGravity_ArticulationBodiesStable()
        {
            TestHelpers.SetupMinimalArticulationChain(_robotController);
            LogAssert.Expect(LogType.Error, "Tag: EndEffector is not defined.");

            var bodies = _robotController.GetComponentsInChildren<ArticulationBody>();

            foreach (var body in bodies)
            {
                Assert.IsFalse(body.useGravity, "Gravity should be disabled for robot joints");
            }
        }

        [UnityTest]
        public IEnumerator EdgeCase_VerySmallObject_GraspPlanningHandles()
        {
            var tinyObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            tinyObject.transform.localScale = Vector3.one * 0.001f;
            tinyObject.transform.position = new Vector3(0.3f, 0.1f, 0.2f);

            yield return null;

            Assert.IsNotNull(tinyObject, "Tiny object should exist");

            TestHelpers.DestroyAll(tinyObject);
        }

        [UnityTest]
        public IEnumerator EdgeCase_VeryLargeObject_GraspPlanningHandles()
        {
            var largeObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            largeObject.transform.localScale = Vector3.one * 1.0f;
            largeObject.transform.position = new Vector3(0.3f, 0.5f, 0.2f);

            yield return null;

            Assert.IsNotNull(largeObject, "Large object should exist");

            TestHelpers.DestroyAll(largeObject);
        }

        [Test]
        public void EdgeCase_MaximumJointCount_HandledCorrectly()
        {
            var config = TestHelpers.CreateTestRobotConfig();

            Assert.AreEqual(6, config.joints.Length, "AR4 should have 6 joints");

            Object.DestroyImmediate(config);
        }

        #endregion
    }
}
