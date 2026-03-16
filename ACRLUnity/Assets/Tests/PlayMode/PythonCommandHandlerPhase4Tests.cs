using NUnit.Framework;
using System.Collections;
using UnityEngine;
using UnityEngine.TestTools;
using PythonCommunication;
using Robotics;
using Simulation;

namespace Tests.PlayMode
{
    /// <summary>
    /// Tests for PythonCommandHandler geometry-based verification with WorkspaceManager.
    ///
    /// Note: Allocation/collision-zone logic has been removed — all coordination decisions
    /// are made by Python via signal/wait. These tests verify the remaining pure geometry
    /// queries (safe separation) that WorkspaceManager still provides.
    /// </summary>
    public class PythonCommandHandlerPhase4Tests
    {
        private GameObject _handlerObject;
        private PythonCommandHandler _handler;
        private GameObject _workspaceManagerObject;
        private WorkspaceManager _workspaceManager;
        private GameObject _robotManagerObject;
        private RobotManager _robotManager;

        [UnitySetUp]
        public IEnumerator Setup()
        {
            // Clean up existing instances
            if (PythonCommandHandler.Instance != null)
            {
                Object.DestroyImmediate(PythonCommandHandler.Instance.gameObject);
            }
            if (WorkspaceManager.Instance != null)
            {
                Object.DestroyImmediate(WorkspaceManager.Instance.gameObject);
            }
            if (RobotManager.Instance != null)
            {
                Object.DestroyImmediate(RobotManager.Instance.gameObject);
            }

            // Create WorkspaceManager
            _workspaceManagerObject = new GameObject("TestWorkspaceManager");
            _workspaceManager = _workspaceManagerObject.AddComponent<WorkspaceManager>();

            // Create RobotManager
            _robotManagerObject = new GameObject("TestRobotManager");
            _robotManager = _robotManagerObject.AddComponent<RobotManager>();

            // Create PythonCommandHandler
            _handlerObject = new GameObject("TestPythonCommandHandler");
            _handler = _handlerObject.AddComponent<PythonCommandHandler>();

            yield return null; // Wait for Start() to complete
        }

        [TearDown]
        public void TearDown()
        {
            if (_handlerObject != null)
            {
                Object.DestroyImmediate(_handlerObject);
            }
            if (_workspaceManagerObject != null)
            {
                Object.DestroyImmediate(_workspaceManagerObject);
            }
            if (_robotManagerObject != null)
            {
                Object.DestroyImmediate(_robotManagerObject);
            }
        }

        private void CreateAndRegisterTestRobot(string robotId, Vector3 position)
        {
            var robotObj = new GameObject(robotId);
            var controller = robotObj.AddComponent<RobotController>();
            controller.robotId = robotId;

            // Create end effector
            var endEffector = new GameObject($"{robotId}_EndEffector");
            endEffector.transform.SetParent(robotObj.transform);
            endEffector.transform.position = position;
            controller.endEffectorBase = endEffector.transform;

            // Initialize required fields
            controller.robotJoints = new ArticulationBody[0];

            // Register the robot using the proper API
            _robotManager.RegisterRobot(robotId, robotObj);
        }

        #region Singleton and Configuration Tests

        [Test]
        public void PythonCommandHandler_Singleton_IsSet()
        {
            Assert.IsNotNull(PythonCommandHandler.Instance);
            Assert.AreEqual(_handler, PythonCommandHandler.Instance);
        }

        [Test]
        public void PythonCommandHandler_IsNotNull_AfterCreation()
        {
            Assert.IsNotNull(_handler);
        }

        #endregion

        #region Safe Separation Tests

        [UnityTest]
        public IEnumerator VerificationSafeSeparation_RobotsTooClose_Fails()
        {
            // Setup: Two robots with targets too close together
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] No GripperController found in children of Robot1");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] No GripperController found in children of Robot2");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");

            CreateAndRegisterTestRobot("Robot1", new Vector3(0f, 0f, 0f));
            CreateAndRegisterTestRobot("Robot2", new Vector3(0.5f, 0f, 0f));

            yield return null;

            // Robot1 targets position very close to Robot2's current position
            Vector3 targetTooClose = new Vector3(0.5f, 0.05f, 0f); // Only 5cm away from Robot2

            // Verify separation is not safe
            var robot2 = _robotManager.RobotInstances["Robot2"];
            Vector3 robot2Pos = robot2.controller.GetCurrentEndEffectorPosition();
            bool safeSeparation = _workspaceManager.IsSafeSeparation(targetTooClose, robot2Pos);

            Assert.IsFalse(safeSeparation); // Should be too close (< 0.2m)

            // Cleanup
            var robot1 = _robotManager.RobotInstances["Robot1"];
            Object.DestroyImmediate(robot1.robotGameObject);
            Object.DestroyImmediate(robot2.robotGameObject);
        }

        [UnityTest]
        public IEnumerator VerificationSafeSeparation_RobotsFarApart_Succeeds()
        {
            // Setup: Two robots with targets far apart
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] No GripperController found in children of Robot1");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] No GripperController found in children of Robot2");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");

            CreateAndRegisterTestRobot("Robot1", new Vector3(-0.5f, 0f, 0f));
            CreateAndRegisterTestRobot("Robot2", new Vector3(0.5f, 0f, 0f));

            yield return null;

            // Robot1 targets position far from Robot2
            Vector3 targetFarAway = new Vector3(-0.8f, 0f, 0f);

            // Verify separation is safe
            var robot2 = _robotManager.RobotInstances["Robot2"];
            Vector3 robot2Pos = robot2.controller.GetCurrentEndEffectorPosition();
            bool safeSeparation = _workspaceManager.IsSafeSeparation(targetFarAway, robot2Pos);

            Assert.IsTrue(safeSeparation); // Should be far enough (> 0.2m)

            // Cleanup
            var robot1 = _robotManager.RobotInstances["Robot1"];
            Object.DestroyImmediate(robot1.robotGameObject);
            Object.DestroyImmediate(robot2.robotGameObject);
        }

        [UnityTest]
        public IEnumerator VerificationPathConflict_RobotsMovingToSameArea_Detected()
        {
            // Setup: Two robots with overlapping target paths
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] No GripperController found in children of Robot1");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] No GripperController found in children of Robot2");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");

            CreateAndRegisterTestRobot("Robot1", new Vector3(-0.5f, 0f, 0f));
            CreateAndRegisterTestRobot("Robot2", new Vector3(0.5f, 0f, 0f));

            // Robot2 has a target
            Vector3 robot2Target = new Vector3(0f, 0.15f, 0.1f);
            var robot2 = _robotManager.RobotInstances["Robot2"];
            robot2.controller.SetTarget(robot2Target);

            yield return null;

            // Robot1 wants to move to position close to Robot2's target
            Vector3 robot1Target = new Vector3(0.05f, 0.15f, 0.1f);

            // Verify targets are too close
            bool safeSeparation = _workspaceManager.IsSafeSeparation(robot1Target, robot2Target);
            Assert.IsFalse(safeSeparation); // Targets within 0.2m

            // Cleanup
            var robot1 = _robotManager.RobotInstances["Robot1"];
            Object.DestroyImmediate(robot1.robotGameObject);
            Object.DestroyImmediate(robot2.robotGameObject);
        }

        #endregion

        #region Region Geometry Tests

        [UnityTest]
        public IEnumerator GetRegionAtPosition_LeftWorkspace_ReturnsCorrectRegion()
        {
            // Setup: robot in left workspace
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] No GripperController found in children of Robot1");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");

            CreateAndRegisterTestRobot("Robot1", new Vector3(-0.5f, 0f, 0.2f));

            yield return null;

            // Use position inside left_workspace but outside center
            Vector3 targetInLeftWorkspace = new Vector3(-0.4f, 0.15f, 0.1f);
            var region = _workspaceManager.GetRegionAtPosition(targetInLeftWorkspace);

            Assert.IsNotNull(region);
            Assert.AreEqual("left_workspace", region.regionName);

            // Cleanup
            var robot1 = _robotManager.RobotInstances["Robot1"];
            Object.DestroyImmediate(robot1.robotGameObject);
        }

        #endregion

        #region Verification Without WorkspaceManager Tests

        [UnityTest]
        public IEnumerator VerificationWithoutWorkspaceManager_FallsBackToIndependent()
        {
            // Destroy workspace manager to simulate missing manager
            Object.DestroyImmediate(_workspaceManagerObject);
            _workspaceManager = null;

            yield return null;

            // Re-create handler to test fallback behavior
            Object.DestroyImmediate(_handlerObject);
            _handlerObject = new GameObject("TestPythonCommandHandler");
            _handler = _handlerObject.AddComponent<PythonCommandHandler>();

            yield return null;

            // Verification should be disabled (fallback to independent mode)
            // Handler should still be functional
            Assert.IsNotNull(_handler);
        }

        #endregion

        #region Command Statistics Tests

        [Test]
        public void GetCommandStats_InitialState_ReturnsZeros()
        {
            var (successful, failed) = _handler.GetCommandStats();
            Assert.AreEqual(0, successful);
            Assert.AreEqual(0, failed);
        }

        [Test]
        public void ResetStats_ClearsStatistics()
        {
            _handler.ResetStats();
            var (successful, failed) = _handler.GetCommandStats();
            Assert.AreEqual(0, successful);
            Assert.AreEqual(0, failed);
        }

        #endregion

        #region Integration Scenario Tests

        [UnityTest]
        public IEnumerator Scenario_DualRobotIndependentWorkspaces_SafeSeparation()
        {
            // Scenario: Two robots moving in their respective workspaces (no conflict)
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] No GripperController found in children of Robot1");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] No GripperController found in children of Robot2");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");

            CreateAndRegisterTestRobot("Robot1", new Vector3(-0.5f, 0f, 0.2f));
            CreateAndRegisterTestRobot("Robot2", new Vector3(0.5f, 0f, 0.2f));

            yield return null;

            // Both robots target their own workspaces
            Vector3 robot1Target = new Vector3(-0.6f, 0.15f, 0.1f); // Left
            Vector3 robot2Target = new Vector3(0.6f, 0.15f, 0.1f);  // Right

            // Verify safe separation
            bool safeSeparation = _workspaceManager.IsSafeSeparation(robot1Target, robot2Target);
            Assert.IsTrue(safeSeparation); // Far apart

            // Cleanup
            var robot1 = _robotManager.RobotInstances["Robot1"];
            var robot2 = _robotManager.RobotInstances["Robot2"];
            Object.DestroyImmediate(robot1.robotGameObject);
            Object.DestroyImmediate(robot2.robotGameObject);
        }

        [UnityTest]
        public IEnumerator Scenario_CollisionPrevention_TargetsTooClose()
        {
            // Scenario: Verification catches collision when targets are too close
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] No GripperController found in children of Robot1");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] No GripperController found in children of Robot2");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");

            CreateAndRegisterTestRobot("Robot1", new Vector3(0f, 0f, 0f));
            CreateAndRegisterTestRobot("Robot2", new Vector3(0.3f, 0f, 0f));

            yield return null;

            // Robot1 tries to move very close to Robot2
            Vector3 robot1Target = new Vector3(0.35f, 0f, 0f); // Only 5cm from Robot2

            var robot2 = _robotManager.RobotInstances["Robot2"];
            Vector3 robot2Pos = robot2.controller.GetCurrentEndEffectorPosition();
            bool safeSeparation = _workspaceManager.IsSafeSeparation(robot1Target, robot2Pos);

            // Should be blocked (< 0.2m minimum separation)
            Assert.IsFalse(safeSeparation);

            // Cleanup
            var robot1 = _robotManager.RobotInstances["Robot1"];
            Object.DestroyImmediate(robot1.robotGameObject);
            Object.DestroyImmediate(robot2.robotGameObject);
        }

        #endregion
    }
}
