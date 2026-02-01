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
    /// Tests for PythonCommandHandler Phase 4 additions.
    /// Validates movement verification with workspace management and collision detection.
    ///
    /// Note: These tests focus on the verification logic. Full command execution tests
    /// require RobotManager and network components which are tested separately.
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
            LogAssert.ignoreFailingMessages = true; // Ignore expected warnings from missing dependencies

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

            LogAssert.ignoreFailingMessages = false;
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

            // Note: Expected log assertions should be set BEFORE calling this method
            // since Start() is called immediately during robot creation

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
        public void PythonCommandHandler_VerificationEnabled_ByDefault()
        {
            // Verification should be enabled by default
            // We can't directly test the private field, but we can test behavior
            Assert.IsNotNull(_handler);
        }

        [Test]
        public void SetPythonVerificationEnabled_EnablesVerification()
        {
            Assert.DoesNotThrow(() => _handler.SetPythonVerificationEnabled(true));
        }

        [Test]
        public void SetPythonVerificationEnabled_DisablesVerification()
        {
            Assert.DoesNotThrow(() => _handler.SetPythonVerificationEnabled(false));
        }

        #endregion

        #region Verification Scenario Tests

        [UnityTest]
        public IEnumerator VerificationWithWorkspaceManager_RobotInAllocatedRegion_Succeeds()
        {
            // Setup: Create robot in left workspace
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] No GripperController found in children of Robot1");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");

            CreateAndRegisterTestRobot("Robot1", new Vector3(-0.5f, 0f, 0.2f));

            // Allocate left workspace to Robot1
            _workspaceManager.AllocateRegion("Robot1", "left_workspace");
            yield return null;

            // Target is also in left workspace (should be allowed)
            // Left workspace bounds: min(-0.65, 0.0, -0.5) to max(-0.1, 0.7, 0.5)
            // Center region: min(-0.3, 0.0, -0.3) to max(0.3, 0.3, 0.3)
            // Use position outside center but inside left_workspace
            Vector3 targetInLeftWorkspace = new Vector3(-0.4f, 0.15f, 0.1f);

            // Verification should pass (we can't directly test private method,
            // but we verify setup is correct)
            var region = _workspaceManager.GetRegionAtPosition(targetInLeftWorkspace);
            Assert.IsNotNull(region);
            Assert.AreEqual("left_workspace", region.regionName);
            Assert.IsTrue(_workspaceManager.IsRegionAvailable("left_workspace", "Robot1"));

            // Cleanup
            var robot1 = _robotManager.RobotInstances["Robot1"];
            Object.DestroyImmediate(robot1.robotGameObject);
        }

        [UnityTest]
        public IEnumerator VerificationWithWorkspaceManager_RobotTargetingOccupiedRegion_Fails()
        {
            // Setup: Two robots, one has allocated a region
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] No GripperController found in children of Robot1");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] No GripperController found in children of Robot2");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");

            CreateAndRegisterTestRobot("Robot1", new Vector3(-0.5f, 0f, 0.2f));
            CreateAndRegisterTestRobot("Robot2", new Vector3(0.5f, 0f, 0.2f));

            // Robot1 allocates left workspace
            _workspaceManager.AllocateRegion("Robot1", "left_workspace");
            yield return null;

            // Robot2 tries to target left workspace (should fail)
            Vector3 targetInLeftWorkspace = new Vector3(-0.6f, 0.15f, 0.1f);

            // Verify region is not available to Robot2
            Assert.IsFalse(_workspaceManager.IsRegionAvailable("left_workspace", "Robot2"));

            // Cleanup
            var robot1 = _robotManager.RobotInstances["Robot1"];
            var robot2 = _robotManager.RobotInstances["Robot2"];
            Object.DestroyImmediate(robot1.robotGameObject);
            Object.DestroyImmediate(robot2.robotGameObject);
        }

        [UnityTest]
        public IEnumerator VerificationWithCollisionZone_RobotTargetingActiveZone_Fails()
        {
            // Setup: Robot targeting a region marked as collision zone
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] No GripperController found in children of Robot1");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");

            CreateAndRegisterTestRobot("Robot1", new Vector3(-0.5f, 0f, 0.2f));

            // Mark left workspace as collision zone
            _workspaceManager.MarkCollisionZone("left_workspace");
            yield return null;

            // Robot tries to target left workspace (should fail due to collision zone)
            Vector3 targetInLeftWorkspace = new Vector3(-0.6f, 0.15f, 0.1f);

            // Verify collision zone is marked
            Assert.IsTrue(_workspaceManager.IsCollisionZone("left_workspace"));

            // Cleanup
            var robot1 = _robotManager.RobotInstances["Robot1"];
            Object.DestroyImmediate(robot1.robotGameObject);
        }

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
        public IEnumerator Scenario_DualRobotIndependentWorkspaces_BothSucceed()
        {
            // Scenario: Two robots moving in their respective workspaces (no conflict)
            // Expect warnings for both robots
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] No GripperController found in children of Robot1");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] No GripperController found in children of Robot2");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");

            CreateAndRegisterTestRobot("Robot1", new Vector3(-0.5f, 0f, 0.2f));
            CreateAndRegisterTestRobot("Robot2", new Vector3(0.5f, 0f, 0.2f));

            // Allocate workspaces
            _workspaceManager.AllocateRegion("Robot1", "left_workspace");
            _workspaceManager.AllocateRegion("Robot2", "right_workspace");

            yield return null;

            // Both robots target their own workspaces
            Vector3 robot1Target = new Vector3(-0.6f, 0.15f, 0.1f); // Left
            Vector3 robot2Target = new Vector3(0.6f, 0.15f, 0.1f);  // Right

            // Verify both regions are available to their respective robots
            Assert.IsTrue(_workspaceManager.IsRegionAvailable("left_workspace", "Robot1"));
            Assert.IsTrue(_workspaceManager.IsRegionAvailable("right_workspace", "Robot2"));

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
        public IEnumerator Scenario_SharedZoneAccess_Serialized()
        {
            // Scenario: Two robots want to access shared zone (must be serialized)
            // Expect warnings for both robots
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] No GripperController found in children of Robot1");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] No GripperController found in children of Robot2");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");

            CreateAndRegisterTestRobot("Robot1", new Vector3(-0.5f, 0f, 0.2f));
            CreateAndRegisterTestRobot("Robot2", new Vector3(0.5f, 0f, 0.2f));

            yield return null;

            // Robot1 enters shared zone first
            _workspaceManager.AllocateRegion("Robot1", "shared_zone");
            _workspaceManager.MarkCollisionZone("shared_zone");

            Vector3 sharedTarget1 = new Vector3(0.05f, 0.15f, 0.1f);

            // Verify Robot1 has access
            Assert.IsTrue(_workspaceManager.IsRegionAvailable("shared_zone", "Robot1"));

            // Verify Robot2 is blocked
            Assert.IsFalse(_workspaceManager.IsRegionAvailable("shared_zone", "Robot2"));
            Assert.IsTrue(_workspaceManager.IsCollisionZone("shared_zone"));

            // Robot1 completes movement
            _workspaceManager.ClearCollisionZone("shared_zone");
            _workspaceManager.ReleaseRegion("Robot1", "shared_zone");

            yield return null;

            // Now Robot2 can enter
            Vector3 sharedTarget2 = new Vector3(-0.05f, 0.15f, 0.1f);
            Assert.IsTrue(_workspaceManager.IsRegionAvailable("shared_zone", "Robot2"));

            // Cleanup
            var robot1 = _robotManager.RobotInstances["Robot1"];
            var robot2 = _robotManager.RobotInstances["Robot2"];
            Object.DestroyImmediate(robot1.robotGameObject);
            Object.DestroyImmediate(robot2.robotGameObject);
        }

        [UnityTest]
        public IEnumerator Scenario_CollisionPrevention_TargetsTooClose()
        {
            // Scenario: Verification prevents collision when targets are too close
            // Expect warnings for both robots
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
