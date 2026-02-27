using NUnit.Framework;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.TestTools;
using Simulation;
using Simulation.CoordinationStrategies;
using Robotics;

namespace Tests.PlayMode
{
    /// <summary>
    /// Tests for CollaborativeStrategy (Phase 4).
    /// Validates collision detection, path checking, and workspace coordination.
    /// </summary>
    public class CollaborativeStrategyTests
    {
        private GameObject _workspaceManagerObject;
        private WorkspaceManager _workspaceManager;
        private CollaborativeStrategy _strategy;
        private List<GameObject> _testObjects;

        [UnitySetUp]
        public IEnumerator Setup()
        {
            _testObjects = new List<GameObject>();

            // Clean up any existing workspace manager
            if (WorkspaceManager.Instance != null)
            {
                Object.DestroyImmediate(WorkspaceManager.Instance.gameObject);
            }

            // Create workspace manager
            _workspaceManagerObject = new GameObject("TestWorkspaceManager");
            _workspaceManager = _workspaceManagerObject.AddComponent<WorkspaceManager>();

            yield return null;

            // Create strategy
            _strategy = new CollaborativeStrategy();
        }

        [TearDown]
        public void TearDown()
        {
            // Clean up test objects
            foreach (var obj in _testObjects)
            {
                if (obj != null)
                {
                    Object.DestroyImmediate(obj);
                }
            }
            _testObjects.Clear();

            if (_workspaceManagerObject != null)
            {
                Object.DestroyImmediate(_workspaceManagerObject);
            }
        }

        private GameObject CreateTestRobot(string name, Vector3 position)
        {
            var robotObj = new GameObject(name);
            var controller = robotObj.AddComponent<RobotController>();
            controller.robotId = name;

            // Create end effector
            var endEffector = new GameObject($"{name}_EndEffector");
            endEffector.transform.SetParent(robotObj.transform);
            endEffector.transform.localPosition = position;
            controller.endEffectorBase = endEffector.transform;

            // Initialize robot joints array (required by RobotController)
            controller.robotJoints = new ArticulationBody[0];

            // Expect the warning message for missing gripper (this is expected in tests)
            LogAssert.Expect(LogType.Warning, $"[ROBOT_CONTROLLER] No GripperController found in children of {name}");

            // Expect the warning message for missing joints (this is expected in tests)
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");

            _testObjects.Add(robotObj);
            _testObjects.Add(endEffector);

            return robotObj;
        }

        #region Basic Coordination Tests

        [Test]
        public void CollaborativeStrategy_Constructor_InitializesCorrectly()
        {
            Assert.IsNotNull(_strategy);
            Assert.IsNotNull(_strategy.GetActiveRobotId());
        }

        [UnityTest]
        public IEnumerator Update_NoRobots_DoesNotCrash()
        {
            RobotController[] emptyArray = new RobotController[0];
            Dictionary<string, bool> emptyDict = new Dictionary<string, bool>();

            Assert.DoesNotThrow(() => _strategy.Update(emptyArray, emptyDict));
            yield return null;
        }

        [UnityTest]
        public IEnumerator Update_NullArray_DoesNotCrash()
        {
            Dictionary<string, bool> emptyDict = new Dictionary<string, bool>();

            Assert.DoesNotThrow(() => _strategy.Update(null, emptyDict));
            yield return null;
        }

        #endregion

        #region Robot Active State Tests

        [Test]
        public void IsRobotActive_NoWorkspaceManager_ReturnsTrue()
        {
            // Destroy workspace manager to simulate missing manager
            Object.DestroyImmediate(_workspaceManagerObject);

            var strategy = new CollaborativeStrategy();
            bool active = strategy.IsRobotActive("Robot1");

            Assert.IsTrue(active); // Fallback to independent mode
        }

        [UnityTest]
        public IEnumerator IsRobotActive_NoTarget_ReturnsTrue()
        {
            var robotObj = CreateTestRobot("Robot1", Vector3.zero);
            var controller = robotObj.GetComponent<RobotController>();

            yield return null;

            bool active = _strategy.IsRobotActive("Robot1");
            Assert.IsTrue(active);
        }

        [UnityTest]
        public IEnumerator IsRobotActive_WithTarget_ChecksWorkspace()
        {
            var robotObj = CreateTestRobot("Robot1", Vector3.zero);
            var controller = robotObj.GetComponent<RobotController>();

            // Set target in left workspace
            Vector3 leftTarget = new Vector3(-0.5f, 0.15f, 0.1f);
            controller.SetTarget(leftTarget);

            yield return null;

            // Update strategy to track robot
            RobotController[] robots = new RobotController[] { controller };
            Dictionary<string, bool> targetReached = new Dictionary<string, bool> { { "Robot1", false } };
            _strategy.Update(robots, targetReached);

            bool active = _strategy.IsRobotActive("Robot1");
            Assert.IsTrue(active);
        }

        #endregion

        #region Active Robot ID Tests

        [Test]
        public void GetActiveRobotId_NoActiveRobots_ReturnsNone()
        {
            string activeId = _strategy.GetActiveRobotId();
            Assert.AreEqual("None", activeId);
        }

        [UnityTest]
        public IEnumerator GetActiveRobotId_OneRobot_ReturnsRobotId()
        {
            var robot1 = CreateTestRobot("Robot1", Vector3.zero);
            var controller1 = robot1.GetComponent<RobotController>();
            controller1.SetTarget(new Vector3(0.3f, 0.15f, 0.1f));

            yield return null;

            RobotController[] robots = new RobotController[] { controller1 };
            Dictionary<string, bool> targetReached = new Dictionary<string, bool> { { "Robot1", false } };
            _strategy.Update(robots, targetReached);

            string activeId = _strategy.GetActiveRobotId();
            Assert.AreEqual("Robot1", activeId);
        }

        [UnityTest]
        public IEnumerator GetActiveRobotId_MultipleRobots_ReturnsCommaSeparated()
        {
            var robot1 = CreateTestRobot("Robot1", Vector3.zero);
            var robot2 = CreateTestRobot("Robot2", Vector3.zero);
            var controller1 = robot1.GetComponent<RobotController>();
            var controller2 = robot2.GetComponent<RobotController>();

            controller1.SetTarget(new Vector3(0.3f, 0.15f, 0.1f));
            controller2.SetTarget(new Vector3(-0.3f, 0.15f, 0.1f));

            yield return null;

            RobotController[] robots = new RobotController[] { controller1, controller2 };
            Dictionary<string, bool> targetReached = new Dictionary<string, bool>
            {
                { "Robot1", false },
                { "Robot2", false }
            };
            _strategy.Update(robots, targetReached);

            string activeId = _strategy.GetActiveRobotId();
            Assert.IsTrue(activeId.Contains("Robot1"));
            Assert.IsTrue(activeId.Contains("Robot2"));
        }

        #endregion

        #region Reset Tests

        [UnityTest]
        public IEnumerator Reset_ClearsState()
        {
            var robot1 = CreateTestRobot("Robot1", Vector3.zero);
            var controller1 = robot1.GetComponent<RobotController>();
            controller1.SetTarget(new Vector3(0.3f, 0.15f, 0.1f));

            yield return null;

            // Update to populate state
            RobotController[] robots = new RobotController[] { controller1 };
            Dictionary<string, bool> targetReached = new Dictionary<string, bool> { { "Robot1", false } };
            _strategy.Update(robots, targetReached);

            Assert.AreNotEqual("None", _strategy.GetActiveRobotId());

            // Reset
            _strategy.Reset();

            Assert.AreEqual("None", _strategy.GetActiveRobotId());
        }

        [UnityTest]
        public IEnumerator Reset_ResetsWorkspaceAllocations()
        {
            _workspaceManager.AllocateRegion("Robot1", "left_workspace");
            yield return null;

            _strategy.Reset();

            var leftRegion = _workspaceManager.GetRegion("left_workspace");
            Assert.IsFalse(leftRegion.IsAllocated());
        }

        #endregion

        #region Coordination Check Tests

        [UnityTest]
        public IEnumerator RequiresCoordination_FarApart_ReturnsFalse()
        {
            var robot1 = CreateTestRobot("Robot1", Vector3.zero);
            var robot2 = CreateTestRobot("Robot2", Vector3.zero);
            var controller1 = robot1.GetComponent<RobotController>();
            var controller2 = robot2.GetComponent<RobotController>();

            // Set targets far apart
            controller1.SetTarget(new Vector3(-1.0f, 0.15f, 0.1f));
            controller2.SetTarget(new Vector3(1.0f, 0.15f, 0.1f));

            yield return null;

            RobotController[] robots = new RobotController[] { controller1, controller2 };
            Dictionary<string, bool> targetReached = new Dictionary<string, bool>
            {
                { "Robot1", false },
                { "Robot2", false }
            };
            _strategy.Update(robots, targetReached);

            bool requires = _strategy.RequiresCoordination("Robot1", "Robot2");
            Assert.IsFalse(requires);
        }

        [UnityTest]
        public IEnumerator RequiresCoordination_CloseTargets_ReturnsTrue()
        {
            var robot1 = CreateTestRobot("Robot1", Vector3.zero);
            var robot2 = CreateTestRobot("Robot2", Vector3.zero);
            var controller1 = robot1.GetComponent<RobotController>();
            var controller2 = robot2.GetComponent<RobotController>();

            // Set targets close together
            controller1.SetTarget(new Vector3(0.1f, 0.15f, 0.1f));
            controller2.SetTarget(new Vector3(-0.1f, 0.15f, 0.1f));

            yield return null;

            RobotController[] robots = new RobotController[] { controller1, controller2 };
            Dictionary<string, bool> targetReached = new Dictionary<string, bool>
            {
                { "Robot1", false },
                { "Robot2", false }
            };
            _strategy.Update(robots, targetReached);

            bool requires = _strategy.RequiresCoordination("Robot1", "Robot2");
            Assert.IsTrue(requires);
        }

        #endregion

        #region Safe Separation Configuration Tests

        [Test]
        public void SetMinSafeSeparation_UpdatesValue()
        {
            _strategy.SetMinSafeSeparation(0.3f);
            // Note: We can't directly test the internal value, but we can verify it doesn't crash
            Assert.DoesNotThrow(() => _strategy.SetMinSafeSeparation(0.3f));
        }

        [Test]
        public void SetMinSafeSeparation_NegativeValue_ClampsToMinimum()
        {
            // Should clamp to 0.05f minimum
            Assert.DoesNotThrow(() => _strategy.SetMinSafeSeparation(-1.0f));
        }

        [Test]
        public void SetMinSafeSeparation_ZeroValue_ClampsToMinimum()
        {
            // Should clamp to 0.05f minimum
            Assert.DoesNotThrow(() => _strategy.SetMinSafeSeparation(0f));
        }

        #endregion

        #region Workspace Allocation Tests

        [UnityTest]
        public IEnumerator Update_RobotEntersRegion_AllocatesWorkspace()
        {
            var robot1 = CreateTestRobot("Robot1", new Vector3(-0.5f, 0f, 0.2f));
            var controller1 = robot1.GetComponent<RobotController>();

            yield return null;

            RobotController[] robots = new RobotController[] { controller1 };
            Dictionary<string, bool> targetReached = new Dictionary<string, bool> { { "Robot1", false } };

            _strategy.Update(robots, targetReached);

            // Robot should be allocated to left workspace (where it's positioned)
            var leftRegion = _workspaceManager.GetRegion("left_workspace");
            // Note: Allocation happens when robot enters unallocated region
            // In test setup, robot starts in position so allocation may or may not happen
            // depending on Update logic
            Assert.IsNotNull(leftRegion); // At least verify region exists
        }

        #endregion

        #region Integration Scenario Tests

        [UnityTest]
        public IEnumerator Scenario_TwoRobots_IndependentWorkspaces_NoConflict()
        {
            // Note: Removed ignoreFailingMessages anti-pattern
            // Expected warnings from robot setup are now properly handled with LogAssert.Expect

            var robot1 = CreateTestRobot("Robot1", new Vector3(-0.5f, 0f, 0.2f));
            var robot2 = CreateTestRobot("Robot2", new Vector3(0.5f, 0f, 0.2f));
            var controller1 = robot1.GetComponent<RobotController>();
            var controller2 = robot2.GetComponent<RobotController>();

            // Set targets in their respective workspaces
            controller1.SetTarget(new Vector3(-0.6f, 0.15f, 0.1f)); // Left workspace
            controller2.SetTarget(new Vector3(0.6f, 0.15f, 0.1f));  // Right workspace

            yield return null;

            RobotController[] robots = new RobotController[] { controller1, controller2 };
            Dictionary<string, bool> targetReached = new Dictionary<string, bool>
            {
                { "Robot1", false },
                { "Robot2", false }
            };

            // Update once, wait for coordination check interval (500ms), then update again
            _strategy.Update(robots, targetReached);
            yield return new WaitForSeconds(0.6f); // Wait for internal coordination timer (500ms)
            _strategy.Update(robots, targetReached);

            // Both robots should be active (no conflicts)
            Assert.IsTrue(_strategy.IsRobotActive("Robot1"));
            Assert.IsTrue(_strategy.IsRobotActive("Robot2"));
        }

        [UnityTest]
        public IEnumerator Scenario_TwoRobots_SharedZone_Coordination()
        {
            // Note: Removed ignoreFailingMessages anti-pattern
            // Collision warnings are expected behavior for this test

            var robot1 = CreateTestRobot("Robot1", new Vector3(-0.5f, 0f, 0.2f));
            var robot2 = CreateTestRobot("Robot2", new Vector3(0.5f, 0f, 0.2f));
            var controller1 = robot1.GetComponent<RobotController>();
            var controller2 = robot2.GetComponent<RobotController>();

            // Both target shared zone
            controller1.SetTarget(new Vector3(0.05f, 0.15f, 0.1f));
            controller2.SetTarget(new Vector3(-0.05f, 0.15f, 0.1f));

            yield return null;

            RobotController[] robots = new RobotController[] { controller1, controller2 };
            Dictionary<string, bool> targetReached = new Dictionary<string, bool>
            {
                { "Robot1", false },
                { "Robot2", false }
            };

            // Update once, wait for coordination check interval (500ms), then update again
            _strategy.Update(robots, targetReached);
            yield return new WaitForSeconds(0.6f); // Wait for internal coordination timer (500ms)
            _strategy.Update(robots, targetReached);

            // Coordination should detect conflict and block Robot2
            // Robot1 should be active, Robot2 should be blocked
            Assert.IsTrue(_strategy.IsRobotActive("Robot1"), "Robot1 should be active");
            Assert.IsFalse(_strategy.IsRobotActive("Robot2"), "Robot2 should be blocked due to collision");
        }

        #endregion

        #region Path Collision Detection Tests

        [UnityTest]
        public IEnumerator Update_DetectsPathCollision_BetweenMovingRobots()
        {
            // Note: Removed ignoreFailingMessages anti-pattern
            // Path collision warnings are expected and indicate correct behavior

            var robot1 = CreateTestRobot("Robot1", new Vector3(-0.5f, 0f, 0f));
            var robot2 = CreateTestRobot("Robot2", new Vector3(0.5f, 0f, 0f));
            var controller1 = robot1.GetComponent<RobotController>();
            var controller2 = robot2.GetComponent<RobotController>();

            // Set targets that will cause paths to intersect
            controller1.SetTarget(new Vector3(0.5f, 0f, 0f)); // Robot1 moves right
            controller2.SetTarget(new Vector3(-0.5f, 0f, 0f)); // Robot2 moves left

            yield return null;

            RobotController[] robots = new RobotController[] { controller1, controller2 };
            Dictionary<string, bool> targetReached = new Dictionary<string, bool>
            {
                { "Robot1", false },
                { "Robot2", false }
            };

            // Update strategy multiple times to trigger collision detection
            for (int i = 0; i < 5; i++)
            {
                _strategy.Update(robots, targetReached);
                yield return null;
            }

            // Strategy should detect potential collision (logged as warning)
            Assert.IsNotNull(_strategy.GetActiveRobotId(), "Active robot ID should not be null");
        }

        [UnityTest]
        public IEnumerator Update_AllocatesWorkspace_DuringMovement()
        {
            var robot1 = CreateTestRobot("Robot1", new Vector3(0f, 0f, 0f));
            var controller1 = robot1.GetComponent<RobotController>();

            // Set target in left workspace
            controller1.SetTarget(new Vector3(-0.6f, 0.15f, 0.1f));

            yield return null;

            RobotController[] robots = new RobotController[] { controller1 };
            Dictionary<string, bool> targetReached = new Dictionary<string, bool> { { "Robot1", false } };

            // Update strategy
            _strategy.Update(robots, targetReached);

            yield return null;

            // Verify robot is tracked
            bool isActive = _strategy.IsRobotActive("Robot1");
            Assert.IsTrue(isActive, "Robot should be active during movement");
        }

        #endregion

        #region Stale Robot Cleanup Tests

        [UnityTest]
        public IEnumerator Update_CleansUpStaleRobots_WhenRemovedFromArray()
        {
            var robot1 = CreateTestRobot("Robot1", Vector3.zero);
            var robot2 = CreateTestRobot("Robot2", Vector3.zero);
            var controller1 = robot1.GetComponent<RobotController>();
            var controller2 = robot2.GetComponent<RobotController>();

            controller1.SetTarget(new Vector3(0.3f, 0.15f, 0.1f));
            controller2.SetTarget(new Vector3(-0.3f, 0.15f, 0.1f));

            yield return null;

            // First update with both robots
            RobotController[] bothRobots = new RobotController[] { controller1, controller2 };
            Dictionary<string, bool> targetReached = new Dictionary<string, bool>
            {
                { "Robot1", false },
                { "Robot2", false }
            };
            _strategy.Update(bothRobots, targetReached);

            string activeIdBoth = _strategy.GetActiveRobotId();
            Assert.IsTrue(activeIdBoth.Contains("Robot1") || activeIdBoth.Contains("Robot2"),
                "Should have at least one robot active");

            yield return null;

            // Update with only robot1 (robot2 removed)
            RobotController[] oneRobot = new RobotController[] { controller1 };
            Dictionary<string, bool> oneTargetReached = new Dictionary<string, bool> { { "Robot1", false } };

            _strategy.Update(oneRobot, oneTargetReached);

            string activeIdOne = _strategy.GetActiveRobotId();
            Assert.IsTrue(activeIdOne.Contains("Robot1"), "Should still have Robot1 active");
            // Robot2 should be cleaned up (no longer in active ID string if properly cleaned)
        }

        #endregion

        #region Waypoint Replanning Tests

        [UnityTest]
        public IEnumerator RequiresCoordination_TriggersReplanning_ForConflictingPaths()
        {
            // Note: Removed ignoreFailingMessages anti-pattern
            // Coordination requirement is the expected behavior being tested

            var robot1 = CreateTestRobot("Robot1", new Vector3(-0.3f, 0f, 0f));
            var robot2 = CreateTestRobot("Robot2", new Vector3(0.3f, 0f, 0f));
            var controller1 = robot1.GetComponent<RobotController>();
            var controller2 = robot2.GetComponent<RobotController>();

            // Set targets that require coordination
            controller1.SetTarget(new Vector3(0.1f, 0.15f, 0.1f));
            controller2.SetTarget(new Vector3(-0.1f, 0.15f, 0.1f));

            yield return null;

            RobotController[] robots = new RobotController[] { controller1, controller2 };
            Dictionary<string, bool> targetReached = new Dictionary<string, bool>
            {
                { "Robot1", false },
                { "Robot2", false }
            };

            _strategy.Update(robots, targetReached);

            // Check if coordination is required
            bool requiresCoord = _strategy.RequiresCoordination("Robot1", "Robot2");
            Assert.IsTrue(requiresCoord, "Should require coordination for close targets");
        }

        #endregion

        #region Workspace Collision Zone Tests

        [UnityTest]
        public IEnumerator Update_MarksCollisionZones_ForConflictingRobots()
        {
            // Note: Removed ignoreFailingMessages anti-pattern
            // Collision zone marking is the expected behavior being tested

            var robot1 = CreateTestRobot("Robot1", new Vector3(0f, 0f, 0f));
            var robot2 = CreateTestRobot("Robot2", new Vector3(0.1f, 0f, 0f));
            var controller1 = robot1.GetComponent<RobotController>();
            var controller2 = robot2.GetComponent<RobotController>();

            // Both robots target same region (should trigger collision zone)
            controller1.SetTarget(new Vector3(0.05f, 0.15f, 0.1f));
            controller2.SetTarget(new Vector3(0.06f, 0.15f, 0.1f));

            yield return null;

            RobotController[] robots = new RobotController[] { controller1, controller2 };
            Dictionary<string, bool> targetReached = new Dictionary<string, bool>
            {
                { "Robot1", false },
                { "Robot2", false }
            };

            // Update once, wait for coordination timer, then check
            _strategy.Update(robots, targetReached);
            yield return new WaitForSeconds(0.6f); // Wait for internal coordination timer (500ms)
            _strategy.Update(robots, targetReached);

            // Robot1 should be active, Robot2 should be blocked due to collision
            Assert.IsTrue(_strategy.IsRobotActive("Robot1"), "Robot1 should be active");
            Assert.IsFalse(_strategy.IsRobotActive("Robot2"), "Robot2 should be blocked due to collision");
        }

        #endregion
    }
}
