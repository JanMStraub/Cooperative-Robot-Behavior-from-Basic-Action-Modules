using NUnit.Framework;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.TestTools;
using Robotics;

namespace Tests.PlayMode
{
    /// <summary>
    /// Tests for RobotManager.
    /// Validates robot registration, target assignment, and lifecycle management.
    /// </summary>
    public class RobotManagerTests
    {
        private GameObject _managerObject;
        private RobotManager _manager;
        private readonly List<GameObject> _tempObjects = new List<GameObject>();

        [UnitySetUp]
        public IEnumerator Setup()
        {
            // Clean up any existing instance
            if (RobotManager.Instance != null)
            {
                UnityEngine.Object.DestroyImmediate(RobotManager.Instance.gameObject);
            }

            _managerObject = new GameObject("TestRobotManager");
            _manager = _managerObject.AddComponent<RobotManager>();

            yield return null; // Allow Awake/Start to run and singleton to be set
        }

        [TearDown]
        public void TearDown()
        {
            if (_managerObject != null)
            {
                UnityEngine.Object.DestroyImmediate(_managerObject);
            }

            foreach (var obj in _tempObjects)
            {
                if (obj != null) UnityEngine.Object.DestroyImmediate(obj);
            }
            _tempObjects.Clear();
        }

        #region Singleton Tests

        [Test]
        public void RobotManager_Singleton_IsSet()
        {
            Assert.IsNotNull(RobotManager.Instance);
            Assert.AreEqual(_manager, RobotManager.Instance);
        }

        [Test]
        public void RobotManager_Singleton_DestroysExtraInstances()
        {
            var secondObject = new GameObject("SecondManager");
            var secondManager = secondObject.AddComponent<RobotManager>();

            Assert.AreEqual(_manager, RobotManager.Instance);

            if (secondObject != null)
            {
                UnityEngine.Object.DestroyImmediate(secondObject);
            }
        }

        #endregion

        #region Initialization Tests

        [Test]
        public void RobotManager_InitialState_HasEmptyRobotInstances()
        {
            Assert.IsNotNull(_manager.RobotInstances);
            Assert.AreEqual(0, _manager.ActiveRobotCount);
        }

        [Test]
        public void RobotManager_CreatesDefaultProfile_WhenNull()
        {
            Assert.IsNotNull(_manager.RobotProfile);
        }

        [Test]
        public void RobotManager_AllRobotIds_ReturnsEmptyListInitially()
        {
            Assert.IsNotNull(_manager.AllRobotIds);
            Assert.AreEqual(0, _manager.AllRobotIds.Count);
        }

        [Test]
        public void RobotManager_Robots_ReturnsEmptyArrayInitially()
        {
            Assert.IsNotNull(_manager.Robots);
            Assert.AreEqual(0, _manager.Robots.Length);
        }

        #endregion

        #region Robot Registration Tests

        [UnityTest]
        public IEnumerator RobotManager_RegisterRobot_AddsToInstances()
        {
            int countBeforeRegistration = _manager.AllRobotIds.Count;

            var robotObject = new GameObject("TestRobot");
            _tempObjects.Add(robotObject);
            robotObject.AddComponent<RobotController>();

            // RegisterRobot takes (robotId, robotGameObject, targetGameObject, profile)
            _manager.RegisterRobot("TestRobot", robotObject, null, null);

            Assert.IsTrue(_manager.RobotInstances.ContainsKey("TestRobot"));
            Assert.AreEqual(countBeforeRegistration + 1, _manager.AllRobotIds.Count);

            yield return null;
        }

        [UnityTest]
        public IEnumerator RobotManager_RegisterRobot_WithAutoId_GeneratesId()
        {
            int countBefore = _manager.AllRobotIds.Count;

            var robotObject = new GameObject("CustomRobot");
            _tempObjects.Add(robotObject);
            robotObject.AddComponent<RobotController>();

            // Pass null/empty robotId to auto-generate
            _manager.RegisterRobot(null, robotObject, null, null);

            // Should have auto-generated an ID based on GameObject name
            Assert.AreEqual(countBefore + 1, _manager.AllRobotIds.Count);
            Assert.IsTrue(_manager.RobotInstances.ContainsKey("CustomRobot"));

            yield return null;
        }

        [UnityTest]
        public IEnumerator RobotManager_RegisterRobot_WithTarget_SetsTarget()
        {
            var robotObject = new GameObject("TestRobot");
            var targetObject = new GameObject("Target");
            _tempObjects.Add(robotObject);
            _tempObjects.Add(targetObject);
            robotObject.AddComponent<RobotController>();

            _manager.RegisterRobot("TestRobot", robotObject, targetObject, null);

            Assert.IsTrue(_manager.RobotInstances.ContainsKey("TestRobot"));
            var instance = _manager.RobotInstances["TestRobot"];
            Assert.AreEqual(targetObject, instance.targetGameObject);

            yield return null;
        }

        #endregion

        #region Robot Query Tests

        [UnityTest]
        public IEnumerator RobotManager_RobotInstances_ReturnsRegisteredRobot()
        {
            var robotObject = new GameObject("TestRobot");
            _tempObjects.Add(robotObject);
            robotObject.AddComponent<RobotController>();

            _manager.RegisterRobot("TestRobot", robotObject, null, null);

            Assert.IsTrue(_manager.RobotInstances.ContainsKey("TestRobot"));
            var instance = _manager.RobotInstances["TestRobot"];
            Assert.AreEqual("TestRobot", instance.robotId);

            yield return null;
        }

        #endregion

        #region Event Tests

        [UnityTest]
        public IEnumerator RobotManager_OnTargetChanged_EventFires()
        {
            var robotObject = new GameObject("TestRobot");
            var targetObject = new GameObject("Target");
            _tempObjects.Add(robotObject);
            _tempObjects.Add(targetObject);
            robotObject.AddComponent<RobotController>();

            _manager.RegisterRobot("TestRobot", robotObject, targetObject, null);

            // Mark the robot active so CheckForTargetChanges processes it
            _manager.RobotInstances["TestRobot"].isActive = true;

            string changedRobotId = null;
            _manager.OnTargetChanged += (id, target) =>
            {
                changedRobotId = id;
            };

            // Move the target beyond the 0.001m threshold to trigger the event
            // OnTargetChanged fires from CheckForTargetChanges() in Update when
            // the target position changes by more than 0.001m.
            targetObject.transform.position = new Vector3(1f, 0f, 0f);

            // Wait until the event fires (driven by Update)
            yield return new WaitUntil(() => changedRobotId != null);

            Assert.AreEqual("TestRobot", changedRobotId,
                "OnTargetChanged should fire with correct robot ID when target moves");
        }

        #endregion

        #region Duplicate Registration Tests

        [UnityTest]
        public IEnumerator RobotManager_RegisterRobot_DuplicateId_LogsWarningAndUpdates()
        {
            var robotObject = new GameObject("TestRobot");
            _tempObjects.Add(robotObject);
            robotObject.AddComponent<RobotController>();

            _manager.RegisterRobot("TestRobot", robotObject, null, null);
            Assert.AreEqual(1, _manager.AllRobotIds.Count, "Should have one robot after first registration");

            // Second registration with same ID should log a warning
            LogAssert.Expect(
                LogType.Warning,
                new System.Text.RegularExpressions.Regex("Robot TestRobot already registered")
            );
            _manager.RegisterRobot("TestRobot", robotObject, null, null);

            // Count should remain 1 — duplicate overwrites, not appends
            Assert.AreEqual(1, _manager.AllRobotIds.Count,
                "Duplicate registration should not increase the robot count");

            yield return null;
        }

        #endregion

        #region Multiple Robot Tests

        [UnityTest]
        public IEnumerator RobotManager_MultipleRobots_TracksAll()
        {
            var robot1 = new GameObject("Robot1");
            var robot2 = new GameObject("Robot2");
            _tempObjects.Add(robot1);
            _tempObjects.Add(robot2);
            robot1.AddComponent<RobotController>();
            robot2.AddComponent<RobotController>();

            _manager.RegisterRobot("Robot1", robot1, null, null);
            _manager.RegisterRobot("Robot2", robot2, null, null);

            Assert.AreEqual(2, _manager.AllRobotIds.Count);
            Assert.IsTrue(_manager.RobotInstances.ContainsKey("Robot1"));
            Assert.IsTrue(_manager.RobotInstances.ContainsKey("Robot2"));

            yield return null;
        }

        #endregion

        #region Active Robot Count Tests

        [UnityTest]
        public IEnumerator RobotManager_ActiveRobotCount_TracksActiveRobots()
        {
            var robot1 = new GameObject("Robot1");
            _tempObjects.Add(robot1);
            robot1.AddComponent<RobotController>();

            _manager.RegisterRobot("Robot1", robot1, null, null);
            var instance = _manager.RobotInstances["Robot1"];
            instance.isActive = true;

            Assert.AreEqual(1, _manager.ActiveRobotCount);

            instance.isActive = false;
            Assert.AreEqual(0, _manager.ActiveRobotCount);

            yield return null;
        }

        #endregion
    }
}
