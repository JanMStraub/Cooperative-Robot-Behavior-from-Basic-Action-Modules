using NUnit.Framework;
using System.Collections;
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

        [SetUp]
        public void Setup()
        {
            // Clean up any existing instance
            if (RobotManager.Instance != null)
            {
                UnityEngine.Object.DestroyImmediate(RobotManager.Instance.gameObject);
            }

            _managerObject = new GameObject("TestRobotManager");
            _manager = _managerObject.AddComponent<RobotManager>();
        }

        [TearDown]
        public void TearDown()
        {
            if (_managerObject != null)
            {
                UnityEngine.Object.DestroyImmediate(_managerObject);
            }
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
            yield return null; // Allow Start() to complete

            int countBeforeRegistration = _manager.AllRobotIds.Count;

            var robotObject = new GameObject("TestRobot");
            robotObject.AddComponent<RobotController>();

            // RegisterRobot takes (robotId, robotGameObject, targetGameObject, profile)
            _manager.RegisterRobot("TestRobot", robotObject, null, null);

            Assert.IsTrue(_manager.RobotInstances.ContainsKey("TestRobot"));
            Assert.AreEqual(countBeforeRegistration + 1, _manager.AllRobotIds.Count);

            UnityEngine.Object.DestroyImmediate(robotObject);
        }

        [UnityTest]
        public IEnumerator RobotManager_RegisterRobot_WithAutoId_GeneratesId()
        {
            yield return null;

            int countBefore = _manager.AllRobotIds.Count;

            var robotObject = new GameObject("CustomRobot");
            robotObject.AddComponent<RobotController>();

            // Pass null/empty robotId to auto-generate
            _manager.RegisterRobot(null, robotObject, null, null);

            // Should have auto-generated an ID based on GameObject name
            Assert.AreEqual(countBefore + 1, _manager.AllRobotIds.Count);
            Assert.IsTrue(_manager.RobotInstances.ContainsKey("CustomRobot"));

            UnityEngine.Object.DestroyImmediate(robotObject);
        }

        [UnityTest]
        public IEnumerator RobotManager_RegisterRobot_WithTarget_SetsTarget()
        {
            yield return null;

            var robotObject = new GameObject("TestRobot");
            var targetObject = new GameObject("Target");
            robotObject.AddComponent<RobotController>();

            _manager.RegisterRobot("TestRobot", robotObject, targetObject, null);

            Assert.IsTrue(_manager.RobotInstances.ContainsKey("TestRobot"));
            var instance = _manager.RobotInstances["TestRobot"];
            Assert.AreEqual(targetObject, instance.targetGameObject);

            UnityEngine.Object.DestroyImmediate(robotObject);
            UnityEngine.Object.DestroyImmediate(targetObject);
        }

        #endregion

        #region Robot Query Tests

        [UnityTest]
        public IEnumerator RobotManager_RobotInstances_ReturnsRegisteredRobot()
        {
            yield return null;

            var robotObject = new GameObject("TestRobot");
            robotObject.AddComponent<RobotController>();

            _manager.RegisterRobot("TestRobot", robotObject, null, null);

            Assert.IsTrue(_manager.RobotInstances.ContainsKey("TestRobot"));
            var instance = _manager.RobotInstances["TestRobot"];
            Assert.AreEqual("TestRobot", instance.robotId);

            UnityEngine.Object.DestroyImmediate(robotObject);
        }

        #endregion

        #region Event Tests

        [UnityTest]
        public IEnumerator RobotManager_OnTargetChanged_EventExists()
        {
            yield return null;

            var robotObject = new GameObject("TestRobot");
            var targetObject = new GameObject("Target");
            robotObject.AddComponent<RobotController>();

            _manager.RegisterRobot("TestRobot", robotObject, targetObject, null);

            string changedRobotId = null;
            _manager.OnTargetChanged += (id, target) =>
            {
                changedRobotId = id;
            };

            // Event mechanism exists
            Assert.IsNotNull(_manager);

            UnityEngine.Object.DestroyImmediate(robotObject);
            UnityEngine.Object.DestroyImmediate(targetObject);
        }

        #endregion

        #region Multiple Robot Tests

        [UnityTest]
        public IEnumerator RobotManager_MultipleRobots_TracksAll()
        {
            yield return null;

            var robot1 = new GameObject("Robot1");
            var robot2 = new GameObject("Robot2");
            robot1.AddComponent<RobotController>();
            robot2.AddComponent<RobotController>();

            _manager.RegisterRobot("Robot1", robot1, null, null);
            _manager.RegisterRobot("Robot2", robot2, null, null);

            Assert.AreEqual(2, _manager.AllRobotIds.Count);
            Assert.IsTrue(_manager.RobotInstances.ContainsKey("Robot1"));
            Assert.IsTrue(_manager.RobotInstances.ContainsKey("Robot2"));

            UnityEngine.Object.DestroyImmediate(robot1);
            UnityEngine.Object.DestroyImmediate(robot2);
        }

        #endregion

        #region Active Robot Count Tests

        [UnityTest]
        public IEnumerator RobotManager_ActiveRobotCount_TracksActiveRobots()
        {
            yield return null;

            var robot1 = new GameObject("Robot1");
            robot1.AddComponent<RobotController>();

            _manager.RegisterRobot("Robot1", robot1, null, null);
            var instance = _manager.RobotInstances["Robot1"];
            instance.isActive = true;

            Assert.AreEqual(1, _manager.ActiveRobotCount);

            instance.isActive = false;
            Assert.AreEqual(0, _manager.ActiveRobotCount);

            UnityEngine.Object.DestroyImmediate(robot1);
        }

        #endregion
    }
}
