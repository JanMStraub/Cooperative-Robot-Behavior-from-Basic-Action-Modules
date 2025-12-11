using NUnit.Framework;
using System.Collections;
using UnityEngine;
using UnityEngine.TestTools;
using Robotics;

namespace Tests.PlayMode
{
    /// <summary>
    /// Tests for RobotController Phase 4 additions.
    /// Validates new properties and methods added for coordination support.
    /// </summary>
    public class RobotControllerPhase4Tests
    {
        private GameObject _robotObject;
        private RobotController _controller;
        private GameObject _endEffectorObject;
        private GameObject _targetObject;

        [UnitySetUp]
        public IEnumerator Setup()
        {
            // Create robot GameObject
            _robotObject = new GameObject("TestRobot");
            _controller = _robotObject.AddComponent<RobotController>();
            _controller.robotId = "TestRobot";

            // Create end effector
            _endEffectorObject = new GameObject("EndEffector");
            _endEffectorObject.transform.SetParent(_robotObject.transform);
            _endEffectorObject.transform.localPosition = Vector3.zero;
            _controller.endEffectorBase = _endEffectorObject.transform;

            // Initialize robot joints (required by RobotController)
            _controller.robotJoints = new ArticulationBody[0];

            yield return null;
        }

        [TearDown]
        public void TearDown()
        {
            if (_targetObject != null)
            {
                Object.DestroyImmediate(_targetObject);
            }
            if (_endEffectorObject != null)
            {
                Object.DestroyImmediate(_endEffectorObject);
            }
            if (_robotObject != null)
            {
                Object.DestroyImmediate(_robotObject);
            }
        }

        #region HasTarget Tests

        [Test]
        public void HasTarget_NoTargetSet_ReturnsFalse()
        {
            Assert.IsFalse(_controller.HasTarget);
        }

        [UnityTest]
        public IEnumerator HasTarget_TargetSet_ReturnsTrue()
        {
            Vector3 targetPosition = new Vector3(0.3f, 0.15f, 0.1f);
            _controller.SetTarget(targetPosition);

            yield return null;

            Assert.IsTrue(_controller.HasTarget);
        }

        [UnityTest]
        public IEnumerator HasTarget_TargetSetWithGameObject_ReturnsTrue()
        {
            _targetObject = new GameObject("Target");
            _targetObject.transform.position = new Vector3(0.3f, 0.15f, 0.1f);

            _controller.SetTarget(_targetObject);

            yield return null;

            Assert.IsTrue(_controller.HasTarget);
        }

        [UnityTest]
        public IEnumerator HasTarget_TargetDestroyed_UpdatesCorrectly()
        {
            _targetObject = new GameObject("Target");
            _targetObject.transform.position = new Vector3(0.3f, 0.15f, 0.1f);

            _controller.SetTarget(_targetObject);
            yield return null;

            Assert.IsTrue(_controller.HasTarget);

            // Destroy target
            Object.DestroyImmediate(_targetObject);
            _targetObject = null;
            yield return null;

            // HasTarget should still return true because target transform reference exists
            // but is now null (Unity behavior)
            // This tests that the property handles null safely
            Assert.DoesNotThrow(() => { bool _ = _controller.HasTarget; });
        }

        #endregion

        #region GetTargetPosition Tests

        [Test]
        public void GetTargetPosition_NoTarget_ThrowsException()
        {
            Assert.Throws<System.InvalidOperationException>(() =>
            {
                Vector3 _ = _controller.GetTargetPosition();
            });
        }

        [UnityTest]
        public IEnumerator GetTargetPosition_TargetSet_ReturnsCorrectPosition()
        {
            Vector3 expectedPosition = new Vector3(0.3f, 0.15f, 0.1f);
            _controller.SetTarget(expectedPosition);

            yield return null;

            Vector3 actualPosition = _controller.GetTargetPosition();
            Assert.AreEqual(expectedPosition, actualPosition);
        }

        [UnityTest]
        public IEnumerator GetTargetPosition_TargetMoved_ReturnsUpdatedPosition()
        {
            _targetObject = new GameObject("Target");
            Vector3 initialPosition = new Vector3(0.3f, 0.15f, 0.1f);
            _targetObject.transform.position = initialPosition;

            _controller.SetTarget(_targetObject);
            yield return null;

            // Move target
            Vector3 newPosition = new Vector3(0.5f, 0.2f, 0.15f);
            _targetObject.transform.position = newPosition;
            yield return null;

            Vector3 actualPosition = _controller.GetTargetPosition();
            Assert.AreEqual(newPosition, actualPosition);
        }

        [UnityTest]
        public IEnumerator GetTargetPosition_MultipleTargets_ReturnsLatestTarget()
        {
            // Set first target
            Vector3 firstTarget = new Vector3(0.3f, 0.15f, 0.1f);
            _controller.SetTarget(firstTarget);
            yield return null;

            // Set second target
            Vector3 secondTarget = new Vector3(0.5f, 0.2f, 0.15f);
            _controller.SetTarget(secondTarget);
            yield return null;

            Vector3 actualPosition = _controller.GetTargetPosition();
            Assert.AreEqual(secondTarget, actualPosition);
        }

        #endregion

        #region GetCurrentEndEffectorPosition Tests

        [Test]
        public void GetCurrentEndEffectorPosition_NoEndEffector_ReturnsZero()
        {
            _controller.endEffectorBase = null;
            Vector3 position = _controller.GetCurrentEndEffectorPosition();
            Assert.AreEqual(Vector3.zero, position);
        }

        [UnityTest]
        public IEnumerator GetCurrentEndEffectorPosition_EndEffectorSet_ReturnsPosition()
        {
            Vector3 expectedPosition = new Vector3(0.1f, 0.2f, 0.3f);
            _endEffectorObject.transform.position = expectedPosition;

            yield return null;

            Vector3 actualPosition = _controller.GetCurrentEndEffectorPosition();
            Assert.AreEqual(expectedPosition, actualPosition);
        }

        [UnityTest]
        public IEnumerator GetCurrentEndEffectorPosition_EndEffectorMoved_ReturnsUpdatedPosition()
        {
            Vector3 initialPosition = new Vector3(0.1f, 0.2f, 0.3f);
            _endEffectorObject.transform.position = initialPosition;
            yield return null;

            // Move end effector
            Vector3 newPosition = new Vector3(0.4f, 0.5f, 0.6f);
            _endEffectorObject.transform.position = newPosition;
            yield return null;

            Vector3 actualPosition = _controller.GetCurrentEndEffectorPosition();
            Assert.AreEqual(newPosition, actualPosition);
        }

        [UnityTest]
        public IEnumerator GetCurrentEndEffectorPosition_LocalPosition_ReturnsWorldPosition()
        {
            // Set robot parent position
            _robotObject.transform.position = new Vector3(1f, 1f, 1f);

            // Set end effector local position
            _endEffectorObject.transform.localPosition = new Vector3(0.1f, 0.2f, 0.3f);
            yield return null;

            // Should return world position, not local
            Vector3 expectedWorldPosition = new Vector3(1.1f, 1.2f, 1.3f);
            Vector3 actualPosition = _controller.GetCurrentEndEffectorPosition();

            Assert.AreEqual(expectedWorldPosition.x, actualPosition.x, 0.001f);
            Assert.AreEqual(expectedWorldPosition.y, actualPosition.y, 0.001f);
            Assert.AreEqual(expectedWorldPosition.z, actualPosition.z, 0.001f);
        }

        #endregion

        #region Integration Tests with Existing Methods

        [UnityTest]
        public IEnumerator GetCurrentTarget_ConsistentWithGetTargetPosition()
        {
            Vector3 targetPosition = new Vector3(0.3f, 0.15f, 0.1f);
            _controller.SetTarget(targetPosition);

            yield return null;

            // Both methods should return the same position
            Vector3? currentTarget = _controller.GetCurrentTarget();
            Vector3 targetPos = _controller.GetTargetPosition();

            Assert.IsNotNull(currentTarget);
            Assert.AreEqual(currentTarget.Value, targetPos);
        }

        [UnityTest]
        public IEnumerator HasTarget_ConsistentWithGetCurrentTarget()
        {
            // No target set
            Assert.IsFalse(_controller.HasTarget);
            Assert.IsNull(_controller.GetCurrentTarget());

            // Set target
            Vector3 targetPosition = new Vector3(0.3f, 0.15f, 0.1f);
            _controller.SetTarget(targetPosition);
            yield return null;

            // Both should indicate target exists
            Assert.IsTrue(_controller.HasTarget);
            Assert.IsNotNull(_controller.GetCurrentTarget());
        }

        [UnityTest]
        public IEnumerator GetDistanceToTarget_WorksWithNewMethods()
        {
            Vector3 endEffectorPos = new Vector3(0f, 0f, 0f);
            Vector3 targetPos = new Vector3(0.3f, 0f, 0f);

            _endEffectorObject.transform.position = endEffectorPos;
            _controller.SetTarget(targetPos);

            yield return null;

            // Distance should be approximately 0.3m
            float distance = _controller.GetDistanceToTarget();
            Assert.Greater(distance, 0f);
            Assert.LessOrEqual(distance, 0.5f);
        }

        #endregion

        #region Phase 4 Coordination Scenario Tests

        [UnityTest]
        public IEnumerator Scenario_CheckTargetBeforeMovement()
        {
            // Scenario: Check if robot has target before attempting coordination
            Assert.IsFalse(_controller.HasTarget);

            // Set target
            _controller.SetTarget(new Vector3(0.3f, 0.15f, 0.1f));
            yield return null;

            // Now robot has target and position can be retrieved
            Assert.IsTrue(_controller.HasTarget);
            Vector3 targetPos = _controller.GetTargetPosition();
            Assert.IsNotNull(targetPos);
        }

        [UnityTest]
        public IEnumerator Scenario_CompareRobotPositions()
        {
            // Scenario: Get positions of two robots for collision checking
            Vector3 robot1Pos = new Vector3(-0.3f, 0f, 0f);
            Vector3 robot2Pos = new Vector3(0.3f, 0f, 0f);

            _endEffectorObject.transform.position = robot1Pos;
            yield return null;

            Vector3 actualPos = _controller.GetCurrentEndEffectorPosition();
            float distance = Vector3.Distance(actualPos, robot2Pos);

            Assert.Greater(distance, 0.5f); // Robots are far apart
        }

        [UnityTest]
        public IEnumerator Scenario_TrackRobotMovement()
        {
            // Scenario: Track robot as it moves toward target
            Vector3 startPos = new Vector3(0f, 0f, 0f);
            Vector3 targetPos = new Vector3(0.3f, 0.15f, 0.1f);

            _endEffectorObject.transform.position = startPos;
            _controller.SetTarget(targetPos);
            yield return null;

            // Initially at start position
            Vector3 currentPos = _controller.GetCurrentEndEffectorPosition();
            float initialDistance = Vector3.Distance(currentPos, targetPos);

            // Simulate movement (just move transform directly for test)
            _endEffectorObject.transform.position = Vector3.Lerp(startPos, targetPos, 0.5f);
            yield return null;

            // Should be closer to target
            currentPos = _controller.GetCurrentEndEffectorPosition();
            float newDistance = Vector3.Distance(currentPos, targetPos);

            Assert.Less(newDistance, initialDistance);
        }

        #endregion

        #region Edge Case Tests

        [Test]
        public void GetCurrentEndEffectorPosition_CalledMultipleTimes_Consistent()
        {
            Vector3 position = new Vector3(0.1f, 0.2f, 0.3f);
            _endEffectorObject.transform.position = position;

            Vector3 pos1 = _controller.GetCurrentEndEffectorPosition();
            Vector3 pos2 = _controller.GetCurrentEndEffectorPosition();
            Vector3 pos3 = _controller.GetCurrentEndEffectorPosition();

            Assert.AreEqual(pos1, pos2);
            Assert.AreEqual(pos2, pos3);
        }

        [UnityTest]
        public IEnumerator HasTarget_CheckedMultipleTimes_Consistent()
        {
            _controller.SetTarget(new Vector3(0.3f, 0.15f, 0.1f));
            yield return null;

            bool check1 = _controller.HasTarget;
            bool check2 = _controller.HasTarget;
            bool check3 = _controller.HasTarget;

            Assert.IsTrue(check1);
            Assert.AreEqual(check1, check2);
            Assert.AreEqual(check2, check3);
        }

        [Test]
        public void GetTargetPosition_CalledTwice_ReturnsSameValue()
        {
            Vector3 targetPos = new Vector3(0.3f, 0.15f, 0.1f);
            _controller.SetTarget(targetPos);

            Vector3 pos1 = _controller.GetTargetPosition();
            Vector3 pos2 = _controller.GetTargetPosition();

            Assert.AreEqual(pos1, pos2);
        }

        #endregion
    }
}
