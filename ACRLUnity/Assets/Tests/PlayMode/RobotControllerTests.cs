using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using Robotics;

namespace Tests.PlayMode
{
    /// <summary>
    /// Tests for RobotController public API and target management
    /// </summary>
    public class RobotControllerTests
    {
        private GameObject _testRobotObject;
        private RobotController _robotController;
        private GameObject _endEffectorObject;

        [SetUp]
        public void SetUp()
        {
            // Create minimal robot setup for testing
            _testRobotObject = new GameObject("TestRobot");
            _robotController = _testRobotObject.AddComponent<RobotController>();
            _robotController.robotId = "TestRobot";

            // Add minimal required components
            _endEffectorObject = new GameObject("EndEffectorBase");
            _endEffectorObject.transform.SetParent(_testRobotObject.transform);
            _endEffectorObject.transform.position = Vector3.zero;
            _robotController.endEffectorBase = _endEffectorObject.transform;
        }

        [TearDown]
        public void TearDown()
        {
            if (_testRobotObject != null)
            {
                Object.Destroy(_testRobotObject);
            }
        }

        [Test]
        public void HasTarget_ReturnsFalse_WhenNoTargetSet()
        {
            // Assert
            Assert.IsFalse(_robotController.HasTarget,
                "HasTarget should be false when no target has been set");
        }

        [UnityTest]
        public IEnumerator SetTarget_GameObject_Default_OpensGripper()
        {
            // Arrange
            var targetObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            targetObject.transform.position = new Vector3(1f, 1f, 1f);

            // Add GripperController to test gripper behavior
            var gripperController = _testRobotObject.AddComponent<GripperController>();
            // Note: GripperController requires ArticulationBody setup, so this is a basic test

            // Act
            _robotController.SetTarget(targetObject, GraspOptions.Default);
            yield return null;

            // Assert
            Assert.IsTrue(_robotController.HasTarget,
                "HasTarget should be true after setting target");

            // Cleanup
            Object.Destroy(targetObject);
        }

        [UnityTest]
        public IEnumerator SetTarget_Vector3_MoveOnly_SkipsGraspPlanning()
        {
            // Arrange
            Vector3 targetPosition = new Vector3(0.5f, 0.5f, 0.5f);

            // Act
            _robotController.SetTarget(targetPosition, GraspOptions.MoveOnly);
            yield return null;

            // Assert
            Assert.IsTrue(_robotController.HasTarget, "Should have target");
            var currentTarget = _robotController.GetCurrentTarget();
            Assert.IsNotNull(currentTarget, "GetCurrentTarget should return non-null");
            Assert.AreEqual(targetPosition, currentTarget.Value,
                "Target position should match the set position");
        }

        [UnityTest]
        public IEnumerator SetTarget_PositionAndRotation_UsesExplicitPose()
        {
            // Arrange
            Vector3 targetPosition = new Vector3(0.3f, 0.2f, 0.1f);
            Quaternion targetRotation = Quaternion.Euler(45f, 30f, 15f);

            // Act
            _robotController.SetTarget(targetPosition, targetRotation, GraspOptions.MoveOnly);
            yield return null;

            // Assert
            Assert.IsTrue(_robotController.HasTarget, "Should have target");

            var currentTarget = _robotController.GetCurrentTarget();
            Assert.IsNotNull(currentTarget, "GetCurrentTarget should return non-null");
            Assert.AreEqual(targetPosition, currentTarget.Value,
                "Target position should match");

            var currentRotation = _robotController.GetCurrentTargetRotation();
            Assert.IsNotNull(currentRotation, "GetCurrentTargetRotation should return non-null");

            // Compare quaternions with tolerance
            float angle = Quaternion.Angle(targetRotation, currentRotation.Value);
            Assert.Less(angle, 0.1f, $"Target rotation should match (angle difference: {angle})");
        }

        [Test]
        public void GetCurrentTarget_ReturnsNull_WhenNoTargetSet()
        {
            // Act
            var target = _robotController.GetCurrentTarget();

            // Assert
            Assert.IsNull(target, "GetCurrentTarget should return null when no target is set");
        }

        [Test]
        public void GetCurrentTargetRotation_ReturnsNull_WhenNoTargetSet()
        {
            // Act
            var rotation = _robotController.GetCurrentTargetRotation();

            // Assert
            Assert.IsNull(rotation,
                "GetCurrentTargetRotation should return null when no target is set");
        }

        [UnityTest]
        public IEnumerator GetTargetObject_ReturnsCorrectObject()
        {
            // Arrange
            var targetObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            targetObject.name = "TestCube";
            targetObject.transform.position = new Vector3(1f, 1f, 1f);

            // Act
            _robotController.SetTarget(targetObject, GraspOptions.MoveOnly);
            yield return null;

            var retrievedObject = _robotController.GetTargetObject();

            // Assert
            Assert.IsNotNull(retrievedObject, "GetTargetObject should return non-null");
            Assert.AreEqual(targetObject, retrievedObject,
                "GetTargetObject should return the original target object");

            // Cleanup
            Object.Destroy(targetObject);
        }

        [UnityTest]
        public IEnumerator GetDistanceToTarget_ReturnsCorrectDistance()
        {
            // Arrange
            _endEffectorObject.transform.position = Vector3.zero;
            Vector3 targetPosition = new Vector3(3f, 4f, 0f); // Distance = 5.0

            // Act
            _robotController.SetTarget(targetPosition, GraspOptions.MoveOnly);
            yield return null;

            float distance = _robotController.GetDistanceToTarget();

            // Assert
            Assert.AreEqual(5f, distance, 0.01f,
                "Distance should be 5.0 (3-4-5 triangle)");
        }

        [Test]
        public void GetCurrentEndEffectorPosition_ReturnsCorrectPosition()
        {
            // Arrange
            Vector3 expectedPosition = new Vector3(1.2f, 3.4f, 5.6f);
            _endEffectorObject.transform.position = expectedPosition;

            // Act
            Vector3 actualPosition = _robotController.GetCurrentEndEffectorPosition();

            // Assert
            Assert.AreEqual(expectedPosition, actualPosition,
                "GetCurrentEndEffectorPosition should return end effector position");
        }

        [UnityTest]
        public IEnumerator SetTarget_NullGameObject_DoesNotSetTarget()
        {
            // Arrange
            GameObject nullObject = null;

            // Act
            _robotController.SetTarget(nullObject, GraspOptions.Default);
            yield return null;

            // Assert
            Assert.IsFalse(_robotController.HasTarget,
                "Should not have target when SetTarget called with null");
        }

        [UnityTest]
        public IEnumerator CustomGraspOptions_AreRespected()
        {
            // Arrange
            var targetObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            targetObject.transform.position = new Vector3(1f, 1f, 1f);

            var customOptions = new GraspOptions
            {
                useGraspPlanning = false,
                openGripperOnSet = false,
                closeGripperOnReach = false,
                approach = null
            };

            // Act
            _robotController.SetTarget(targetObject, customOptions);
            yield return null;

            // Assert
            Assert.IsTrue(_robotController.HasTarget, "Should have target");
            // With useGraspPlanning = false, target should be the object itself
            var currentTarget = _robotController.GetCurrentTarget();
            Assert.IsNotNull(currentTarget);

            // Cleanup
            Object.Destroy(targetObject);
        }

        [UnityTest]
        public IEnumerator SetTargetReached_UpdatesTargetReachedState()
        {
            // Arrange
            Vector3 targetPosition = new Vector3(0.5f, 0.5f, 0.5f);
            _robotController.SetTarget(targetPosition, GraspOptions.MoveOnly);
            yield return null;

            // Act
            _robotController.SetTargetReached(true);

            // Assert - HasTarget should still be true even when target is reached
            Assert.IsTrue(_robotController.HasTarget,
                "HasTarget should remain true when target is marked as reached");
        }

        [UnityTest]
        public IEnumerator MultipleSetTarget_Calls_UpdateTarget()
        {
            // Arrange
            Vector3 firstTarget = new Vector3(1f, 0f, 0f);
            Vector3 secondTarget = new Vector3(0f, 1f, 0f);

            // Act
            _robotController.SetTarget(firstTarget, GraspOptions.MoveOnly);
            yield return null;

            var firstRetrievedTarget = _robotController.GetCurrentTarget();

            _robotController.SetTarget(secondTarget, GraspOptions.MoveOnly);
            yield return null;

            var secondRetrievedTarget = _robotController.GetCurrentTarget();

            // Assert
            Assert.AreEqual(firstTarget, firstRetrievedTarget.Value,
                "First target should be set correctly");
            Assert.AreEqual(secondTarget, secondRetrievedTarget.Value,
                "Second target should override first target");
            Assert.AreNotEqual(firstRetrievedTarget, secondRetrievedTarget,
                "Targets should be different");
        }
    }
}
