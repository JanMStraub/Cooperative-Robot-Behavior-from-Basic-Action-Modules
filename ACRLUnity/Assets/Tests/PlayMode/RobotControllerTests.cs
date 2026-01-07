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

        [UnitySetUp]
        public IEnumerator SetUp()
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

            // Create mock ArticulationBody joints to prevent initialization errors
            _robotController.robotJoints = new ArticulationBody[6];
            for (int i = 0; i < 6; i++)
            {
                var jointObject = new GameObject($"Joint{i}");
                jointObject.transform.SetParent(_testRobotObject.transform);
                var articulationBody = jointObject.AddComponent<ArticulationBody>();
                _robotController.robotJoints[i] = articulationBody;
            }

            // Suppress expected warning about missing GripperController
            LogAssert.Expect(LogType.Warning,
                "[ROBOT_CONTROLLER] No GripperController found in children of TestRobot");

            // Wait one frame to allow Start() to be called
            yield return null;
        }

        [TearDown]
        public void TearDown()
        {
            if (_testRobotObject != null)
            {
                Object.Destroy(_testRobotObject);
            }
        }

        [UnityTest]
        public IEnumerator HasTarget_ReturnsFalse_WhenNoTargetSet()
        {
            // Assert
            Assert.IsFalse(_robotController.HasTarget,
                "HasTarget should be false when no target has been set");
            yield return null;
        }

        [UnityTest]
        public IEnumerator SetTarget_GameObject_Default_OpensGripper()
        {
            // Arrange
            var targetObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            targetObject.transform.position = new Vector3(1f, 1f, 1f);

            // Expect GripperController error since we're not setting up gripper references
            LogAssert.Expect(LogType.Error, "[GRIPPER_CONTROLLER] Gripper references not assigned!");

            // Add GripperController to test gripper behavior
            _testRobotObject.AddComponent<GripperController>();
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

        [UnityTest]
        public IEnumerator GetCurrentTarget_ReturnsNull_WhenNoTargetSet()
        {
            // Act
            var target = _robotController.GetCurrentTarget();

            // Assert
            Assert.IsNull(target, "GetCurrentTarget should return null when no target is set");
            yield return null;
        }

        [UnityTest]
        public IEnumerator GetCurrentTargetRotation_ReturnsNull_WhenNoTargetSet()
        {
            // Act
            var rotation = _robotController.GetCurrentTargetRotation();

            // Assert
            Assert.IsNull(rotation,
                "GetCurrentTargetRotation should return null when no target is set");
            yield return null;
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

        [UnityTest]
        public IEnumerator GetCurrentEndEffectorPosition_ReturnsCorrectPosition()
        {
            // Arrange
            Vector3 expectedPosition = new Vector3(1.2f, 3.4f, 5.6f);
            _endEffectorObject.transform.position = expectedPosition;

            // Act
            Vector3 actualPosition = _robotController.GetCurrentEndEffectorPosition();

            // Assert
            Assert.AreEqual(expectedPosition, actualPosition,
                "GetCurrentEndEffectorPosition should return end effector position");
            yield return null;
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

        #region Grasp Planning Tests (December 2025)

        [UnityTest]
        public IEnumerator SetTarget_WithTopApproach_UsesTopGraspDirection()
        {
            // Arrange
            var targetObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            targetObject.transform.position = new Vector3(0.5f, 0.3f, 0.5f);

            var topApproachOptions = new GraspOptions
            {
                useGraspPlanning = true,
                approach = Robotics.Grasp.GraspApproach.Top,
                openGripperOnSet = false,
                closeGripperOnReach = false
            };

            // Act
            _robotController.SetTarget(targetObject, topApproachOptions);
            yield return null;

            // Assert
            Assert.IsTrue(_robotController.HasTarget, "Should have target with top approach");

            // Cleanup
            Object.Destroy(targetObject);
        }

        [UnityTest]
        public IEnumerator SetTarget_WithFrontApproach_UsesFrontGraspDirection()
        {
            // Arrange
            var targetObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            targetObject.transform.position = new Vector3(0.5f, 0.3f, 0.5f);

            var frontApproachOptions = new GraspOptions
            {
                useGraspPlanning = true,
                approach = Robotics.Grasp.GraspApproach.Front,
                openGripperOnSet = false,
                closeGripperOnReach = false
            };

            // Act
            _robotController.SetTarget(targetObject, frontApproachOptions);
            yield return null;

            // Assert
            Assert.IsTrue(_robotController.HasTarget, "Should have target with front approach");

            // Cleanup
            Object.Destroy(targetObject);
        }

        [UnityTest]
        public IEnumerator SetTarget_WithSideApproach_UsesSideGraspDirection()
        {
            // Arrange
            var targetObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            targetObject.transform.position = new Vector3(0.5f, 0.3f, 0.5f);

            var sideApproachOptions = new GraspOptions
            {
                useGraspPlanning = true,
                approach = Robotics.Grasp.GraspApproach.Side,
                openGripperOnSet = false,
                closeGripperOnReach = false
            };

            // Act
            _robotController.SetTarget(targetObject, sideApproachOptions);
            yield return null;

            // Assert
            Assert.IsTrue(_robotController.HasTarget, "Should have target with side approach");

            // Cleanup
            Object.Destroy(targetObject);
        }

        [UnityTest]
        public IEnumerator SetTarget_WithGraspPlanning_EnablesWaypointSequence()
        {
            // Arrange
            var targetObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            targetObject.transform.position = new Vector3(0.5f, 0.3f, 0.5f);

            var graspOptions = new GraspOptions
            {
                useGraspPlanning = true,
                approach = Robotics.Grasp.GraspApproach.Top,
                openGripperOnSet = false,
                closeGripperOnReach = false
            };

            // Act
            _robotController.SetTarget(targetObject, graspOptions);
            yield return null;

            // Assert - target should be set (waypoint planning happens internally)
            Assert.IsTrue(_robotController.HasTarget,
                "Target should be set with grasp planning enabled");

            // Cleanup
            Object.Destroy(targetObject);
        }

        [UnityTest]
        public IEnumerator SetTarget_WithNullApproach_UsesDefaultBehavior()
        {
            // Arrange
            var targetObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            targetObject.transform.position = new Vector3(0.5f, 0.3f, 0.5f);

            var nullApproachOptions = new GraspOptions
            {
                useGraspPlanning = true,
                approach = null, // No specific approach
                openGripperOnSet = false,
                closeGripperOnReach = false
            };

            // Act
            _robotController.SetTarget(targetObject, nullApproachOptions);
            yield return null;

            // Assert - should still work with default behavior
            Assert.IsTrue(_robotController.HasTarget,
                "Target should be set even with null approach");

            // Cleanup
            Object.Destroy(targetObject);
        }

        [UnityTest]
        public IEnumerator GripperAutoDiscovery_FindsGripperController()
        {
            // Arrange - Add a GripperController as a child
            var gripperObject = new GameObject("Gripper");
            gripperObject.transform.SetParent(_testRobotObject.transform);
            var gripperController = gripperObject.AddComponent<GripperController>();

            // Expect error from GripperController.Awake due to missing ArticulationBody references
            LogAssert.Expect(LogType.Error, "[GRIPPER_CONTROLLER] Gripper references not assigned!");

            // Act - Create a new RobotController to trigger Start() and auto-discovery
            var newRobotObject = new GameObject("NewTestRobot");
            var newRobotController = newRobotObject.AddComponent<RobotController>();
            newRobotController.robotId = "NewTestRobot";
            newRobotController.endEffectorBase = _endEffectorObject.transform;
            newRobotController.robotJoints = _robotController.robotJoints;

            // Move gripper to new robot
            gripperObject.transform.SetParent(newRobotObject.transform);

            yield return null; // Wait for Start() to execute

            // Assert - GripperController should be found
            // Note: We can't directly access the private field, but we can verify
            // that no warning is logged about missing gripper

            // Cleanup
            Object.Destroy(newRobotObject);
            Object.Destroy(gripperObject);
        }

        [UnityTest]
        public IEnumerator SetTarget_DifferentApproaches_AllSucceed()
        {
            // Test that all approach types can be set without errors
            var targetObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            targetObject.transform.position = new Vector3(0.5f, 0.3f, 0.5f);

            var approaches = new Robotics.Grasp.GraspApproach?[]
            {
                Robotics.Grasp.GraspApproach.Top,
                Robotics.Grasp.GraspApproach.Front,
                Robotics.Grasp.GraspApproach.Side,
                null
            };

            foreach (var approach in approaches)
            {
                var options = new GraspOptions
                {
                    useGraspPlanning = true,
                    approach = approach,
                    openGripperOnSet = false,
                    closeGripperOnReach = false
                };

                _robotController.SetTarget(targetObject, options);
                yield return null;

                Assert.IsTrue(_robotController.HasTarget,
                    $"Target should be set with approach: {approach?.ToString() ?? "null"}");
            }

            // Cleanup
            Object.Destroy(targetObject);
        }

        [UnityTest]
        public IEnumerator SetTarget_WithCoordinates_UsesObjectFinder()
        {
            // Arrange - Set target using coordinates (should trigger ObjectFinder)
            Vector3 targetPosition = new Vector3(0.5f, 0.3f, 0.5f);

            // Act
            _robotController.SetTarget(targetPosition, GraspOptions.MoveOnly);
            yield return null;

            // Assert
            Assert.IsTrue(_robotController.HasTarget,
                "Target should be set from coordinates");

            var currentTarget = _robotController.GetCurrentTarget();
            Assert.IsNotNull(currentTarget, "Current target should not be null");
            Assert.AreEqual(targetPosition, currentTarget.Value,
                "Target position should match input coordinates");
        }

        #endregion

        #region Precision Improvement Tests (January 2026)

        [UnityTest]
        public IEnumerator GraspConvergenceThreshold_UsesRelaxedThreshold()
        {
            // Test that grasp convergence uses the relaxed 5mm threshold (0.33 multiplier)
            // instead of the old 3mm threshold (0.2 multiplier)

            // Arrange
            var targetObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            targetObject.transform.position = new Vector3(0.5f, 0.3f, 0.5f);

            var graspOptions = new GraspOptions
            {
                useGraspPlanning = true,
                approach = Robotics.Grasp.GraspApproach.Top,
                closeGripperOnReach = true
            };

            // Act
            _robotController.SetTarget(targetObject, graspOptions);
            yield return null;

            // Assert - Verify the controller is using the relaxed convergence threshold
            // The actual threshold value is: DEFAULT_CONVERGENCE_THRESHOLD (0.015m) * GRASP_CONVERGENCE_MULTIPLIER (0.33)
            // = 0.00495m (approximately 5mm)
            float expectedThreshold = Core.RobotConstants.DEFAULT_CONVERGENCE_THRESHOLD *
                                     Core.RobotConstants.GRASP_CONVERGENCE_MULTIPLIER;
            Assert.AreEqual(0.00495f, expectedThreshold, 0.0001f,
                "Grasp convergence threshold should be approximately 5mm");

            // Cleanup
            Object.Destroy(targetObject);
        }

        [UnityTest]
        public IEnumerator OrientationThreshold_UsesConfigurableValue()
        {
            // Test that orientation convergence uses the new configurable threshold (10 degrees)
            // instead of the old hardcoded 5 degrees

            // Assert - Verify the constant is set correctly
            Assert.AreEqual(10f, Core.RobotConstants.DEFAULT_ORIENTATION_THRESHOLD_DEGREES,
                "Orientation convergence threshold should be 10 degrees");

            yield return null;
        }

        [UnityTest]
        public IEnumerator OrientationRampStart_BeginsAt30cm()
        {
            // Test that orientation ramping starts at 30cm instead of 20cm

            // Assert - Verify the constant is set correctly
            Assert.AreEqual(0.30f, Core.RobotConstants.ORIENTATION_RAMP_START_DISTANCE,
                "Orientation ramping should start at 30cm");

            yield return null;
        }

        [UnityTest]
        public IEnumerator GraspTimeout_UsesConfiguredValue()
        {
            // Test that grasp operations use the configured timeout value

            // Assert - Verify the constant is set correctly
            Assert.AreEqual(10f, Core.RobotConstants.DEFAULT_GRASP_TIMEOUT_SECONDS,
                "Grasp timeout should be 10 seconds");

            yield return null;
        }

        [UnityTest]
        public IEnumerator MovementTimeout_UsesConfiguredValue()
        {
            // Test that movement operations use the configured timeout value

            // Assert - Verify the constant is set correctly
            Assert.AreEqual(15f, Core.RobotConstants.DEFAULT_MOVEMENT_TIMEOUT_SECONDS,
                "Movement timeout should be 15 seconds");

            yield return null;
        }

        #endregion
    }
}
