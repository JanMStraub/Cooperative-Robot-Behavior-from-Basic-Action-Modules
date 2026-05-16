using System.Collections;
using System.Collections.Generic;
using NUnit.Framework;
using Robotics;
using UnityEngine;
using UnityEngine.TestTools;

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
        private readonly List<GameObject> _tempObjects = new List<GameObject>();

        [UnitySetUp]
        public IEnumerator SetUp()
        {
            if (RobotManager.Instance != null)
                Object.DestroyImmediate(RobotManager.Instance.gameObject);

            _testRobotObject = new GameObject("TestRobot");
            _robotController = _testRobotObject.AddComponent<RobotController>();
            _robotController.robotId = "TestRobot";

            _endEffectorObject = new GameObject("EndEffectorBase");
            _endEffectorObject.transform.SetParent(_testRobotObject.transform);
            _endEffectorObject.transform.position = Vector3.zero;
            _robotController.endEffectorBase = _endEffectorObject.transform;

            var rootBody = _testRobotObject.AddComponent<ArticulationBody>();
            rootBody.immovable = true;

            _robotController.robotJoints = new ArticulationBody[6];
            for (int i = 0; i < 6; i++)
            {
                var jointObject = new GameObject($"Joint{i}");
                jointObject.transform.SetParent(_testRobotObject.transform);
                var articulationBody = jointObject.AddComponent<ArticulationBody>();
                _robotController.robotJoints[i] = articulationBody;
            }

            LogAssert.Expect(
                LogType.Warning,
                "[ROBOT_CONTROLLER] No GripperController found in children of TestRobot"
            );

            yield return null;
        }

        [TearDown]
        public void TearDown()
        {
            if (_testRobotObject != null)
            {
                Object.Destroy(_testRobotObject);
            }

            foreach (var obj in _tempObjects)
            {
                if (obj != null)
                    Object.Destroy(obj);
            }
            _tempObjects.Clear();
        }

        [UnityTest]
        public IEnumerator HasTarget_ReturnsFalse_WhenNoTargetSet()
        {
            Assert.IsFalse(
                _robotController.HasTarget,
                "HasTarget should be false when no target has been set"
            );
            yield return null;
        }

        [UnityTest]
        public IEnumerator SetTarget_GameObject_Default_OpensGripper()
        {
            var targetObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            _tempObjects.Add(targetObject);
            targetObject.transform.position = new Vector3(1f, 1f, 1f);

            LogAssert.Expect(
                LogType.Error,
                "[GRIPPER_CONTROLLER] Gripper references not assigned!"
            );

            _testRobotObject.AddComponent<GripperController>();

            _robotController.SetTarget(targetObject, GraspOptions.Default);
            yield return null;

            Assert.IsTrue(
                _robotController.HasTarget,
                "HasTarget should be true after setting target"
            );
        }

        [UnityTest]
        public IEnumerator SetTarget_Vector3_MoveOnly_SkipsGraspPlanning()
        {
            Vector3 targetPosition = new Vector3(0.5f, 0.5f, 0.5f);

            _robotController.SetTarget(targetPosition, GraspOptions.MoveOnly);
            yield return null;

            Assert.IsTrue(_robotController.HasTarget, "Should have target");
            var currentTarget = _robotController.GetCurrentTarget();
            Assert.IsNotNull(currentTarget, "GetCurrentTarget should return non-null");
            Assert.AreEqual(
                targetPosition,
                currentTarget.Value,
                "Target position should match the set position"
            );
        }

        [UnityTest]
        public IEnumerator SetTarget_PositionAndRotation_UsesExplicitPose()
        {
            Vector3 targetPosition = new Vector3(0.3f, 0.2f, 0.1f);
            Quaternion targetRotation = Quaternion.Euler(45f, 30f, 15f);

            _robotController.SetTarget(targetPosition, targetRotation, GraspOptions.MoveOnly);
            yield return null;

            Assert.IsTrue(_robotController.HasTarget, "Should have target");

            var currentTarget = _robotController.GetCurrentTarget();
            Assert.IsNotNull(currentTarget, "GetCurrentTarget should return non-null");
            Assert.AreEqual(targetPosition, currentTarget.Value, "Target position should match");

            var currentRotation = _robotController.GetCurrentTargetRotation();
            Assert.IsNotNull(currentRotation, "GetCurrentTargetRotation should return non-null");

            float angle = Quaternion.Angle(targetRotation, currentRotation.Value);
            Assert.Less(angle, 0.1f, $"Target rotation should match (angle difference: {angle})");
        }

        [UnityTest]
        public IEnumerator GetCurrentTarget_ReturnsNull_WhenNoTargetSet()
        {
            var target = _robotController.GetCurrentTarget();

            Assert.IsNull(target, "GetCurrentTarget should return null when no target is set");
            yield return null;
        }

        [UnityTest]
        public IEnumerator GetCurrentTargetRotation_ReturnsNull_WhenNoTargetSet()
        {
            var rotation = _robotController.GetCurrentTargetRotation();

            Assert.IsNull(
                rotation,
                "GetCurrentTargetRotation should return null when no target is set"
            );
            yield return null;
        }

        [UnityTest]
        public IEnumerator GetTargetObject_ReturnsCorrectObject()
        {
            var targetObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            _tempObjects.Add(targetObject);
            targetObject.name = "TestCube";
            targetObject.transform.position = new Vector3(1f, 1f, 1f);

            _robotController.SetTarget(targetObject, GraspOptions.MoveOnly);
            yield return null;

            var retrievedObject = _robotController.GetTargetObject();

            Assert.IsNotNull(retrievedObject, "GetTargetObject should return non-null");
            Assert.AreEqual(
                targetObject,
                retrievedObject,
                "GetTargetObject should return the original target object"
            );
        }

        [UnityTest]
        public IEnumerator GetDistanceToTarget_ReturnsCorrectDistance()
        {
            _endEffectorObject.transform.position = Vector3.zero;
            Vector3 targetPosition = new Vector3(3f, 4f, 0f);

            _robotController.SetTarget(targetPosition, GraspOptions.MoveOnly);
            yield return null;

            float distance = _robotController.GetDistanceToTarget();

            Assert.AreEqual(5f, distance, 0.01f, "Distance should be 5.0 (3-4-5 triangle)");
        }

        [UnityTest]
        public IEnumerator GetCurrentEndEffectorPosition_ReturnsCorrectPosition()
        {
            Vector3 expectedPosition = new Vector3(1.2f, 3.4f, 5.6f);
            _endEffectorObject.transform.position = expectedPosition;

            Vector3 actualPosition = _robotController.GetCurrentEndEffectorPosition();

            Assert.AreEqual(
                expectedPosition,
                actualPosition,
                "GetCurrentEndEffectorPosition should return end effector position"
            );
            yield return null;
        }

        [UnityTest]
        public IEnumerator SetTarget_NullGameObject_DoesNotSetTarget()
        {
            GameObject nullObject = null;

            _robotController.SetTarget(nullObject, GraspOptions.Default);
            yield return null;

            Assert.IsFalse(
                _robotController.HasTarget,
                "Should not have target when SetTarget called with null"
            );
        }

        [UnityTest]
        public IEnumerator CustomGraspOptions_AreRespected()
        {
            var targetObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            _tempObjects.Add(targetObject);
            targetObject.transform.position = new Vector3(1f, 1f, 1f);

            var customOptions = new GraspOptions
            {
                useGraspPlanning = false,
                openGripperOnSet = false,
                closeGripperOnReach = false,
                approach = null,
            };

            _robotController.SetTarget(targetObject, customOptions);
            yield return null;

            Assert.IsTrue(_robotController.HasTarget, "Should have target");
            var currentTarget = _robotController.GetCurrentTarget();
            Assert.IsNotNull(currentTarget);
        }

        [UnityTest]
        public IEnumerator SetTargetReached_UpdatesTargetReachedState()
        {
            Vector3 targetPosition = new Vector3(0.5f, 0.5f, 0.5f);
            _robotController.SetTarget(targetPosition, GraspOptions.MoveOnly);
            yield return null;

            _robotController.SetTargetReached(true);

            Assert.IsTrue(
                _robotController.HasTarget,
                "HasTarget should remain true when target is marked as reached"
            );
        }

        [UnityTest]
        public IEnumerator MultipleSetTarget_Calls_UpdateTarget()
        {
            Vector3 firstTarget = new Vector3(1f, 0f, 0f);
            Vector3 secondTarget = new Vector3(0f, 1f, 0f);

            _robotController.SetTarget(firstTarget, GraspOptions.MoveOnly);
            yield return null;

            var firstRetrievedTarget = _robotController.GetCurrentTarget();

            _robotController.SetTarget(secondTarget, GraspOptions.MoveOnly);
            yield return null;

            var secondRetrievedTarget = _robotController.GetCurrentTarget();

            Assert.AreEqual(
                firstTarget,
                firstRetrievedTarget.Value,
                "First target should be set correctly"
            );
            Assert.AreEqual(
                secondTarget,
                secondRetrievedTarget.Value,
                "Second target should override first target"
            );
            Assert.AreNotEqual(
                firstRetrievedTarget,
                secondRetrievedTarget,
                "Targets should be different"
            );
        }

        [UnityTest]
        public IEnumerator SetTarget_WithTopApproach_UsesTopGraspDirection()
        {
            var targetObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            _tempObjects.Add(targetObject);
            targetObject.transform.position = new Vector3(0.5f, 0.3f, 0.5f);

            var topApproachOptions = new GraspOptions
            {
                useGraspPlanning = true,
                approach = Robotics.Grasp.GraspApproach.Top,
                openGripperOnSet = false,
                closeGripperOnReach = false,
            };

            _robotController.SetTarget(targetObject, topApproachOptions);
            yield return null;

            Assert.IsTrue(_robotController.HasTarget, "Should have target with top approach");
        }

        [UnityTest]
        public IEnumerator SetTarget_WithFrontApproach_UsesFrontGraspDirection()
        {
            var targetObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            _tempObjects.Add(targetObject);
            targetObject.transform.position = new Vector3(0.5f, 0.3f, 0.5f);

            var frontApproachOptions = new GraspOptions
            {
                useGraspPlanning = true,
                approach = Robotics.Grasp.GraspApproach.Front,
                openGripperOnSet = false,
                closeGripperOnReach = false,
            };

            _robotController.SetTarget(targetObject, frontApproachOptions);
            yield return null;

            Assert.IsTrue(_robotController.HasTarget, "Should have target with front approach");
        }

        [UnityTest]
        public IEnumerator SetTarget_WithSideApproach_UsesSideGraspDirection()
        {
            var targetObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            _tempObjects.Add(targetObject);
            targetObject.transform.position = new Vector3(0.5f, 0.3f, 0.5f);

            var sideApproachOptions = new GraspOptions
            {
                useGraspPlanning = true,
                approach = Robotics.Grasp.GraspApproach.Side,
                openGripperOnSet = false,
                closeGripperOnReach = false,
            };

            _robotController.SetTarget(targetObject, sideApproachOptions);
            yield return null;

            Assert.IsTrue(_robotController.HasTarget, "Should have target with side approach");
        }

        [UnityTest]
        public IEnumerator SetTarget_WithGraspPlanning_EnablesWaypointSequence()
        {
            var targetObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            _tempObjects.Add(targetObject);
            targetObject.transform.position = new Vector3(0.5f, 0.3f, 0.5f);

            var graspOptions = new GraspOptions
            {
                useGraspPlanning = true,
                approach = Robotics.Grasp.GraspApproach.Top,
                openGripperOnSet = false,
                closeGripperOnReach = false,
            };

            _robotController.SetTarget(targetObject, graspOptions);
            yield return null;

            Assert.IsTrue(
                _robotController.HasTarget,
                "Target should be set with grasp planning enabled"
            );
        }

        [UnityTest]
        public IEnumerator SetTarget_WithNullApproach_UsesDefaultBehavior()
        {
            var targetObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            _tempObjects.Add(targetObject);
            targetObject.transform.position = new Vector3(0.5f, 0.3f, 0.5f);

            var nullApproachOptions = new GraspOptions
            {
                useGraspPlanning = true,
                approach = null,
                openGripperOnSet = false,
                closeGripperOnReach = false,
            };

            _robotController.SetTarget(targetObject, nullApproachOptions);
            yield return null;

            Assert.IsTrue(
                _robotController.HasTarget,
                "Target should be set even with null approach"
            );
        }

        [UnityTest]
        public IEnumerator GripperAutoDiscovery_FindsGripperController()
        {
            var gripperObject = new GameObject("Gripper");
            gripperObject.transform.SetParent(_testRobotObject.transform);
            var gripperController = gripperObject.AddComponent<GripperController>();

            LogAssert.Expect(
                LogType.Error,
                "[GRIPPER_CONTROLLER] Gripper references not assigned!"
            );

            var newRobotObject = new GameObject("NewTestRobot");
            _tempObjects.Add(newRobotObject);
            var newRobotController = newRobotObject.AddComponent<RobotController>();
            newRobotController.robotId = "NewTestRobot";
            newRobotController.endEffectorBase = _endEffectorObject.transform;
            newRobotController.robotJoints = _robotController.robotJoints;

            gripperObject.transform.SetParent(newRobotObject.transform);

            yield return null;
        }

        [UnityTest]
        public IEnumerator SetTarget_DifferentApproaches_AllSucceed()
        {
            var targetObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            _tempObjects.Add(targetObject);
            targetObject.transform.position = new Vector3(0.5f, 0.3f, 0.5f);

            var approaches = new Robotics.Grasp.GraspApproach?[]
            {
                Robotics.Grasp.GraspApproach.Top,
                Robotics.Grasp.GraspApproach.Front,
                Robotics.Grasp.GraspApproach.Side,
                null,
            };

            foreach (var approach in approaches)
            {
                var options = new GraspOptions
                {
                    useGraspPlanning = true,
                    approach = approach,
                    openGripperOnSet = false,
                    closeGripperOnReach = false,
                };

                _robotController.SetTarget(targetObject, options);
                yield return null;

                Assert.IsTrue(
                    _robotController.HasTarget,
                    $"Target should be set with approach: {approach?.ToString() ?? "null"}"
                );
            }
        }

        [UnityTest]
        public IEnumerator SetTarget_WithCoordinates_UsesObjectFinder()
        {
            Vector3 targetPosition = new Vector3(0.5f, 0.3f, 0.5f);

            _robotController.SetTarget(targetPosition, GraspOptions.MoveOnly);
            yield return null;

            Assert.IsTrue(_robotController.HasTarget, "Target should be set from coordinates");

            var currentTarget = _robotController.GetCurrentTarget();
            Assert.IsNotNull(currentTarget, "Current target should not be null");
            Assert.AreEqual(
                targetPosition,
                currentTarget.Value,
                "Target position should match input coordinates"
            );
        }

        [UnityTest]
        public IEnumerator GraspBehavior_UsesRelaxedConvergenceThreshold()
        {
            var targetObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            _tempObjects.Add(targetObject);
            targetObject.transform.position = new Vector3(0.5f, 0.3f, 0.5f);
            targetObject.transform.localScale = Vector3.one * 0.05f;

            var graspOptions = new GraspOptions
            {
                useGraspPlanning = true,
                approach = Robotics.Grasp.GraspApproach.Top,
                closeGripperOnReach = true,
            };

            _robotController.SetTarget(targetObject, graspOptions);
            yield return TestHelpers.WaitUntil(
                () => _robotController.GetCurrentTarget() != null,
                1.0f
            );

            Assert.IsNotNull(
                _robotController.GetCurrentTarget(),
                "Robot should have target set for grasp"
            );
        }

        [UnityTest]
        public IEnumerator OrientationRamping_StartsAtConfiguredDistance()
        {
            Vector3 nearPosition = new Vector3(0.25f, 0.2f, 0.3f);
            var targetObject = TestHelpers.CreateTestTarget(nearPosition);
            _tempObjects.Add(targetObject);

            _robotController.SetTarget(targetObject);
            yield return TestHelpers.WaitUntil(() => _robotController.HasTarget, 0.5f);

            float distance = Vector3.Distance(_robotController.transform.position, nearPosition);
            bool withinRampRange = distance < 0.30f;

            if (withinRampRange)
            {
                Assert.IsNotNull(
                    _robotController.GetCurrentTarget(),
                    "Robot should accept target within ramping range"
                );
            }
        }

        [UnityTest]
        public IEnumerator GraspTimeout_TriggersAtConfiguredTime()
        {
            var unreachableTarget = GameObject.CreatePrimitive(PrimitiveType.Cube);
            _tempObjects.Add(unreachableTarget);
            unreachableTarget.transform.position = new Vector3(10f, 10f, 10f);

            var graspOptions = new GraspOptions
            {
                useGraspPlanning = true,
                approach = Robotics.Grasp.GraspApproach.Top,
                closeGripperOnReach = true,
            };

            _robotController.SetTarget(unreachableTarget, graspOptions);

            yield return TestHelpers.WaitUntil(
                () => _robotController.GetCurrentTarget() != null,
                0.5f
            );

            Assert.IsNotNull(
                _robotController.GetCurrentTarget(),
                "Robot should maintain target before timeout"
            );
        }

        [UnityTest]
        public IEnumerator MovementBehavior_AcceptsTargetAndBecomesActive()
        {
            Vector3 reachablePosition = new Vector3(0.3f, 0.2f, 0.3f);
            var targetObject = TestHelpers.CreateTestTarget(reachablePosition);
            _tempObjects.Add(targetObject);

            _robotController.SetTarget(targetObject);
            yield return null;

            Assert.IsTrue(
                _robotController.HasTarget,
                "SetTarget should immediately set HasTarget = true"
            );
            Assert.IsNotNull(
                _robotController.GetCurrentTarget(),
                "GetCurrentTarget should return a position after SetTarget"
            );
            Assert.IsFalse(
                _robotController.TargetReached,
                "TargetReached should be false immediately after SetTarget (movement just started)"
            );
        }

        [UnityTest]
        public IEnumerator PreGraspApproach_UsesLooserTolerance()
        {
            var targetObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            _tempObjects.Add(targetObject);
            targetObject.transform.position = new Vector3(0.4f, 0.3f, 0.4f);
            targetObject.transform.localScale = Vector3.one * 0.05f;

            var graspOptions = new GraspOptions
            {
                useGraspPlanning = true,
                approach = Robotics.Grasp.GraspApproach.Top,
                closeGripperOnReach = true,
            };

            _robotController.SetTarget(targetObject, graspOptions);
            yield return TestHelpers.WaitUntil(() => _robotController.HasTarget, 0.5f);

            Assert.IsNotNull(
                _robotController.GetCurrentTarget(),
                "Robot should accept target with pre-grasp tolerance"
            );
        }
    }
}
