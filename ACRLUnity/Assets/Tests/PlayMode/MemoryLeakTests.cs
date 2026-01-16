using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using Robotics;

namespace Tests.PlayMode
{
    /// <summary>
    /// Tests to verify memory leak fixes in RobotController
    /// Specifically tests GameObject caching to prevent memory leaks
    /// </summary>
    public class MemoryLeakTests
    {
        private GameObject _testRobotObject;
        private RobotController _robotController;

        [UnitySetUp]
        public IEnumerator SetUp()
        {
            // Create minimal robot setup for testing
            _testRobotObject = new GameObject("TestRobot");
            _robotController = _testRobotObject.AddComponent<RobotController>();
            _robotController.robotId = "TestRobot";

            // Add minimal required components
            var endEffectorBase = new GameObject("EndEffectorBase");
            endEffectorBase.transform.SetParent(_testRobotObject.transform);
            _robotController.endEffectorBase = endEffectorBase.transform;

            // Expect initialization warnings since we're not setting up full ArticulationBody chain
            // Must be called BEFORE the logs appear (before Start() is invoked)
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] No GripperController found in children of TestRobot");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");

            // Wait for Start() to be called - this will trigger the expected logs
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
        public IEnumerator SetTarget_Vector3_DoesNotLeakGameObjects()
        {
            // Arrange
            int initialObjectCount = Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None).Length;
            Vector3 targetPosition = new Vector3(1f, 2f, 3f);

            // Act - Call SetTarget multiple times (this used to create new GameObjects each time)
            for (int i = 0; i < 100; i++)
            {
                _robotController.SetTarget(
                    new Vector3(i * 0.01f, i * 0.01f, i * 0.01f),
                    GraspOptions.MoveOnly
                );
                yield return null;
            }

            // Allow garbage collection
            yield return null;
            System.GC.Collect();
            yield return null;

            // Assert
            int finalObjectCount = Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None).Length;
            int objectDelta = finalObjectCount - initialObjectCount;

            // Should create at most 1 cached temporary object, not 100
            Assert.LessOrEqual(objectDelta, 1,
                $"Expected at most 1 new GameObject (cached temp target), but found {objectDelta} new objects. " +
                "This indicates a memory leak from repeated SetTarget calls.");
        }

        [UnityTest]
        public IEnumerator SetTarget_GameObject_WithGraspPlanning_DoesNotLeakGameObjects()
        {
            // Arrange
            var targetObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            targetObject.name = "TargetCube";
            targetObject.transform.position = new Vector3(1f, 1f, 1f);

            int initialObjectCount = Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None).Length;

            // Act - Call SetTarget with grasp planning multiple times
            for (int i = 0; i < 100; i++)
            {
                targetObject.transform.position = new Vector3(i * 0.01f, 1f, 1f);
                _robotController.SetTarget(targetObject, GraspOptions.Default);
                yield return null;
            }

            // Allow garbage collection
            yield return null;
            System.GC.Collect();
            yield return null;

            // Assert
            int finalObjectCount = Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None).Length;
            int objectDelta = finalObjectCount - initialObjectCount;

            // Should create at most 1 cached grasp target object, not 100
            Assert.LessOrEqual(objectDelta, 1,
                $"Expected at most 1 new GameObject (cached grasp target), but found {objectDelta} new objects. " +
                "This indicates a memory leak from repeated grasp planning.");

            // Cleanup
            Object.Destroy(targetObject);
        }

        [UnityTest]
        public IEnumerator SetTarget_WithRotation_DoesNotLeakGameObjects()
        {
            // Arrange
            int initialObjectCount = Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None).Length;

            // Act - Call SetTarget with rotation multiple times
            for (int i = 0; i < 100; i++)
            {
                _robotController.SetTarget(
                    position: new Vector3(i * 0.01f, i * 0.01f, i * 0.01f),
                    rotation: Quaternion.Euler(i, 0f, 0f),
                    options: GraspOptions.MoveOnly
                );
                yield return null;
            }

            // Allow garbage collection
            yield return null;
            System.GC.Collect();
            yield return null;

            // Assert
            int finalObjectCount = Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None).Length;
            int objectDelta = finalObjectCount - initialObjectCount;

            // Should create at most 1 cached temporary object, not 100
            Assert.LessOrEqual(objectDelta, 1,
                $"Expected at most 1 new GameObject (cached temp target), but found {objectDelta} new objects. " +
                "This indicates a memory leak from repeated SetTarget with rotation.");
        }

        [UnityTest]
        public IEnumerator OnDestroy_CleansUp_CachedGameObjects()
        {
            // Arrange
            var targetObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            targetObject.transform.position = new Vector3(1f, 1f, 1f);

            // Create cached objects by calling SetTarget
            _robotController.SetTarget(new Vector3(1f, 0f, 0f), GraspOptions.MoveOnly);
            yield return null;
            _robotController.SetTarget(targetObject, GraspOptions.Default);
            yield return null;

            // Get count before destruction
            int countBeforeDestroy = Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None).Length;

            // Act - Destroy the robot controller
            Object.Destroy(_testRobotObject);
            _testRobotObject = null; // Prevent TearDown from double-destroying

            yield return null; // Allow Unity to process destruction

            // Assert
            int countAfterDestroy = Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None).Length;

            // Should have fewer objects after destruction (cached objects cleaned up)
            Assert.Less(countAfterDestroy, countBeforeDestroy,
                "Cached GameObjects should be destroyed when RobotController is destroyed");

            // Cleanup
            Object.Destroy(targetObject);
        }

        [UnityTest]
        public IEnumerator Mixed_SetTarget_Calls_ShareCachedObjects()
        {
            // Arrange
            var targetObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            targetObject.transform.position = new Vector3(1f, 1f, 1f);

            int initialObjectCount = Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None).Length;

            // Act - Mix different SetTarget calls
            for (int i = 0; i < 50; i++)
            {
                // Alternate between different SetTarget overloads
                if (i % 3 == 0)
                {
                    _robotController.SetTarget(new Vector3(i * 0.01f, 0f, 0f), GraspOptions.MoveOnly);
                }
                else if (i % 3 == 1)
                {
                    _robotController.SetTarget(
                        new Vector3(i * 0.01f, 0f, 0f),
                        Quaternion.identity,
                        GraspOptions.MoveOnly);
                }
                else
                {
                    targetObject.transform.position = new Vector3(i * 0.01f, 1f, 1f);
                    _robotController.SetTarget(targetObject, GraspOptions.Default);
                }
                yield return null;
            }

            // Allow garbage collection
            yield return null;
            System.GC.Collect();
            yield return null;

            // Assert
            int finalObjectCount = Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None).Length;
            int objectDelta = finalObjectCount - initialObjectCount;

            // Should create at most 2 cached objects (grasp target + temp target), not 50
            Assert.LessOrEqual(objectDelta, 2,
                $"Expected at most 2 new GameObjects (cached targets), but found {objectDelta} new objects. " +
                "This indicates cached objects are not being reused.");

            // Cleanup
            Object.Destroy(targetObject);
        }
    }
}
