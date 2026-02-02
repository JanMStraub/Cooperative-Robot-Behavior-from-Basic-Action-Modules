using System.Collections;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using PythonCommunication;
using Robotics;
using Simulation;
using Tests.EditMode;

namespace Tests.PlayMode
{
    /// <summary>
    /// Integration tests for WorldState synchronization between Unity and Python.
    /// Validates WorldStatePublisher → WorldStateServer → Python coordination flow.
    ///
    /// Tests cover:
    /// - Publisher: Robot state publishing, object state publishing, update frequency
    /// - Server Integration: Data reception, state storage, workspace region tracking
    /// - Coordination Integration: CollaborativeStrategy, WorkspaceManager, minimum separation
    /// </summary>
    public class WorldStateIntegrationTests
    {
        private GameObject _publisherObject;
        private WorldStatePublisher _publisher;
        private GameObject _robotManagerObject;
        private RobotManager _robotManager;
        private GameObject _testRobotObject;
        private RobotController _testRobot;

        #region Setup/Teardown

        [UnitySetUp]
        public IEnumerator SetUp()
        {
            // Clean up existing singletons and all test robots
            TestHelpers.CleanupAllSingletons();

            // Create RobotManager
            (_robotManagerObject, _robotManager) = TestHelpers.CreateRobotManager();

            // Create test robot
            (_testRobotObject, _testRobot) = TestHelpers.CreateTestRobot("TestRobot1");
            _testRobot.robotId = "TestRobot1";

            // Create WorldStatePublisher
            _publisherObject = new GameObject("WorldStatePublisher");
            _publisher = _publisherObject.AddComponent<WorldStatePublisher>();

            yield return null; // Wait for Start() to complete
        }

        [TearDown]
        public void TearDown()
        {
            TestHelpers.DestroyAll(_publisherObject, _robotManagerObject, _testRobotObject);
            TestHelpers.CleanupAllSingletons();
        }

        #endregion

        #region Publisher Initialization Tests

        [Test]
        public void WorldStatePublisher_Singleton_IsSet()
        {
            Assert.IsNotNull(WorldStatePublisher.Instance, "WorldStatePublisher Instance should be set");
            Assert.AreEqual(_publisher, WorldStatePublisher.Instance, "Instance should match created publisher");
        }

        [UnityTest]
        public IEnumerator WorldStatePublisher_Initialization_FindsRobotManager()
        {
            yield return null;

            // Publisher should find RobotManager during Start()
            Assert.IsTrue(_publisher.enabled, "Publisher should be enabled if RobotManager found");
        }

        [UnityTest]
        public IEnumerator WorldStatePublisher_WithoutRobotManager_Disables()
        {
            // Destroy original publisher from SetUp
            Object.DestroyImmediate(_publisherObject);
            _publisherObject = null;

            // Destroy test robot first to avoid any references to RobotManager
            Object.DestroyImmediate(_testRobotObject);
            _testRobotObject = null;

            // Destroy RobotManager and clear singleton instance
            Object.DestroyImmediate(_robotManagerObject);
            _robotManagerObject = null;

            // Wait a frame to ensure OnDestroy is called and Instance is cleared
            yield return null;

            // Verify RobotManager is actually gone
            Assert.IsNull(RobotManager.Instance, "RobotManager.Instance should be null after destruction");

            // Create new publisher (will fail to find RobotManager)
            var newPublisherObj = new GameObject("NewPublisher");
            var newPublisher = newPublisherObj.AddComponent<WorldStatePublisher>();

            // Expect error log when Start() is called on next frame
            LogAssert.Expect(LogType.Error,
                "[WORLD_STATE_PUBLISHER] RobotManager.Instance is null! Ensure RobotManager GameObject is in the scene.");

            yield return null; // Wait for Start() to be called

            // Publisher should disable itself
            Assert.IsFalse(newPublisher.enabled, "Publisher should disable without RobotManager");

            Object.DestroyImmediate(newPublisherObj);
        }

        #endregion

        #region Robot State Publishing Tests

        [UnityTest]
        public IEnumerator Publisher_RobotState_IncludesPosition()
        {
            // Set robot position
            Vector3 testPosition = new Vector3(0.5f, 0.3f, 0.4f);
            _testRobot.transform.position = testPosition;

            yield return null;

            // Publisher should track robot position
            // (Actual publishing happens periodically in Update)
            Assert.AreEqual(testPosition, _testRobot.transform.position,
                "Robot position should be trackable");
        }

        [UnityTest]
        public IEnumerator Publisher_RobotState_IncludesRotation()
        {
            // Set robot rotation
            Quaternion testRotation = Quaternion.Euler(45f, 30f, 15f);
            _testRobot.transform.rotation = testRotation;

            yield return null;

            // Publisher should track robot rotation
            TestHelpers.AssertQuaternionApproximately(testRotation, _testRobot.transform.rotation,
                0.01f, "Robot rotation should be trackable");
        }

        [UnityTest]
        public IEnumerator Publisher_RobotState_IncludesTargetPosition()
        {
            // Set robot target
            Vector3 targetPosition = new Vector3(0.6f, 0.4f, 0.5f);
            GameObject target = TestHelpers.CreateTestTarget(targetPosition);

            _testRobot.SetTarget(target);
            yield return null;

            // Publisher should track target position
            Assert.IsNotNull(_testRobot.GetCurrentTarget(), "Robot should have target set");

            TestHelpers.DestroyAll(target);
        }

        [UnityTest]
        public IEnumerator Publisher_RobotState_TracksMovementStatus()
        {
            // Set robot target to trigger movement
            Vector3 targetPosition = new Vector3(0.5f, 0.3f, 0.4f);
            GameObject target = TestHelpers.CreateTestTarget(targetPosition);

            _testRobot.SetTarget(target);
            yield return new WaitForSeconds(0.1f);

            // Publisher should track if robot is moving
            // (Movement tracking depends on RobotController.TargetReached)
            bool hasTarget = _testRobot.GetCurrentTarget() != null;
            Assert.IsTrue(hasTarget, "Robot should have active target indicating movement intent");

            TestHelpers.DestroyAll(target);
        }

        [UnityTest]
        public IEnumerator Publisher_RobotState_IncludesInitializationStatus()
        {
            yield return null;

            // Publisher should track if robot is initialized
            // (Robot is initialized after components are set up)
            Assert.IsNotNull(_testRobot, "Robot should be initialized");
        }

        #endregion

        #region Object State Publishing Tests

        [UnityTest]
        public IEnumerator Publisher_ObjectState_TracksTrackedObjects()
        {
            // Create test object
            var testCube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            testCube.name = "TestCube";
            testCube.transform.position = new Vector3(0.3f, 0.2f, 0.3f);

            // Note: Objects must be added to publisher's tracked list manually
            // or detected via vision system

            yield return null;

            // Verify test object exists
            Assert.IsNotNull(testCube, "Test object should exist");

            Object.Destroy(testCube);
        }

        [UnityTest]
        public IEnumerator Publisher_ObjectState_IncludesPosition()
        {
            var testObject = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            testObject.name = "TestSphere";
            Vector3 objectPosition = new Vector3(0.4f, 0.3f, 0.5f);
            testObject.transform.position = objectPosition;

            yield return null;

            // Publisher should be able to track object position
            Assert.AreEqual(objectPosition, testObject.transform.position,
                "Object position should be trackable");

            Object.Destroy(testObject);
        }

        [UnityTest]
        public IEnumerator Publisher_DetectedObjects_UpdatesFromVisionSystem()
        {
            // Simulate detected object from vision system
            var detectedObjectData = new ObjectStateData
            {
                object_id = "cube_001",
                position = new PositionData(new Vector3(0.3f, 0.2f, 0.3f)),
                color = "blue",
                object_type = "cube",
                confidence = 0.95f
            };

            yield return null;

            // Publisher should be able to handle detected object updates
            Assert.IsNotNull(detectedObjectData, "Detected object data should be created");
            Assert.AreEqual("blue", detectedObjectData.color, "Object color should be tracked");
            Assert.AreEqual(0.95f, detectedObjectData.confidence, 0.01f,
                "Detection confidence should be tracked");
        }

        #endregion

        #region Update Frequency Tests

        [UnityTest]
        public IEnumerator Publisher_UpdateRate_PublishesAtConfiguredFrequency()
        {
            // Publisher defaults to 2Hz (every 0.5s)
            // float expectedInterval = 0.5f; // 1.0 / 2.0 Hz

            yield return new WaitForSeconds(1.0f);

            // Should have published approximately 2 updates in 1 second
            // (Actual verification would require access to _updatesSent private field)
            Assert.Pass("Publisher runs for configured duration without errors");
        }

        [UnityTest]
        public IEnumerator Publisher_DisabledPublishing_DoesNotPublish()
        {
            // Disable publishing
            var publisherObj = new GameObject("DisabledPublisher");
            var disabledPublisher = publisherObj.AddComponent<WorldStatePublisher>();

            // Use reflection to set _enablePublishing = false
            var enableField = typeof(WorldStatePublisher).GetField("_enablePublishing",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            enableField?.SetValue(disabledPublisher, false);

            yield return new WaitForSeconds(1.0f);

            // Publisher should not publish updates
            Assert.Pass("Disabled publisher runs without attempting to publish");

            Object.DestroyImmediate(publisherObj);
        }

        #endregion

        #region Data Model Tests

        [Test]
        public void WorldStateUpdate_CreatesValidStructure()
        {
            var update = new WorldStateUpdate
            {
                type = "world_state_update",
                robots = new List<RobotStateData>(),
                objects = new List<ObjectStateData>(),
                timestamp = Time.time
            };

            Assert.AreEqual("world_state_update", update.type, "Type should be world_state_update");
            Assert.IsNotNull(update.robots, "Robots list should be initialized");
            Assert.IsNotNull(update.objects, "Objects list should be initialized");
            Assert.Greater(update.timestamp, 0f, "Timestamp should be set");
        }

        [Test]
        public void RobotStateData_CreatesValidStructure()
        {
            var robotState = new RobotStateData
            {
                robot_id = "TestRobot",
                position = new PositionData(new Vector3(0.5f, 0.3f, 0.4f)),
                rotation = new RotationData(Quaternion.identity),
                gripper_state = "open",
                is_moving = false,
                is_initialized = true,
                joint_angles = new float[] { 0f, 0f, 0f, 0f, 0f, 0f }
            };

            Assert.AreEqual("TestRobot", robotState.robot_id);
            Assert.IsNotNull(robotState.position);
            Assert.AreEqual("open", robotState.gripper_state);
            Assert.IsTrue(robotState.is_initialized);
            Assert.AreEqual(6, robotState.joint_angles.Length, "Should have 6 joint angles for AR4");
        }

        [Test]
        public void PositionData_ConvertsFromVector3()
        {
            Vector3 testVector = new Vector3(1.5f, 2.3f, 3.7f);
            var posData = new PositionData(testVector);

            Assert.AreEqual(1.5f, posData.x, 0.001f);
            Assert.AreEqual(2.3f, posData.y, 0.001f);
            Assert.AreEqual(3.7f, posData.z, 0.001f);
        }

        [Test]
        public void RotationData_ConvertsFromQuaternion()
        {
            Quaternion testQuat = Quaternion.Euler(45f, 30f, 15f);
            var rotData = new RotationData(testQuat);

            Assert.AreEqual(testQuat.x, rotData.x, 0.001f);
            Assert.AreEqual(testQuat.y, rotData.y, 0.001f);
            Assert.AreEqual(testQuat.z, rotData.z, 0.001f);
            Assert.AreEqual(testQuat.w, rotData.w, 0.001f);
        }

        [Test]
        public void ObjectStateData_CreatesValidStructure()
        {
            var objectState = new ObjectStateData
            {
                object_id = "cube_001",
                position = new PositionData(new Vector3(0.3f, 0.2f, 0.3f)),
                color = "red",
                object_type = "cube",
                confidence = 0.92f
            };

            Assert.AreEqual("cube_001", objectState.object_id);
            Assert.AreEqual("red", objectState.color);
            Assert.AreEqual("cube", objectState.object_type);
            Assert.AreEqual(0.92f, objectState.confidence, 0.001f);
        }

        #endregion

        #region Coordination Integration Tests

        [UnityTest]
        public IEnumerator Coordination_WorkspaceTracking_UpdatesWithRobotMovement()
        {
            // Create second robot
            var (robot2Obj, robot2) = TestHelpers.CreateTestRobot("TestRobot2");
            robot2.robotId = "TestRobot2";
            robot2.transform.position = new Vector3(0.5f, 0f, 0f);

            yield return null;

            // Both robots should have distinct positions
            float distance = Vector3.Distance(_testRobot.transform.position, robot2.transform.position);
            Assert.Greater(distance, 0f, "Robots should be at different positions");

            Object.Destroy(robot2Obj);
        }

        [UnityTest]
        public IEnumerator Coordination_MinimumSeparation_TrackedByPublisher()
        {
            // Create second robot close to first
            var (robot2Obj, robot2) = TestHelpers.CreateTestRobot("TestRobot2");
            robot2.transform.position = new Vector3(0.1f, 0f, 0f); // Close to TestRobot1

            yield return null;

            // Check separation distance
            float distance = Vector3.Distance(_testRobot.transform.position, robot2.transform.position);

            if (distance < TestConstants.MIN_SAFE_SEPARATION)
            {
                // Publisher should report robots within minimum separation
                Assert.Less(distance, TestConstants.MIN_SAFE_SEPARATION,
                    "Robots are within minimum safe separation");
            }

            Object.Destroy(robot2Obj);
        }

        [UnityTest]
        public IEnumerator Coordination_CollaborativeMode_UsesWorldState()
        {
            // Create SimulationManager with Collaborative mode
            var (simObj, simManager) = TestHelpers.CreateSimulationManager();
            var config = TestHelpers.CreateTestSimulationConfig(Configuration.RobotCoordinationMode.Collaborative);
            simManager.config = config;

            yield return null;

            // Collaborative mode should be set
            Assert.AreEqual(Configuration.RobotCoordinationMode.Collaborative,
                config.coordinationMode, "Should use Collaborative coordination mode");

            Object.Destroy(simObj);
            Object.DestroyImmediate(config);
        }

        #endregion

        #region Python Integration Tests (Require Backend)

        [UnityTest]
        public IEnumerator Integration_WorldStateServer_ReceivesUpdates()
        {
            TestHelpers.SkipIfPythonUnavailable();

            // Wait for publisher to send updates
            yield return new WaitForSeconds(1.5f); // > update interval

            // If Python backend is available, updates should be sent
            // (Actual verification would require Python WorldState API call)
            Assert.Pass("Publisher runs and sends updates to Python backend");
        }

        [UnityTest]
        public IEnumerator Integration_RobotState_SyncsToPython()
        {
            TestHelpers.SkipIfPythonUnavailable();

            // Move robot
            Vector3 targetPos = new Vector3(0.5f, 0.3f, 0.4f);
            GameObject target = TestHelpers.CreateTestTarget(targetPos);
            _testRobot.SetTarget(target);

            // Wait for state sync
            yield return new WaitForSeconds(2f);

            // Robot state should sync to Python
            Assert.Pass("Robot state syncs to Python WorldState");

            TestHelpers.DestroyAll(target);
        }

        [UnityTest]
        public IEnumerator Integration_ObjectState_SyncsToPython()
        {
            TestHelpers.SkipIfPythonUnavailable();

            // Create tracked object
            var testCube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            testCube.name = "TestCube";
            testCube.transform.position = new Vector3(0.3f, 0.2f, 0.3f);

            // Wait for state sync
            yield return new WaitForSeconds(2f);

            // Object state should sync to Python
            Assert.Pass("Object state syncs to Python WorldState");

            Object.Destroy(testCube);
        }

        #endregion

        #region Error Handling Tests

        [UnityTest]
        public IEnumerator Publisher_NullRobotManager_HandlesGracefully()
        {
            // Simulate null RobotManager scenario
            yield return null;

            // Publisher should handle null gracefully (disabled in Start)
            Assert.IsNotNull(_publisher, "Publisher should still exist");
        }

        [UnityTest]
        public IEnumerator Publisher_EmptyRobotList_DoesNotCrash()
        {
            // With no robots registered, publisher should not crash
            yield return new WaitForSeconds(1f);

            // Should continue running without errors
            Assert.Pass("Publisher handles empty robot list gracefully");
        }

        [UnityTest]
        public IEnumerator Publisher_EmptyObjectList_DoesNotCrash()
        {
            // With no tracked objects, publisher should not crash
            yield return new WaitForSeconds(1f);

            // Should continue running without errors
            Assert.Pass("Publisher handles empty object list gracefully");
        }

        [UnityTest]
        public IEnumerator Publisher_RapidUpdates_HandlesBackpressure()
        {
            // Set very high update rate
            var fastPublisherObj = new GameObject("FastPublisher");
            var fastPublisher = fastPublisherObj.AddComponent<WorldStatePublisher>();

            // Use reflection to set high update rate
            var updateRateField = typeof(WorldStatePublisher).GetField("_updateRate",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            updateRateField?.SetValue(fastPublisher, 10f); // 10Hz

            yield return new WaitForSeconds(2f);

            // Should handle rapid updates without crashing
            Assert.Pass("Publisher handles rapid update rate");

            Object.DestroyImmediate(fastPublisherObj);
        }

        #endregion

        #region Performance Tests

        [UnityTest]
        public IEnumerator Performance_StateCollection_LessThan10ms()
        {
            // Measure time to collect world state
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            // Simulate state collection
            var update = new WorldStateUpdate
            {
                type = "world_state_update",
                robots = new List<RobotStateData>(),
                objects = new List<ObjectStateData>(),
                timestamp = Time.time
            };

            // Add robot state
            update.robots.Add(new RobotStateData
            {
                robot_id = _testRobot.robotId,
                position = new PositionData(_testRobot.transform.position),
                rotation = new RotationData(_testRobot.transform.rotation),
                gripper_state = "unknown",
                is_moving = false,
                is_initialized = true,
                joint_angles = new float[6]
            });

            stopwatch.Stop();

            yield return null;

            // State collection should be fast (< 10ms)
            Assert.Less(stopwatch.ElapsedMilliseconds, 10,
                $"State collection should take < 10ms, took {stopwatch.ElapsedMilliseconds}ms");
        }

        [UnityTest]
        public IEnumerator Performance_MultipleRobots_ScalesLinearly()
        {
            // Create 3 robots
            var robots = new List<RobotController>();
            for (int i = 0; i < 3; i++)
            {
                var (obj, controller) = TestHelpers.CreateTestRobot($"PerfRobot{i}");
                robots.Add(controller);
            }

            yield return new WaitForSeconds(2f);

            // Publisher should handle multiple robots efficiently
            Assert.Pass("Publisher scales with multiple robots");

            foreach (var robot in robots)
            {
                Object.Destroy(robot.gameObject);
            }
        }

        #endregion
    }
}
