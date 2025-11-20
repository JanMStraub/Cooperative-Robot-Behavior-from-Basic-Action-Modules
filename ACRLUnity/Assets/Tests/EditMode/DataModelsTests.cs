using NUnit.Framework;
using System.Collections.Generic;
using UnityEngine;
using Logging;

namespace Tests.EditMode
{
    /// <summary>
    /// Tests for Logging.DataModels classes.
    /// Validates data model initialization, serialization, and field access.
    /// </summary>
    public class DataModelsTests
    {
        #region RobotAction Tests

        [Test]
        public void RobotAction_DefaultConstructor_InitializesArrays()
        {
            var action = new RobotAction();

            Assert.IsNotNull(action.robotIds);
            Assert.IsNotNull(action.objectIds);
            Assert.IsNotNull(action.trajectoryPoints);
            Assert.IsNotNull(action.metrics);
            Assert.IsNotNull(action.childActionIds);
            Assert.AreEqual(0, action.robotIds.Length);
            Assert.AreEqual(0, action.objectIds.Length);
            Assert.AreEqual(0, action.trajectoryPoints.Length);
            Assert.AreEqual(0, action.metrics.Count);
            Assert.AreEqual(0, action.childActionIds.Length);
        }

        [Test]
        public void RobotAction_CanSetAllFields()
        {
            var action = new RobotAction
            {
                actionId = "test-action-001",
                actionName = "move_to_target",
                description = "Move robot to pickup location",
                type = ActionType.Movement,
                status = ActionStatus.InProgress,
                robotIds = new[] { "Robot1", "Robot2" },
                objectIds = new[] { "Cube_01" },
                timestamp = "2024-01-01T00:00:00Z",
                gameTime = 10.5f,
                duration = 2.3f,
                startPosition = new Vector3(0, 0, 0),
                targetPosition = new Vector3(1, 1, 1),
                trajectoryPoints = new[] { new Vector3(0.5f, 0.5f, 0.5f) },
                detectedTargetWorldPosition = new Vector3(1, 1, 1),
                depthEstimationConfidence = 0.95f,
                depthEstimationMethod = "stereo_disparity",
                success = true,
                errorMessage = null,
                qualityScore = 0.85f,
                humanReadable = "Robot moved to target",
                parentActionId = "parent-001",
                childActionIds = new[] { "child-001" }
            };

            action.metrics["distance_traveled"] = 1.73f;

            Assert.AreEqual("test-action-001", action.actionId);
            Assert.AreEqual(ActionType.Movement, action.type);
            Assert.AreEqual(ActionStatus.InProgress, action.status);
            Assert.AreEqual(2, action.robotIds.Length);
            Assert.AreEqual(1.73f, action.metrics["distance_traveled"]);
            Assert.AreEqual(0.95f, action.depthEstimationConfidence);
        }

        [Test]
        public void RobotAction_NullableFieldsAreNullByDefault()
        {
            var action = new RobotAction();

            Assert.IsNull(action.detectedTargetWorldPosition);
            Assert.IsNull(action.depthEstimationConfidence);
        }

        [Test]
        public void ActionType_HasExpectedValues()
        {
            Assert.AreEqual(0, (int)ActionType.Task);
            Assert.AreEqual(1, (int)ActionType.Movement);
            Assert.AreEqual(2, (int)ActionType.Manipulation);
            Assert.AreEqual(3, (int)ActionType.Coordination);
            Assert.AreEqual(4, (int)ActionType.Observation);
        }

        [Test]
        public void ActionStatus_HasExpectedValues()
        {
            Assert.AreEqual(0, (int)ActionStatus.Started);
            Assert.AreEqual(1, (int)ActionStatus.InProgress);
            Assert.AreEqual(2, (int)ActionStatus.Completed);
            Assert.AreEqual(3, (int)ActionStatus.Failed);
        }

        #endregion

        #region SceneSnapshot Tests

        [Test]
        public void SceneSnapshot_DefaultConstructor_InitializesArrays()
        {
            var snapshot = new SceneSnapshot();

            Assert.IsNotNull(snapshot.objects);
            Assert.IsNotNull(snapshot.robots);
            Assert.AreEqual(0, snapshot.objects.Length);
            Assert.AreEqual(0, snapshot.robots.Length);
        }

        [Test]
        public void SceneSnapshot_CanSetAllFields()
        {
            var snapshot = new SceneSnapshot
            {
                snapshotId = "snap-001",
                timestamp = "2024-01-01T00:00:00Z",
                gameTime = 15.0f,
                totalObjects = 5,
                graspableObjects = 3,
                sceneDescription = "Test scene with cubes",
                objects = new[]
                {
                    new Logging.Object { id = "obj-1", name = "Cube1", type = "cube" }
                },
                robots = new[]
                {
                    new RobotState { robotId = "Robot1" }
                }
            };

            Assert.AreEqual("snap-001", snapshot.snapshotId);
            Assert.AreEqual(1, snapshot.objects.Length);
            Assert.AreEqual(1, snapshot.robots.Length);
        }

        #endregion

        #region Object Tests

        [Test]
        public void Object_CanSetAllFields()
        {
            var obj = new Logging.Object
            {
                id = "cube-001",
                name = "RedCube",
                type = "cube",
                position = new Vector3(1, 2, 3),
                rotation = Quaternion.identity,
                isGraspable = true,
                isMovable = true,
                mass = 0.5f
            };

            Assert.AreEqual("cube-001", obj.id);
            Assert.AreEqual("RedCube", obj.name);
            Assert.AreEqual(new Vector3(1, 2, 3), obj.position);
            Assert.IsTrue(obj.isGraspable);
            Assert.AreEqual(0.5f, obj.mass);
        }

        #endregion

        #region RobotState Tests

        [Test]
        public void RobotState_DefaultConstructor_InitializesArrays()
        {
            var state = new RobotState();

            Assert.IsNotNull(state.jointAngles);
            Assert.AreEqual(0, state.jointAngles.Length);
        }

        [Test]
        public void RobotState_CanSetAllFields()
        {
            var state = new RobotState
            {
                robotId = "Robot1",
                position = new Vector3(1, 2, 3),
                rotation = Quaternion.Euler(0, 90, 0),
                jointAngles = new[] { 0f, 0.5f, 1f, 1.5f, 2f, 2.5f },
                targetPosition = new Vector3(2, 3, 4),
                distanceToTarget = 1.73f,
                isMoving = true,
                currentAction = "moving_to_target"
            };

            Assert.AreEqual("Robot1", state.robotId);
            Assert.AreEqual(6, state.jointAngles.Length);
            Assert.AreEqual(0.5f, state.jointAngles[1]);
            Assert.IsTrue(state.isMoving);
        }

        #endregion

        #region LogEntry Tests

        [Test]
        public void LogEntry_CanSetAllFields()
        {
            var entry = new LogEntry
            {
                logId = "log-001",
                timestamp = "2024-01-01T00:00:00Z",
                gameTime = 20.0f,
                logType = "action",
                action = new RobotAction { actionId = "action-001" },
                scene = new SceneSnapshot { snapshotId = "snap-001" }
            };

            Assert.AreEqual("log-001", entry.logId);
            Assert.AreEqual("action", entry.logType);
            Assert.IsNotNull(entry.action);
            Assert.IsNotNull(entry.scene);
        }

        [Test]
        public void LogEntry_FieldsAreNullByDefault()
        {
            var entry = new LogEntry();

            Assert.IsNull(entry.action);
            Assert.IsNull(entry.scene);
            Assert.IsNull(entry.logId);
        }

        #endregion

        #region Serialization Tests

        [Test]
        public void RobotAction_CanSerializeToJson()
        {
            var action = new RobotAction
            {
                actionId = "test-001",
                actionName = "test_action",
                type = ActionType.Movement,
                status = ActionStatus.Completed,
                success = true
            };

            string json = JsonUtility.ToJson(action);

            Assert.IsNotNull(json);
            Assert.IsTrue(json.Contains("test-001"));
            Assert.IsTrue(json.Contains("test_action"));
        }

        [Test]
        public void RobotAction_CanDeserializeFromJson()
        {
            var original = new RobotAction
            {
                actionId = "test-002",
                actionName = "deserialize_test",
                type = ActionType.Manipulation,
                startPosition = new Vector3(1, 2, 3)
            };

            string json = JsonUtility.ToJson(original);
            var deserialized = JsonUtility.FromJson<RobotAction>(json);

            Assert.AreEqual(original.actionId, deserialized.actionId);
            Assert.AreEqual(original.actionName, deserialized.actionName);
            Assert.AreEqual(original.type, deserialized.type);
            Assert.AreEqual(original.startPosition, deserialized.startPosition);
        }

        [Test]
        public void SceneSnapshot_CanRoundTripJson()
        {
            var original = new SceneSnapshot
            {
                snapshotId = "snap-test",
                gameTime = 30.5f,
                totalObjects = 10
            };

            string json = JsonUtility.ToJson(original);
            var deserialized = JsonUtility.FromJson<SceneSnapshot>(json);

            Assert.AreEqual(original.snapshotId, deserialized.snapshotId);
            Assert.AreEqual(original.gameTime, deserialized.gameTime);
            Assert.AreEqual(original.totalObjects, deserialized.totalObjects);
        }

        #endregion
    }
}
