using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

namespace Logging
{
    /// <summary>
    ///  unified robot logger for LLM training data
    /// Replaces: EnhancedRobotActionLogger + TaskLogger + OperationLogger +
    ///           EnvironmentTracker + CoordinationLogger
    /// </summary>
    public class RobotLogger : MonoBehaviour
    {
        public static RobotLogger Instance { get; private set; }

        [Header("Configuration")]
        [Tooltip("Enable/disable logging")]
        public bool enableLogging = true;

        [Tooltip("Log file directory (leave empty for default)")]
        public string logDirectory = "";

        [Tooltip("Capture environment snapshots")]
        public bool captureEnvironment = true;

        [Tooltip("Environment sampling rate (seconds)")]
        public float environmentSampleRate = 2f;

        [Tooltip("Track robot trajectories")]
        public bool trackTrajectories = true;

        [Tooltip("Trajectory point sampling rate (seconds)")]
        public float trajectorySampleRate = 0.2f;

        // Active tracking
        private readonly Dictionary<string, RobotAction> _activeActions = new();
        private readonly Dictionary<string, List<Vector3>> _trajectories = new();
        private readonly Dictionary<string, float> _actionStartTimes = new();

        // Environment tracking
        private float _lastEnvironmentCapture;
        private readonly Dictionary<string, Object> _trackedObjects = new();

        // File management
        private string _logFilePath;
        private StreamWriter _logWriter;
        private string _sessionId;

        private void Awake()
        {
            if (Instance == null)
            {
                Instance = this;
                DontDestroyOnLoad(gameObject);
                Initialize();
            }
            else
            {
                Destroy(gameObject);
            }
        }

        private void Initialize()
        {
            _sessionId = DateTime.Now.ToString("yyyyMMdd_HHmmss");

            if (string.IsNullOrEmpty(logDirectory))
                logDirectory = Path.Combine(Application.persistentDataPath, "RobotLogs");

            Directory.CreateDirectory(logDirectory);
            _logFilePath = Path.Combine(logDirectory, $"robot_actions_{_sessionId}.jsonl");

            _logWriter = new StreamWriter(_logFilePath, true);
            _logWriter.AutoFlush = true;

            LogSessionStart();
            Debug.Log($"RobotLogger initialized. Logs: {_logFilePath}");
        }

        private void Update()
        {
            if (!enableLogging)
                return;

            // Update trajectories for active actions
            if (trackTrajectories)
            {
                UpdateTrajectories();
            }

            // Capture environment periodically
            if (captureEnvironment && Time.time - _lastEnvironmentCapture >= environmentSampleRate)
            {
                CaptureEnvironment();
                _lastEnvironmentCapture = Time.time;
            }
        }

        // ==================== PUBLIC API ====================

        /// <summary>
        /// Start a new action (task, movement, manipulation, etc.)
        /// </summary>
        public string StartAction(
            string actionName,
            ActionType type,
            string[] robotIds,
            Vector3? startPos = null,
            Vector3? targetPos = null,
            string[] objectIds = null,
            string description = null
        )
        {
            if (!enableLogging)
                return "";

            string actionId = GenerateActionId(actionName, type);

            var action = new RobotAction
            {
                actionId = actionId,
                actionName = actionName,
                type = type,
                status = ActionStatus.Started,
                robotIds = robotIds ?? new string[0],
                objectIds = objectIds ?? new string[0],
                timestamp = DateTime.Now.ToString("yyyy-MM-ddTHH:mm:ss.fffZ"),
                gameTime = Time.time,
                startPosition = startPos ?? Vector3.zero,
                targetPosition = targetPos ?? Vector3.zero,
                description = description ?? actionName,
                humanReadable = GenerateHumanReadable(actionName, type, robotIds, description),
                capabilities = DetermineCapabilities(type, actionName),
                complexityLevel = CalculateComplexity(type, robotIds?.Length ?? 0),
            };

            _activeActions[actionId] = action;
            _actionStartTimes[actionId] = Time.time;

            if (trackTrajectories && startPos.HasValue)
            {
                _trajectories[actionId] = new List<Vector3> { startPos.Value };
            }

            LogAction(action);
            return actionId;
        }

        /// <summary>
        /// Complete an action with outcome
        /// </summary>
        public void CompleteAction(
            string actionId,
            bool success,
            float qualityScore = 0f,
            string errorMessage = null,
            Dictionary<string, float> metrics = null
        )
        {
            if (!enableLogging || !_activeActions.TryGetValue(actionId, out RobotAction action))
                return;

            action.status = success ? ActionStatus.Completed : ActionStatus.Failed;
            action.success = success;
            action.qualityScore = qualityScore;
            action.errorMessage = errorMessage;
            action.duration = Time.time - _actionStartTimes.GetValueOrDefault(actionId, Time.time);
            action.metrics = metrics ?? new Dictionary<string, float>();

            // Add trajectory if tracked
            if (_trajectories.TryGetValue(actionId, out List<Vector3> trajectory))
            {
                action.trajectoryPoints = trajectory.ToArray();
                _trajectories.Remove(actionId);
            }

            // Update human-readable with outcome
            action.humanReadable = UpdateHumanReadableWithOutcome(action);

            LogAction(action);

            _activeActions.Remove(actionId);
            _actionStartTimes.Remove(actionId);
        }

        /// <summary>
        /// Log a multi-robot coordination event
        /// </summary>
        public string LogCoordination(
            string coordinationName,
            string[] robotIds,
            string description = null,
            string[] objectIds = null
        )
        {
            return StartAction(
                coordinationName,
                ActionType.Coordination,
                robotIds,
                objectIds: objectIds,
                description: description
            );
        }

        /// <summary>
        /// Capture current environment state
        /// </summary>
        public void CaptureEnvironment(string snapshotId = null)
        {
            if (!enableLogging || !captureEnvironment)
                return;

            var snapshot = new SceneSnapshot
            {
                snapshotId = snapshotId ?? $"env_{Time.time:F2}",
                timestamp = DateTime.Now.ToString("yyyy-MM-ddTHH:mm:ss.fffZ"),
                gameTime = Time.time,
                objects = GetSceneObjects(),
                robots = GetRobotStates(),
                sceneDescription = GenerateSceneDescription(),
            };

            snapshot.totalObjects = snapshot.objects.Length;
            snapshot.graspableObjects = snapshot.objects.Count(o => o.isGraspable);

            LogScene(snapshot);
        }

        /// <summary>
        /// Register an object for tracking
        /// </summary>
        public void RegisterObject(
            GameObject obj,
            string objectType = null,
            bool isGraspable = true
        )
        {
            if (!enableLogging)
                return;

            var rb = obj.GetComponent<Rigidbody>();
            var simpleObj = new Object
            {
                id = obj.name,
                name = obj.name,
                type = objectType ?? DetermineObjectType(obj),
                position = obj.transform.position,
                rotation = obj.transform.rotation,
                isGraspable = isGraspable,
                isMovable = rb != null,
                mass = rb != null ? rb.mass : 0f,
            };

            _trackedObjects[obj.name] = simpleObj;
        }

        /// <summary>
        /// Backward compatibility with RobotActionLogger
        /// </summary>
        public void LogAction(
            string type,
            string robotId,
            string objectName = null,
            Vector3? target = null,
            float[] jointAngles = null,
            float speed = 0f,
            bool success = true,
            string errorMessage = null
        )
        {
            // Map old API to new system
            ActionType actionType = type.ToLower() switch
            {
                "move" or "settarget" => ActionType.Movement,
                "grip" or "grasp" or "release" => ActionType.Manipulation,
                "observe" or "scan" => ActionType.Observation,
                _ => ActionType.Task,
            };

            string actionId = StartAction(
                type,
                actionType,
                new[] { robotId },
                targetPos: target,
                objectIds: objectName != null ? new[] { objectName } : null
            );

            var metrics = new Dictionary<string, float>();
            if (speed > 0)
                metrics["speed"] = speed;
            if (jointAngles != null)
                metrics["joint_count"] = jointAngles.Length;

            CompleteAction(actionId, success, success ? 0.8f : 0f, errorMessage, metrics);
        }

        // ==================== INTERNAL METHODS ====================

        private void UpdateTrajectories()
        {
            var robotManager = RobotManager.Instance;
            if (robotManager == null)
                return;

            foreach (
                var kvp in _activeActions.Where(a =>
                    a.Value.status == ActionStatus.Started
                    || a.Value.status == ActionStatus.InProgress
                )
            )
            {
                string actionId = kvp.Key;
                var action = kvp.Value;

                if (!_trajectories.ContainsKey(actionId))
                    _trajectories[actionId] = new List<Vector3>();

                // Get robot position
                foreach (string robotId in action.robotIds)
                {
                    if (
                        robotManager.RobotInstances.TryGetValue(robotId, out var robotInstance)
                        && robotInstance.controller != null
                    )
                    {
                        var currentPos = robotInstance.controller.endEffectorBase.position;
                        var trajectory = _trajectories[actionId];

                        // Sample at specified rate
                        if (
                            trajectory.Count == 0
                            || Time.time - _actionStartTimes[actionId]
                                >= trajectory.Count * trajectorySampleRate
                        )
                        {
                            trajectory.Add(currentPos);
                        }
                    }
                }
            }
        }

        private Object[] GetSceneObjects()
        {
            return _trackedObjects.Values.ToArray();
        }

        private RobotState[] GetRobotStates()
        {
            var robotManager = RobotManager.Instance;
            if (robotManager == null)
                return new RobotState[0];

            var states = new List<RobotState>();

            foreach (var robotInstance in robotManager.RobotInstances.Values)
            {
                if (robotInstance.controller == null)
                    continue;

                var controller = robotInstance.controller;
                var state = new RobotState
                {
                    robotId = robotInstance.robotId,
                    position = controller.endEffectorBase.position,
                    rotation = controller.endEffectorBase.rotation,
                    jointAngles =
                        controller.robotJoints?.Select(j => j.jointPosition[0]).ToArray()
                        ?? new float[0],
                    targetPosition =
                        robotInstance.targetGameObject?.transform.position ?? Vector3.zero,
                    distanceToTarget = controller.GetDistanceToTarget(),
                    isMoving = controller.GetDistanceToTarget() > 0.1f,
                    currentAction =
                        _activeActions
                            .Values.FirstOrDefault(a => a.robotIds.Contains(robotInstance.robotId))
                            ?.actionName ?? "idle",
                };

                states.Add(state);
            }

            return states.ToArray();
        }

        private string GenerateSceneDescription()
        {
            int objects = _trackedObjects.Count;
            int graspable = _trackedObjects.Values.Count(o => o.isGraspable);
            int robots = RobotManager.Instance?.RobotInstances.Count ?? 0;

            return $"Scene with {robots} robot(s), {objects} object(s) ({graspable} graspable)";
        }

        private string GenerateHumanReadable(
            string actionName,
            ActionType type,
            string[] robotIds,
            string description
        )
        {
            string robotList =
                robotIds != null && robotIds.Length > 0 ? string.Join(" and ", robotIds) : "robot";

            string action = type switch
            {
                ActionType.Movement => $"{robotList} moving",
                ActionType.Manipulation => $"{robotList} manipulating object",
                ActionType.Coordination => $"{string.Join(", ", robotIds)} coordinating",
                ActionType.Observation => $"{robotList} observing",
                _ => $"{robotList} performing {actionName}",
            };

            return description != null ? $"{action}: {description}" : action;
        }

        private string UpdateHumanReadableWithOutcome(RobotAction action)
        {
            string outcome = action.success ? "successfully" : "unsuccessfully";
            return $"{action.humanReadable} ({outcome} completed in {action.duration:F1}s, quality: {action.qualityScore:F2})";
        }

        private string[] DetermineCapabilities(ActionType type, string actionName)
        {
            var caps = new List<string>();

            switch (type)
            {
                case ActionType.Movement:
                    caps.Add("movement");
                    break;
                case ActionType.Manipulation:
                    caps.AddRange(new[] { "manipulation", "grasping" });
                    break;
                case ActionType.Coordination:
                    caps.AddRange(new[] { "coordination", "communication" });
                    break;
                case ActionType.Observation:
                    caps.Add("perception");
                    break;
            }

            return caps.ToArray();
        }

        private int CalculateComplexity(ActionType type, int robotCount)
        {
            int complexity = 1;

            if (type == ActionType.Coordination)
                complexity += 2;
            if (type == ActionType.Manipulation)
                complexity += 1;
            if (robotCount > 1)
                complexity += robotCount;

            return Mathf.Clamp(complexity, 1, 4);
        }

        private string DetermineObjectType(GameObject obj)
        {
            string name = obj.name.ToLower();
            if (name.Contains("cube"))
                return "cube";
            if (name.Contains("sphere"))
                return "sphere";
            if (name.Contains("target"))
                return "target";
            return "object";
        }

        private string GenerateActionId(string actionName, ActionType type)
        {
            string cleanName = actionName.Replace(" ", "_").ToLower();
            return $"{type}_{cleanName}_{Time.time:F2}_{UnityEngine.Random.Range(100, 999)}";
        }

        private void LogSessionStart()
        {
            var entry = new LogEntry
            {
                logId = "session_start",
                timestamp = DateTime.Now.ToString("yyyy-MM-ddTHH:mm:ss.fffZ"),
                gameTime = Time.time,
                logType = "session",
                trainingPrompt = "Robot simulation session started",
                trainingResponse = $"Session {_sessionId} initialized with logging enabled",
                difficultyLevel = "simple",
            };

            WriteLog(entry);
        }

        private void LogAction(RobotAction action)
        {
            var entry = new LogEntry
            {
                logId = action.actionId,
                timestamp = action.timestamp,
                gameTime = action.gameTime,
                logType = "action",
                action = action,
                trainingPrompt = GenerateTrainingPrompt(action),
                trainingResponse = action.humanReadable,
                learningPoints = GenerateLearningPoints(action),
                difficultyLevel = action.complexityLevel switch
                {
                    1 => "simple",
                    2 => "moderate",
                    3 => "complex",
                    _ => "expert",
                },
            };

            WriteLog(entry);
        }

        private void LogScene(SceneSnapshot scene)
        {
            var entry = new LogEntry
            {
                logId = scene.snapshotId,
                timestamp = scene.timestamp,
                gameTime = scene.gameTime,
                logType = "scene",
                scene = scene,
                trainingPrompt = "Describe the current scene",
                trainingResponse = scene.sceneDescription,
                difficultyLevel = "simple",
            };

            WriteLog(entry);
        }

        private string GenerateTrainingPrompt(RobotAction action)
        {
            string robots = string.Join(", ", action.robotIds);
            string objects =
                action.objectIds.Length > 0
                    ? $" involving {string.Join(", ", action.objectIds)}"
                    : "";

            return $"Task: {action.actionName} with robot(s) {robots}{objects}. "
                + $"Required capabilities: {string.Join(", ", action.capabilities)}. "
                + $"What should happen?";
        }

        private string[] GenerateLearningPoints(RobotAction action)
        {
            var points = new List<string>();

            if (!action.success)
            {
                points.Add($"Action failed: {action.errorMessage ?? "unknown reason"}");
            }

            if (action.robotIds.Length > 1)
            {
                points.Add("Multi-robot coordination demonstrated");
            }

            if (action.type == ActionType.Manipulation)
            {
                points.Add("Object manipulation skills used");
            }

            if (action.qualityScore < 0.5f && action.success)
            {
                points.Add("Action succeeded but with low quality - improvement possible");
            }

            return points.ToArray();
        }

        private void WriteLog(LogEntry entry)
        {
            if (_logWriter == null)
                return;

            try
            {
                string json = JsonUtility.ToJson(entry);
                _logWriter.WriteLine(json);
            }
            catch (Exception ex)
            {
                Debug.LogError($"Failed to write log: {ex.Message}");
            }
        }

        private void OnDestroy()
        {
            if (Instance == this)
            {
                _logWriter?.Close();
                _logWriter?.Dispose();
                _logWriter = null;
                Instance = null;

                Debug.Log($"RobotLogger shutdown. Logs saved to: {_logFilePath}");
            }
        }

        private void OnApplicationQuit()
        {
            _logWriter?.Flush();
        }
    }
}
