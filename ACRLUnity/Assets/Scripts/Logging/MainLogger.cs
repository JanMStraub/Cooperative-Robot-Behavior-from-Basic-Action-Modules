using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

namespace Logging
{
    /// <summary>
    /// Unified logger for LLM training data
    /// </summary>
    public class MainLogger : MonoBehaviour
    {
        public static MainLogger Instance { get; private set; }

        [Header("Configuration")]
        [Tooltip("Enable/disable logging")]
        public bool enableLogging = true;

        [Tooltip("Log file directory (leave empty for default)")]
        public string logDirectory = "";

        [Tooltip("Folder name for the operation performed")]
        public string operationType = "default";

        [Tooltip("Use per-robot log files (true) or single session file (false)")]
        public bool perRobotFiles = true;

        [Header("Environment Tracking")]
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
        private string _fullLogPath;
        private StreamWriter _logWriter;
        private string _sessionId;

        // Per-robot file management
        private readonly Dictionary<string, StreamWriter> _robotFiles = new();

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
            _sessionId = DateTime.Now.ToString("yyyyMMddHHmmss");

            if (string.IsNullOrEmpty(logDirectory))
                logDirectory = Path.Combine(Application.persistentDataPath, "RobotLogs");

            // Add operation type subfolder and timestamp folder
            _fullLogPath = Path.Combine(logDirectory, operationType, _sessionId);
            Directory.CreateDirectory(_fullLogPath);

            if (!perRobotFiles)
            {
                // Single session file
                _logFilePath = Path.Combine(_fullLogPath, $"robot_actions_{_sessionId}.jsonl");
                _logWriter = new StreamWriter(_logFilePath, true);
                _logWriter.AutoFlush = true;
                Debug.Log($"[MAIN_LOGGER] Initialized. Session log: {_logFilePath}");
            }
            else
            {
                Debug.Log($"[MAIN_LOGGER] Initialized. Per-robot logs in: {_fullLogPath}");
            }

            LogSessionStart();
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
                timestamp = DateTime.Now.ToString("dd-MM-yyyy HH:mm:ss.ff"),
                gameTime = Time.time,
                startPosition = startPos ?? Vector3.zero,
                targetPosition = targetPos ?? Vector3.zero,
                description = description ?? actionName,
                humanReadable = GenerateHumanReadable(actionName, type, robotIds, description),
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
                timestamp = DateTime.Now.ToString("dd-MM-yyyy HH:mm:ss.ff"),
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

            return $"[MAIN_LOGGER] Scene with {robots} robot(s), {objects} object(s) ({graspable} graspable)";
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
                timestamp = DateTime.Now.ToString("dd-MM-yyyy HH:mm:ss.ff"),
                gameTime = Time.time,
                logType = "session",
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
            };

            WriteLog(entry);
        }

        private void WriteLog(LogEntry entry)
        {
            if (!enableLogging)
                return;

            try
            {
                string json = JsonUtility.ToJson(entry);

                if (perRobotFiles && entry.action != null && entry.action.robotIds.Length > 0)
                {
                    // Write to per-robot files
                    foreach (string robotId in entry.action.robotIds)
                    {
                        WriteToRobotFile(robotId, json);
                    }
                }
                else if (_logWriter != null)
                {
                    // Write to session file
                    _logWriter.WriteLine(json);
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"[MAIN_LOGGER] Failed to write log: {ex.Message}");
            }
        }

        /// <summary>
        /// Get or create per-robot file writer
        /// </summary>
        private StreamWriter GetOrCreateRobotFile(string robotId)
        {
            if (_robotFiles.TryGetValue(robotId, out StreamWriter writer))
                return writer;

            // Create new robot file
            string safeRobotId = string.IsNullOrEmpty(robotId)
                ? "Unknown"
                : robotId.Replace("/", "_").Replace("\\", "_");

            string fileName = $"{safeRobotId}_actions.json";
            string filePath = Path.Combine(_fullLogPath, fileName);

            writer = new StreamWriter(filePath, true) { AutoFlush = true };
            _robotFiles[robotId] = writer;
            Debug.Log($"[MAIN_LOGGER] Created log file for robot '{safeRobotId}': {filePath}");

            return writer;
        }

        /// <summary>
        /// Write to per-robot file
        /// </summary>
        private void WriteToRobotFile(string robotId, string json)
        {
            StreamWriter writer = GetOrCreateRobotFile(robotId);
            writer.WriteLine(json);
        }

        private void OnDestroy()
        {
            if (Instance == this)
            {
                // Close session writer
                _logWriter?.Close();
                _logWriter?.Dispose();
                _logWriter = null;

                // Close all per-robot writers
                foreach (var writer in _robotFiles.Values)
                {
                    writer?.Close();
                    writer?.Dispose();
                }
                _robotFiles.Clear();

                Instance = null;
                Debug.Log($"[MAIN_LOGGER] Shutdown. Logs saved.");
            }
        }

        private void OnApplicationQuit()
        {
            _logWriter?.Flush();

            foreach (var writer in _robotFiles.Values)
            {
                writer?.Flush();
            }
        }

        /// <summary>
        /// Flush all logs immediately (for compatibility with RobotActionLogger and FileLogger)
        /// </summary>
        public void FlushLogs()
        {
            _logWriter?.Flush();

            foreach (var writer in _robotFiles.Values)
            {
                writer?.Flush();
            }
        }

        /// <summary>
        /// Flush logs for specific robot (for compatibility with RobotActionLogger)
        /// </summary>
        public void FlushLogs(string robotId)
        {
            if (_robotFiles.TryGetValue(robotId, out StreamWriter writer))
            {
                writer?.Flush();
            }
        }
    }
}
