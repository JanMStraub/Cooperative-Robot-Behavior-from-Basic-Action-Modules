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

        [Tooltip("Maximum log file size in MB before rotation")]
        public float maxFileSizeMB = 10f;

        [Tooltip("Maximum number of rotated log files to keep")]
        public int maxRotatedFiles = 5;

        [Header("Console Logging")]
        [Tooltip("Capture Unity console logs (Debug.Log, etc.)")]
        public bool captureUnityLogs = true;

        [Tooltip("Enable simulation state logging")]
        public bool logSimulationState = true;

        [Tooltip("Interval for periodic state logging (seconds)")]
        public float stateLogInterval = 10f;

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
        private StreamWriter _logWriter;
        private string _sessionId;
        private StreamWriter _consoleLogWriter; // Separate console log file

        // Simulation state tracking
        private float _nextStateLogTime;
        private float _startTime;
        private SimulationManager _simulationManager;
        private RobotController[] _robotControllers;

        // Per-robot file management
        private readonly Dictionary<string, RobotFileData> _robotFiles = new();

        private class RobotFileData
        {
            public string FilePath { get; set; }
            public StreamWriter Writer { get; set; }
            public long CurrentFileSize { get; set; }
        }

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
            _startTime = Time.time;

            if (string.IsNullOrEmpty(logDirectory))
                logDirectory = Path.Combine(Application.persistentDataPath, "RobotLogs");

            // Add operation type subfolder
            string fullLogPath = Path.Combine(logDirectory, operationType);
            Directory.CreateDirectory(fullLogPath);

            if (!perRobotFiles)
            {
                // Single session file
                _logFilePath = Path.Combine(fullLogPath, $"robot_actions_{_sessionId}.jsonl");
                _logWriter = new StreamWriter(_logFilePath, true);
                _logWriter.AutoFlush = true;
                Debug.Log($"MainLogger initialized. Session log: {_logFilePath}");
            }
            else
            {
                Debug.Log($"MainLogger initialized. Per-robot logs in: {fullLogPath}");
            }

            // Create console log file if enabled
            if (captureUnityLogs)
            {
                string consoleLogPath = Path.Combine(fullLogPath, $"console_{_sessionId}.log");
                _consoleLogWriter = new StreamWriter(consoleLogPath, true);
                _consoleLogWriter.AutoFlush = true;
                Application.logMessageReceived += HandleUnityLog;
                Debug.Log($"Console logging enabled: {consoleLogPath}");
            }

            LogSessionStart();
            LogSystemInformation();
        }

        private void Start()
        {
            // Get references to other managers
            _simulationManager = SimulationManager.Instance;
            _robotControllers = FindObjectsByType<RobotController>(
                FindObjectsInactive.Exclude,
                FindObjectsSortMode.None
            );

            _nextStateLogTime = Time.time + stateLogInterval;
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

            // Log simulation state periodically
            if (logSimulationState && Time.time >= _nextStateLogTime)
            {
                LogCurrentSimulationState();
                _nextStateLogTime = Time.time + stateLogInterval;
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

        /// <summary>
        /// Log joint states for IK debugging (for compatibility with RobotActionLogger)
        /// </summary>
        public void LogJointState(
            string robotId,
            float[] jointAngles,
            Vector3? targetPosition = null,
            bool success = true,
            string errorMessage = null
        )
        {
            LogAction(
                "joint_state",
                robotId,
                null,
                targetPosition,
                jointAngles,
                0f,
                success,
                errorMessage
            );
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
                Debug.LogError($"Failed to write log: {ex.Message}");
            }
        }

        /// <summary>
        /// Get or create per-robot file writer
        /// </summary>
        private RobotFileData GetOrCreateRobotFile(string robotId)
        {
            if (_robotFiles.TryGetValue(robotId, out RobotFileData fileData))
                return fileData;

            // Create new robot file
            string safeRobotId = string.IsNullOrEmpty(robotId)
                ? "Unknown"
                : robotId.Replace("/", "_").Replace("\\", "_");

            string fullLogPath = Path.Combine(logDirectory, operationType);
            string fileName = $"{safeRobotId}_actions.json";
            string filePath = Path.Combine(fullLogPath, fileName);

            fileData = new RobotFileData
            {
                FilePath = filePath,
                Writer = new StreamWriter(filePath, true) { AutoFlush = true },
                CurrentFileSize = File.Exists(filePath) ? new FileInfo(filePath).Length : 0,
            };

            _robotFiles[robotId] = fileData;
            Debug.Log($"Created log file for robot '{safeRobotId}': {filePath}");

            return fileData;
        }

        /// <summary>
        /// Write to per-robot file with rotation
        /// </summary>
        private void WriteToRobotFile(string robotId, string json)
        {
            RobotFileData fileData = GetOrCreateRobotFile(robotId);

            // Check if rotation needed
            if (fileData.CurrentFileSize > maxFileSizeMB * 1024 * 1024)
            {
                RotateLogFile(robotId, fileData);
            }

            fileData.Writer.WriteLine(json);
            fileData.CurrentFileSize += System.Text.Encoding.UTF8.GetByteCount(json) + 1; // +1 for newline
        }

        /// <summary>
        /// Rotate log file when size limit reached
        /// </summary>
        private void RotateLogFile(string robotId, RobotFileData fileData)
        {
            try
            {
                string baseFileName = Path.GetFileNameWithoutExtension(fileData.FilePath);
                string extension = Path.GetExtension(fileData.FilePath);
                string directory = Path.GetDirectoryName(fileData.FilePath);

                // Close current writer
                fileData.Writer.Close();
                fileData.Writer.Dispose();

                // Rotate existing backup files
                for (int i = maxRotatedFiles; i > 1; i--)
                {
                    string oldFile = Path.Combine(directory, $"{baseFileName}.{i - 1}{extension}");
                    string newFile = Path.Combine(directory, $"{baseFileName}.{i}{extension}");

                    if (File.Exists(oldFile))
                    {
                        if (File.Exists(newFile))
                            File.Delete(newFile);
                        File.Move(oldFile, newFile);
                    }
                }

                // Move current file to .1
                if (File.Exists(fileData.FilePath))
                {
                    string firstBackup = Path.Combine(directory, $"{baseFileName}.1{extension}");
                    if (File.Exists(firstBackup))
                        File.Delete(firstBackup);
                    File.Move(fileData.FilePath, firstBackup);
                }

                // Create new writer
                fileData.Writer = new StreamWriter(fileData.FilePath, true) { AutoFlush = true };
                fileData.CurrentFileSize = 0;

                Debug.Log($"Rotated log file for {robotId}. New log: {fileData.FilePath}");
            }
            catch (Exception ex)
            {
                Debug.LogError($"Failed to rotate log file for robot {robotId}: {ex.Message}");
                // Recreate writer on error
                fileData.Writer = new StreamWriter(fileData.FilePath, true) { AutoFlush = true };
            }
        }

        private void OnDestroy()
        {
            if (Instance == this)
            {
                LogSimulationEvent("session_end", "Simulation session ended", false);

                // Close session writer
                _logWriter?.Close();
                _logWriter?.Dispose();
                _logWriter = null;

                // Close console writer
                _consoleLogWriter?.Close();
                _consoleLogWriter?.Dispose();
                _consoleLogWriter = null;

                // Close all per-robot writers
                foreach (var fileData in _robotFiles.Values)
                {
                    fileData.Writer?.Close();
                    fileData.Writer?.Dispose();
                }
                _robotFiles.Clear();

                Instance = null;
                Debug.Log($"MainLogger shutdown. Logs saved.");
            }
        }

        private void OnApplicationQuit()
        {
            _logWriter?.Flush();
            _consoleLogWriter?.Flush();

            foreach (var fileData in _robotFiles.Values)
            {
                fileData.Writer?.Flush();
            }
        }

        /// <summary>
        /// Flush all logs immediately (for compatibility with RobotActionLogger and FileLogger)
        /// </summary>
        public void FlushLogs()
        {
            _logWriter?.Flush();
            _consoleLogWriter?.Flush();

            foreach (var fileData in _robotFiles.Values)
            {
                fileData.Writer?.Flush();
            }
        }

        /// <summary>
        /// Flush logs for specific robot (for compatibility with RobotActionLogger)
        /// </summary>
        public void FlushLogs(string robotId)
        {
            if (_robotFiles.TryGetValue(robotId, out RobotFileData fileData))
            {
                fileData.Writer?.Flush();
            }
        }

        // ==================== CONSOLE LOGGING (from FileLogger) ====================

        /// <summary>
        /// Handle Unity console log messages
        /// </summary>
        private void HandleUnityLog(string logString, string stackTrace, LogType type)
        {
            if (_consoleLogWriter == null)
                return;

            try
            {
                string logEntry = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff} [{type}] {logString}";
                if (type == LogType.Exception || type == LogType.Error)
                {
                    logEntry += $"\nStack Trace: {stackTrace}";
                }

                _consoleLogWriter.WriteLine(logEntry);
            }
            catch (Exception ex)
            {
                Debug.LogError($"Failed to write Unity log: {ex.Message}");
            }
        }

        /// <summary>
        /// Log simulation event (for compatibility with FileLogger)
        /// </summary>
        public void LogSimulationEvent(string eventType, string details, bool isActive = true)
        {
            if (!enableLogging)
                return;

            try
            {
                var state = new
                {
                    timestamp = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ"),
                    gameTime = Time.time,
                    eventType,
                    simulationActive = isActive,
                    frameCount = Time.frameCount,
                    frameRate = 1f / Time.deltaTime,
                    robotCount = _robotControllers?.Length ?? 0,
                    activeRobots = GetActiveRobotIds(),
                    memoryUsageMB = System.GC.GetTotalMemory(false) / (1024f * 1024f),
                    unityVersion = Application.unityVersion,
                    details,
                };

                string json = JsonUtility.ToJson(state);
                _consoleLogWriter?.WriteLine($"[SIM] {json}");
            }
            catch (Exception ex)
            {
                Debug.LogError($"Failed to log simulation event: {ex.Message}");
            }
        }

        /// <summary>
        /// Log current simulation state
        /// </summary>
        private void LogCurrentSimulationState()
        {
            if (_simulationManager != null)
            {
                string details = $"StopRobot={_simulationManager.ShouldStopRobots}";
                LogSimulationEvent("periodic_state", details, !_simulationManager.ShouldStopRobots);
            }
        }

        /// <summary>
        /// Log system information at startup
        /// </summary>
        private void LogSystemInformation()
        {
            try
            {
                string sysInfo = $"System Information - ";
                sysInfo += $"Unity: {Application.unityVersion}, ";
                sysInfo += $"Platform: {Application.platform}, ";
                sysInfo += $"Device: {SystemInfo.deviceModel}, ";
                sysInfo += $"OS: {SystemInfo.operatingSystem}, ";
                sysInfo += $"CPU: {SystemInfo.processorType} ({SystemInfo.processorCount} cores), ";
                sysInfo += $"Memory: {SystemInfo.systemMemorySize}MB, ";
                sysInfo += $"GPU: {SystemInfo.graphicsDeviceName}";

                LogSimulationEvent("system_info", sysInfo);
            }
            catch (Exception ex)
            {
                Debug.LogError($"Failed to log system information: {ex.Message}");
            }
        }

        /// <summary>
        /// Get active robot IDs
        /// </summary>
        private string[] GetActiveRobotIds()
        {
            if (_robotControllers == null)
                return new string[0];

            List<string> activeIds = new List<string>();
            foreach (var controller in _robotControllers)
            {
                if (controller != null && controller.gameObject.activeInHierarchy)
                {
                    activeIds.Add(controller.robotId);
                }
            }
            return activeIds.ToArray();
        }

        private void OnDisable()
        {
            if (captureUnityLogs)
            {
                Application.logMessageReceived -= HandleUnityLog;
            }
        }

        private void OnApplicationPause(bool pauseStatus)
        {
            LogSimulationEvent(
                pauseStatus ? "app_pause" : "app_resume",
                $"Application paused: {pauseStatus}",
                !pauseStatus
            );
            if (pauseStatus)
                FlushLogs();
        }

        private void OnApplicationFocus(bool hasFocus)
        {
            LogSimulationEvent(
                hasFocus ? "app_focus" : "app_unfocus",
                $"Application focus: {hasFocus}",
                hasFocus
            );
            if (!hasFocus)
                FlushLogs();
        }
    }
}
