using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using UnityEngine;

[System.Serializable]
public class RobotAction
{
    public string actionType; // e.g., "move", "grip", "release", "rotate", "joint_state"
    public string robotId; // robot identifier (e.g., "AR4Left", "AR4Right")
    public string objectName; // optional: object being manipulated
    public float[] target; // optional: position [x,y,z] or rotation [x,y,z]
    public float[] jointAngles; // optional: current joint angles for IK debugging
    public float speed; // optional: used for "move" or "rotate"
    public string timestamp; // ISO 8601 timestamp
    public float gameTime; // Unity Time.time for correlation
    public bool success; // optional: if the action succeeded
    public string errorMessage; // optional: error details if success is false
}

public class RobotActionLogger : MonoBehaviour
{
    public static RobotActionLogger Instance { get; private set; }

    [Tooltip("Folder name for the operation performed")]
    public string operationType = "default";

    [Tooltip("Directory where log file will be saved")]
    public string logFilePath = "";

    [Tooltip("Maximum log file size in MB before rotation")]
    public float maxFileSizeMB = 10f;

    [Tooltip("Maximum number of rotated log files to keep")]
    public int maxRotatedFiles = 5;

    [Tooltip("Buffer size for batched writing")]
    public int bufferSize = 100;

    private string _logDirectory;
    private Thread _writerThread;
    private AutoResetEvent _bufferEvent;
    private volatile bool _shouldStop;

    // Per-robot file management
    private readonly ConcurrentDictionary<string, RobotLogData> _robotLogData =
        new ConcurrentDictionary<string, RobotLogData>();

    private class RobotLogData
    {
        public string FilePath { get; set; }
        public Queue<string> LogBuffer { get; set; } = new Queue<string>();
        public long CurrentFileSize { get; set; }
        public readonly object BufferLock = new object();
    }

    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
            InitializeLogger();
        }
        else
        {
            Destroy(gameObject);
        }
    }

    private void InitializeLogger()
    {
        try
        {
            if (string.IsNullOrEmpty(logFilePath))
                logFilePath = Application.persistentDataPath;

            _logDirectory = Path.Combine(logFilePath, "RobotLogs", operationType);

            // Auto-create directory
            if (!Directory.Exists(_logDirectory))
            {
                Directory.CreateDirectory(_logDirectory);
                Debug.Log($"Created log directory: {_logDirectory}");
            }

            _bufferEvent = new AutoResetEvent(false);
            _shouldStop = false;

            // Start background writer thread
            _writerThread = new Thread(BackgroundWriter)
            {
                IsBackground = true,
                Name = "RobotActionLogWriter",
            };
            _writerThread.Start();

            Debug.Log($"Robot action logger initialized. Logs will be saved to: {_logDirectory}");
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to initialize RobotActionLogger: {ex.Message}");
        }
    }

    private void BackgroundWriter()
    {
        while (!_shouldStop)
        {
            try
            {
                _bufferEvent.WaitOne(1000); // Wait for signal or timeout after 1 second

                // Process logs for each robot
                foreach (var robotEntry in _robotLogData)
                {
                    string robotId = robotEntry.Key;
                    RobotLogData logData = robotEntry.Value;
                    List<string> logsToWrite = new List<string>();

                    lock (logData.BufferLock)
                    {
                        while (logData.LogBuffer.Count > 0)
                        {
                            logsToWrite.Add(logData.LogBuffer.Dequeue());
                        }
                    }

                    if (logsToWrite.Count > 0)
                    {
                        WriteLogsToFile(robotId, logsToWrite);
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"Error in background log writer: {ex.Message}");
            }
        }
    }

    private void WriteLogsToFile(string robotId, List<string> logs)
    {
        try
        {
            if (!_robotLogData.TryGetValue(robotId, out RobotLogData logData))
            {
                Debug.LogError($"Robot log data not found for robotId: {robotId}");
                return;
            }

            // Check if file rotation is needed
            if (logData.CurrentFileSize > maxFileSizeMB * 1024 * 1024)
            {
                RotateLogFile(robotId);
            }

            string logDataString = string.Join("\n", logs) + "\n";
            File.AppendAllText(logData.FilePath, logDataString);
            logData.CurrentFileSize += System.Text.Encoding.UTF8.GetByteCount(logDataString);
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to write logs to file for robot {robotId}: {ex.Message}");
        }
    }

    private void RotateLogFile(string robotId)
    {
        try
        {
            if (!_robotLogData.TryGetValue(robotId, out RobotLogData logData))
            {
                Debug.LogError($"Robot log data not found for robotId: {robotId}");
                return;
            }

            string baseFileName = Path.GetFileNameWithoutExtension(logData.FilePath);
            string extension = Path.GetExtension(logData.FilePath);

            // Rotate existing backup files
            for (int i = maxRotatedFiles; i > 1; i--)
            {
                string oldFile = Path.Combine(_logDirectory, $"{baseFileName}.{i - 1}{extension}");
                string newFile = Path.Combine(_logDirectory, $"{baseFileName}.{i}{extension}");

                if (File.Exists(oldFile))
                {
                    if (File.Exists(newFile))
                        File.Delete(newFile);
                    File.Move(oldFile, newFile);
                }
            }

            // Move current file to .1
            if (File.Exists(logData.FilePath))
            {
                string firstBackup = Path.Combine(_logDirectory, $"{baseFileName}.1{extension}");
                if (File.Exists(firstBackup))
                    File.Delete(firstBackup);
                File.Move(logData.FilePath, firstBackup);
            }

            logData.CurrentFileSize = 0;
            Debug.Log($"Rotated log file for {robotId}. New log: {logData.FilePath}");
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to rotate log file for robot {robotId}: {ex.Message}");
        }
    }

    private RobotLogData GetOrCreateRobotLogData(string robotId)
    {
        return _robotLogData.GetOrAdd(
            robotId,
            id =>
            {
                string safeRobotId = string.IsNullOrEmpty(id)
                    ? "Unknown"
                    : id.Replace("/", "_").Replace("\\", "_");
                string fileName = $"{safeRobotId}_actions.json";
                string filePath = Path.Combine(_logDirectory, fileName);

                var logData = new RobotLogData { FilePath = filePath };

                // Get current file size if it exists
                if (File.Exists(filePath))
                {
                    logData.CurrentFileSize = new FileInfo(filePath).Length;
                }
                else
                {
                    logData.CurrentFileSize = 0;
                }

                Debug.Log($"Created log file for robot '{safeRobotId}': {filePath}");
                return logData;
            }
        );
    }

    // Log a standard robot action
    public void LogAction(
        string type,
        string robotId = null,
        string objectName = null,
        Vector3? target = null,
        float speed = 0f,
        bool success = true,
        string errorMessage = null
    )
    {
        LogAction(type, robotId, objectName, target, null, speed, success, errorMessage);
    }

    // Log a robot action with joint angles for IK debugging
    public void LogAction(
        string type,
        string robotId = null,
        string objectName = null,
        Vector3? target = null,
        float[] jointAngles = null,
        float speed = 0f,
        bool success = true,
        string errorMessage = null
    )
    {
        try
        {
            string actualRobotId = robotId ?? "Unknown";
            RobotLogData logData = GetOrCreateRobotLogData(actualRobotId);

            RobotAction action = new RobotAction
            {
                actionType = type,
                robotId = actualRobotId,
                objectName = objectName,
                target = target.HasValue
                    ? new float[] { target.Value.x, target.Value.y, target.Value.z }
                    : null,
                jointAngles = jointAngles,
                speed = speed,
                timestamp = DateTime.Now.ToString("yyyy-MM-ddTHH:mm:ss.fffZ"),
                gameTime = Time.time,
                success = success,
                errorMessage = errorMessage,
            };

            string json = JsonUtility.ToJson(action);

            lock (logData.BufferLock)
            {
                logData.LogBuffer.Enqueue(json);

                // Signal writer thread if buffer is getting full or immediately if it's an error
                if (logData.LogBuffer.Count >= bufferSize || !success)
                {
                    _bufferEvent.Set();
                }
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to queue log action for robot {robotId}: {ex.Message}");
        }
    }

    // Log joint states for IK debugging
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

    // Force flush all buffered logs for all robots
    public void FlushLogs()
    {
        bool hasLogs = false;

        foreach (var robotEntry in _robotLogData)
        {
            RobotLogData logData = robotEntry.Value;
            lock (logData.BufferLock)
            {
                if (logData.LogBuffer.Count > 0)
                {
                    hasLogs = true;
                }
            }
        }

        if (hasLogs)
        {
            _bufferEvent.Set();
            // Wait a bit for the background thread to process
            Thread.Sleep(100);
        }
    }

    // Force flush logs for a specific robot
    public void FlushLogs(string robotId)
    {
        if (_robotLogData.TryGetValue(robotId, out RobotLogData logData))
        {
            lock (logData.BufferLock)
            {
                if (logData.LogBuffer.Count > 0)
                {
                    _bufferEvent.Set();
                    // Wait a bit for the background thread to process
                    Thread.Sleep(100);
                }
            }
        }
    }

    private void OnApplicationPause(bool pauseStatus)
    {
        if (pauseStatus)
        {
            FlushLogs();
        }
    }

    private void OnApplicationFocus(bool hasFocus)
    {
        if (!hasFocus)
        {
            FlushLogs();
        }
    }

    private void OnDestroy()
    {
        if (Instance == this)
        {
            // Graceful shutdown
            _shouldStop = true;
            _bufferEvent?.Set();

            if (_writerThread != null && _writerThread.IsAlive)
            {
                if (!_writerThread.Join(2000)) // Wait up to 2 seconds
                {
                    Debug.LogWarning("Log writer thread did not shutdown gracefully");
                }
            }

            _bufferEvent?.Dispose();
            Instance = null;
        }
    }
}
