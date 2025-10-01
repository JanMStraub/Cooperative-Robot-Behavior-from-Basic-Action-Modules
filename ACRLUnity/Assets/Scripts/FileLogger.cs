using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using UnityEngine;

[System.Serializable]
public class SimulationLoggingState
{
    public string timestamp;
    public float gameTime;
    public string eventType;
    public bool simulationActive;
    public int frameCount;
    public float frameRate;
    public int robotCount;
    public string[] activeRobots;
    public float memoryUsageMB;
    public string unityVersion;
    public string details;
}

public class FileLogger : MonoBehaviour
{
    public static FileLogger Instance { get; private set; }

    [Header("Log Configuration")]
    [Tooltip("Base directory for all simulation logs")]
    public string logFilePath;

    [Tooltip("Operation type for organizing logs")]
    public string operationType = "simulation";

    [Tooltip("Enable Unity console log capture")]
    public bool captureUnityLogs = true;

    [Tooltip("Enable simulation state logging")]
    public bool logSimulationState = true;

    [Tooltip("Interval for periodic state logging (seconds)")]
    public float stateLogInterval = 10f;

    private string _logDirectory;
    private string _logFile;
    private StreamWriter _logWriter;

    private SimulationManager _simulationManager;
    private RobotActionLogger _robotActionLogger;
    private RobotController[] _robotControllers;

    private float _nextStateLogTime;
    private float _startTime;

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

            _logDirectory = Path.Combine(logFilePath, "SimulationLogs", operationType);

            if (!Directory.Exists(_logDirectory))
            {
                Directory.CreateDirectory(_logDirectory);
                Debug.Log($"Created simulation log directory: {_logDirectory}");
            }

            string sessionId = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            _logFile = Path.Combine(_logDirectory, $"simulation_{sessionId}.log");
            _logWriter = new StreamWriter(_logFile, true);
            _startTime = Time.time;

            Debug.Log($"Simulation logger initialized. Logs: {_logDirectory}");

            // Log session start
            LogSimulationEvent("session_start", "Simulation session started", true);

            // Log system information
            LogSystemInformation();
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to initialize FileLogger: {ex.Message}");
        }
    }

    private void Start()
    {
        // Get references to other managers
        _simulationManager = SimulationManager.Instance;
        _robotActionLogger = RobotActionLogger.Instance;
        _robotControllers = FindObjectsByType<RobotController>(FindObjectsInactive.Exclude, FindObjectsSortMode.None);

        if (captureUnityLogs)
        {
            Application.logMessageReceived += HandleUnityLog;
        }

        _nextStateLogTime = Time.time + stateLogInterval;
    }

    private void Update()
    {
        if (logSimulationState && Time.time >= _nextStateLogTime)
        {
            LogCurrentSimulationState();
            _nextStateLogTime = Time.time + stateLogInterval;
        }
    }

    private void HandleUnityLog(string logString, string stackTrace, LogType type)
    {
        try
        {
            string logEntry = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff} [{type}] {logString}";
            if (type == LogType.Exception || type == LogType.Error)
            {
                logEntry += $"\nStack Trace: {stackTrace}";
            }

            _logWriter?.WriteLine(logEntry);
            _logWriter?.Flush();
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to write Unity log: {ex.Message}");
        }
    }

    public void LogSimulationEvent(string eventType, string details, bool isActive = true)
    {
        try
        {
            SimulationLoggingState state = new SimulationLoggingState
            {
                timestamp = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ"),
                gameTime = Time.time,
                eventType = eventType,
                simulationActive = isActive,
                frameCount = Time.frameCount,
                frameRate = 1f / Time.deltaTime,
                robotCount = _robotControllers?.Length ?? 0,
                activeRobots = GetActiveRobotIds(),
                memoryUsageMB = GC.GetTotalMemory(false) / (1024f * 1024f),
                unityVersion = Application.unityVersion,
                details = details,
            };

            string json = JsonUtility.ToJson(state);
            try
            {
                _logWriter?.WriteLine($"[SIM] {json}");
                _logWriter?.Flush();
            }
            catch (Exception ex)
            {
                Debug.LogError($"Failed to write simulation event: {ex.Message}");
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to log simulation event: {ex.Message}");
        }
    }

    private void LogCurrentSimulationState()
    {
        if (_simulationManager != null)
        {
            string details = $"StopRobot={_simulationManager.ShouldStopRobots}";
            LogSimulationEvent("periodic_state", details, !_simulationManager.ShouldStopRobots);
        }
    }

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

    private string[] GetActiveRobotIds()
    {
        if (_robotControllers == null)
            return new string[0];

        List<string> activeIds = new List<string>();
        foreach (var controller in _robotControllers)
        {
            if (controller != null && controller.gameObject.activeInHierarchy)
            {
                activeIds.Add(controller.gameObject.name);
            }
        }
        return activeIds.ToArray();
    }

    public void FlushLogs()
    {
        try
        {
            _logWriter?.Flush();
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to flush logs: {ex.Message}");
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

    private void OnDisable()
    {
        if (captureUnityLogs)
        {
            Application.logMessageReceived -= HandleUnityLog;
        }
    }

    private void OnDestroy()
    {
        if (Instance == this)
        {
            LogSimulationEvent("session_end", "Simulation session ended", false);
            FlushLogs();
            _logWriter?.Close();
            _logWriter?.Dispose();
            Instance = null;
        }
    }
}
