using System;
using System.Collections.Generic;
using UnityEngine;
using Logging;
#if UNITY_EDITOR
using UnityEditor;
#endif

/// <summary>
/// Configuration for a Python server process
/// </summary>
[System.Serializable]
public class PythonServerConfig
{
    [Tooltip("Unique identifier for this server (used for tracking)")]
    public string serverName = "MyServer";

    [Tooltip("Relative path to Python script from project root (e.g., ACRLPython/LLMcommunication/RunAnalyzer.py)")]
    public string scriptPath = "";

    [Tooltip("Command-line arguments to pass to the script")]
    [TextArea(2, 4)]
    public string arguments = "";

    [Tooltip("Start this server automatically when Unity starts")]
    public bool autoStart = true;

    [Tooltip("Enable/disable this server")]
    public bool enabled = true;

    [Tooltip("Description of what this server does")]
    [TextArea(1, 3)]
    public string description = "";

    // Runtime state (not serialized)
    [System.NonSerialized]
    public int processId = -1;

    [System.NonSerialized]
    public bool isRunning = false;
}

#if UNITY_EDITOR
[CustomEditor(typeof(PythonServerManager))]
public class PythonServerManagerEditor : Editor
{
    /// <summary>
    /// Custom inspector GUI for PythonServerManager
    /// </summary>
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        PythonServerManager manager = (PythonServerManager)target;

        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Server Controls", EditorStyles.boldLabel);

        // Global controls
        EditorGUILayout.BeginHorizontal();
        if (GUILayout.Button("Start All Servers"))
        {
            manager.StartAllServers();
        }
        if (GUILayout.Button("Stop All Servers"))
        {
            manager.StopAllServers();
        }
        EditorGUILayout.EndHorizontal();

        EditorGUILayout.Space();

        // Individual server controls
        if (manager.servers != null && manager.servers.Count > 0)
        {
            EditorGUILayout.LabelField("Individual Server Controls", EditorStyles.boldLabel);

            foreach (var server in manager.servers)
            {
                if (server == null || !server.enabled)
                    continue;

                EditorGUILayout.BeginHorizontal();

                // Status indicator
                string statusIcon = server.isRunning ? "✓" : "○";
                Color statusColor = server.isRunning ? Color.green : Color.gray;
                GUI.color = statusColor;
                EditorGUILayout.LabelField(statusIcon, GUILayout.Width(20));
                GUI.color = Color.white;

                // Server name and info
                string info = $"{server.serverName}";
                if (server.isRunning && manager.GetServerProcessInfo(server.serverName, out string scriptPath, out float elapsed))
                {
                    info += $" (running {elapsed:F1}s)";
                }
                EditorGUILayout.LabelField(info, GUILayout.ExpandWidth(true));

                // Control buttons
                GUI.enabled = !server.isRunning && Application.isPlaying;
                if (GUILayout.Button("Start", GUILayout.Width(60)))
                {
                    manager.StartServer(server.serverName);
                }
                GUI.enabled = server.isRunning && Application.isPlaying;
                if (GUILayout.Button("Stop", GUILayout.Width(60)))
                {
                    manager.StopServer(server.serverName);
                }
                GUI.enabled = true;

                EditorGUILayout.EndHorizontal();

                // Show description if available
                if (!string.IsNullOrEmpty(server.description))
                {
                    EditorGUILayout.LabelField($"  → {server.description}", EditorStyles.miniLabel);
                }
            }
        }

        // Repaint while playing to update elapsed time
        if (Application.isPlaying)
        {
            Repaint();
        }
    }
}
#endif

/// <summary>
/// Manages automatic startup and lifecycle of Python server processes.
/// Uses PythonCaller to launch and monitor background Python scripts.
/// Designed for easy extension - add new servers via Inspector.
/// </summary>
public class PythonServerManager : MonoBehaviour
{
    public static PythonServerManager Instance { get; private set; }

    [Header("Server Configuration")]
    [Tooltip("List of Python servers to manage. Add new entries to register additional scripts.")]
    public List<PythonServerConfig> servers = new List<PythonServerConfig>();

    [Header("Settings")]
    [Tooltip("Wait this many seconds after Unity starts before launching servers")]
    [SerializeField]
    private float _startupDelay = 1.0f;

    // Runtime state
    private PythonCaller _pythonCaller;
    private MainLogger _logger;
    private Dictionary<string, PythonServerConfig> _serverRegistry = new Dictionary<string, PythonServerConfig>();
    private bool _initialized = false;

    #region Unity Lifecycle

    /// <summary>
    /// Singleton initialization
    /// </summary>
    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }

    /// <summary>
    /// Initialize component references and server registry
    /// </summary>
    private void Start()
    {
        try
        {
            _pythonCaller = PythonCaller.Instance;
            _logger = MainLogger.Instance;

            if (_pythonCaller == null)
            {
                LogError("PythonCaller not found. Make sure PythonCaller GameObject exists in the scene.");
                return;
            }

            if (!_pythonCaller.IsActive())
            {
                LogWarning("PythonCaller is not active. Python servers will not start. Check Python environment setup.");
                return;
            }

            // Build server registry
            _serverRegistry.Clear();
            foreach (var server in servers)
            {
                if (server == null || !server.enabled)
                    continue;

                if (_serverRegistry.ContainsKey(server.serverName))
                {
                    LogWarning($"Duplicate server name '{server.serverName}' - only the first will be used");
                    continue;
                }

                _serverRegistry[server.serverName] = server;
                server.isRunning = false;
                server.processId = -1;
            }

            _initialized = true;
            LogMessage($"PythonServerManager initialized with {_serverRegistry.Count} server(s)");

            // Start servers after delay
            if (_startupDelay > 0)
            {
                Invoke(nameof(StartAllServers), _startupDelay);
            }
            else
            {
                StartAllServers();
            }
        }
        catch (Exception ex)
        {
            LogError($"Failed to initialize PythonServerManager: {ex.Message}\n{ex.StackTrace}");
        }
    }

    /// <summary>
    /// Clean up all running servers on application quit
    /// </summary>
    private void OnApplicationQuit()
    {
        LogMessage("Unity shutting down - gracefully stopping Python servers...");
        StopAllServersGracefully();
    }

    /// <summary>
    /// Clean up all running servers when destroyed
    /// </summary>
    private void OnDestroy()
    {
        if (Instance == this)
        {
            StopAllServersGracefully();
            Instance = null;
        }
    }

    #endregion

    #region Public API

    /// <summary>
    /// Starts all servers that have autoStart and enabled set to true
    /// </summary>
    public void StartAllServers()
    {
        if (!_initialized)
        {
            LogWarning("Cannot start servers - manager not initialized");
            return;
        }

        LogMessage("Starting all auto-start servers...");

        int startedCount = 0;
        foreach (var server in _serverRegistry.Values)
        {
            if (server.autoStart && server.enabled && !server.isRunning)
            {
                if (StartServer(server.serverName))
                {
                    startedCount++;
                }
            }
        }

        LogMessage($"Started {startedCount} server(s)");
    }

    /// <summary>
    /// Starts a specific server by name
    /// </summary>
    /// <param name="serverName">Name of the server to start</param>
    /// <returns>True if server was started successfully</returns>
    public bool StartServer(string serverName)
    {
        if (!_initialized)
        {
            LogWarning("Cannot start server - manager not initialized");
            return false;
        }

        if (!_serverRegistry.TryGetValue(serverName, out PythonServerConfig server))
        {
            LogError($"Server '{serverName}' not found in registry");
            return false;
        }

        if (server.isRunning)
        {
            LogWarning($"Server '{serverName}' is already running");
            return false;
        }

        if (string.IsNullOrEmpty(server.scriptPath))
        {
            LogError($"Server '{serverName}' has no script path configured");
            return false;
        }

        try
        {
            // Start the server process (timeout -1 = runs indefinitely)
            int processId = _pythonCaller.ExecuteAsync(
                server.scriptPath,
                server.arguments,
                onComplete: (result) => OnServerProcessComplete(serverName, result),
                timeoutSeconds: -1  // No timeout - server runs until stopped
            );

            if (processId < 0)
            {
                LogError($"Failed to start server '{serverName}' - PythonCaller returned invalid process ID");
                return false;
            }

            server.processId = processId;
            server.isRunning = true;

            LogMessage($"Started server '{serverName}' (PID: {processId})\n  Script: {server.scriptPath}\n  Args: {server.arguments}");
            return true;
        }
        catch (Exception ex)
        {
            LogError($"Exception starting server '{serverName}': {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Stops a specific server by name
    /// </summary>
    /// <param name="serverName">Name of the server to stop</param>
    /// <returns>True if server was stopped successfully</returns>
    public bool StopServer(string serverName)
    {
        if (!_initialized)
        {
            LogWarning("Cannot stop server - manager not initialized");
            return false;
        }

        if (!_serverRegistry.TryGetValue(serverName, out PythonServerConfig server))
        {
            LogError($"Server '{serverName}' not found in registry");
            return false;
        }

        if (!server.isRunning)
        {
            LogWarning($"Server '{serverName}' is not running");
            return false;
        }

        try
        {
            bool stopped = _pythonCaller.StopProcess(server.processId);

            if (stopped)
            {
                server.isRunning = false;
                server.processId = -1;
                LogMessage($"Stopped server '{serverName}'");
            }
            else
            {
                LogWarning($"Failed to stop server '{serverName}' - process may have already exited");
            }

            return stopped;
        }
        catch (Exception ex)
        {
            LogError($"Exception stopping server '{serverName}': {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Stops all running servers
    /// </summary>
    public void StopAllServers()
    {
        if (!_initialized)
            return;

        LogMessage("Stopping all servers...");

        int stoppedCount = 0;
        foreach (var server in _serverRegistry.Values)
        {
            if (server.isRunning)
            {
                if (StopServer(server.serverName))
                {
                    stoppedCount++;
                }
            }
        }

        LogMessage($"Stopped {stoppedCount} server(s)");
    }

    /// <summary>
    /// Gracefully stops all running servers, suppressing warnings for expected shutdown behavior
    /// </summary>
    private void StopAllServersGracefully()
    {
        if (!_initialized)
            return;

        if (_serverRegistry == null || _serverRegistry.Count == 0)
            return;

        List<string> runningServers = new List<string>();
        foreach (var server in _serverRegistry.Values)
        {
            if (server.isRunning)
            {
                runningServers.Add(server.serverName);
            }
        }

        if (runningServers.Count == 0)
            return;

        LogMessage($"Gracefully stopping {runningServers.Count} running server(s)...");

        foreach (string serverName in runningServers)
        {
            if (!_serverRegistry.TryGetValue(serverName, out PythonServerConfig server))
                continue;

            try
            {
                // Attempt to stop the process
                bool stopped = _pythonCaller.StopProcess(server.processId);

                // Update state regardless of whether stop succeeded
                // (process may have already exited, which is fine during shutdown)
                server.isRunning = false;
                server.processId = -1;

                if (stopped)
                {
                    LogMessage($"  ✓ Stopped '{serverName}' gracefully");
                }
                else
                {
                    // During shutdown, it's normal for processes to already be gone
                    LogMessage($"  → '{serverName}' already stopped");
                }
            }
            catch (Exception ex)
            {
                // Don't treat exceptions as errors during shutdown
                LogMessage($"  → '{serverName}' cleanup: {ex.Message}");
                server.isRunning = false;
                server.processId = -1;
            }
        }

        LogMessage("All servers stopped successfully");
    }

    /// <summary>
    /// Checks if a server is currently running
    /// </summary>
    /// <param name="serverName">Name of the server to check</param>
    /// <returns>True if server is running</returns>
    public bool IsServerRunning(string serverName)
    {
        if (!_initialized || !_serverRegistry.TryGetValue(serverName, out PythonServerConfig server))
            return false;

        return server.isRunning;
    }

    /// <summary>
    /// Gets process information for a running server
    /// </summary>
    /// <param name="serverName">Name of the server</param>
    /// <param name="scriptPath">Output: script path</param>
    /// <param name="elapsedSeconds">Output: elapsed execution time</param>
    /// <returns>True if server is running and info was retrieved</returns>
    public bool GetServerProcessInfo(string serverName, out string scriptPath, out float elapsedSeconds)
    {
        scriptPath = null;
        elapsedSeconds = 0;

        if (!_initialized || !_serverRegistry.TryGetValue(serverName, out PythonServerConfig server))
            return false;

        if (!server.isRunning)
            return false;

        return _pythonCaller.GetProcessInfo(server.processId, out scriptPath, out elapsedSeconds);
    }

    /// <summary>
    /// Gets the list of all registered server names
    /// </summary>
    /// <returns>List of server names</returns>
    public List<string> GetServerNames()
    {
        return new List<string>(_serverRegistry.Keys);
    }

    #endregion

    #region Private Methods

    /// <summary>
    /// Callback invoked when a server process completes (either normally or due to error)
    /// </summary>
    private void OnServerProcessComplete(string serverName, PythonCaller.PythonResult result)
    {
        if (!_serverRegistry.TryGetValue(serverName, out PythonServerConfig server))
            return;

        server.isRunning = false;
        server.processId = -1;

        if (result.Success)
        {
            LogMessage($"Server '{serverName}' completed successfully after {result.ExecutionTimeSeconds:F1}s");
        }
        else
        {
            if (result.TimedOut)
            {
                LogWarning($"Server '{serverName}' timed out");
            }
            else
            {
                LogError($"Server '{serverName}' failed with exit code {result.ExitCode}:\n{result.Error}");
            }
        }
    }

    #endregion

    #region Logging

    /// <summary>
    /// Logs a message
    /// </summary>
    private void LogMessage(string message)
    {
        Debug.Log($"[PythonServerManager] {message}");
    }

    /// <summary>
    /// Logs a warning
    /// </summary>
    private void LogWarning(string message)
    {
        Debug.LogWarning($"[PythonServerManager] {message}");
    }

    /// <summary>
    /// Logs an error
    /// </summary>
    private void LogError(string message)
    {
        Debug.LogError($"[PythonServerManager] {message}");
    }

    #endregion
}
