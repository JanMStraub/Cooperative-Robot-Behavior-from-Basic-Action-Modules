using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using Logging;

/// <summary>
/// Manages Python process execution for ML-Agents and external data processing.
/// Supports async execution, process monitoring, and proper resource management.
/// </summary>
public class PythonCaller : MonoBehaviour
{
    [Header("Python Environment")]
    [SerializeField]
    [Tooltip("Base path for Python environment (leave empty to auto-detect from Unity project)")]
    private string _basePath;

    [SerializeField]
    [Tooltip("Relative path to Python executable from base path")]
    private string _pythonEnvPath;

    [Header("Process Settings")]
    [SerializeField]
    [Tooltip("Default timeout in seconds for Python processes (0 = no timeout)")]
    private int _defaultTimeoutSeconds = 300; // 5 minutes

    [SerializeField]
    [Tooltip("Maximum number of concurrent Python processes")]
    private int _maxConcurrentProcesses = 3;

    [SerializeField]
    [Tooltip("Buffer size for streaming process output (bytes)")]
    private int _outputBufferSize = 4096;

    // Singleton
    public static PythonCaller Instance { get; private set; }

    // State
    private bool _isActive = false;
    private string _fullPythonPath;
    private readonly Dictionary<int, ProcessInfo> _runningProcesses = new Dictionary<int, ProcessInfo>();
    private int _nextProcessId = 1;

    // Logging integration
    private MainLogger _logger;

    /// <summary>
    /// Information about a running Python process
    /// </summary>
    private class ProcessInfo
    {
        public Process Process;
        public CancellationTokenSource CancellationToken;
        public string ScriptPath;
        public string Arguments;
        public DateTime StartTime;
        public StringBuilder OutputBuffer;
        public StringBuilder ErrorBuffer;
    }

    /// <summary>
    /// Result of a Python process execution
    /// </summary>
    public class PythonResult
    {
        public bool Success;
        public int ExitCode;
        public string Output;
        public string Error;
        public float ExecutionTimeSeconds;
        public bool TimedOut;
    }

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
    /// Initialize Python environment and validate paths
    /// </summary>
    private void Start()
    {
        try
        {
            // Get RobotLogger reference
            _logger = FindFirstObjectByType<MainLogger>();

            // Auto-detect base path if not set
            if (string.IsNullOrEmpty(_basePath))
            {
                // Navigate to grandparent of Assets folder
                _basePath = Directory.GetParent(Directory.GetParent(Application.dataPath).FullName).FullName;
            }

            // Auto-detect Python path if not set
            if (string.IsNullOrEmpty(_pythonEnvPath))
            {
                if (SystemInfo.operatingSystemFamily == OperatingSystemFamily.Windows)
                    _pythonEnvPath = "roboscan/Scripts/python.exe";
                else
                    _pythonEnvPath = "roboscan/bin/python";
            }

            // Validate Python environment exists
            _fullPythonPath = NormalizePath(Path.Combine(_basePath, _pythonEnvPath));

            if (File.Exists(_fullPythonPath))
            {
                _isActive = true;
                LogMessage($"PythonCaller initialized: Python={_fullPythonPath}, Timeout={_defaultTimeoutSeconds}s, MaxProcesses={_maxConcurrentProcesses}");
            }
            else
            {
                _isActive = false;
                LogWarning($"Python environment not found at: {_fullPythonPath}. PythonCaller will be inactive.");
            }
        }
        catch (Exception ex)
        {
            _isActive = false;
            LogError($"Failed to initialize PythonCaller: {ex.Message}\n{ex.StackTrace}");
        }
    }

    /// <summary>
    /// Clean up any running processes on application quit
    /// </summary>
    private void OnApplicationQuit()
    {
        StopAllProcesses();
    }

    /// <summary>
    /// Clean up any running processes when destroyed
    /// </summary>
    private void OnDestroy()
    {
        if (Instance == this)
        {
            StopAllProcesses();
        }
    }

    #endregion

    #region Public API

    /// <summary>
    /// Returns whether the Python environment is active and ready to execute scripts
    /// </summary>
    public bool IsActive() => _isActive;

    /// <summary>
    /// Returns the number of currently running Python processes
    /// </summary>
    public int GetActiveProcessCount() => _runningProcesses.Count;

    /// <summary>
    /// Executes a Python script asynchronously
    /// </summary>
    /// <param name="scriptPath">Path to Python script (absolute or relative to base path)</param>
    /// <param name="arguments">Command line arguments for the script</param>
    /// <param name="onComplete">Callback invoked when process completes</param>
    /// <param name="timeoutSeconds">Timeout in seconds (0 = use default, -1 = no timeout)</param>
    /// <returns>Process ID for tracking, or -1 if failed to start</returns>
    public int ExecuteAsync(string scriptPath, string arguments = "", Action<PythonResult> onComplete = null, int timeoutSeconds = 0)
    {
        if (!_isActive)
        {
            LogError("Cannot execute Python script: PythonCaller is not active");
            onComplete?.Invoke(new PythonResult
            {
                Success = false,
                Error = "PythonCaller is not active",
                ExitCode = -1
            });
            return -1;
        }

        if (_runningProcesses.Count >= _maxConcurrentProcesses)
        {
            LogWarning($"Cannot execute Python script: Maximum concurrent processes ({_maxConcurrentProcesses}) reached");
            onComplete?.Invoke(new PythonResult
            {
                Success = false,
                Error = $"Maximum concurrent processes ({_maxConcurrentProcesses}) reached",
                ExitCode = -1
            });
            return -1;
        }

        // Resolve script path
        string fullScriptPath = Path.IsPathRooted(scriptPath)
            ? scriptPath
            : Path.Combine(_basePath, scriptPath);
        fullScriptPath = NormalizePath(fullScriptPath);

        if (!File.Exists(fullScriptPath))
        {
            LogError($"Python script not found: {fullScriptPath}");
            onComplete?.Invoke(new PythonResult
            {
                Success = false,
                Error = $"Script not found: {fullScriptPath}",
                ExitCode = -1
            });
            return -1;
        }

        // Determine timeout
        int timeout = timeoutSeconds == 0 ? _defaultTimeoutSeconds : timeoutSeconds;

        // Start process asynchronously
        int processId = _nextProcessId++;
        StartCoroutine(ExecuteProcessAsync(processId, fullScriptPath, arguments, onComplete, timeout));

        return processId;
    }

    /// <summary>
    /// Executes a Python script synchronously (blocks Unity until complete)
    /// WARNING: Only use for quick scripts - prefer ExecuteAsync for longer operations
    /// </summary>
    /// <param name="scriptPath">Path to Python script (absolute or relative to base path)</param>
    /// <param name="arguments">Command line arguments for the script</param>
    /// <param name="timeoutSeconds">Timeout in seconds (0 = use default, -1 = no timeout)</param>
    /// <returns>Result of the execution</returns>
    public PythonResult ExecuteSync(string scriptPath, string arguments = "", int timeoutSeconds = 0)
    {
        if (!_isActive)
        {
            return new PythonResult
            {
                Success = false,
                Error = "PythonCaller is not active",
                ExitCode = -1
            };
        }

        // Resolve script path
        string fullScriptPath = Path.IsPathRooted(scriptPath)
            ? scriptPath
            : Path.Combine(_basePath, scriptPath);
        fullScriptPath = NormalizePath(fullScriptPath);

        if (!File.Exists(fullScriptPath))
        {
            return new PythonResult
            {
                Success = false,
                Error = $"Script not found: {fullScriptPath}",
                ExitCode = -1
            };
        }

        // Determine timeout
        int timeout = timeoutSeconds == 0 ? _defaultTimeoutSeconds : timeoutSeconds;

        return ExecuteProcessSync(fullScriptPath, arguments, timeout);
    }

    /// <summary>
    /// Stops a running Python process by ID
    /// </summary>
    /// <param name="processId">Process ID returned from ExecuteAsync</param>
    /// <returns>True if process was found and stopped</returns>
    public bool StopProcess(int processId)
    {
        if (!_runningProcesses.TryGetValue(processId, out ProcessInfo info))
            return false;

        try
        {
            info.CancellationToken.Cancel();

            if (!info.Process.HasExited)
            {
                info.Process.Kill();
                LogMessage($"Stopped Python process {processId}: {Path.GetFileName(info.ScriptPath)}");
            }

            CleanupProcess(processId);
            return true;
        }
        catch (Exception ex)
        {
            LogError($"Error stopping process {processId}: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Stops all running Python processes
    /// </summary>
    public void StopAllProcesses()
    {
        var processIds = new List<int>(_runningProcesses.Keys);
        foreach (int processId in processIds)
        {
            StopProcess(processId);
        }
    }

    /// <summary>
    /// Gets information about a running process
    /// </summary>
    /// <param name="processId">Process ID</param>
    /// <param name="scriptPath">Output: script path</param>
    /// <param name="elapsedSeconds">Output: elapsed execution time</param>
    /// <returns>True if process was found</returns>
    public bool GetProcessInfo(int processId, out string scriptPath, out float elapsedSeconds)
    {
        if (_runningProcesses.TryGetValue(processId, out ProcessInfo info))
        {
            scriptPath = info.ScriptPath;
            elapsedSeconds = (float)(DateTime.Now - info.StartTime).TotalSeconds;
            return true;
        }

        scriptPath = null;
        elapsedSeconds = 0;
        return false;
    }

    #endregion

    #region Process Execution

    /// <summary>
    /// Executes a Python process asynchronously using coroutine
    /// </summary>
    private IEnumerator ExecuteProcessAsync(int processId, string scriptPath, string arguments, Action<PythonResult> onComplete, int timeoutSeconds)
    {
        DateTime startTime = DateTime.Now;
        ProcessInfo info = null;
        string exceptionMessage = null;

        // Create process
        ProcessStartInfo psi = new ProcessStartInfo
        {
            FileName = _fullPythonPath,
            Arguments = NormalizePath($"\"{scriptPath}\" {arguments}"),
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
            WorkingDirectory = NormalizePath(_basePath)
        };

        Process process = new Process { StartInfo = psi };
        CancellationTokenSource cts = new CancellationTokenSource();

        info = new ProcessInfo
        {
            Process = process,
            CancellationToken = cts,
            ScriptPath = scriptPath,
            Arguments = arguments,
            StartTime = startTime,
            OutputBuffer = new StringBuilder(),
            ErrorBuffer = new StringBuilder()
        };

        _runningProcesses[processId] = info;

        // Start process
        try
        {
            process.Start();
            LogMessage($"Started Python process {processId}: {Path.GetFileName(scriptPath)} {arguments}");
        }
        catch (Exception startEx)
        {
            exceptionMessage = $"Failed to start process: {startEx.Message}";
            LogError($"Python process {processId} failed: {exceptionMessage}");
            CleanupProcess(processId);

            onComplete?.Invoke(new PythonResult
            {
                Success = false,
                ExitCode = -1,
                Output = "",
                Error = exceptionMessage,
                ExecutionTimeSeconds = (float)(DateTime.Now - startTime).TotalSeconds,
                TimedOut = false
            });

            yield break;
        }

        // Read output asynchronously
        Task<string> outputTask = ReadStreamAsync(process.StandardOutput, info.OutputBuffer, cts.Token);
        Task<string> errorTask = ReadStreamAsync(process.StandardError, info.ErrorBuffer, cts.Token);

        // Wait for process to complete or timeout
        bool completed = false;
        bool timedOut = false;
        float elapsed = 0f;

        while (!completed && !timedOut)
        {
            if (process.HasExited)
            {
                completed = true;
            }
            else if (timeoutSeconds > 0)
            {
                elapsed = (float)(DateTime.Now - startTime).TotalSeconds;
                if (elapsed >= timeoutSeconds)
                {
                    timedOut = true;
                }
            }

            yield return null;
        }

        // Handle timeout
        if (timedOut)
        {
            LogWarning($"Python process {processId} timed out after {timeoutSeconds}s: {Path.GetFileName(scriptPath)}");

            try
            {
                cts.Cancel();
                if (!process.HasExited)
                    process.Kill();
            }
            catch (Exception killEx)
            {
                LogError($"Error killing timed out process: {killEx.Message}");
            }

            CleanupProcess(processId);

            onComplete?.Invoke(new PythonResult
            {
                Success = false,
                ExitCode = -1,
                Output = info.OutputBuffer.ToString(),
                Error = info.ErrorBuffer.ToString(),
                ExecutionTimeSeconds = (float)(DateTime.Now - startTime).TotalSeconds,
                TimedOut = true
            });

            yield break;
        }

        // Wait for output/error streams to finish reading
        while (!outputTask.IsCompleted || !errorTask.IsCompleted)
        {
            yield return null;
        }

        // Get results
        string output = outputTask.Result;
        string error = errorTask.Result;
        int exitCode = process.ExitCode;
        float executionTime = (float)(DateTime.Now - startTime).TotalSeconds;

        CleanupProcess(processId);

        // Log results
        if (exitCode == 0)
        {
            LogMessage($"Python process {processId} completed successfully in {executionTime:F2}s");
        }
        else
        {
            LogError($"Python process {processId} failed with exit code {exitCode}:\n{error}");
        }

        if (!string.IsNullOrEmpty(output))
        {
            UnityEngine.Debug.Log($"Python Output [{processId}]:\n{output}");
        }

        // Invoke callback
        onComplete?.Invoke(new PythonResult
        {
            Success = exitCode == 0,
            ExitCode = exitCode,
            Output = output,
            Error = error,
            ExecutionTimeSeconds = executionTime,
            TimedOut = false
        });
    }

    /// <summary>
    /// Executes a Python process synchronously (blocking)
    /// </summary>
    private PythonResult ExecuteProcessSync(string scriptPath, string arguments, int timeoutSeconds)
    {
        DateTime startTime = DateTime.Now;

        try
        {
            ProcessStartInfo psi = new ProcessStartInfo
            {
                FileName = _fullPythonPath,
                Arguments = NormalizePath($"\"{scriptPath}\" {arguments}"),
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                WorkingDirectory = NormalizePath(_basePath)
            };

            using (Process process = new Process { StartInfo = psi })
            {
                process.Start();

                // Read output and error
                string output = process.StandardOutput.ReadToEnd();
                string error = process.StandardError.ReadToEnd();

                // Wait with timeout
                bool completed;
                if (timeoutSeconds > 0)
                {
                    completed = process.WaitForExit(timeoutSeconds * 1000);
                }
                else
                {
                    process.WaitForExit();
                    completed = true;
                }

                float executionTime = (float)(DateTime.Now - startTime).TotalSeconds;

                if (!completed)
                {
                    try { process.Kill(); } catch { }

                    return new PythonResult
                    {
                        Success = false,
                        ExitCode = -1,
                        Output = output,
                        Error = error,
                        ExecutionTimeSeconds = executionTime,
                        TimedOut = true
                    };
                }

                return new PythonResult
                {
                    Success = process.ExitCode == 0,
                    ExitCode = process.ExitCode,
                    Output = output,
                    Error = error,
                    ExecutionTimeSeconds = executionTime,
                    TimedOut = false
                };
            }
        }
        catch (Exception ex)
        {
            return new PythonResult
            {
                Success = false,
                ExitCode = -1,
                Output = "",
                Error = $"Exception: {ex.Message}",
                ExecutionTimeSeconds = (float)(DateTime.Now - startTime).TotalSeconds,
                TimedOut = false
            };
        }
    }

    /// <summary>
    /// Reads a stream asynchronously with buffering
    /// </summary>
    private async Task<string> ReadStreamAsync(StreamReader reader, StringBuilder buffer, CancellationToken token)
    {
        try
        {
            char[] charBuffer = new char[_outputBufferSize];
            int bytesRead;

            while (!token.IsCancellationRequested && (bytesRead = await reader.ReadAsync(charBuffer, 0, charBuffer.Length)) > 0)
            {
                buffer.Append(charBuffer, 0, bytesRead);
            }

            return buffer.ToString();
        }
        catch (Exception ex)
        {
            if (!token.IsCancellationRequested)
            {
                LogError($"Error reading stream: {ex.Message}");
            }
            return buffer.ToString();
        }
    }

    #endregion

    #region Helpers

    /// <summary>
    /// Normalizes file paths for the current operating system
    /// </summary>
    private string NormalizePath(string path)
    {
        if (string.IsNullOrEmpty(path))
            return path;

        if (SystemInfo.operatingSystemFamily == OperatingSystemFamily.Windows)
            return path.Replace("/", "\\");
        else
            return path.Replace("\\", "/");
    }

    /// <summary>
    /// Cleans up a process and removes it from tracking
    /// </summary>
    private void CleanupProcess(int processId)
    {
        if (_runningProcesses.TryGetValue(processId, out ProcessInfo info))
        {
            try
            {
                info.CancellationToken?.Dispose();
                info.Process?.Dispose();
            }
            catch (Exception ex)
            {
                LogError($"Error cleaning up process {processId}: {ex.Message}");
            }
            finally
            {
                _runningProcesses.Remove(processId);
            }
        }
    }

    /// <summary>
    /// Logs a message using RobotLogger if available, otherwise uses Unity Debug
    /// </summary>
    private void LogMessage(string message)
    {
        UnityEngine.Debug.Log($"[PythonCaller] {message}");
    }

    /// <summary>
    /// Logs a warning using Unity Debug
    /// </summary>
    private void LogWarning(string message)
    {
        UnityEngine.Debug.LogWarning($"[PythonCaller] {message}");
    }

    /// <summary>
    /// Logs an error using Unity Debug
    /// </summary>
    private void LogError(string message)
    {
        UnityEngine.Debug.LogError($"[PythonCaller] {message}");
    }

    #endregion
}
