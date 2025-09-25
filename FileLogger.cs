using System.IO;
using UnityEngine;

public class FileLogger : MonoBehaviour
{
    public string logFilePath;
    private string _logFile;

    void OnEnable()
    {
        // Set file path
        _logFile = Path.Combine(logFilePath, "unity_log.txt");
        Application.logMessageReceived += HandleLog;
    }

    void OnDisable()
    {
        Application.logMessageReceived -= HandleLog;
    }

    void HandleLog(string logString, string stackTrace, LogType type)
    {
        string logEntry = $"{System.DateTime.Now:yyyy-MM-dd HH:mm:ss} [{type}] {logString}\n";
        if (type == LogType.Exception || type == LogType.Error || type == LogType.Log)
        {
            logEntry += $"{stackTrace}\n";
        }

        File.AppendAllText(_logFile, logEntry);
    }
}
