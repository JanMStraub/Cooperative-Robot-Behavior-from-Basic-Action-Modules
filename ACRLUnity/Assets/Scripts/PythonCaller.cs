using System.Diagnostics;
using System.IO;
using UnityEngine;

public class PythonCaller : MonoBehaviour
{
    private bool _isActive = false;

    public string basePath;
    public string pythonEnvPath;
    public static PythonCaller Instance { get; private set; }

    /// <summary>
    /// Checks if we're on windows and replaces forward slashes with backslashes.
    /// </summary>
    private string NormalizePath(string path)
    {
        if (SystemInfo.operatingSystemFamily == OperatingSystemFamily.Windows)
            return path.Replace("/", "\\");
        else
            return path;
    }

    /// <summary>
    /// Starts a Python process with the specified script and arguments.
    /// </summary>
    private void StartPythonProcess(string scriptPath, string arguments)
    {
        ProcessStartInfo psi = new ProcessStartInfo
        {
            // FileName = @"C:\Users\ioana\AppData\Local\Programs\Python\Python312\python.exe", // for Windows (for some reason just with pythonEnvPath it doesn't work) TODO: investigate
            FileName = NormalizePath(Path.Combine(basePath, pythonEnvPath)),
            Arguments = NormalizePath($"\"{scriptPath}\" {arguments}"),
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
            WorkingDirectory = NormalizePath(basePath)
        };

        using (Process process = new Process { StartInfo = psi })
        {
            process.Start();

            string output = process.StandardOutput.ReadToEnd();
            string error = process.StandardError.ReadToEnd();
            process.WaitForExit();

            UnityEngine.Debug.Log("Python Output: " + output);
            if (!string.IsNullOrEmpty(error))
            {
                UnityEngine.Debug.LogError("Python Error: " + error);
            }
        }
    }

    public bool IsActive() => _isActive;

    // Singleton pattern initialization
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

    private void Start()
    {
        _isActive = true;

        if (string.IsNullOrEmpty(basePath))
            basePath = Directory.GetParent(Directory.GetParent(Application.dataPath).FullName).FullName;

        if (string.IsNullOrEmpty(pythonEnvPath))
            if (SystemInfo.operatingSystemFamily == OperatingSystemFamily.Windows)
                pythonEnvPath = "roboscan/Scripts/python.exe";
            else
                pythonEnvPath = "roboscan/bin/python";
    }
}
