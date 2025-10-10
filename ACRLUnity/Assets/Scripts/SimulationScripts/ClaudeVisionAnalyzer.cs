using System;
using System.Collections;
using Logging;
using UnityEngine;

/// <summary>
/// Example integration of Claude API vision analysis with Unity robot system.
/// Demonstrates how to capture screenshots and send them to Claude for analysis.
/// </summary>
public class ClaudeVisionAnalyzer : MonoBehaviour
{
    [Header("Configuration")]
    [SerializeField]
    [Tooltip("Robot name to analyze (e.g., AR4Left, AR4Right)")]
    private string _robotName = "AR4Left";

    [SerializeField]
    [Tooltip("Prompt to send to Claude for analysis")]
    private string _analysisPrompt =
        "Describe what the robot camera sees, including object positions and any notable features.";

    [SerializeField]
    [Tooltip("Claude model to use")]
    private ClaudeModel _model = ClaudeModel.Haiku;

    [SerializeField]
    [Tooltip("Number of most recent screenshots to analyze")]
    private int _screenshotCount = 1;

    [SerializeField]
    [Tooltip("Timeout for Claude API requests in seconds")]
    private int _timeoutSeconds = 60;

    [Header("Auto-Analysis")]
    [SerializeField]
    [Tooltip("Enable automatic analysis after screenshot capture")]
    private bool _autoAnalyze = false;

    [SerializeField]
    [Tooltip("Delay in seconds after capture before analysis")]
    private float _analysisDelay = 1.0f;

    [Header("References")]
    [SerializeField]
    [Tooltip("Camera controller to monitor for captures")]
    private CameraController _cameraController;

    // Component references
    private PythonCaller _pythonCaller;
    private MainLogger _logger;

    // State
    private bool _isAnalyzing = false;
    private int _activeProcessId = -1;

    /// <summary>
    /// Claude model options
    /// </summary>
    public enum ClaudeModel
    {
        Sonnet, // claude-3-5-sonnet-20241022 (balanced)
        Haiku, // claude-3-5-haiku-20241022 (fast/cheap)
        Opus, // claude-3-opus-20240229 (most capable)
    }

    /// <summary>
    /// Initialize component references
    /// </summary>
    private void Start()
    {
        // Get required components
        _pythonCaller = PythonCaller.Instance;
        _logger = MainLogger.Instance;

        // Validate setup
        if (_pythonCaller == null || !_pythonCaller.IsActive())
        {
            Debug.LogWarning(
                "ClaudeVisionAnalyzer: PythonCaller is not active. Vision analysis will not work."
            );
            enabled = false;
            return;
        }

        // Auto-find camera controller if not set
        if (_cameraController == null)
        {
            _cameraController = GetComponent<CameraController>();
        }

        Debug.Log($"ClaudeVisionAnalyzer initialized for robot: {_robotName}");
    }

    /// <summary>
    /// Update loop - check for keyboard shortcuts
    /// </summary>
    private void Update()
    {
        // Manual trigger: Press V to analyze latest screenshots
        if (Input.GetKeyDown(KeyCode.V))
        {
            AnalyzeLatestScreenshots();
        }

        // Cancel current analysis: Press Escape
        if (Input.GetKeyDown(KeyCode.Escape) && _isAnalyzing)
        {
            CancelAnalysis();
        }
    }

    /// <summary>
    /// Analyzes the latest screenshot(s) for the configured robot
    /// </summary>
    public void AnalyzeLatestScreenshots()
    {
        if (_isAnalyzing)
        {
            Debug.LogWarning("ClaudeVisionAnalyzer: Analysis already in progress");
            return;
        }

        if (_pythonCaller == null || !_pythonCaller.IsActive())
        {
            Debug.LogError("ClaudeVisionAnalyzer: PythonCaller is not active");
            return;
        }

        StartCoroutine(AnalyzeCoroutine());
    }

    /// <summary>
    /// Analyzes screenshots after a screenshot capture
    /// </summary>
    public void AnalyzeAfterCapture()
    {
        if (!_autoAnalyze)
            return;

        StartCoroutine(AnalyzeAfterCaptureCoroutine());
    }

    /// <summary>
    /// Cancels the current analysis if one is running
    /// </summary>
    public void CancelAnalysis()
    {
        if (!_isAnalyzing || _activeProcessId < 0)
            return;

        if (_pythonCaller.StopProcess(_activeProcessId))
        {
            Debug.Log($"ClaudeVisionAnalyzer: Cancelled analysis (process {_activeProcessId})");
            _isAnalyzing = false;
            _activeProcessId = -1;
        }
    }

    /// <summary>
    /// Coroutine to handle analysis with delay after capture
    /// </summary>
    private IEnumerator AnalyzeAfterCaptureCoroutine()
    {
        // Wait for capture to complete
        yield return new WaitForSeconds(_analysisDelay);

        // Perform analysis
        AnalyzeLatestScreenshots();
    }

    /// <summary>
    /// Coroutine to perform Claude API analysis
    /// </summary>
    private IEnumerator AnalyzeCoroutine()
    {
        _isAnalyzing = true;

        Debug.Log($"ClaudeVisionAnalyzer: Starting analysis for {_robotName}...");

        // Log analysis start
        Debug.Log($"[VISION] Analysis start: Robot={_robotName}, Prompt={_analysisPrompt}, Model={GetModelName()}");

        // Build Python script arguments
        string scriptPath = "Assets/Scripts/LLMcommunication/SendScreenshots.py";
        string args = BuildArguments();

        Debug.Log($"ClaudeVisionAnalyzer: Executing: {scriptPath} {args}");

        // Execute Python script asynchronously
        _activeProcessId = _pythonCaller.ExecuteAsync(
            scriptPath,
            args,
            OnAnalysisComplete,
            _timeoutSeconds
        );

        if (_activeProcessId < 0)
        {
            Debug.LogError("ClaudeVisionAnalyzer: Failed to start analysis process");
            _isAnalyzing = false;
            yield break;
        }

        Debug.Log($"ClaudeVisionAnalyzer: Analysis started (process {_activeProcessId})");

        // Wait for completion (process will call OnAnalysisComplete)
        yield return null;
    }

    /// <summary>
    /// Callback when Claude API analysis completes
    /// </summary>
    private void OnAnalysisComplete(PythonCaller.PythonResult result)
    {
        _isAnalyzing = false;
        _activeProcessId = -1;

        if (result.Success)
        {
            Debug.Log(
                $"ClaudeVisionAnalyzer: Analysis completed in {result.ExecutionTimeSeconds:F2}s"
            );
            Debug.Log($"Claude's response:\n{result.Output}");

            // Log successful analysis
            if (_logger != null)
            {
                string actionId = _logger.StartAction(
                    actionName: "vision_analysis",
                    type: Logging.ActionType.Observation,
                    robotIds: new[] { _robotName },
                    description: $"Vision analysis with {GetModelName()}"
                );
                var metrics = new System.Collections.Generic.Dictionary<string, float>
                {
                    ["execution_time"] = result.ExecutionTimeSeconds
                };
                _logger.CompleteAction(actionId, success: true, qualityScore: 1f, metrics: metrics);
            }
        }
        else
        {
            if (result.TimedOut)
            {
                Debug.LogError(
                    $"ClaudeVisionAnalyzer: Analysis timed out after {result.ExecutionTimeSeconds:F2}s"
                );
            }
            else
            {
                Debug.LogError($"ClaudeVisionAnalyzer: Analysis failed: {result.Error}");
            }

            // Log failure
            if (_logger != null)
            {
                string actionId = _logger.StartAction(
                    actionName: "vision_analysis",
                    type: Logging.ActionType.Observation,
                    robotIds: new[] { _robotName },
                    description: $"Vision analysis with {GetModelName()}"
                );
                var metrics = new System.Collections.Generic.Dictionary<string, float>
                {
                    ["execution_time"] = result.ExecutionTimeSeconds
                };
                _logger.CompleteAction(actionId, success: false, qualityScore: 0f, errorMessage: result.Error, metrics: metrics);
            }
        }
    }

    /// <summary>
    /// Builds command-line arguments for the Python script
    /// </summary>
    private string BuildArguments()
    {
        string modelName = GetModelName();

        return $"--robot {_robotName} "
            + $"--latest {_screenshotCount} "
            + $"--model {modelName} "
            + $"--prompt \"{_analysisPrompt}\" "
            + $"--quiet";
    }

    /// <summary>
    /// Gets the Claude model name string
    /// </summary>
    private string GetModelName()
    {
        switch (_model)
        {
            case ClaudeModel.Haiku:
                return "claude-3-5-haiku-20241022";
            case ClaudeModel.Opus:
                return "claude-3-opus-20240229";
            case ClaudeModel.Sonnet:
            default:
                return "claude-3-5-sonnet-20241022";
        }
    }

    /// <summary>
    /// Returns whether an analysis is currently running
    /// </summary>
    public bool IsAnalyzing() => _isAnalyzing;

    /// <summary>
    /// Gets the current robot name
    /// </summary>
    public string GetRobotName() => _robotName;

    /// <summary>
    /// Sets a new robot name for analysis
    /// </summary>
    public void SetRobotName(string robotName)
    {
        _robotName = robotName;
    }

    /// <summary>
    /// Gets the current analysis prompt
    /// </summary>
    public string GetPrompt() => _analysisPrompt;

    /// <summary>
    /// Sets a new analysis prompt
    /// </summary>
    public void SetPrompt(string prompt)
    {
        _analysisPrompt = prompt;
    }
}
