using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public enum SimulationState
{
    Stopped,
    Initializing,
    Running,
    Paused,
    Resetting,
    Error,
}

public enum RobotCoordinationMode
{
    Independent, // Robots operate independently
    Sequential, // Robots take turns
}

[System.Serializable]
public class SimulationConfig
{
    [Header("Simulation Settings")]
    public float timeScale = 1f;
    public bool autoStart = true;
    public bool resetOnError = true;

    [Header("Robot Coordination")]
    public RobotCoordinationMode coordinationMode = RobotCoordinationMode.Independent;

    [Header("Performance")]
    public int targetFrameRate = 60;
    public bool enableVSync = true;
}

public class SimulationManager : MonoBehaviour
{
    public static SimulationManager Instance { get; private set; }

    [Header("Configuration")]
    [SerializeField]
    private SimulationConfig config = new SimulationConfig();

    [Header("Runtime Control")]
    [SerializeField]
    private bool stopRobot = false;

    [SerializeField]
    private bool pauseSimulation = false;

    // Core components
    private PythonCaller _pythonCaller;
    private FileLogger _fileLogger;
    private RobotActionLogger _robotActionLogger;
    private RobotController[] _robotControllers;
    private RobotManager _robotManager;

    // State management
    private SimulationState _currentState = SimulationState.Stopped;
    private SimulationState _previousState = SimulationState.Stopped;
    private float _stateChangeTime;
    private string _lastErrorMessage;

    // Robot coordination
    private int _activeRobotIndex = 0;
    private Dictionary<string, bool> _robotTargetReached = new Dictionary<string, bool>();

    // Session tracking
    private float _sessionStartTime;

    // Events
    public event System.Action<SimulationState, SimulationState> OnStateChanged;

    // Properties
    public SimulationState CurrentState => _currentState;
    public bool IsRunning => _currentState == SimulationState.Running;
    public bool IsPaused => _currentState == SimulationState.Paused || pauseSimulation;
    public bool ShouldStopRobots =>
        stopRobot || IsPaused || _currentState != SimulationState.Running;

    // Singleton pattern initialization
    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
            InitializeSimulation();
        }
        else
        {
            Destroy(gameObject);
        }
    }

    private void InitializeSimulation()
    {
        try
        {
            // Apply performance settings
            Application.targetFrameRate = config.targetFrameRate;
            QualitySettings.vSyncCount = config.enableVSync ? 1 : 0;
            Time.timeScale = config.timeScale;

            ChangeState(SimulationState.Initializing);

            _sessionStartTime = Time.time;

            Debug.Log(
                $"SimulationManager initialized: {config.coordinationMode} mode, {config.targetFrameRate}fps"
            );
        }
        catch (Exception ex)
        {
            HandleError($"Failed to initialize simulation: {ex.Message}");
        }
    }

    private void Start()
    {
        try
        {
            // Get component references
            _pythonCaller = PythonCaller.Instance;
            _fileLogger = FileLogger.Instance;
            _robotActionLogger = RobotActionLogger.Instance;
            _robotManager = RobotManager.Instance;

            // Find all robot controllers
            _robotControllers = FindObjectsByType<RobotController>(FindObjectsSortMode.None);

            if (_robotControllers.Length == 0)
            {
                HandleError("No RobotController components found in scene");
                return;
            }

            // Initialize robot tracking
            foreach (var robot in _robotControllers)
            {
                string robotId = robot.gameObject.name;
                _robotTargetReached[robotId] = true; // Start with no active targets
            }

            // Log simulation start
            _fileLogger?.LogSimulationEvent(
                "simulation_initialized",
                $"Found {_robotControllers.Length} robots. Mode: {config.coordinationMode}"
            );

            // Auto-start if configured
            if (config.autoStart)
            {
                StartSimulation();
            }
            else
            {
                ChangeState(SimulationState.Stopped);
            }
        }
        catch (Exception ex)
        {
            HandleError($"Failed to start simulation: {ex.Message}");
        }
    }

    private void Update()
    {
        try
        {
            UpdateRobotCoordination();
        }
        catch (Exception ex)
        {
            HandleError($"Error in simulation update: {ex.Message}");
        }
    }

    private void LateUpdate()
    {
        try
        {
            if (_pythonCaller != null && _pythonCaller.IsActive())
            {
                // Python process is active - could add ML-Agents coordination here
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"Non-critical error in LateUpdate: {ex.Message}");
        }
    }

    private void UpdateRobotCoordination()
    {
        if (!IsRunning || _robotControllers == null)
            return;

        switch (config.coordinationMode)
        {
            case RobotCoordinationMode.Sequential:
                HandleSequentialMode();
                break;

            case RobotCoordinationMode.Independent:
            default:
                // All robots operate independently - no coordination needed
                break;
        }
    }

    private void HandleSequentialMode()
    {
        // Simple sequential mode - robots take turns automatically
        if (
            _robotTargetReached.GetValueOrDefault(
                _robotControllers[_activeRobotIndex].gameObject.name,
                true
            )
        )
        {
            // Switch to next robot
            _activeRobotIndex = (_activeRobotIndex + 1) % _robotControllers.Length;

            if (_fileLogger != null)
            {
                _fileLogger.LogSimulationEvent(
                    "robot_switch",
                    $"Switched to robot {_activeRobotIndex}"
                );
            }
        }
    }

    private void ChangeState(SimulationState newState)
    {
        if (_currentState == newState)
            return;

        SimulationState oldState = _currentState;
        _previousState = _currentState;
        _currentState = newState;
        _stateChangeTime = Time.time;

        OnStateChanged?.Invoke(oldState, newState);

        if (_fileLogger != null)
        {
            _fileLogger.LogSimulationEvent(
                "state_change",
                $"Changed from {oldState} to {newState}",
                newState == SimulationState.Running
            );
        }

        Debug.Log($"SimulationManager state: {oldState} -> {newState}");
    }

    private void HandleError(string errorMessage)
    {
        _lastErrorMessage = errorMessage;
        ChangeState(SimulationState.Error);

        Debug.LogError($"SimulationManager Error: {errorMessage}");

        if (_fileLogger != null)
        {
            _fileLogger.LogSimulationEvent("simulation_error", errorMessage, false);
        }

        if (config.resetOnError)
        {
            Invoke(nameof(ResetSimulation), 2f); // Reset after 2 seconds
        }
    }

    // Public control methods
    public void StartSimulation()
    {
        if (_currentState == SimulationState.Error && !string.IsNullOrEmpty(_lastErrorMessage))
        {
            Debug.LogWarning("Cannot start simulation while in error state. Reset first.");
            return;
        }

        ChangeState(SimulationState.Running);
        stopRobot = false;
        pauseSimulation = false;

        if (_fileLogger != null)
        {
            _fileLogger.LogSimulationEvent(
                "simulation_started",
                "Simulation started by user request"
            );
        }
    }

    public void StopSimulation()
    {
        ChangeState(SimulationState.Stopped);
        stopRobot = true;

        if (_fileLogger != null)
        {
            _fileLogger.LogSimulationEvent(
                "simulation_stopped",
                "Simulation stopped by user request",
                false
            );
        }
    }

    public void PauseSimulation()
    {
        if (IsRunning)
        {
            ChangeState(SimulationState.Paused);
            pauseSimulation = true;

            if (_fileLogger != null)
            {
                _fileLogger.LogSimulationEvent("simulation_paused", "Simulation paused", false);
            }
        }
    }

    public void ResumeSimulation()
    {
        if (IsPaused)
        {
            ChangeState(SimulationState.Running);
            pauseSimulation = false;

            if (_fileLogger != null)
            {
                _fileLogger.LogSimulationEvent("simulation_resumed", "Simulation resumed");
            }
        }
    }

    public void ResetSimulation()
    {
        ChangeState(SimulationState.Resetting);

        try
        {
            // Reset all robots
            foreach (var robot in _robotControllers)
            {
                if (robot != null)
                {
                    robot.ResetJointTargets();
                    _robotTargetReached[robot.gameObject.name] = true;
                }
            }

            // Reset coordination
            _activeRobotIndex = 0;

            _lastErrorMessage = string.Empty;

            if (_fileLogger != null)
            {
                _fileLogger.LogSimulationEvent("simulation_reset", "Simulation reset completed");
            }

            // Restart if configured
            if (config.autoStart)
            {
                StartSimulation();
            }
            else
            {
                ChangeState(SimulationState.Stopped);
            }
        }
        catch (Exception ex)
        {
            HandleError($"Failed to reset simulation: {ex.Message}");
        }
    }

    // Public getters
    public float GetSessionTime() => Time.time - _sessionStartTime;

    public string GetActiveRobotId() =>
        _robotControllers?[_activeRobotIndex]?.gameObject.name ?? "None";

    public string GetLastError() => _lastErrorMessage;

    private void OnApplicationPause(bool pauseStatus)
    {
        if (pauseStatus && IsRunning)
        {
            PauseSimulation();
        }
    }

    private void OnDestroy()
    {
        if (Instance == this)
        {
            if (_fileLogger != null)
            {
                _fileLogger.LogSimulationEvent(
                    "simulation_destroyed",
                    "SimulationManager destroyed",
                    false
                );
            }
            Instance = null;
        }
    }
}
