using System;
using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

public enum SimulationState
{
    Initializing,
    Running,
    Paused,
    Resetting,
    Error,
}

#if UNITY_EDITOR
[CustomEditor(typeof(SimulationManager))]
public class SimulationManagerEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        SimulationManager manager = (SimulationManager)target;

        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Simulation Controls", EditorStyles.boldLabel);

        // Display current state
        EditorGUILayout.LabelField($"Current State: {manager.CurrentState}");
        EditorGUILayout.LabelField($"Active Robot: {manager.GetActiveRobotId()}");

        EditorGUILayout.Space();

        // Control buttons
        EditorGUILayout.BeginHorizontal();

        if (GUILayout.Button("Start"))
        {
            manager.StartSimulation();
        }

        if (GUILayout.Button("Pause"))
        {
            manager.PauseSimulation();
        }

        if (GUILayout.Button("Resume"))
        {
            manager.ResumeSimulation();
        }

        if (GUILayout.Button("Reset"))
        {
            manager.ResetSimulation();
        }

        EditorGUILayout.EndHorizontal();
    }
}
#endif

public class SimulationManager : MonoBehaviour
{
    public static SimulationManager Instance { get; private set; } // Singleton instance

    [Header("Configuration")]
    [SerializeField]
    public SimulationConfig config;

    // Core components
    private PythonCaller _pythonCaller;
    private FileLogger _fileLogger;
    private RobotController[] _robotControllers;

    // State management
    private SimulationState _currentState = SimulationState.Paused;
    private SimulationState _previousState = SimulationState.Paused;

    // Robot coordination
    private int _activeRobotIndex = 0;
    private Dictionary<string, bool> _robotTargetReached = new Dictionary<string, bool>();

    // Events
    public event System.Action<SimulationState, SimulationState> OnStateChanged;

    // Properties
    public SimulationState CurrentState => _currentState;
    public bool IsRunning => _currentState == SimulationState.Running;
    public bool IsPaused => _currentState == SimulationState.Paused;
    public bool ShouldStopRobots => _currentState != SimulationState.Running;

    /// <summary>
    /// Unity Awake callback - initializes singleton instance and simulation.
    /// </summary>
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

    /// <summary>
    /// Initializes the simulation with default configuration and performance settings.
    /// </summary>
    private void InitializeSimulation()
    {
        try
        {
            // Create default config if not assigned
            if (config == null)
            {
                Debug.LogWarning("SimulationConfig not assigned. Creating default configuration.");
                config = ScriptableObject.CreateInstance<SimulationConfig>();
            }

            // Apply performance settings
            Application.targetFrameRate = config.targetFrameRate;
            QualitySettings.vSyncCount = config.enableVSync ? 1 : 0;
            Time.timeScale = config.timeScale;

            ChangeState(SimulationState.Initializing);

            Debug.Log(
                $"SimulationManager initialized: {config.coordinationMode} mode, {config.targetFrameRate}fps"
            );
        }
        catch (Exception ex)
        {
            HandleError($"Failed to initialize simulation: {ex.Message}");
        }
    }

    /// <summary>
    /// Unity Start callback - initializes component references and robot tracking.
    /// </summary>
    private void Start()
    {
        try
        {
            // Get component references
            _pythonCaller = PythonCaller.Instance;
            _fileLogger = FileLogger.Instance;

            // Find all robot controllers
            _robotControllers = FindObjectsByType<RobotController>(FindObjectsInactive.Exclude, FindObjectsSortMode.None);

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
                ChangeState(SimulationState.Paused);
            }
        }
        catch (Exception ex)
        {
            HandleError($"Failed to start simulation: {ex.Message}");
        }
    }

    /// <summary>
    /// Unity Update callback - updates robot coordination based on configuration mode.
    /// </summary>
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

    /// <summary>
    /// Unity LateUpdate callback - handles ML-Agents coordination if Python process is active.
    /// </summary>
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

    /// <summary>
    /// Updates robot coordination based on the current coordination mode.
    /// </summary>
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

            case RobotCoordinationMode.Collaborative:
            default:
                // All robots operate independently - no coordination needed
                break;
        }
    }

    /// <summary>
    /// Handles robot coordination in sequential mode, switching to the next robot when current robot reaches its target.
    /// </summary>
    private void HandleSequentialMode()
    {
        if (_robotControllers == null || _robotControllers.Length == 0)
            return;

        if (_activeRobotIndex < 0 || _activeRobotIndex >= _robotControllers.Length)
            _activeRobotIndex = 0;

        var currentRobot = _robotControllers[_activeRobotIndex];
        if (currentRobot == null)
            return;

        string currentRobotId = currentRobot.gameObject.name;

        // Check if current robot has reached its target
        if (_robotTargetReached.GetValueOrDefault(currentRobotId, true))
        {
            // Switch to next robot
            int previousIndex = _activeRobotIndex;
            _activeRobotIndex = (_activeRobotIndex + 1) % _robotControllers.Length;

            if (_fileLogger != null)
            {
                _fileLogger.LogSimulationEvent(
                    "robot_switch",
                    $"Switched from {currentRobotId} to {GetActiveRobotId()}"
                );
            }

            Debug.Log(
                $"Sequential mode: Switched from robot {previousIndex} ({currentRobotId}) to robot {_activeRobotIndex} ({GetActiveRobotId()})"
            );
        }
    }

    /// <summary>
    /// Changes the simulation state and triggers state change event.
    /// </summary>
    /// <param name="newState">The new simulation state to transition to</param>
    private void ChangeState(SimulationState newState)
    {
        if (_currentState == newState)
            return;

        _previousState = _currentState;
        _currentState = newState;

        OnStateChanged?.Invoke(_previousState, newState);

        if (_fileLogger != null)
        {
            _fileLogger.LogSimulationEvent(
                "state_change",
                $"Changed from {_previousState} to {newState}",
                newState == SimulationState.Running
            );
        }

        Debug.Log($"SimulationManager state: {_previousState} -> {newState}");
    }

    /// <summary>
    /// Handles simulation errors by changing state to error and optionally scheduling a reset.
    /// </summary>
    /// <param name="errorMessage">The error message to log</param>
    private void HandleError(string errorMessage)
    {
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

    /// <summary>
    /// Starts the simulation and changes state to Running.
    /// </summary>
    public void StartSimulation()
    {
        if (_currentState == SimulationState.Error)
        {
            Debug.LogWarning("Cannot start simulation while in error state. Reset first.");
            return;
        }

        ChangeState(SimulationState.Running);

        if (_fileLogger != null)
        {
            _fileLogger.LogSimulationEvent(
                "simulation_started",
                "Simulation started by user request"
            );
        }
    }

    /// <summary>
    /// Pauses the simulation if currently running.
    /// </summary>
    public void PauseSimulation()
    {
        if (IsRunning)
        {
            ChangeState(SimulationState.Paused);

            if (_fileLogger != null)
            {
                _fileLogger.LogSimulationEvent("simulation_paused", "Simulation paused", false);
            }
        }
    }

    /// <summary>
    /// Resumes the simulation if currently paused.
    /// </summary>
    public void ResumeSimulation()
    {
        if (IsPaused)
        {
            ChangeState(SimulationState.Running);

            if (_fileLogger != null)
            {
                _fileLogger.LogSimulationEvent("simulation_resumed", "Simulation resumed");
            }
        }
    }

    /// <summary>
    /// Resets the simulation by resetting all robot joint targets and coordination state.
    /// </summary>
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
                ChangeState(SimulationState.Paused);
            }
        }
        catch (Exception ex)
        {
            HandleError($"Failed to reset simulation: {ex.Message}");
        }
    }

    /// <summary>
    /// Gets the identifier of the currently active robot.
    /// </summary>
    /// <returns>The robot identifier or "None" if no active robot</returns>
    public string GetActiveRobotId()
    {
        if (_robotControllers == null || _robotControllers.Length == 0)
            return "None";

        if (_activeRobotIndex < 0 || _activeRobotIndex >= _robotControllers.Length)
            return "None";

        var controller = _robotControllers[_activeRobotIndex];
        if (controller == null)
            return "None";

        return controller.gameObject.name;
    }

    /// <summary>
    /// Checks if a specific robot is allowed to move based on coordination mode.
    /// </summary>
    /// <param name="robotId">The robot identifier to check</param>
    /// <returns>True if the robot is allowed to move, false otherwise</returns>
    public bool IsRobotActive(string robotId)
    {
        if (!IsRunning)
            return false;

        // In Sequential mode, only the active robot can move
        if (config.coordinationMode == RobotCoordinationMode.Sequential)
        {
            return GetActiveRobotId() == robotId;
        }

        // In all other modes, all robots can move
        return true;
    }

    /// <summary>
    /// Notifies the manager that a robot has reached or is moving towards its target.
    /// </summary>
    /// <param name="robotId">The robot identifier</param>
    /// <param name="reached">True if target is reached, false if still moving</param>
    public void NotifyTargetReached(string robotId, bool reached)
    {
        _robotTargetReached[robotId] = reached;

        if (reached && config.coordinationMode == RobotCoordinationMode.Sequential)
        {
            _fileLogger?.LogSimulationEvent(
                "robot_target_reached",
                $"Robot {robotId} reached target in sequential mode"
            );
        }
    }

    /// <summary>
    /// Unity OnDestroy callback - cleans up singleton instance.
    /// </summary>
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
