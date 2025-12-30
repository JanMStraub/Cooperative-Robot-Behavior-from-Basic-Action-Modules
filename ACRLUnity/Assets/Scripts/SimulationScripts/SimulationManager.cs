using System;
using System.Collections.Generic;
using Configuration;
using Robotics;
using Simulation.CoordinationStrategies;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace Simulation
{
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
        private RobotController[] _robotControllers;

        // State management
        private SimulationState _currentState = SimulationState.Paused;
        private SimulationState _previousState = SimulationState.Paused;

        // Robot coordination
        private ICoordinationStrategy _coordinationStrategy;
        private Dictionary<string, bool> _robotTargetReached = new Dictionary<string, bool>();

        // Events
        public event System.Action<SimulationState, SimulationState> OnStateChanged;

        // Properties
        public SimulationState CurrentState => _currentState;
        public bool IsRunning => _currentState == SimulationState.Running;
        public bool IsPaused => _currentState == SimulationState.Paused;
        public bool ShouldStopRobots => _currentState != SimulationState.Running;

        // Helper variables
        private const string _logPrefix = "[SIMULATION_MANAGER]";

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
                    Debug.LogWarning(
                        $"{_logPrefix} SimulationConfig not assigned. Creating default configuration."
                    );
                    config = ScriptableObject.CreateInstance<SimulationConfig>();
                }

                // Apply performance settings
                Application.targetFrameRate = config.targetFrameRate;
                QualitySettings.vSyncCount = config.enableVSync ? 1 : 0;
                Time.timeScale = config.timeScale;

                ChangeState(SimulationState.Initializing);

                Debug.Log(
                    $"{_logPrefix} Initialized: {config.coordinationMode} mode, {config.targetFrameRate}fps"
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
                // Find all robot controllers
                _robotControllers = FindObjectsByType<RobotController>(
                    FindObjectsInactive.Exclude,
                    FindObjectsSortMode.None
                );

                if (_robotControllers.Length == 0)
                {
                    Debug.LogWarning($"{_logPrefix} No RobotController components found in scene");
                    // Continue with initialization - robots may be added dynamically
                }
                else
                {
                    // Initialize robot tracking
                    foreach (var robot in _robotControllers)
                    {
                        string robotId = robot.gameObject.name;
                        _robotTargetReached[robotId] = true; // Start with no active targets
                    }
                }

                // Initialize coordination strategy based on config
                InitializeCoordinationStrategy();

                // Log simulation start
                Debug.Log(
                    $"{_logPrefix} Initialized: {_robotControllers.Length} robots found. Mode: {config.coordinationMode}"
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
        /// Initializes the coordination strategy based on the current configuration.
        /// Uses the Strategy Pattern to allow different coordination modes.
        /// Phase 4: Now includes CollaborativeStrategy with Python verification
        /// </summary>
        private void InitializeCoordinationStrategy()
        {
            _coordinationStrategy = config.coordinationMode switch
            {
                RobotCoordinationMode.Sequential => new SequentialStrategy(),
                RobotCoordinationMode.Independent => new IndependentStrategy(),
                RobotCoordinationMode.Collaborative => new CollaborativeStrategy(), // Phase 4: Implemented
                RobotCoordinationMode.MasterSlave => new IndependentStrategy(), // TODO: Implement MasterSlaveStrategy
                RobotCoordinationMode.Distributed => new IndependentStrategy(), // TODO: Implement DistributedStrategy
                _ => new IndependentStrategy(),
            };

            Debug.Log(
                $"{_logPrefix} Initialized coordination strategy: {_coordinationStrategy.GetType().Name}"
            );

            // Phase 4: Log workspace manager status
            if (config.coordinationMode == RobotCoordinationMode.Collaborative)
            {
                if (WorkspaceManager.Instance != null)
                {
                    Debug.Log(
                        $"{_logPrefix} [Phase 4] WorkspaceManager active for collaborative coordination"
                    );
                }
                else
                {
                    Debug.LogWarning(
                        $"{_logPrefix} [Phase 4] WorkspaceManager not found! "
                            + "Collaborative mode will operate without workspace management. "
                            + "Add WorkspaceManager GameObject to scene for full Phase 4 features."
                    );
                }
            }
        }

        /// <summary>
        /// Updates robot coordination using the current strategy.
        /// Delegates to the strategy pattern for mode-specific behavior.
        /// </summary>
        private void UpdateRobotCoordination()
        {
            if (
                !IsRunning
                || _robotControllers == null
                || _robotControllers.Length == 0
                || _coordinationStrategy == null
            )
                return;

            // Delegate coordination logic to the strategy
            _coordinationStrategy.Update(_robotControllers, _robotTargetReached);
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

            Debug.Log($"{_logPrefix} State: {_previousState} -> {newState}");
        }

        /// <summary>
        /// Handles simulation errors by changing state to error and optionally scheduling a reset.
        /// </summary>
        /// <param name="errorMessage">The error message to log</param>
        private void HandleError(string errorMessage)
        {
            ChangeState(SimulationState.Error);

            Debug.LogError($"{_logPrefix} Error: {errorMessage}");

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
                Debug.LogWarning(
                    $"{_logPrefix} Cannot start simulation while in error state. Reset first."
                );
                return;
            }

            ChangeState(SimulationState.Running);

            Debug.Log($"{_logPrefix} Started by user request");
        }

        /// <summary>
        /// Pauses the simulation if currently running.
        /// </summary>
        public void PauseSimulation()
        {
            if (IsRunning)
            {
                ChangeState(SimulationState.Paused);

                Debug.Log($"{_logPrefix} Paused");
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

                Debug.Log($"{_logPrefix} Resumed");
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
                // Reset all robots (if any exist)
                if (_robotControllers != null && _robotControllers.Length > 0)
                {
                    foreach (var robot in _robotControllers)
                    {
                        if (robot != null)
                        {
                            robot.ResetJointTargets();
                            _robotTargetReached[robot.gameObject.name] = true;
                        }
                    }
                }

                // Reset coordination strategy
                _coordinationStrategy?.Reset();

                Debug.Log($"{_logPrefix} Reset completed");

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
        /// <summary>
        /// Gets the ID of the currently active robot based on the coordination strategy.
        /// </summary>
        /// <returns>The active robot ID, or descriptive string based on strategy</returns>
        public string GetActiveRobotId()
        {
            if (_coordinationStrategy == null)
                return "None";

            return _coordinationStrategy.GetActiveRobotId();
        }

        /// <summary>
        /// Checks if a specific robot is allowed to move based on the coordination strategy.
        /// </summary>
        /// <param name="robotId">The robot identifier to check</param>
        /// <returns>True if the robot is allowed to move, false otherwise</returns>
        public bool IsRobotActive(string robotId)
        {
            if (!IsRunning || _coordinationStrategy == null)
                return false;

            // Delegate to the coordination strategy
            return _coordinationStrategy.IsRobotActive(robotId);
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
                Debug.Log($"{_logPrefix} Robot {robotId} reached target in sequential mode");
            }
        }

        /// <summary>
        /// Unity OnDestroy callback - cleans up singleton instance.
        /// </summary>
        private void OnDestroy()
        {
            if (Instance == this)
            {
                Debug.Log($"{_logPrefix} SimulationManager destroyed");

                Instance = null;
            }
        }
    }
}
