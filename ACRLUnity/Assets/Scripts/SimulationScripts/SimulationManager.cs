using System;
using System.Collections;
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

            EditorGUILayout.LabelField($"Current State: {manager.CurrentState}");
            EditorGUILayout.LabelField($"Active Robot: {manager.GetActiveRobotId()}");

            EditorGUILayout.Space();

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
        public static SimulationManager Instance { get; private set; }

        [Header("Configuration")]
        [SerializeField]
        public SimulationConfig config;

        [SerializeField]
        private CoordinationConfig _coordinationConfig;

        private RobotController[] _robotControllers;
        private SimulationState _currentState = SimulationState.Paused;
        private SimulationState _previousState = SimulationState.Paused;
        private ICoordinationStrategy _coordinationStrategy;

        /// <summary>
        /// Set to true in Start() when a fatal configuration error prevents
        /// the simulation from running (e.g. no robots found, strategy null).
        /// Guards Update() and other periodic methods from executing with bad state.
        /// </summary>
        private bool _initializationFailed = false;
        private Dictionary<string, bool> _robotTargetReached = new Dictionary<string, bool>();
        private Dictionary<
            ArticulationBody,
            (Vector3 position, Quaternion rotation)
        > _initialRobotPoses = new Dictionary<ArticulationBody, (Vector3, Quaternion)>();
        private Coroutine _activeResetCoroutine;

        public event System.Action<SimulationState, SimulationState> OnStateChanged;

        public SimulationState CurrentState => _currentState;
        public bool IsRunning => _currentState == SimulationState.Running;
        public bool IsPaused => _currentState == SimulationState.Paused;
        public bool ShouldStopRobots => _currentState != SimulationState.Running;

        /// <summary>
        /// Cached ROSControlModeManagers found in the scene.
        /// Populated during Start() alongside _robotControllers.
        /// </summary>
        private Dictionary<string, ROSControlModeManager> _rosControlModeManagers =
            new Dictionary<string, ROSControlModeManager>();

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
                if (config == null)
                {
                    Debug.LogWarning(
                        $"{_logPrefix} SimulationConfig not assigned. Creating default configuration."
                    );
                    config = ScriptableObject.CreateInstance<SimulationConfig>();
                }

                Application.targetFrameRate = config.targetFrameRate;
                QualitySettings.vSyncCount = config.enableVSync ? 1 : 0;
                Time.timeScale = config.timeScale;

                ChangeState(SimulationState.Initializing);
            }
            catch (Exception ex)
            {
                HandleError($"Failed to initialize simulation: {ex.Message}");
            }
        }

        /// <summary>
        /// Initializes component references and robot tracking.
        /// Sorts robots by name to ensure deterministic sequential execution order.
        /// </summary>
        private void Start()
        {
            try
            {
                var regularControllers = FindObjectsByType<RobotController>(
                    FindObjectsInactive.Exclude,
                    FindObjectsSortMode.None
                );

                System.Array.Sort(
                    regularControllers,
                    (a, b) =>
                        string.Compare(
                            a.gameObject.name,
                            b.gameObject.name,
                            System.StringComparison.Ordinal
                        )
                );

                _robotControllers = regularControllers;
                int totalRobots = regularControllers.Length;

                foreach (var robot in regularControllers)
                {
                    string robotId = robot.gameObject.name;
                    _robotTargetReached[robotId] = true;

                    ArticulationBody rootBody = robot.GetComponent<ArticulationBody>();
                    if (rootBody != null)
                    {
                        _initialRobotPoses[rootBody] = (
                            rootBody.transform.position,
                            rootBody.transform.rotation
                        );
                    }

                    // Cache ROSControlModeManager if present
                    var rosControlMode = robot.GetComponent<ROSControlModeManager>();
                    if (rosControlMode != null)
                    {
                        _rosControlModeManagers[robotId] = rosControlMode;
                    }
                }

                if (totalRobots == 0)
                {
                    _initializationFailed = true;
                    HandleError("No robots found. Scene may be misconfigured.");
                    return;
                }

                InitializeCoordinationStrategy();

                if (_coordinationStrategy == null)
                {
                    _initializationFailed = true;
                    HandleError("Coordination strategy failed to initialize.");
                    return;
                }

                Debug.Log(
                    $"{_logPrefix} Initialized: {totalRobots} robots found. Mode: {config.coordinationMode}"
                );

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
        /// Updates robot coordination each frame when simulation is running.
        /// </summary>
        private void Update()
        {
            if (_initializationFailed || !IsRunning || _coordinationStrategy == null || _robotControllers == null)
            {
                return;
            }

            UpdateRobotCoordination();
        }

        /// <summary>
        /// Initializes the coordination strategy based on the current configuration.
        /// </summary>
        private void InitializeCoordinationStrategy()
        {
            if (
                _coordinationConfig == null
                && config.coordinationMode == RobotCoordinationMode.Collaborative
            )
            {
                Debug.LogWarning(
                    $"{_logPrefix} CoordinationConfig not assigned. Creating default configuration."
                );
                _coordinationConfig = ScriptableObject.CreateInstance<CoordinationConfig>();
            }

            _coordinationStrategy = config.coordinationMode switch
            {
                RobotCoordinationMode.Sequential => new SequentialStrategy(),
                RobotCoordinationMode.Independent => new IndependentStrategy(),
                RobotCoordinationMode.Collaborative => new CollaborativeStrategy(
                    _coordinationConfig
                ),
                RobotCoordinationMode.MasterSlave => LogFallbackAndReturn(
                    "MasterSlave coordination mode is not implemented. Falling back to Independent.",
                    new IndependentStrategy()
                ),
                RobotCoordinationMode.Distributed => LogFallbackAndReturn(
                    "Distributed coordination mode is not implemented. Falling back to Independent.",
                    new IndependentStrategy()
                ),
                RobotCoordinationMode.Negotiated => new NegotiatedStrategy(),
                _ => new IndependentStrategy(),
            };

            if (config.coordinationMode == RobotCoordinationMode.Collaborative)
            {
                if (WorkspaceManager.Instance != null)
                {
                    Debug.Log(
                        $"{_logPrefix} WorkspaceManager active for collaborative coordination"
                    );
                }
                else
                {
                    Debug.LogWarning(
                        $"{_logPrefix} WorkspaceManager not found! "
                            + "Collaborative mode will operate without workspace management. "
                            + "Add WorkspaceManager GameObject to scene for full features."
                    );
                }
            }
        }

        /// <summary>
        /// Emits a warning log and returns the given strategy. Used in switch expressions
        /// where unimplemented modes fall back to IndependentStrategy.
        /// </summary>
        private ICoordinationStrategy LogFallbackAndReturn(string message, ICoordinationStrategy fallback)
        {
            Debug.LogWarning($"{_logPrefix} {message}");
            return fallback;
        }

        /// <summary>
        /// Updates robot coordination using the current strategy.
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
        /// Handles simulation errors and optionally schedules a reset.
        /// </summary>
        /// <param name="errorMessage">The error message to log</param>
        private void HandleError(string errorMessage)
        {
            ChangeState(SimulationState.Error);

            Debug.LogError($"{_logPrefix} Error: {errorMessage}");

            if (config.resetOnError)
            {
                Invoke(nameof(ResetSimulation), 2f);
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

            // Re-initialize coordination strategy in case the mode was changed after Initialize()
            InitializeCoordinationStrategy();

            ChangeState(SimulationState.Running);
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
        /// Resets all robots to their initial poses and coordination state.
        /// Uses coroutine to ensure physics engine processes changes before resuming.
        /// </summary>
        public void ResetSimulation()
        {
            if (_activeResetCoroutine != null)
            {
                StopCoroutine(_activeResetCoroutine);
                _activeResetCoroutine = null;
            }

            _activeResetCoroutine = StartCoroutine(ResetSimulationCoroutine());
        }

        /// <summary>
        /// Performs physics-safe reset with proper frame timing.
        /// Waits for FixedUpdate to allow physics engine to process teleport before resuming.
        /// </summary>
        private IEnumerator ResetSimulationCoroutine()
        {
            ChangeState(SimulationState.Resetting);

            if (_robotControllers == null)
            {
                Debug.LogWarning($"{_logPrefix} Cannot reset - robot controllers not initialized");
                ChangeState(SimulationState.Paused);
                yield break;
            }

            if (_robotControllers.Length > 0)
            {
                foreach (var robot in _robotControllers)
                {
                    if (robot != null && robot.gameObject != null)
                    {
                        ArticulationBody rootBody = robot.GetComponent<ArticulationBody>();
                        if (rootBody != null)
                        {
                            if (_initialRobotPoses.TryGetValue(rootBody, out var initialPose))
                            {
                                rootBody.TeleportRoot(initialPose.position, initialPose.rotation);
                            }
                            else
                            {
                                Debug.LogWarning(
                                    $"{_logPrefix} No cached initial pose for {robot.gameObject.name}, "
                                        + "clearing velocities only"
                                );
                                rootBody.TeleportRoot(
                                    rootBody.transform.position,
                                    rootBody.transform.rotation
                                );
                            }
                        }

                        robot.ResetJointTargets();
                        string robotId = robot.gameObject.name;
                        _robotTargetReached[robotId] = true;
                    }
                }
            }

            yield return new WaitForFixedUpdate();

            _coordinationStrategy?.Reset();

            Debug.Log($"{_logPrefix} Reset completed");

            _activeResetCoroutine = null;

            if (config.autoStart)
            {
                StartSimulation();
            }
            else
            {
                ChangeState(SimulationState.Paused);
            }
        }

        /// <summary>
        /// Gets the ID of the currently active robot based on the coordination strategy.
        /// </summary>
        /// <returns>The active robot ID, or "None" if no active robot</returns>
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

            return _coordinationStrategy.IsRobotActive(robotId);
        }

        /// <summary>
        /// Notifies the manager that a robot has reached or is moving towards its target.
        /// Works for both Unity IK and ROS trajectory paths.
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
        /// Get the ROS control mode for a specific robot.
        /// Returns null if the robot doesn't have a ROSControlModeManager.
        /// </summary>
        public ControlMode? GetROSControlMode(string robotId)
        {
            if (_rosControlModeManagers.TryGetValue(robotId, out var manager))
                return manager.CurrentMode;
            return null;
        }

        /// <summary>
        /// Check if a robot is currently controlled by ROS (not Unity IK).
        /// Returns false if no ROSControlModeManager is attached.
        /// </summary>
        public bool IsRobotROSControlled(string robotId)
        {
            if (_rosControlModeManagers.TryGetValue(robotId, out var manager))
                return !manager.ShouldUnityIKBeActive;
            return false;
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
