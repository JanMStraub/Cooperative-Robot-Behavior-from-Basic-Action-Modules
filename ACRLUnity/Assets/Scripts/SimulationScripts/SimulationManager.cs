using System;
using System.Collections;
using System.Collections.Generic;
using Configuration;
using Robotics;
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
            EditorGUILayout.LabelField("Active Robot: All");

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

        private RobotController[] _robotControllers;
        private SimulationState _currentState = SimulationState.Paused;
        private SimulationState _previousState = SimulationState.Paused;

        /// <summary>
        /// Set to true in Start() when a fatal configuration error prevents
        /// the simulation from running (e.g. no robots found).
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

                Debug.Log(
                    $"{_logPrefix} Initialized: {totalRobots} robots found"
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
        /// Unity Update callback - guards against bad initialization state.
        /// </summary>
        private void Update()
        {
            if (_initializationFailed || !IsRunning || _robotControllers == null)
            {
                return;
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
        /// Notifies the manager that a robot has reached or is moving towards its target.
        /// Works for both Unity IK and ROS trajectory paths.
        /// </summary>
        /// <param name="robotId">The robot identifier</param>
        /// <param name="reached">True if target is reached, false if still moving</param>
        public void NotifyTargetReached(string robotId, bool reached)
        {
            _robotTargetReached[robotId] = reached;
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
