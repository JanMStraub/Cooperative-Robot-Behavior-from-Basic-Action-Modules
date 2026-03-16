using System;
using System.Collections;
using System.Linq;
using Core;
using Robotics;
using Robotics.Grasp;
using UnityEngine;
using Vision;

namespace PythonCommunication
{
    /// <summary>
    /// Data structure for robot commands received from Python operations
    /// </summary>
    [System.Serializable]
    public class RobotCommand
    {
        public string command_type;
        public string target_type; // "robot" or "camera" - determines routing
        public string robot_id;
        public string camera_id; // For camera-targeted commands
        public CommandParameters parameters;
        public float timestamp;
        public uint request_id; // Protocol V2: for request/response correlation
    }

    [System.Serializable]
    public class CommandParameters
    {
        public TargetPosition target_position;
        public TargetPosition original_target;
        public float speed_multiplier;
        public float approach_offset;
        public bool detailed; // For status queries
        public bool open_gripper; // For gripper control

        // Grasp planning parameters
        public string object_id; // Object to grasp
        public bool use_advanced_planning; // Use full grasp planning pipeline
        public string preferred_approach; // "auto", "top", "front", "side"
        public float pre_grasp_distance; // Custom pre-grasp distance (0 = use config)
        public bool enable_retreat; // Whether to retreat after grasping
        public float retreat_distance; // Custom retreat distance (0 = use config)
        public TargetPosition custom_approach_vector; // Custom approach direction

        // New operation parameters (Phase 2 - Atomic Operations)
        public TargetPosition point_a; // For move_from_a_to_b
        public TargetPosition point_b; // For move_from_a_to_b
        public float roll; // For adjust_orientation
        public float pitch; // For adjust_orientation
        public float yaw; // For adjust_orientation
        public TargetPosition[] waypoints; // For follow_path
        public string alignment_type; // For align_object
        public TargetPosition target_orientation; // For align_object
        public string shape; // For draw_with_pen
        public TargetPosition pen_position; // For draw_with_pen
        public TargetPosition paper_position; // For draw_with_pen
        public string target_robot_id; // For mirror_movement
        public string mirror_axis; // For mirror_movement
        public float scale_factor; // For mirror_movement
        public int duration_ms; // For stabilize_object
        public float force_limit; // For stabilize_object

        // GraspNet: optional pre-computed neural grasp candidates.
        // When non-empty, GraspPlanningPipeline skips geometric candidate generation
        // and feeds these directly into IK/collision filtering.
        public GraspCandidateData[] precomputed_candidates;
    }

    /// <summary>
    /// A single neural grasp candidate produced by Contact-GraspNet and
    /// transformed into Unity world frame by the Python backend.
    /// Consumed by GraspPlanningPipeline.PlanGraspWithExternalCandidates().
    /// </summary>
    [System.Serializable]
    public class GraspCandidateData
    {
        public TargetPosition pre_grasp_position;
        public TargetRotation pre_grasp_rotation;
        public TargetPosition grasp_position;
        public TargetRotation grasp_rotation;
        public TargetPosition approach_direction;
        public float grasp_depth;
        public float antipodal_score;
        public float graspnet_score;
        public string approach_type;
    }

    /// <summary>
    /// Quaternion (x, y, z, w) for grasp rotation data from Python.
    /// A separate class is needed because TargetPosition only carries x/y/z.
    /// </summary>
    [System.Serializable]
    public class TargetRotation
    {
        public float x;
        public float y;
        public float z;
        public float w;
    }

    [System.Serializable]
    public class TargetPosition
    {
        public float x;
        public float y;
        public float z;
    }

    [System.Serializable]
    public class CommandCompletionData
    {
        public string type;
        public string robot_id;
        public string command_type;
        public bool success;
        public uint request_id;
        public float timestamp;
    }

    /// <summary>
    /// Handles commands from Python operations and executes them on Unity robots.
    ///
    /// This handler listens to SequenceClient for incoming commands from Python
    /// SequenceServer and executes them using RobotManager and RobotController.
    ///
    /// Supported Commands:
    /// - move_to_coordinate: Move robot end effector to target position
    /// - control_gripper: Open or close the gripper
    /// - check_robot_status: Get current robot state
    /// - return_to_start_position: Move robot back to initial position
    ///
    /// Usage:
    /// 1. Attach this component to a GameObject in your scene
    /// 2. Ensure SequenceClient and RobotManager are active
    /// 3. Python SequenceServer will send commands via port 5013
    /// </summary>
    public class PythonCommandHandler : MonoBehaviour
    {
        public static PythonCommandHandler Instance { get; private set; }

        [Header("Configuration")]
        [Tooltip("Enable detailed logging of command processing")]
        [SerializeField]
        private bool _verboseLogging = true;

        [Tooltip("Apply speed multiplier from commands")]
        [SerializeField]
        private bool _applySpeedMultiplier = true;

        [Header("Runtime Info")]
        [Tooltip("Number of commands processed successfully")]
        [SerializeField]
        private int _successfulCommands = 0;

        [Tooltip("Number of commands that failed")]
        [SerializeField]
        private int _failedCommands = 0;

        private RobotManager _robotManager;

        /// <summary>
        /// Track active commands for completion notification
        /// </summary>
        private System.Collections.Generic.Dictionary<string, uint> _activeCommands =
            new System.Collections.Generic.Dictionary<string, uint>();

        /// <summary>
        /// Manages active event listeners per robot to prevent zombie delegates.
        ///
        /// Replaces the four separate Dictionary&lt;string, Action&gt; fields that
        /// previously implemented identical subscribe/fire/clear patterns.
        /// </summary>
        private sealed class CommandListenerManager
        {
            private readonly System.Collections.Generic.Dictionary<string, System.Action> _listeners
                = new System.Collections.Generic.Dictionary<string, System.Action>();

            /// <summary>Register a callback for the given robot key, overwriting any prior entry.</summary>
            public void Register(string key, System.Action callback)
            {
                _listeners[key] = callback;
            }

            /// <summary>Return the stored callback without removing it, or null if absent.</summary>
            public System.Action Get(string key)
            {
                _listeners.TryGetValue(key, out System.Action cb);
                return cb;
            }

            /// <summary>Remove and return the stored callback, or null if absent.</summary>
            public System.Action Remove(string key)
            {
                if (_listeners.TryGetValue(key, out System.Action cb))
                {
                    _listeners.Remove(key);
                    return cb;
                }
                return null;
            }

            /// <summary>Return true if a callback is registered for the given key.</summary>
            public bool Contains(string key) => _listeners.ContainsKey(key);

            /// <summary>Remove all entries.</summary>
            public void ClearAll() => _listeners.Clear();
        }

        /// <summary>
        /// Per-command-type active listener managers — prevent zombie delegates on controller events.
        /// </summary>
        private readonly CommandListenerManager _activeMoveListeners = new CommandListenerManager();
        private readonly CommandListenerManager _activeGraspListeners = new CommandListenerManager();
        private readonly CommandListenerManager _activeGripperListeners = new CommandListenerManager();
        private readonly CommandListenerManager _activeOrientationListeners = new CommandListenerManager();

        /// <summary>
        /// Object lookup cache to avoid expensive FindObjectsByType calls
        /// </summary>
        private System.Collections.Generic.Dictionary<string, GameObject> _objectCache =
            new System.Collections.Generic.Dictionary<string, GameObject>();
        private const float OBJECT_CACHE_VALIDITY = 5.0f;
        private float _lastCacheRefreshTime = 0f;

        // Helper variable
        private const string _logPrefix = "[PYTHON_COMMAND_HANDLER]";

        #region Unity Lifecycle

        /// <summary>
        /// Initialize singleton and subscribe to events
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
                return;
            }
        }

        /// <summary>
        /// Subscribe to CommandReceiver events when component starts
        /// </summary>
        private void Start()
        {
            _robotManager = RobotManager.Instance;
            if (_robotManager == null)
            {
                Debug.LogError(
                    $"{_logPrefix} RobotManager.Instance is null! "
                        + "Ensure RobotManager GameObject is in the scene."
                );
                return;
            }

            Debug.Log($"{_logPrefix} Initialized and listening for Python commands");
        }

        #endregion

        #region Command Processing

        /// <summary>
        /// Public method to handle commands routed from UnifiedPythonReceiver
        /// </summary>
        public void HandleCommand(RobotCommand command)
        {
            HandlePythonCommand(command);
        }

        /// <summary>
        /// Handle incoming commands from Python SequenceServer
        /// </summary>
        private void HandlePythonCommand(RobotCommand command)
        {
            if (command == null)
                return;

            if (_verboseLogging)
            {
                Debug.Log(
                    $"{_logPrefix} [req={command.request_id}] Received command: {command.command_type}"
                );
            }

            ProcessCommand(command);
        }

        /// <summary>
        /// Process a validated robot command
        /// </summary>
        private void ProcessCommand(RobotCommand command)
        {
            // Determine target type for logging
            string targetInfo =
                command.target_type == "camera"
                    ? $"camera: {command.camera_id}"
                    : $"robot: {command.robot_id}";

            if (_verboseLogging)
            {
                Debug.Log(
                    $"{_logPrefix} Processing command: {command.command_type} for {targetInfo}"
                );
            }

            switch (command.command_type)
            {
                case "move_to_coordinate":
                    ExecuteMoveToCoordinate(command);
                    break;

                case "control_gripper":
                    ExecuteControlGripper(command);
                    break;

                case "check_robot_status":
                    ExecuteCheckRobotStatus(command);
                    break;

                case "return_to_start_position":
                    ExecuteReturnToStartPosition(command);
                    break;

                case "grasp_object":
                    ExecuteGraspObject(command);
                    break;

                case "release_object":
                    ExecuteReleaseObject(command);
                    break;

                case "move_from_a_to_b":
                    ExecuteMoveFromAToB(command);
                    break;

                case "adjust_end_effector_orientation":
                    ExecuteAdjustOrientation(command);
                    break;


                case "align_object":
                    ExecuteAlignObject(command);
                    break;

                case "follow_path":
                    ExecuteFollowPath(command);
                    break;

                case "draw_with_pen":
                    ExecuteDrawWithPen(command);
                    break;

                case "mirror_movement":
                    ExecuteMirrorMovement(command);
                    break;

                case "stabilize_object":
                    ExecuteStabilizeObject(command);
                    break;

                case "capture_stereo_images":
                    ExecuteCaptureSteroImages(command);
                    break;

                default:
                    Debug.LogWarning($"{_logPrefix} Unknown command type: {command.command_type}");
                    _failedCommands++;
                    break;
            }
        }

        #endregion

        #region Validation Helpers

        /// <summary>
        /// Validate and retrieve robot instance with controller.
        /// Handles error logging and failed command counting.
        /// Supports both RobotController and SimpleRobotController.
        /// </summary>
        /// <param name="robotId">Robot identifier</param>
        /// <param name="commandName">Command name for error messages</param>
        /// <param name="robotInstance">Output: robot instance if found</param>
        /// <param name="controller">Output: robot controller if found (null if using SimpleRobotController)</param>
        /// <returns>True if validation passed, false if failed</returns>
        private bool ValidateAndGetRobot(
            string robotId,
            string commandName,
            out RobotInstance robotInstance,
            out RobotController controller
        )
        {
            robotInstance = null;
            controller = null;

            if (string.IsNullOrEmpty(robotId))
            {
                Debug.LogError($"{_logPrefix} {commandName}: robot_id is null or empty");
                _failedCommands++;
                return false;
            }

            if (!_robotManager.RobotInstances.TryGetValue(robotId, out robotInstance))
            {
                Debug.LogError(
                    $"{_logPrefix} {commandName}: Robot '{robotId}' not found in RobotManager. "
                        + $"Available robots: {string.Join(", ", _robotManager.RobotInstances.Keys)}"
                );
                _failedCommands++;
                return false;
            }

            controller = robotInstance.controller;
            if (controller == null && robotInstance.simpleController == null)
            {
                Debug.LogError(
                    $"{_logPrefix} {commandName}: Robot '{robotId}' has no RobotController or SimpleRobotController component"
                );
                _failedCommands++;
                return false;
            }

            return true;
        }

        /// <summary>
        /// Check if the robot is in ROS control mode and should skip Unity execution.
        /// When in ROS mode, movement is handled by Python -> ROS -> MoveIt -> ROSTrajectorySubscriber.
        /// Unity sends an immediate completion response since it won't process the command itself.
        /// </summary>
        /// <param name="controller">The RobotController to check.</param>
        /// <param name="robotId">Robot ID for logging.</param>
        /// <param name="commandName">Command name for logging and completion.</param>
        /// <param name="requestId">Request ID for completion response.</param>
        /// <returns>True if ROS mode is active and Unity should skip execution.</returns>
        private bool ShouldSkipForROSMode(
            RobotController controller,
            string robotId,
            string commandName,
            uint requestId
        )
        {
            if (controller == null)
                return false;

            var rosMode = controller.GetComponent<ROSControlModeManager>();
            if (rosMode != null && !rosMode.ShouldUnityIKBeActive)
            {
                Debug.Log(
                    $"{_logPrefix} Robot {robotId} is in ROS mode. "
                        + $"Skipping Unity execution for {commandName} - handled by MoveIt via ROS."
                );
                SendCommandCompletion(robotId, commandName, true, requestId);
                _successfulCommands++;
                return true;
            }

            return false;
        }

        #endregion

        #region Command Implementations

        // TODO: Future enhancement - expose grasp planning parameters to Python
        // Example: Add to CommandParameters struct:
        //   public bool use_grasp_planning;
        //   public string grasp_approach;  // "top", "front", "side", or null for auto
        // Then pass to controller.SetTarget() via GraspOptions struct
        // This would enable LLM-driven grasp planning from Python operations

        /// <summary>
        /// Execute move_to_coordinate command
        /// Phase 4: Now includes Python CoordinationVerifier integration
        /// Supports both RobotController and SimpleRobotController
        /// When robot is in ROS control mode, movement is handled by MoveIt
        /// via ROSTrajectorySubscriber, not by Unity IK.
        /// </summary>
        private void ExecuteMoveToCoordinate(RobotCommand command)
        {
            try
            {
                // Validate robot and get controller
                if (
                    !ValidateAndGetRobot(
                        command.robot_id,
                        "move_to_coordinate",
                        out RobotInstance robotInstance,
                        out RobotController controller
                    )
                )
                {
                    return;
                }

                // Check if robot is in ROS control mode - if so, skip Unity IK
                // Movement will be handled by Python -> ROS -> MoveIt -> ROSTrajectorySubscriber
                if (
                    ShouldSkipForROSMode(
                        controller,
                        command.robot_id,
                        "move_to_coordinate",
                        command.request_id
                    )
                )
                    return;

                if (command.parameters == null || command.parameters.target_position == null)
                {
                    Debug.LogError(
                        $"{_logPrefix} move_to_coordinate: Missing parameters or target_position"
                    );
                    _failedCommands++;
                    return;
                }

                // FIX #4: Get target position using helper
                Vector3 targetPosition = TargetPositionToVector3(
                    command.parameters.target_position
                );

                if (_applySpeedMultiplier && command.parameters.speed_multiplier > 0)
                {
                    if (_verboseLogging)
                    {
                        Debug.Log(
                            $"{_logPrefix} Speed multiplier: {command.parameters.speed_multiplier:F2}"
                        );
                    }
                }

                string commandKey = $"move_{command.robot_id}_{command.request_id}";
                _activeCommands[commandKey] = command.request_id;

                // FIX #2: CLEANUP - Remove old listener for this robot if one exists
                string robotListenerKey = $"move_{command.robot_id}";
                if (_activeMoveListeners.Contains(robotListenerKey))
                {
                    System.Action oldListener = _activeMoveListeners.Remove(robotListenerKey);
                    if (controller != null)
                        controller.OnTargetReached -= oldListener;
                    else if (robotInstance.simpleController != null)
                        robotInstance.simpleController.OnTargetReached -= oldListener;

                    _activeMoveListeners.Remove(robotListenerKey);
                }

                System.Action onComplete = null;
                onComplete = () =>
                {
                    if (controller != null)
                        controller.OnTargetReached -= onComplete;
                    else if (robotInstance.simpleController != null)
                        robotInstance.simpleController.OnTargetReached -= onComplete;

                    _activeMoveListeners.Remove(robotListenerKey);

                    if (_activeCommands.ContainsKey(commandKey))
                    {
                        _activeCommands.Remove(commandKey);
                        SendCommandCompletion(
                            command.robot_id,
                            "move_to_coordinate",
                            true,
                            command.request_id
                        );
                    }
                };

                _activeMoveListeners.Register(robotListenerKey, onComplete);

                if (controller != null)
                {
                    controller.OnTargetReached += onComplete;
                }
                else if (robotInstance.simpleController != null)
                {
                    robotInstance.simpleController.OnTargetReached += onComplete;
                }

                if (controller != null)
                {
                    controller.SetTarget(targetPosition);
                }
                else if (robotInstance.simpleController != null)
                {
                    robotInstance.simpleController.SetTarget(targetPosition);
                }

                if (_verboseLogging)
                {
                    string offsetInfo =
                        command.parameters.approach_offset > 0
                            ? $" (offset: {command.parameters.approach_offset:F3}m)"
                            : "";
                    Debug.Log(
                        $"{_logPrefix} Moving {command.robot_id} to position: "
                            + $"({targetPosition.x:F3}, {targetPosition.y:F3}, {targetPosition.z:F3}){offsetInfo}"
                    );
                }

                _successfulCommands++;
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"{_logPrefix} Error executing move_to_coordinate: {ex.Message}\n{ex.StackTrace}"
                );
                _failedCommands++;
            }
        }

        /// <summary>
        /// Execute control_gripper command
        /// </summary>
        private void ExecuteControlGripper(RobotCommand command)
        {
            try
            {
                if (
                    !ValidateAndGetRobot(
                        command.robot_id,
                        "control_gripper",
                        out RobotInstance robotInstance,
                        out _
                    )
                )
                {
                    return;
                }

                if (command.parameters == null)
                {
                    Debug.LogError($"{_logPrefix} control_gripper: Missing parameters");
                    _failedCommands++;
                    return;
                }

                GripperController gripperController =
                    robotInstance.robotGameObject.GetComponentInChildren<GripperController>();
                if (gripperController == null)
                {
                    Debug.LogError(
                        $"{_logPrefix} control_gripper: Robot '{command.robot_id}' has no GripperController component"
                    );
                    _failedCommands++;
                    return;
                }

                string commandKey = $"gripper_{command.robot_id}_{command.request_id}";
                _activeCommands[commandKey] = command.request_id;

                // FIX #2: CLEANUP - Remove old gripper listener for this robot if one exists
                string listenerKey = $"gripper_{command.robot_id}";
                if (_activeGripperListeners.Contains(listenerKey))
                {
                    System.Action oldListener = _activeGripperListeners.Remove(listenerKey);
                    gripperController.OnGripperActionComplete -= oldListener;
                    _activeGripperListeners.Remove(listenerKey);
                }

                System.Action onComplete = null;
                onComplete = () =>
                {
                    gripperController.OnGripperActionComplete -= onComplete;
                    _activeGripperListeners.Remove(listenerKey); // FIX #2
                    if (_activeCommands.ContainsKey(commandKey))
                    {
                        _activeCommands.Remove(commandKey);
                        SendCommandCompletion(
                            command.robot_id,
                            "control_gripper",
                            true,
                            command.request_id
                        );
                    }
                };
                _activeGripperListeners.Register(listenerKey, onComplete); // FIX #2: Track it
                gripperController.OnGripperActionComplete += onComplete;

                bool openGripper = command.parameters.open_gripper;
                if (openGripper)
                {
                    gripperController.OpenGrippers();

                    RobotController controller =
                        robotInstance.robotGameObject.GetComponent<RobotController>();
                    if (controller != null)
                    {
                        controller.ClearTarget();
                        if (_verboseLogging)
                        {
                            Debug.Log($"{_logPrefix} Cleared robot target after opening gripper");
                        }
                    }
                }
                else
                {
                    if (!string.IsNullOrEmpty(command.parameters.object_id))
                    {
                        GameObject targetObj = GameObject.Find(command.parameters.object_id);
                        if (targetObj != null)
                        {
                            gripperController.SetTargetObject(targetObj);
                            Debug.Log(
                                $"{_logPrefix} control_gripper: Set target object '{command.parameters.object_id}' for attachment"
                            );
                        }
                        else
                        {
                            Debug.LogWarning(
                                $"{_logPrefix} control_gripper: Target object '{command.parameters.object_id}' not found"
                            );
                        }
                    }
                    gripperController.CloseGrippers();
                }

                if (_verboseLogging)
                {
                    string action = openGripper ? "Opening" : "Closing";
                    Debug.Log($"{_logPrefix} {action} gripper on {command.robot_id}");
                }

                _successfulCommands++;
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"{_logPrefix} Error executing control_gripper: {ex.Message}\n{ex.StackTrace}"
                );
                _failedCommands++;
            }
        }

        /// <summary>
        /// Execute grasp_object command using advanced grasp planning pipeline.
        /// Supports both RobotController (with GraspOptions) and SimpleRobotController (basic grasp).
        /// </summary>
        /// <param name="command">Robot command with grasp parameters</param>
        private void ExecuteGraspObject(RobotCommand command)
        {
            try
            {
                // Validate robot and get instance
                if (
                    !ValidateAndGetRobot(
                        command.robot_id,
                        "grasp_object",
                        out RobotInstance robotInstance,
                        out RobotController controller
                    )
                )
                {
                    return;
                }
                if (
                    ShouldSkipForROSMode(
                        controller,
                        command.robot_id,
                        "grasp_object",
                        command.request_id
                    )
                )
                    return;

                if (command.parameters == null)
                {
                    Debug.LogError($"{_logPrefix} grasp_object: Missing parameters");
                    _failedCommands++;
                    return;
                }

                string objectId = command.parameters.object_id;
                if (string.IsNullOrEmpty(objectId))
                {
                    Debug.LogError($"{_logPrefix} grasp_object: Missing object_id");
                    _failedCommands++;
                    return;
                }

                GameObject targetObject = FindObjectFlexible(objectId);
                if (targetObject == null)
                {
                    Debug.LogError(
                        $"{_logPrefix} grasp_object: Object '{objectId}' not found in scene. "
                            + "Tried exact match, case-insensitive, and snake_case->camelCase conversion."
                    );
                    _failedCommands++;
                    return;
                }

                string commandKey = $"grasp_{command.robot_id}_{command.request_id}";
                _activeCommands[commandKey] = command.request_id;

                // FIX #2: CLEANUP - Remove old grasp listener for this robot if one exists
                string robotGraspKey = $"grasp_{command.robot_id}";

                if (controller != null)
                {
                    if (_activeGraspListeners.Contains(robotGraspKey))
                    {
                        System.Action oldListener = _activeGraspListeners.Remove(robotGraspKey);
                        controller.OnTargetReached -= oldListener;
                        _activeGraspListeners.Remove(robotGraspKey);
                    }

                    GraspOptions options = new GraspOptions
                    {
                        useGraspPlanning = true,
                        openGripperOnSet = true,
                        closeGripperOnReach = true,
                        useAdvancedPlanning = command.parameters.use_advanced_planning,
                        approach = ParseApproachType(command.parameters.preferred_approach),
                        overridePreGraspDistance = command.parameters.pre_grasp_distance,
                        customApproachVector = ParseApproachVector(
                            command.parameters.custom_approach_vector
                        ),
                        graspConfig = null,
                    };

                    System.Action onComplete = null;
                    onComplete = () =>
                    {
                        controller.OnTargetReached -= onComplete;
                        _activeGraspListeners.Remove(robotGraspKey); // FIX #2
                        if (_activeCommands.ContainsKey(commandKey))
                        {
                            _activeCommands.Remove(commandKey);
                            SendCommandCompletion(
                                command.robot_id,
                                "grasp_object",
                                true,
                                command.request_id
                            );
                        }
                    };
                    _activeGraspListeners.Register(robotGraspKey, onComplete);
                    controller.OnTargetReached += onComplete;

                    // Branch: use GraspNet pre-computed candidates if provided,
                    // otherwise run the standard geometric GraspCandidateGenerator.
                    if (
                        command.parameters.precomputed_candidates != null
                        && command.parameters.precomputed_candidates.Length > 0
                    )
                    {
                        var externalCandidates = ConvertExternalCandidates(
                            command.parameters.precomputed_candidates
                        );
                        controller.SetTargetWithExternalCandidates(
                            targetObject,
                            externalCandidates,
                            options
                        );
                        if (_verboseLogging)
                            Debug.Log(
                                $"{_logPrefix} GraspNet grasp: {command.robot_id} -> {objectId} "
                                    + $"({command.parameters.precomputed_candidates.Length} candidates)"
                            );
                    }
                    else
                    {
                        controller.SetTarget(targetObject, options);
                        if (_verboseLogging)
                        {
                            string planningMode =
                                options.useAdvancedPlanning ? "Advanced" : "Standard";
                            Debug.Log(
                                $"{_logPrefix} {planningMode} grasp planning: {command.robot_id} -> {objectId}"
                            );
                        }
                    }
                }
                else if (robotInstance.simpleController != null)
                {
                    SimpleRobotController simpleController = robotInstance.simpleController;

                    if (_activeGraspListeners.Contains(robotGraspKey))
                    {
                        System.Action oldListener = _activeGraspListeners.Remove(robotGraspKey);
                        simpleController.OnTargetReached -= oldListener;
                        _activeGraspListeners.Remove(robotGraspKey);
                    }

                    System.Action onComplete = null;
                    onComplete = () =>
                    {
                        simpleController.OnTargetReached -= onComplete;
                        _activeGraspListeners.Remove(robotGraspKey); // FIX #2
                        if (_activeCommands.ContainsKey(commandKey))
                        {
                            _activeCommands.Remove(commandKey);
                            SendCommandCompletion(
                                command.robot_id,
                                "grasp_object",
                                true,
                                command.request_id
                            );
                        }
                    };
                    _activeGraspListeners.Register(robotGraspKey, onComplete);
                    simpleController.OnTargetReached += onComplete;

                    simpleController.SetTarget(targetObject);

                    if (_verboseLogging)
                    {
                        Debug.Log($"{_logPrefix} Simple grasp: {command.robot_id} -> {objectId}");
                    }
                }
                else
                {
                    Debug.LogError(
                        $"{_logPrefix} grasp_object: No valid controller found for robot '{command.robot_id}'"
                    );
                    _failedCommands++;
                    return;
                }

                _successfulCommands++;
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"{_logPrefix} Error executing grasp_object: {ex.Message}\n{ex.StackTrace}"
                );
                _failedCommands++;
            }
        }

        /// <summary>
        /// Parse approach type string to GraspApproach enum.
        /// </summary>
        /// <param name="approachStr">Approach string ("auto", "top", "front", "side")</param>
        /// <returns>GraspApproach enum value, or null for auto</returns>
        private GraspApproach? ParseApproachType(string approachStr)
        {
            if (string.IsNullOrEmpty(approachStr) || approachStr.ToLower() == "auto")
                return null;

            switch (approachStr.ToLower())
            {
                case "top":
                    return GraspApproach.Top;
                case "front":
                    return GraspApproach.Front;
                case "side":
                    return GraspApproach.Side;
                default:
                    Debug.LogWarning(
                        $"{_logPrefix} Unknown approach type '{approachStr}', using auto"
                    );
                    return null;
            }
        }

        /// <summary>
        /// Convert GraspCandidateData array from Python into GraspCandidate list for Unity pipeline.
        /// Preserves all scoring metadata so GraspScorer can rank candidates correctly.
        /// </summary>
        /// <param name="data">Serialized candidate data from Python GraspNet backend</param>
        /// <returns>List of GraspCandidate objects ready for GraspIKFilter</returns>
        private System.Collections.Generic.List<Robotics.Grasp.GraspCandidate> ConvertExternalCandidates(
            GraspCandidateData[] data
        )
        {
            var result = new System.Collections.Generic.List<Robotics.Grasp.GraspCandidate>(
                data.Length
            );
            foreach (var d in data)
            {
                if (d == null)
                    continue;

                Vector3 preGraspPos = new Vector3(
                    d.pre_grasp_position?.x ?? 0f,
                    d.pre_grasp_position?.y ?? 0f,
                    d.pre_grasp_position?.z ?? 0f
                );
                Quaternion preGraspRot =
                    d.pre_grasp_rotation != null
                        ? new Quaternion(
                            d.pre_grasp_rotation.x,
                            d.pre_grasp_rotation.y,
                            d.pre_grasp_rotation.z,
                            d.pre_grasp_rotation.w
                        )
                        : Quaternion.identity;

                Vector3 graspPos = new Vector3(
                    d.grasp_position?.x ?? 0f,
                    d.grasp_position?.y ?? 0f,
                    d.grasp_position?.z ?? 0f
                );
                Quaternion graspRot =
                    d.grasp_rotation != null
                        ? new Quaternion(
                            d.grasp_rotation.x,
                            d.grasp_rotation.y,
                            d.grasp_rotation.z,
                            d.grasp_rotation.w
                        )
                        : Quaternion.identity;

                Robotics.Grasp.GraspApproach approach = ParseApproachType(d.approach_type)
                    ?? Robotics.Grasp.GraspApproach.Top;

                var candidate = Robotics.Grasp.GraspCandidate.Create(
                    preGraspPos,
                    preGraspRot,
                    graspPos,
                    graspRot,
                    approach
                );

                // Carry over GraspNet-specific scoring metadata
                candidate.graspDepth = d.grasp_depth;
                candidate.antipodalScore = d.antipodal_score;
                // Store the raw GraspNet quality score in totalScore as initial estimate;
                // GraspScorer will overwrite this with its weighted composite score.
                candidate.totalScore = d.graspnet_score;

                result.Add(candidate);
            }
            return result;
        }

        /// <summary>
        /// Parse approach vector from TargetPosition.
        /// </summary>
        /// <param name="vector">Target position representing direction vector</param>
        /// <returns>Vector3 or null if not provided</returns>
        private Vector3? ParseApproachVector(TargetPosition vector)
        {
            if (vector == null)
                return null;

            return TargetPositionToVector3(vector); // FIX #3: Use helper
        }

        /// <summary>
        /// Execute release_object command - ATOMIC operation that ONLY opens gripper.
        /// Does NOT move the robot. For positioned release, chain with move_to_coordinate first.
        /// </summary>
        private void ExecuteReleaseObject(RobotCommand command)
        {
            try
            {
                if (
                    !ValidateAndGetRobot(
                        command.robot_id,
                        "release_object",
                        out RobotInstance robotInstance,
                        out _
                    )
                )
                {
                    return;
                }

                GripperController gripperController =
                    robotInstance.robotGameObject.GetComponentInChildren<GripperController>();
                if (gripperController == null)
                {
                    Debug.LogError(
                        $"{_logPrefix} release_object: Robot '{command.robot_id}' has no GripperController component"
                    );
                    _failedCommands++;
                    return;
                }

                string commandKey = $"release_object_{command.robot_id}_{command.request_id}";
                _activeCommands[commandKey] = command.request_id;

                System.Action onComplete = null;
                onComplete = () =>
                {
                    gripperController.OnGripperActionComplete -= onComplete;
                    if (_activeCommands.ContainsKey(commandKey))
                    {
                        _activeCommands.Remove(commandKey);
                        SendCommandCompletion(
                            command.robot_id,
                            "release_object",
                            true,
                            command.request_id
                        );
                    }
                };
                gripperController.OnGripperActionComplete += onComplete;

                gripperController.OpenGrippers();

                if (_verboseLogging)
                {
                    Debug.Log(
                        $"{_logPrefix} Releasing object (atomic): {command.robot_id} - gripper only, no movement"
                    );
                }

                _successfulCommands++;
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"{_logPrefix} Error executing release_object: {ex.Message}\n{ex.StackTrace}"
                );
                _failedCommands++;
            }
        }

        /// <summary>
        /// Execute move_from_a_to_b command - waypoint-based movement.
        /// Moves robot from point A to point B in a straight line trajectory.
        /// </summary>
        private void ExecuteMoveFromAToB(RobotCommand command)
        {
            try
            {
                if (
                    !ValidateAndGetRobot(
                        command.robot_id,
                        "move_from_a_to_b",
                        out RobotInstance robotInstance,
                        out RobotController controller
                    )
                )
                {
                    return;
                }
                if (
                    ShouldSkipForROSMode(
                        controller,
                        command.robot_id,
                        "move_from_a_to_b",
                        command.request_id
                    )
                )
                    return;
                if (
                    command.parameters == null
                    || command.parameters.point_a == null
                    || command.parameters.point_b == null
                )
                {
                    Debug.LogError(
                        $"{_logPrefix} move_from_a_to_b: Missing parameters or waypoints"
                    );
                    _failedCommands++;
                    return;
                }

                Vector3 pointA = TargetPositionToVector3(command.parameters.point_a);
                Vector3 pointB = TargetPositionToVector3(command.parameters.point_b);

                StartCoroutine(
                    MoveFromAToBCoroutine(
                        controller,
                        robotInstance.simpleController,
                        pointA,
                        pointB,
                        command.robot_id,
                        command.request_id
                    )
                );

                if (_verboseLogging)
                {
                    Debug.Log(
                        $"{_logPrefix} Moving {command.robot_id} from A({pointA.x:F3}, {pointA.y:F3}, {pointA.z:F3}) to B({pointB.x:F3}, {pointB.y:F3}, {pointB.z:F3})"
                    );
                }

                _successfulCommands++;
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"{_logPrefix} Error executing move_from_a_to_b: {ex.Message}\n{ex.StackTrace}"
                );
                _failedCommands++;
            }
        }

        /// <summary>
        /// Coroutine for move_from_a_to_b - moves through point A then to point B.
        /// </summary>
        private IEnumerator MoveFromAToBCoroutine(
            RobotController controller,
            SimpleRobotController simpleController,
            Vector3 pointA,
            Vector3 pointB,
            string robotId,
            uint requestId
        )
        {
            bool usingSimpleController = (controller == null && simpleController != null);
            float segmentTimeout = 15.0f;

            if (controller != null)
            {
                controller.SetTarget(pointA);
            }
            else if (simpleController != null)
            {
                simpleController.SetTarget(pointA);
            }

            float pointATimer = 0f;
            if (usingSimpleController)
            {
                while (!simpleController.HasReachedTarget)
                {
                    pointATimer += 0.1f;
                    if (pointATimer > segmentTimeout)
                    {
                        Debug.LogWarning(
                            $"{_logPrefix} MoveFromAToB: Timeout waiting for point A after {segmentTimeout}s. "
                                + $"Distance: {simpleController.DistanceToTarget:F4}m"
                        );
                        SendCommandCompletion(robotId, "move_from_a_to_b", false, requestId);
                        yield break;
                    }
                    yield return new WaitForSeconds(0.1f);
                }
            }
            else
            {
                while (
                    controller != null
                    && controller.GetDistanceToTarget() > RobotConstants.MOVEMENT_THRESHOLD
                )
                {
                    pointATimer += 0.1f;
                    if (pointATimer > segmentTimeout)
                    {
                        Debug.LogWarning(
                            $"{_logPrefix} MoveFromAToB: Timeout waiting for point A after {segmentTimeout}s. "
                                + $"Distance: {controller.GetDistanceToTarget():F4}m"
                        );
                        SendCommandCompletion(robotId, "move_from_a_to_b", false, requestId);
                        yield break;
                    }
                    yield return new WaitForSeconds(0.1f);
                }
            }

            // Move to point B
            if (controller != null)
            {
                controller.SetTarget(pointB);
            }
            else if (simpleController != null)
            {
                simpleController.SetTarget(pointB);
            }

            // FIX #1: Wait for arrival at point B with TIMEOUT
            float pointBTimer = 0f;
            if (usingSimpleController)
            {
                while (!simpleController.HasReachedTarget)
                {
                    pointBTimer += 0.1f;
                    if (pointBTimer > segmentTimeout)
                    {
                        Debug.LogWarning(
                            $"{_logPrefix} MoveFromAToB: Timeout waiting for point B after {segmentTimeout}s. "
                                + $"Distance: {simpleController.DistanceToTarget:F4}m"
                        );
                        SendCommandCompletion(robotId, "move_from_a_to_b", false, requestId);
                        yield break;
                    }
                    yield return new WaitForSeconds(0.1f);
                }
            }
            else
            {
                while (
                    controller != null
                    && controller.GetDistanceToTarget() > RobotConstants.MOVEMENT_THRESHOLD
                )
                {
                    pointBTimer += 0.1f;
                    if (pointBTimer > segmentTimeout)
                    {
                        Debug.LogWarning(
                            $"{_logPrefix} MoveFromAToB: Timeout waiting for point B after {segmentTimeout}s. "
                                + $"Distance: {controller.GetDistanceToTarget():F4}m"
                        );
                        SendCommandCompletion(robotId, "move_from_a_to_b", false, requestId);
                        yield break;
                    }
                    yield return new WaitForSeconds(0.1f);
                }
            }

            SendCommandCompletion(robotId, "move_from_a_to_b", true, requestId);
        }

        /// <summary>
        /// Execute adjust_end_effector_orientation command - rotate end effector without moving position.
        /// </summary>
        private void ExecuteAdjustOrientation(RobotCommand command)
        {
            try
            {
                if (
                    !ValidateAndGetRobot(
                        command.robot_id,
                        "adjust_end_effector_orientation",
                        out RobotInstance robotInstance,
                        out RobotController controller
                    )
                )
                {
                    return;
                }
                if (
                    ShouldSkipForROSMode(
                        controller,
                        command.robot_id,
                        "adjust_end_effector_orientation",
                        command.request_id
                    )
                )
                    return;

                if (command.parameters == null)
                {
                    Debug.LogError(
                        $"{_logPrefix} adjust_end_effector_orientation: Missing parameters"
                    );
                    _failedCommands++;
                    return;
                }

                Vector3 currentPosition =
                    controller != null
                        ? controller.GetCurrentEndEffectorPosition()
                        : robotInstance.simpleController.GetCurrentEndEffectorPosition();

                Quaternion targetRotation = Quaternion.Euler(
                    command.parameters.roll,
                    command.parameters.pitch,
                    command.parameters.yaw
                );

                string commandKey = $"adjust_orientation_{command.robot_id}_{command.request_id}";
                _activeCommands[commandKey] = command.request_id;

                // FIX #2: CLEANUP - Remove old orientation listener for this robot if one exists
                string listenerKey = $"orientation_{command.robot_id}";
                if (_activeOrientationListeners.Contains(listenerKey))
                {
                    System.Action oldListener = _activeOrientationListeners.Remove(listenerKey);
                    if (controller != null)
                        controller.OnTargetReached -= oldListener;
                    else if (robotInstance.simpleController != null)
                        robotInstance.simpleController.OnTargetReached -= oldListener;
                    _activeOrientationListeners.Remove(listenerKey);
                }

                System.Action onComplete = null;
                onComplete = () =>
                {
                    if (controller != null)
                        controller.OnTargetReached -= onComplete;
                    else if (robotInstance.simpleController != null)
                        robotInstance.simpleController.OnTargetReached -= onComplete;

                    _activeOrientationListeners.Remove(listenerKey); // FIX #2

                    if (_activeCommands.ContainsKey(commandKey))
                    {
                        _activeCommands.Remove(commandKey);
                        SendCommandCompletion(
                            command.robot_id,
                            "adjust_end_effector_orientation",
                            true,
                            command.request_id
                        );
                    }
                };

                _activeOrientationListeners.Register(listenerKey, onComplete);

                if (controller != null)
                {
                    controller.OnTargetReached += onComplete;
                    controller.SetTarget(currentPosition, targetRotation);
                }
                else if (robotInstance.simpleController != null)
                {
                    robotInstance.simpleController.OnTargetReached += onComplete;
                    robotInstance.simpleController.SetTarget(currentPosition, targetRotation);
                }

                if (_verboseLogging)
                {
                    Debug.Log(
                        $"{_logPrefix} Adjusting orientation for {command.robot_id}: roll={command.parameters.roll:F1}, pitch={command.parameters.pitch:F1}, yaw={command.parameters.yaw:F1}"
                    );
                }

                _successfulCommands++;
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"{_logPrefix} Error executing adjust_end_effector_orientation: {ex.Message}\n{ex.StackTrace}"
                );
                _failedCommands++;
            }
        }

        /// <summary>
        /// Execute align_object command - align held object to target orientation.
        /// </summary>
        private void ExecuteAlignObject(RobotCommand command)
        {
            try
            {
                // Validate robot and get controller
                if (
                    !ValidateAndGetRobot(
                        command.robot_id,
                        "align_object",
                        out RobotInstance robotInstance,
                        out RobotController controller
                    )
                )
                {
                    return;
                }
                if (
                    ShouldSkipForROSMode(
                        controller,
                        command.robot_id,
                        "align_object",
                        command.request_id
                    )
                )
                    return;

                // Validate parameters
                if (command.parameters == null || command.parameters.target_orientation == null)
                {
                    Debug.LogError(
                        $"{_logPrefix} align_object: Missing parameters or target_orientation"
                    );
                    _failedCommands++;
                    return;
                }

                // Get current position (maintain it while rotating)
                Vector3 currentPosition =
                    controller != null
                        ? controller.GetCurrentEndEffectorPosition()
                        : robotInstance.simpleController.GetCurrentEndEffectorPosition();

                // Parse target orientation
                Quaternion targetRotation = Quaternion.Euler(
                    command.parameters.target_orientation.x,
                    command.parameters.target_orientation.y,
                    command.parameters.target_orientation.z
                );

                // Subscribe to completion event
                string commandKey = $"align_object_{command.robot_id}_{command.request_id}";
                _activeCommands[commandKey] = command.request_id;

                // FIX #2: CLEANUP - Remove old alignment listener for this robot if one exists
                string listenerKey = $"align_{command.robot_id}";
                if (_activeOrientationListeners.Contains(listenerKey))
                {
                    System.Action oldListener = _activeOrientationListeners.Remove(listenerKey);
                    if (controller != null)
                        controller.OnTargetReached -= oldListener;
                    else if (robotInstance.simpleController != null)
                        robotInstance.simpleController.OnTargetReached -= oldListener;
                    _activeOrientationListeners.Remove(listenerKey);
                }

                System.Action onComplete = null;
                onComplete = () =>
                {
                    if (controller != null)
                        controller.OnTargetReached -= onComplete;
                    else if (robotInstance.simpleController != null)
                        robotInstance.simpleController.OnTargetReached -= onComplete;

                    _activeOrientationListeners.Remove(listenerKey); // FIX #2

                    if (_activeCommands.ContainsKey(commandKey))
                    {
                        _activeCommands.Remove(commandKey);
                        SendCommandCompletion(
                            command.robot_id,
                            "align_object",
                            true,
                            command.request_id
                        );
                    }
                };

                _activeOrientationListeners.Register(listenerKey, onComplete); // FIX #2: Track it

                // Execute alignment
                if (controller != null)
                {
                    controller.OnTargetReached += onComplete;
                    controller.SetTarget(currentPosition, targetRotation);
                }
                else if (robotInstance.simpleController != null)
                {
                    robotInstance.simpleController.OnTargetReached += onComplete;
                    robotInstance.simpleController.SetTarget(currentPosition, targetRotation);
                }

                if (_verboseLogging)
                {
                    Debug.Log($"{_logPrefix} Aligning object for {command.robot_id}");
                }

                _successfulCommands++;
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"{_logPrefix} Error executing align_object: {ex.Message}\n{ex.StackTrace}"
                );
                _failedCommands++;
            }
        }

        /// <summary>
        /// Execute follow_path command - multi-waypoint trajectory following.
        /// </summary>
        private void ExecuteFollowPath(RobotCommand command)
        {
            try
            {
                // Validate robot and get controller
                if (
                    !ValidateAndGetRobot(
                        command.robot_id,
                        "follow_path",
                        out RobotInstance robotInstance,
                        out RobotController controller
                    )
                )
                {
                    return;
                }
                if (
                    ShouldSkipForROSMode(
                        controller,
                        command.robot_id,
                        "follow_path",
                        command.request_id
                    )
                )
                    return;

                // Validate parameters
                if (
                    command.parameters == null
                    || command.parameters.waypoints == null
                    || command.parameters.waypoints.Length == 0
                )
                {
                    Debug.LogError($"{_logPrefix} follow_path: Missing waypoints");
                    _failedCommands++;
                    return;
                }

                // FIX #3: Convert waypoints to Vector3 array using helper
                Vector3[] waypoints = new Vector3[command.parameters.waypoints.Length];
                for (int i = 0; i < command.parameters.waypoints.Length; i++)
                {
                    waypoints[i] = TargetPositionToVector3(command.parameters.waypoints[i]);
                }

                // Start coroutine for path following
                StartCoroutine(
                    FollowPathCoroutine(
                        controller,
                        robotInstance.simpleController,
                        waypoints,
                        command.robot_id,
                        command.request_id
                    )
                );

                if (_verboseLogging)
                {
                    Debug.Log(
                        $"{_logPrefix} Following path with {waypoints.Length} waypoints for {command.robot_id}"
                    );
                }

                _successfulCommands++;
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"{_logPrefix} Error executing follow_path: {ex.Message}\n{ex.StackTrace}"
                );
                _failedCommands++;
            }
        }

        /// <summary>
        /// Coroutine for follow_path - moves through all waypoints sequentially.
        /// </summary>
        private IEnumerator FollowPathCoroutine(
            RobotController controller,
            SimpleRobotController simpleController,
            Vector3[] waypoints,
            string robotId,
            uint requestId
        )
        {
            bool usingSimpleController = (controller == null && simpleController != null);
            float segmentTimeout = 15.0f; // FIX #1: Timeout per waypoint

            // Move through each waypoint
            for (int i = 0; i < waypoints.Length; i++)
            {
                Vector3 waypoint = waypoints[i];

                // Set target
                if (controller != null)
                {
                    controller.SetTarget(waypoint);
                }
                else if (simpleController != null)
                {
                    simpleController.SetTarget(waypoint);
                }

                // FIX #1: Wait for arrival with TIMEOUT
                float timer = 0f;
                if (usingSimpleController)
                {
                    while (!simpleController.HasReachedTarget)
                    {
                        timer += 0.1f;
                        if (timer > segmentTimeout)
                        {
                            Debug.LogWarning(
                                $"{_logPrefix} FollowPath: Timeout waiting for waypoint {i} after {segmentTimeout}s. "
                                    + $"Distance: {simpleController.DistanceToTarget:F4}m"
                            );
                            SendCommandCompletion(robotId, "follow_path", false, requestId);
                            yield break;
                        }
                        yield return new WaitForSeconds(0.1f);
                    }
                }
                else
                {
                    while (
                        controller != null
                        && controller.GetDistanceToTarget() > RobotConstants.MOVEMENT_THRESHOLD
                    )
                    {
                        timer += 0.1f;
                        if (timer > segmentTimeout)
                        {
                            Debug.LogWarning(
                                $"{_logPrefix} FollowPath: Timeout waiting for waypoint {i} after {segmentTimeout}s. "
                                    + $"Distance: {controller.GetDistanceToTarget():F4}m"
                            );
                            SendCommandCompletion(robotId, "follow_path", false, requestId);
                            yield break;
                        }
                        yield return new WaitForSeconds(0.1f);
                    }
                }
            }

            // Send completion after all waypoints reached
            SendCommandCompletion(robotId, "follow_path", true, requestId);
        }

        /// <summary>
        /// Execute draw_with_pen command - tool manipulation for drawing.
        /// </summary>
        private void ExecuteDrawWithPen(RobotCommand command)
        {
            try
            {
                // Validate robot and get controller
                if (
                    !ValidateAndGetRobot(
                        command.robot_id,
                        "draw_with_pen",
                        out RobotInstance robotInstance,
                        out RobotController controller
                    )
                )
                {
                    return;
                }
                if (
                    ShouldSkipForROSMode(
                        controller,
                        command.robot_id,
                        "draw_with_pen",
                        command.request_id
                    )
                )
                    return;

                // Validate parameters
                if (
                    command.parameters == null
                    || command.parameters.pen_position == null
                    || command.parameters.paper_position == null
                )
                {
                    Debug.LogError(
                        $"{_logPrefix} draw_with_pen: Missing pen_position or paper_position"
                    );
                    _failedCommands++;
                    return;
                }

                // FIX #3: Use helper for Vector3 conversion
                Vector3 penPosition = TargetPositionToVector3(command.parameters.pen_position);
                Vector3 paperPosition = TargetPositionToVector3(command.parameters.paper_position);

                string shape = command.parameters.shape ?? "line";

                // Start coroutine for drawing operation
                StartCoroutine(
                    DrawWithPenCoroutine(
                        controller,
                        robotInstance.simpleController,
                        penPosition,
                        paperPosition,
                        shape,
                        command.robot_id,
                        command.request_id
                    )
                );

                if (_verboseLogging)
                {
                    Debug.Log($"{_logPrefix} Drawing {shape} with pen for {command.robot_id}");
                }

                _successfulCommands++;
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"{_logPrefix} Error executing draw_with_pen: {ex.Message}\n{ex.StackTrace}"
                );
                _failedCommands++;
            }
        }

        /// <summary>
        /// Coroutine for draw_with_pen - pick up pen, move to paper, draw shape.
        /// </summary>
        private IEnumerator DrawWithPenCoroutine(
            RobotController controller,
            SimpleRobotController simpleController,
            Vector3 penPosition,
            Vector3 paperPosition,
            string shape,
            string robotId,
            uint requestId
        )
        {
            float timeout = 10.0f; // Safety timeout
            float timer = 0f;
            bool usingSimpleController = (controller == null && simpleController != null);

            // Step 1: Move to pen
            if (controller != null)
            {
                controller.SetTarget(penPosition);
            }
            else if (simpleController != null)
            {
                simpleController.SetTarget(penPosition);
            }

            // FIX: Wait for arrival at pen with TIMEOUT
            if (usingSimpleController)
            {
                while (!simpleController.HasReachedTarget)
                {
                    timer += 0.1f;
                    if (timer > timeout)
                    {
                        Debug.LogWarning($"{_logPrefix} DrawWithPen timed out reaching pen.");
                        SendCommandCompletion(robotId, "draw_with_pen", false, requestId);
                        yield break;
                    }
                    yield return new WaitForSeconds(0.1f);
                }
            }
            else
            {
                while (
                    controller != null
                    && controller.GetDistanceToTarget() > RobotConstants.MOVEMENT_THRESHOLD
                )
                {
                    timer += 0.1f;
                    if (timer > timeout)
                    {
                        Debug.LogWarning($"{_logPrefix} DrawWithPen timed out reaching pen.");
                        SendCommandCompletion(robotId, "draw_with_pen", false, requestId);
                        yield break;
                    }
                    yield return new WaitForSeconds(0.1f);
                }
            }

            // Step 2: Close gripper (pick up pen)
            // Get gripper controller
            GripperController gripperController = null;
            if (controller != null)
            {
                gripperController =
                    controller.gameObject.GetComponentInChildren<GripperController>();
            }
            else if (simpleController != null)
            {
                gripperController =
                    simpleController.gameObject.GetComponentInChildren<GripperController>();
            }

            if (gripperController != null)
            {
                gripperController.CloseGrippers();
                yield return new WaitForSeconds(0.5f); // Wait for gripper to close
            }

            // Step 3: Move to paper
            if (controller != null)
            {
                controller.SetTarget(paperPosition);
            }
            else if (simpleController != null)
            {
                simpleController.SetTarget(paperPosition);
            }

            // FIX: Wait for arrival at paper with TIMEOUT
            timer = 0f; // Reset timer for paper segment
            if (usingSimpleController)
            {
                while (!simpleController.HasReachedTarget)
                {
                    timer += 0.1f;
                    if (timer > timeout)
                    {
                        Debug.LogWarning($"{_logPrefix} DrawWithPen timed out moving to paper.");
                        SendCommandCompletion(robotId, "draw_with_pen", false, requestId);
                        yield break;
                    }
                    yield return new WaitForSeconds(0.1f);
                }
            }
            else
            {
                while (
                    controller != null
                    && controller.GetDistanceToTarget() > RobotConstants.MOVEMENT_THRESHOLD
                )
                {
                    timer += 0.1f;
                    if (timer > timeout)
                    {
                        Debug.LogWarning($"{_logPrefix} DrawWithPen timed out moving to paper.");
                        SendCommandCompletion(robotId, "draw_with_pen", false, requestId);
                        yield break;
                    }
                    yield return new WaitForSeconds(0.1f);
                }
            }

            // Step 4: Draw shape (placeholder - actual drawing would require LineRenderer or similar)
            Debug.Log($"Drawing {shape} at paper position");
            yield return new WaitForSeconds(1.0f);

            // Send completion
            SendCommandCompletion(robotId, "draw_with_pen", true, requestId);
        }

        /// <summary>
        /// Execute mirror_movement command - mirror movements of another robot.
        /// </summary>
        private void ExecuteMirrorMovement(RobotCommand command)
        {
            try
            {
                // Validate robot and get controller
                if (
                    !ValidateAndGetRobot(
                        command.robot_id,
                        "mirror_movement",
                        out RobotInstance robotInstance,
                        out RobotController controller
                    )
                )
                {
                    return;
                }
                if (
                    ShouldSkipForROSMode(
                        controller,
                        command.robot_id,
                        "mirror_movement",
                        command.request_id
                    )
                )
                    return;

                // Validate parameters
                if (
                    command.parameters == null
                    || string.IsNullOrEmpty(command.parameters.target_robot_id)
                )
                {
                    Debug.LogError($"{_logPrefix} mirror_movement: Missing target_robot_id");
                    _failedCommands++;
                    return;
                }

                // Get target robot
                if (
                    !_robotManager.RobotInstances.TryGetValue(
                        command.parameters.target_robot_id,
                        out RobotInstance targetRobotInstance
                    )
                )
                {
                    Debug.LogError(
                        $"{_logPrefix} mirror_movement: Target robot '{command.parameters.target_robot_id}' not found"
                    );
                    _failedCommands++;
                    return;
                }

                string mirrorAxis = command.parameters.mirror_axis ?? "x";
                float scaleFactor =
                    command.parameters.scale_factor > 0 ? command.parameters.scale_factor : 1.0f;

                // Start coroutine for mirroring
                StartCoroutine(
                    MirrorMovementCoroutine(
                        controller,
                        robotInstance.simpleController,
                        targetRobotInstance,
                        mirrorAxis,
                        scaleFactor,
                        command.robot_id,
                        command.request_id
                    )
                );

                if (_verboseLogging)
                {
                    Debug.Log(
                        $"{_logPrefix} Mirroring movement of {command.parameters.target_robot_id} on {command.robot_id} (axis: {mirrorAxis}, scale: {scaleFactor})"
                    );
                }

                _successfulCommands++;
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"{_logPrefix} Error executing mirror_movement: {ex.Message}\n{ex.StackTrace}"
                );
                _failedCommands++;
            }
        }

        /// <summary>
        /// Coroutine for mirror_movement - continuously mirror target robot's movements.
        /// </summary>
        private IEnumerator MirrorMovementCoroutine(
            RobotController controller,
            SimpleRobotController simpleController,
            RobotInstance targetRobotInstance,
            string mirrorAxis,
            float scaleFactor,
            string robotId,
            uint requestId
        )
        {
            // Mirror for a fixed duration (or until stopped by another command)
            float duration = 10.0f; // Default 10 seconds
            float elapsed = 0f;

            while (elapsed < duration)
            {
                // Get target robot position
                Vector3 targetPosition = Vector3.zero;
                if (targetRobotInstance.controller != null)
                {
                    targetPosition = targetRobotInstance.controller.GetCurrentEndEffectorPosition();
                }
                else if (targetRobotInstance.simpleController != null)
                {
                    targetPosition =
                        targetRobotInstance.simpleController.GetCurrentEndEffectorPosition();
                }

                // Mirror position based on axis
                Vector3 mirroredPosition = targetPosition;
                if (mirrorAxis == "x")
                {
                    mirroredPosition.x = -mirroredPosition.x * scaleFactor;
                }
                else if (mirrorAxis == "y")
                {
                    mirroredPosition.y = -mirroredPosition.y * scaleFactor;
                }
                else if (mirrorAxis == "z")
                {
                    mirroredPosition.z = -mirroredPosition.z * scaleFactor;
                }
                else if (mirrorAxis == "none")
                {
                    mirroredPosition = targetPosition * scaleFactor;
                }

                // Set mirrored position
                if (controller != null)
                {
                    controller.SetTarget(mirroredPosition);
                }
                else if (simpleController != null)
                {
                    simpleController.SetTarget(mirroredPosition);
                }

                elapsed += 0.1f;
                yield return new WaitForSeconds(0.1f);
            }

            // Send completion
            SendCommandCompletion(robotId, "mirror_movement", true, requestId);
        }

        /// <summary>
        /// Execute stabilize_object command - hold object stable with force control.
        /// </summary>
        private void ExecuteStabilizeObject(RobotCommand command)
        {
            try
            {
                // Validate robot and get instance
                if (
                    !ValidateAndGetRobot(
                        command.robot_id,
                        "stabilize_object",
                        out RobotInstance robotInstance,
                        out RobotController controller
                    )
                )
                {
                    return;
                }
                if (
                    ShouldSkipForROSMode(
                        controller,
                        command.robot_id,
                        "stabilize_object",
                        command.request_id
                    )
                )
                    return;

                // Validate parameters
                if (command.parameters == null)
                {
                    Debug.LogError($"{_logPrefix} stabilize_object: Missing parameters");
                    _failedCommands++;
                    return;
                }

                int duration_ms =
                    command.parameters.duration_ms > 0 ? command.parameters.duration_ms : 5000;
                float force_limit =
                    command.parameters.force_limit > 0 ? command.parameters.force_limit : 10.0f;

                // Get gripper controller
                GripperController gripperController =
                    robotInstance.robotGameObject.GetComponentInChildren<GripperController>();
                if (gripperController == null)
                {
                    Debug.LogError(
                        $"{_logPrefix} stabilize_object: Robot '{command.robot_id}' has no GripperController component"
                    );
                    _failedCommands++;
                    return;
                }

                // Start coroutine for stabilization
                StartCoroutine(
                    StabilizeObjectCoroutine(
                        gripperController,
                        duration_ms,
                        force_limit,
                        command.robot_id,
                        command.request_id
                    )
                );

                if (_verboseLogging)
                {
                    Debug.Log(
                        $"{_logPrefix} Stabilizing object for {command.robot_id}: duration={duration_ms}ms, force_limit={force_limit}N"
                    );
                }

                _successfulCommands++;
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"{_logPrefix} Error executing stabilize_object: {ex.Message}\n{ex.StackTrace}"
                );
                _failedCommands++;
            }
        }

        /// <summary>
        /// Coroutine for stabilize_object - hold gripper closed with force control for specified duration.
        /// </summary>
        private IEnumerator StabilizeObjectCoroutine(
            GripperController gripperController,
            int duration_ms,
            float force_limit,
            string robotId,
            uint requestId
        )
        {
            // Ensure gripper is closed (targetPosition < 0.1 means closed)
            if (gripperController.targetPosition > 0.1f)
            {
                gripperController.CloseGrippers();
                yield return new WaitForSeconds(0.5f); // Wait for gripper to close
            }

            // Hold for duration (force control would be applied here in a more advanced implementation)
            float duration_seconds = duration_ms / 1000.0f;
            yield return new WaitForSeconds(duration_seconds);

            // Send completion
            SendCommandCompletion(robotId, "stabilize_object", true, requestId);
        }

        /// <summary>
        /// Execute return_to_start_position command - move robot joints back to initial positions
        /// Supports both RobotController and SimpleRobotController
        /// </summary>
        private void ExecuteReturnToStartPosition(RobotCommand command)
        {
            try
            {
                // Validate robot and get controller
                if (
                    !ValidateAndGetRobot(
                        command.robot_id,
                        "return_to_start_position",
                        out RobotInstance robotInstance,
                        out RobotController controller
                    )
                )
                {
                    return;
                }
                if (
                    ShouldSkipForROSMode(
                        controller,
                        command.robot_id,
                        "return_to_start_position",
                        command.request_id
                    )
                )
                    return;

                // Validate start joint targets exist
                if (
                    robotInstance.startJointTargets == null
                    || robotInstance.startJointTargets.Length == 0
                )
                {
                    Debug.LogError(
                        $"{_logPrefix} return_to_start_position: No start joint targets saved for robot '{command.robot_id}'"
                    );
                    _failedCommands++;
                    return;
                }

                if (_verboseLogging)
                {
                    Debug.Log(
                        $"{_logPrefix} Returning {command.robot_id} to start position with joint targets: "
                            + $"[{string.Join(", ", robotInstance.startJointTargets.Select(a => a.ToString("F3")))}]"
                    );
                }

                // Start coroutine based on controller type
                if (controller != null)
                {
                    // RobotController
                    StartCoroutine(
                        ReturnToStartPositionCoroutine(
                            controller,
                            robotInstance.startJointTargets,
                            command.robot_id,
                            command.request_id,
                            2.0f
                        )
                    );
                }
                else if (robotInstance.simpleController != null)
                {
                    // SimpleRobotController
                    StartCoroutine(
                        ReturnToStartPositionSimpleCoroutine(
                            robotInstance.simpleController,
                            robotInstance.startJointTargets,
                            command.robot_id,
                            command.request_id,
                            2.0f
                        )
                    );
                }

                _successfulCommands++;
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"{_logPrefix} Error executing return_to_start_position: {ex.Message}\n{ex.StackTrace}"
                );
                _failedCommands++;
            }
        }

        /// <summary>
        /// Coroutine to smoothly interpolate joint targets to start position
        /// </summary>
        private IEnumerator ReturnToStartPositionCoroutine(
            RobotController controller,
            float[] targetJoints,
            string robotId,
            uint requestId,
            float duration
        )
        {
            if (controller == null || controller.robotJoints == null)
            {
                SendCommandCompletion(robotId, "return_to_start_position", false, requestId);
                yield break;
            }

            // FIX #5: DISABLE IK during manual joint interpolation
            controller.IsManuallyDriven = true;

            try
            {
                // Store initial joint positions from the actual drive targets
                float[] startJoints = new float[controller.robotJoints.Length];
                for (int i = 0; i < controller.robotJoints.Length && i < startJoints.Length; i++)
                {
                    startJoints[i] = controller.robotJoints[i].xDrive.target;
                }

                float elapsed = 0f;

                while (elapsed < duration)
                {
                    // FIX #4: Safety check in case robot died during movement
                    if (controller == null)
                    {
                        Debug.LogWarning(
                            $"{_logPrefix} Robot destroyed during ReturnToStartPosition"
                        );
                        yield break;
                    }

                    elapsed += Time.fixedDeltaTime;
                    float t = Mathf.Clamp01(elapsed / duration);

                    // Smooth interpolation using ease-in-out
                    float smoothT = t * t * (3f - 2f * t);

                    // Interpolate each joint
                    for (
                        int i = 0;
                        i < controller.robotJoints.Length && i < targetJoints.Length;
                        i++
                    )
                    {
                        // startJointTargets are cloned from jointDriveTargets which stores degrees
                        float currentTarget = Mathf.Lerp(startJoints[i], targetJoints[i], smoothT);

                        controller.jointDriveTargets[i] = currentTarget;

                        var joint = controller.robotJoints[i];
                        if (joint != null)
                        {
                            var drive = joint.xDrive;
                            drive.target = currentTarget;
                            joint.xDrive = drive;
                        }
                    }

                    yield return new WaitForFixedUpdate();
                }

                // Ensure final positions are exact
                for (int i = 0; i < controller.robotJoints.Length && i < targetJoints.Length; i++)
                {
                    controller.jointDriveTargets[i] = targetJoints[i];

                    var joint = controller.robotJoints[i];
                    if (joint != null)
                    {
                        var drive = joint.xDrive;
                        drive.target = targetJoints[i];
                        joint.xDrive = drive;
                    }
                }

                // Wait for joint velocities to fall below threshold before releasing
                // manual drive. ArticulationBody joints carry inertia from the
                // interpolation — releasing too early causes a sway/overshoot.
                const float settleVelThreshold = 2.0f; // deg/s
                const int maxSettleFrames = 60;         // ~1 s safety cap
                int settleFrames = 0;
                while (settleFrames < maxSettleFrames)
                {
                    bool allSettled = true;
                    for (int i = 0; i < controller.robotJoints.Length; i++)
                    {
                        var joint = controller.robotJoints[i];
                        if (joint != null && joint.jointVelocity.dofCount > 0)
                        {
                            float velDegPerSec = Mathf.Abs(joint.jointVelocity[0]) * Mathf.Rad2Deg;
                            if (velDegPerSec > settleVelThreshold)
                            {
                                allSettled = false;
                                break;
                            }
                        }
                    }
                    if (allSettled)
                        break;
                    settleFrames++;
                    yield return new WaitForFixedUpdate();
                }

                // FIX #5: Clear the target and mark as reached to allow IK to run again
                controller.ClearTarget(); // This resets _targetTransform and _hasReachedTarget

                if (_verboseLogging)
                {
                    Debug.Log($"{_logPrefix} Return to start position completed for {robotId}");
                }

                // Send completion notification
                SendCommandCompletion(robotId, "return_to_start_position", true, requestId);
            }
            finally
            {
                // FIX #5: ALWAYS re-enable IK, even if something fails
                controller.IsManuallyDriven = false;
            }
        }

        /// <summary>
        /// Coroutine to smoothly interpolate joint targets to start position for SimpleRobotController
        /// </summary>
        private IEnumerator ReturnToStartPositionSimpleCoroutine(
            SimpleRobotController controller,
            float[] targetJoints,
            string robotId,
            uint requestId,
            float duration
        )
        {
            if (controller == null || controller.robotJoints == null)
            {
                SendCommandCompletion(robotId, "return_to_start_position", false, requestId);
                yield break;
            }

            // Store initial joint positions from the actual drive targets
            float[] startJoints = new float[controller.robotJoints.Length];
            for (int i = 0; i < controller.robotJoints.Length && i < startJoints.Length; i++)
            {
                startJoints[i] = controller.robotJoints[i].xDrive.target;
            }

            float elapsed = 0f;

            while (elapsed < duration)
            {
                elapsed += Time.fixedDeltaTime;
                float t = Mathf.Clamp01(elapsed / duration);

                // Smooth interpolation using ease-in-out
                float smoothT = t * t * (3f - 2f * t);

                // Interpolate each joint
                for (int i = 0; i < controller.robotJoints.Length && i < targetJoints.Length; i++)
                {
                    // Target joints are in radians, drive targets are in degrees
                    float startDeg = startJoints[i];
                    float targetDeg = targetJoints[i] * Mathf.Rad2Deg;
                    float currentTarget = Mathf.Lerp(startDeg, targetDeg, smoothT);

                    var joint = controller.robotJoints[i];
                    if (joint != null)
                    {
                        var drive = joint.xDrive;
                        drive.target = currentTarget;
                        joint.xDrive = drive;
                    }
                }

                yield return new WaitForFixedUpdate();
            }

            // Ensure final positions are exact
            for (int i = 0; i < controller.robotJoints.Length && i < targetJoints.Length; i++)
            {
                var joint = controller.robotJoints[i];
                if (joint != null)
                {
                    var drive = joint.xDrive;
                    drive.target = targetJoints[i] * Mathf.Rad2Deg;
                    joint.xDrive = drive;
                }
            }

            // Clear the target and mark as reached
            controller.ClearTarget();

            if (_verboseLogging)
            {
                Debug.Log($"{_logPrefix} Return to start position completed for {robotId}");
            }

            // Send completion notification
            SendCommandCompletion(robotId, "return_to_start_position", true, requestId);
        }

        /// <summary>
        /// Execute capture_stereo_images command - capture and send stereo images to Python
        /// </summary>
        private void ExecuteCaptureSteroImages(RobotCommand command)
        {
            try
            {
                string cameraId = command.camera_id;
                if (string.IsNullOrEmpty(cameraId))
                {
                    Debug.LogError(
                        $"{_logPrefix} capture_stereo_images: camera_id is null or empty"
                    );
                    _failedCommands++;
                    return;
                }

                // Find StereoCameraController in scene
                var stereoController = FindFirstObjectByType<StereoCameraController>();
                if (stereoController == null)
                {
                    Debug.LogError(
                        $"{_logPrefix} capture_stereo_images: No StereoCameraController found in scene"
                    );
                    _failedCommands++;
                    return;
                }

                if (_verboseLogging)
                {
                    Debug.Log(
                        $"{_logPrefix} [req={command.request_id}] Capturing stereo images for {cameraId}"
                    );
                }

                // Capture and send stereo images
                stereoController.CaptureAndSendToServer(cameraId);

                _successfulCommands++;
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"{_logPrefix} Error executing capture_stereo_images: {ex.Message}\n{ex.StackTrace}"
                );
                _failedCommands++;
            }
        }

        /// <summary>
        /// Execute get_robot_status command - gather robot state and send back to Python
        /// </summary>
        private void ExecuteCheckRobotStatus(RobotCommand command)
        {
            try
            {
                if (
                    !ValidateAndGetRobot(
                        command.robot_id,
                        "get_robot_status",
                        out _,
                        out RobotController controller
                    )
                )
                {
                    SendStatusErrorResponse(
                        command.robot_id,
                        "VALIDATION_FAILED",
                        $"Failed to validate robot '{command.robot_id}'",
                        command.request_id
                    );
                    return;
                }

                bool detailed = command.parameters?.detailed ?? false;
                string statusJson = GatherRobotStatus(command.robot_id, controller, detailed);

                if (!SendStatusResponseToPython(statusJson, command.request_id))
                {
                    Debug.LogWarning(
                        $"{_logPrefix} [req={command.request_id}] Failed to send status response to Python"
                    );
                }

                _successfulCommands++;
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"{_logPrefix} Error executing get_robot_status: {ex.Message}\n{ex.StackTrace}"
                );
                _failedCommands++;

                SendStatusErrorResponse(
                    command.robot_id ?? "unknown",
                    "EXCEPTION",
                    $"Error gathering status: {ex.Message}",
                    command.request_id
                );
            }
        }

        /// <summary>
        /// Gather robot status data and return as JSON
        /// Supports both RobotController and SimpleRobotController
        /// </summary>
        private string GatherRobotStatus(string robotId, RobotController controller, bool detailed)
        {
            // Find the robot instance to check for SimpleRobotController
            RobotInstance robotInstance = null;
            _robotManager.RobotInstances.TryGetValue(robotId, out robotInstance);

            Vector3 currentPosition = Vector3.zero;
            Quaternion currentRotation = Quaternion.identity;
            Vector3 targetPosition = Vector3.zero;
            float distanceToTarget = 0f;
            bool isMoving = false;
            float[] jointAngles = new float[0];

            if (controller != null)
            {
                // Use RobotController
                currentPosition =
                    controller.endEffectorBase != null
                        ? controller.endEffectorBase.position
                        : Vector3.zero;

                currentRotation =
                    controller.endEffectorBase != null
                        ? controller.endEffectorBase.rotation
                        : Quaternion.identity;

                jointAngles = GatherJointAngles(controller, detailed);
                targetPosition = controller.GetCurrentTarget() ?? Vector3.zero;
                distanceToTarget = controller.GetDistanceToTarget();
                isMoving = distanceToTarget > RobotConstants.MOVEMENT_THRESHOLD;
            }
            else if (robotInstance != null && robotInstance.simpleController != null)
            {
                // Use SimpleRobotController
                var simpleController = robotInstance.simpleController;

                currentPosition = simpleController.GetCurrentEndEffectorPosition();
                currentRotation = simpleController.GetCurrentEndEffectorRotation();

                if (detailed && simpleController.robotJoints != null)
                {
                    jointAngles = new float[simpleController.robotJoints.Length];
                    for (int i = 0; i < simpleController.robotJoints.Length; i++)
                    {
                        var joint = simpleController.robotJoints[i];
                        if (joint != null && joint.jointType == ArticulationJointType.RevoluteJoint)
                        {
                            jointAngles[i] = joint.jointPosition[0];
                        }
                    }
                }

                targetPosition = simpleController.GetCurrentTarget() ?? Vector3.zero;
                distanceToTarget = simpleController.DistanceToTarget;
                isMoving = !simpleController.HasReachedTarget && simpleController.HasTarget;
            }

            var statusResponse = new
            {
                success = true,
                robot_id = robotId,
                detailed = detailed,
                status = new
                {
                    position = new
                    {
                        x = currentPosition.x,
                        y = currentPosition.y,
                        z = currentPosition.z,
                    },
                    rotation = new
                    {
                        x = currentRotation.x,
                        y = currentRotation.y,
                        z = currentRotation.z,
                        w = currentRotation.w,
                    },
                    joint_angles = jointAngles,
                    target_position = new
                    {
                        x = targetPosition.x,
                        y = targetPosition.y,
                        z = targetPosition.z,
                    },
                    distance_to_target = distanceToTarget,
                    is_moving = isMoving,
                    current_action = isMoving ? "moving_to_target" : "idle",
                },
                error = (object)null,
            };

            if (_verboseLogging)
            {
                Debug.Log(
                    $"{_logPrefix} Gathered status for {robotId}:\n"
                        + $"  Position: ({currentPosition.x:F3}, {currentPosition.y:F3}, {currentPosition.z:F3})\n"
                        + $"  Target: ({targetPosition.x:F3}, {targetPosition.y:F3}, {targetPosition.z:F3})\n"
                        + $"  Distance: {distanceToTarget:F3}m, Moving: {isMoving}"
                );
            }

            return JsonUtility.ToJson(statusResponse);
        }

        /// <summary>
        /// Gather joint angles from robot controller
        /// </summary>
        private float[] GatherJointAngles(RobotController controller, bool detailed)
        {
            if (!detailed || controller.robotJoints == null)
            {
                return new float[0];
            }

            float[] jointAngles = new float[controller.robotJoints.Length];
            for (int i = 0; i < controller.robotJoints.Length; i++)
            {
                var joint = controller.robotJoints[i];
                if (joint != null && joint.jointType == ArticulationJointType.RevoluteJoint)
                {
                    jointAngles[i] = joint.jointPosition[0];
                }
            }

            return jointAngles;
        }

        /// <summary>
        /// Send command completion notification to Python via CommandServer
        /// </summary>
        private void SendCommandCompletion(
            string robotId,
            string commandType,
            bool success,
            uint requestId
        )
        {
            var completionData = new CommandCompletionData
            {
                type = "command_completion",
                robot_id = robotId,
                command_type = commandType,
                success = success,
                request_id = requestId,
                timestamp = Time.time,
            };

            string json = JsonUtility.ToJson(completionData);

            if (_verboseLogging)
            {
                Debug.Log(
                    $"{_logPrefix} [req={requestId}] Sending completion for {commandType}: success={success}"
                );
            }

            // Send via StatusResponseSender
            if (!SendStatusResponseToPython(json, requestId))
            {
                Debug.LogWarning(
                    $"{_logPrefix} [req={requestId}] Failed to send completion notification"
                );
            }
        }

        /// <summary>
        /// Send status error response to Python (Protocol V2)
        /// </summary>
        private void SendStatusErrorResponse(
            string robotId,
            string errorCode,
            string errorMessage,
            uint requestId = 0
        )
        {
            var errorResponse = new
            {
                success = false,
                robot_id = robotId,
                detailed = false,
                status = (object)null,
                error = new { code = errorCode, message = errorMessage },
            };

            string errorJson = JsonUtility.ToJson(errorResponse);

            // Send error response to Python (Protocol V2)
            bool sent = SendStatusResponseToPython(errorJson, requestId);

            if (!sent)
            {
                Debug.LogWarning(
                    $"{_logPrefix} [req={requestId}] Failed to send error response to Python: {errorJson}"
                );
            }
        }

        /// <summary>
        /// Send status response (success or error) to Python on the same connection (Protocol V2)
        /// </summary>
        private bool SendStatusResponseToPython(string statusJson, uint requestId)
        {
            // Use UnifiedPythonReceiver to send response on the same connection that received the command
            if (UnifiedPythonReceiver.Instance == null)
            {
                Debug.LogWarning(
                    $"{_logPrefix} UnifiedPythonReceiver not available. Ensure UnifiedPythonReceiver GameObject is in the scene."
                );
                return false;
            }

            return UnifiedPythonReceiver.Instance.SendCompletion(statusJson, requestId);
        }

        #endregion

        #region Helper Methods

        /// <summary>
        /// Find GameObject with flexible name matching.
        /// Tries multiple strategies to handle naming mismatches between Python and Unity.
        /// </summary>
        /// <param name="objectId">Object identifier (e.g., "blue_cube", "blueCube", "Cube_01")</param>
        /// <returns>GameObject if found, null otherwise</returns>
        private GameObject FindObjectFlexible(string objectId)
        {
            if (string.IsNullOrEmpty(objectId))
                return null;

            // FIX #3: Check cache first
            if (_objectCache.TryGetValue(objectId, out GameObject cachedObj))
            {
                if (cachedObj != null) // Object still exists in scene
                {
                    if (_verboseLogging)
                        Debug.Log($"{_logPrefix} Found object '{objectId}' via cache");
                    return cachedObj;
                }
                else
                {
                    // Object was destroyed, remove from cache
                    _objectCache.Remove(objectId);
                }
            }

            // FIX #3: Periodically clear stale entries (every 5 seconds)
            if (Time.time - _lastCacheRefreshTime > OBJECT_CACHE_VALIDITY)
            {
                var deadKeys = new System.Collections.Generic.List<string>();
                foreach (var kvp in _objectCache)
                {
                    if (kvp.Value == null)
                        deadKeys.Add(kvp.Key);
                }
                foreach (var key in deadKeys)
                {
                    _objectCache.Remove(key);
                }
                _lastCacheRefreshTime = Time.time;
            }

            // Strategy 1: Exact match
            GameObject obj = GameObject.Find(objectId);
            if (obj != null)
            {
                _objectCache[objectId] = obj; // FIX #3: Cache it
                if (_verboseLogging)
                    Debug.Log($"{_logPrefix} Found object '{objectId}' via exact match");
                return obj;
            }

            // Strategy 2: Case-insensitive search through all GameObjects
            GameObject[] allObjects = FindObjectsByType<GameObject>(FindObjectsSortMode.None);
            foreach (var candidate in allObjects)
            {
                if (candidate.name.Equals(objectId, System.StringComparison.OrdinalIgnoreCase))
                {
                    _objectCache[objectId] = candidate; // FIX #3: Cache it
                    if (_verboseLogging)
                        Debug.Log(
                            $"{_logPrefix} Found object '{candidate.name}' via case-insensitive match for '{objectId}'"
                        );
                    return candidate;
                }
            }

            // Strategy 3: Convert snake_case to camelCase and search
            // "blue_cube" -> "blueCube"
            string camelCase = SnakeToCamelCase(objectId);
            if (camelCase != objectId)
            {
                obj = GameObject.Find(camelCase);
                if (obj != null)
                {
                    _objectCache[objectId] = obj; // FIX #3: Cache it
                    if (_verboseLogging)
                        Debug.Log(
                            $"{_logPrefix} Found object '{camelCase}' via snake_case->camelCase conversion from '{objectId}'"
                        );
                    return obj;
                }

                // Try case-insensitive camelCase search
                foreach (var candidate in allObjects)
                {
                    if (candidate.name.Equals(camelCase, System.StringComparison.OrdinalIgnoreCase))
                    {
                        _objectCache[objectId] = candidate; // FIX #3: Cache it
                        if (_verboseLogging)
                            Debug.Log(
                                $"{_logPrefix} Found object '{candidate.name}' via camelCase case-insensitive match for '{objectId}'"
                            );
                        return candidate;
                    }
                }
            }

            // Strategy 4: Partial match (contains)
            foreach (var candidate in allObjects)
            {
                if (
                    candidate.name.IndexOf(objectId, System.StringComparison.OrdinalIgnoreCase) >= 0
                )
                {
                    _objectCache[objectId] = candidate; // FIX #3: Cache it
                    if (_verboseLogging)
                        Debug.Log(
                            $"{_logPrefix} Found object '{candidate.name}' via partial match for '{objectId}'"
                        );
                    return candidate;
                }
            }

            return null;
        }

        /// <summary>
        /// FIX #4: Convert TargetPosition to Vector3.
        /// </summary>
        private Vector3 TargetPositionToVector3(TargetPosition target)
        {
            if (target == null)
                return Vector3.zero;
            return new Vector3(target.x, target.y, target.z);
        }

        /// <summary>
        /// Convert snake_case to camelCase.
        /// Example: "blue_cube" -> "blueCube"
        /// </summary>
        private string SnakeToCamelCase(string snakeCase)
        {
            if (string.IsNullOrEmpty(snakeCase) || !snakeCase.Contains("_"))
                return snakeCase;

            string[] parts = snakeCase.Split('_');
            string result = parts[0].ToLower();

            for (int i = 1; i < parts.Length; i++)
            {
                if (parts[i].Length > 0)
                {
                    result += char.ToUpper(parts[i][0]) + parts[i].Substring(1).ToLower();
                }
            }

            return result;
        }

        #endregion

        #region Public API

        /// <summary>
        /// Get command processing statistics
        /// </summary>
        public (int successful, int failed) GetCommandStats()
        {
            return (_successfulCommands, _failedCommands);
        }

        /// <summary>
        /// Reset command statistics
        /// </summary>
        public void ResetStats()
        {
            _successfulCommands = 0;
            _failedCommands = 0;
            Debug.Log($"{_logPrefix} Statistics reset");
        }

        #endregion
    }
}
