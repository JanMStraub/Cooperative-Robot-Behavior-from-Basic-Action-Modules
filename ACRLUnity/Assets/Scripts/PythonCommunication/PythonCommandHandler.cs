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

        [Tooltip("Enable Python CoordinationVerifier checks before executing movements (Phase 4)")]
        [SerializeField]
        private bool _enablePythonVerification = true;

        [Tooltip("Coordination configuration for verification and collision avoidance")]
        [SerializeField]
        private Configuration.CoordinationConfig _coordinationConfig;

        [Tooltip("Workspace Manager for coordination (Phase 4)")]
        private Simulation.WorkspaceManager _workspaceManager;

        // Coordination verifier (initialized based on config)
        private ICoordinationVerifier _coordinationVerifier;

        [Header("Runtime Info")]
        [Tooltip("Number of commands processed successfully")]
        [SerializeField]
        private int _successfulCommands = 0;

        [Tooltip("Number of commands that failed")]
        [SerializeField]
        private int _failedCommands = 0;

        private RobotManager _robotManager;

        // Track active commands for completion notification
        private System.Collections.Generic.Dictionary<string, uint> _activeCommands =
            new System.Collections.Generic.Dictionary<string, uint>();

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
            // Get RobotManager instance
            _robotManager = RobotManager.Instance;
            if (_robotManager == null)
            {
                Debug.LogError(
                    $"{_logPrefix} RobotManager.Instance is null! "
                        + "Ensure RobotManager GameObject is in the scene."
                );
                return;
            }

            // Get WorkspaceManager instance (Phase 4)
            _workspaceManager = Simulation.WorkspaceManager.Instance;
            if (_workspaceManager == null && _enablePythonVerification)
            {
                Debug.LogWarning(
                    $"{_logPrefix} WorkspaceManager.Instance is null! "
                        + "Workspace coordination will be disabled. "
                        + "Add WorkspaceManager GameObject to enable Phase 4 features."
                );
            }

            // Initialize coordination verifier based on configuration
            InitializeCoordinationVerifier();

            Debug.Log($"{_logPrefix} Initialized and listening for Python commands");
        }

        /// <summary>
        /// Initialize coordination verifier based on configuration mode.
        /// </summary>
        private void InitializeCoordinationVerifier()
        {
            if (_coordinationConfig == null)
            {
                Debug.LogWarning($"{_logPrefix} CoordinationConfig not assigned, using Unity-only verification");
                _coordinationVerifier = new UnityCoordinationVerifier(0.2f);
                return;
            }

            float minSafeSeparation = _coordinationConfig.minSafeSeparation;

            switch (_coordinationConfig.verificationMode)
            {
                case Configuration.VerificationMode.UnityOnly:
                    _coordinationVerifier = new UnityCoordinationVerifier(minSafeSeparation);
                    Debug.Log($"{_logPrefix} Coordination verification mode: Unity-only (fast)");
                    break;

                case Configuration.VerificationMode.PythonVerified:
                    _coordinationVerifier = new PythonCoordinationVerifier(
                        _coordinationConfig.pythonVerificationTimeout,
                        _coordinationConfig.fallbackToUnityOnTimeout,
                        minSafeSeparation
                    );
                    Debug.Log($"{_logPrefix} Coordination verification mode: Python-verified");
                    break;

                case Configuration.VerificationMode.Hybrid:
                    // Hybrid mode: Use Unity for initial check, Python for conflicts
                    // For now, use Unity-only as hybrid implementation requires more complex logic
                    _coordinationVerifier = new UnityCoordinationVerifier(minSafeSeparation);
                    Debug.Log($"{_logPrefix} Coordination verification mode: Hybrid (using Unity for now)");
                    break;

                default:
                    _coordinationVerifier = new UnityCoordinationVerifier(minSafeSeparation);
                    Debug.LogWarning($"{_logPrefix} Unknown verification mode, defaulting to Unity-only");
                    break;
            }
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
                // Robot-targeted commands
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

                // Camera-targeted commands
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

            // Validate robot ID
            if (string.IsNullOrEmpty(robotId))
            {
                Debug.LogError($"{_logPrefix} {commandName}: robot_id is null or empty");
                _failedCommands++;
                return false;
            }

            // Find robot instance
            if (!_robotManager.RobotInstances.TryGetValue(robotId, out robotInstance))
            {
                Debug.LogError(
                    $"{_logPrefix} {commandName}: Robot '{robotId}' not found in RobotManager. "
                        + $"Available robots: {string.Join(", ", _robotManager.RobotInstances.Keys)}"
                );
                _failedCommands++;
                return false;
            }

            // Get robot controller (either RobotController or SimpleRobotController)
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

                // Validate parameters
                if (command.parameters == null || command.parameters.target_position == null)
                {
                    Debug.LogError(
                        $"{_logPrefix} move_to_coordinate: Missing parameters or target_position"
                    );
                    _failedCommands++;
                    return;
                }

                // Get target position
                Vector3 targetPosition = new Vector3(
                    command.parameters.target_position.x,
                    command.parameters.target_position.y,
                    command.parameters.target_position.z
                );

                // Phase 4: Check Python CoordinationVerifier before executing
                if (_enablePythonVerification)
                {
                    if (!VerifyMovementSafety(command.robot_id, targetPosition))
                    {
                        Debug.LogWarning(
                            $"{_logPrefix} [Phase 4] Movement verification failed for {command.robot_id} "
                                + $"to ({targetPosition.x:F3}, {targetPosition.y:F3}, {targetPosition.z:F3}). "
                                + "Movement blocked by coordination check."
                        );
                        SendCommandCompletion(
                            command.robot_id,
                            "move_to_coordinate",
                            false,
                            command.request_id
                        );
                        _failedCommands++;
                        return;
                    }
                }

                // Apply speed multiplier if enabled
                if (_applySpeedMultiplier && command.parameters.speed_multiplier > 0)
                {
                    // Note: You may need to add a speed multiplier property to RobotController
                    // For now, we just log it
                    if (_verboseLogging)
                    {
                        Debug.Log(
                            $"{_logPrefix} Speed multiplier: {command.parameters.speed_multiplier:F2}"
                        );
                    }
                }

                // Subscribe to completion event
                string commandKey = $"move_{command.robot_id}_{command.request_id}";
                _activeCommands[commandKey] = command.request_id;

                // Create completion handler
                System.Action onComplete = null;
                onComplete = () =>
                {
                    // Unsubscribe from the appropriate event
                    if (controller != null)
                        controller.OnTargetReached -= onComplete;
                    else if (robotInstance.simpleController != null)
                        robotInstance.simpleController.OnTargetReached -= onComplete;

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

                    // Phase 4: Release workspace region after movement completes
                    if (_workspaceManager != null)
                    {
                        var region = _workspaceManager.GetRegionAtPosition(targetPosition);
                        if (region != null && region.allocatedRobotId == command.robot_id)
                        {
                            _workspaceManager.ClearCollisionZone(region.regionName);
                        }
                    }
                };

                // Subscribe to the appropriate controller's event
                if (controller != null)
                {
                    controller.OnTargetReached += onComplete;
                }
                else if (robotInstance.simpleController != null)
                {
                    robotInstance.simpleController.OnTargetReached += onComplete;
                }

                // Phase 4: Allocate workspace region before movement starts
                if (_workspaceManager != null)
                {
                    var region = _workspaceManager.GetRegionAtPosition(targetPosition);
                    if (region != null)
                    {
                        _workspaceManager.AllocateRegion(command.robot_id, region.regionName);
                        _workspaceManager.MarkCollisionZone(region.regionName);
                    }
                }

                // Execute movement on the appropriate controller
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
                // Validate robot and get instance
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

                // Validate parameters
                if (command.parameters == null)
                {
                    Debug.LogError($"{_logPrefix} control_gripper: Missing parameters");
                    _failedCommands++;
                    return;
                }

                // Get gripper controller from robot
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

                // Subscribe to completion event
                string commandKey = $"gripper_{command.robot_id}_{command.request_id}";
                _activeCommands[commandKey] = command.request_id;

                // Create completion handler
                System.Action onComplete = null;
                onComplete = () =>
                {
                    gripperController.OnGripperActionComplete -= onComplete;
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
                gripperController.OnGripperActionComplete += onComplete;

                // Execute gripper action
                bool openGripper = command.parameters.open_gripper;
                if (openGripper)
                {
                    gripperController.OpenGrippers();
                }
                else
                {
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

                // Validate parameters
                if (command.parameters == null)
                {
                    Debug.LogError($"{_logPrefix} grasp_object: Missing parameters");
                    _failedCommands++;
                    return;
                }

                // Validate object_id
                string objectId = command.parameters.object_id;
                if (string.IsNullOrEmpty(objectId))
                {
                    Debug.LogError($"{_logPrefix} grasp_object: Missing object_id");
                    _failedCommands++;
                    return;
                }

                // Find target object in scene (with flexible matching)
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

                // Build GraspOptions from command parameters
                GraspOptions options = new GraspOptions
                {
                    useGraspPlanning = true,
                    openGripperOnSet = true,
                    closeGripperOnReach = true,
                    useAdvancedPlanning = command.parameters.use_advanced_planning,
                    approach = ParseApproachType(command.parameters.preferred_approach),
                    overridePreGraspDistance = command.parameters.pre_grasp_distance,
                    customApproachVector = ParseApproachVector(command.parameters.custom_approach_vector),
                    graspConfig = null // Use default config from RobotController
                };

                // Subscribe to target reached event for completion notification
                string commandKey = $"grasp_{command.robot_id}_{command.request_id}";
                _activeCommands[commandKey] = command.request_id;

                System.Action onComplete = null;
                onComplete = () =>
                {
                    controller.OnTargetReached -= onComplete;
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
                controller.OnTargetReached += onComplete;

                // Execute grasp
                controller.SetTarget(targetObject, options);

                if (_verboseLogging)
                {
                    string planningMode = options.useAdvancedPlanning ? "Advanced" : "Standard";
                    Debug.Log(
                        $"{_logPrefix} {planningMode} grasp planning: {command.robot_id} -> {objectId}"
                    );
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
                    Debug.LogWarning($"{_logPrefix} Unknown approach type '{approachStr}', using auto");
                    return null;
            }
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

            return new Vector3(vector.x, vector.y, vector.z);
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

                    controller.jointDriveTargets[i] = currentTarget * Mathf.Deg2Rad;

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
                    drive.target = targetJoints[i] * Mathf.Rad2Deg;
                    joint.xDrive = drive;
                }
            }

            // CRITICAL: Clear the target and mark as reached to prevent IK from running
            // Without this, FixedUpdate() continues to call PerformInverseKinematicsStep()
            // which pulls the robot back to the old target position
            controller.SetTargetReached(true);

            if (_verboseLogging)
            {
                Debug.Log($"{_logPrefix} Return to start position completed for {robotId}");
            }

            // Send completion notification
            SendCommandCompletion(robotId, "return_to_start_position", true, requestId);
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
        /// Send command completion notification to Python via StatusServer
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

        #region Phase 4: Python Verification Integration

        /// <summary>
        /// Verify movement safety using workspace manager and coordination checks
        /// Phase 4: Integration with Python CoordinationVerifier
        /// Supports both RobotController and SimpleRobotController
        /// </summary>
        /// <param name="robotId">Robot requesting movement</param>
        /// <param name="targetPosition">Target position</param>
        /// <returns>True if movement is safe, false if blocked</returns>
        private bool VerifyMovementSafety(string robotId, Vector3 targetPosition)
        {
            // Early exit if verification is disabled
            if (!_enablePythonVerification || _coordinationVerifier == null)
            {
                return true;
            }

            // Get current robot position
            Vector3 currentPosition = Vector3.zero;
            if (_robotManager != null && _robotManager.RobotInstances.TryGetValue(robotId, out var robotInstance))
            {
                if (robotInstance.controller != null)
                {
                    currentPosition = robotInstance.controller.GetCurrentEndEffectorPosition();
                }
                else if (robotInstance.simpleController != null)
                {
                    currentPosition = robotInstance.simpleController.GetCurrentEndEffectorPosition();
                }
            }

            // Use coordination verifier
            var result = _coordinationVerifier.VerifyMovement(robotId, targetPosition, currentPosition);

            // Log warnings if any
            if (_verboseLogging && result.warnings != null && result.warnings.Count > 0)
            {
                foreach (var warning in result.warnings)
                {
                    Debug.LogWarning($"{_logPrefix} [Verification] {warning}");
                }
            }

            // Log result
            if (_verboseLogging)
            {
                if (result.isSafe)
                {
                    Debug.Log($"{_logPrefix} [Verification] Movement safe for {robotId} using {_coordinationVerifier.VerifierName}: {result.reason}");
                }
                else
                {
                    Debug.LogWarning($"{_logPrefix} [Verification] Movement blocked for {robotId} using {_coordinationVerifier.VerifierName}: {result.reason}");
                }
            }

            return result.isSafe;
        }

        /// <summary>
        /// Enable or disable Python verification
        /// </summary>
        public void SetPythonVerificationEnabled(bool enabled)
        {
            _enablePythonVerification = enabled;
            Debug.Log(
                $"{_logPrefix} [Phase 4] Python verification {(enabled ? "enabled" : "disabled")}"
            );
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

            // Strategy 1: Exact match
            GameObject obj = GameObject.Find(objectId);
            if (obj != null)
            {
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
