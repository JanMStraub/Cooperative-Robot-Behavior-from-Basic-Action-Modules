using System;
using System.Collections;
using System.Linq;
using Core;
using Robotics;
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

        // Perception parameters (for depth detection)
        public string[] object_types;
        public float min_confidence;
        public float max_distance;
    }

    [System.Serializable]
    public class TargetPosition
    {
        public float x;
        public float y;
        public float z;
    }

    /// <summary>
    /// Handles commands from Python operations and executes them on Unity robots.
    ///
    /// This handler listens to UnifiedPythonReceiver for incoming commands from Python
    /// operations (e.g., move_to_coordinate) and executes them using RobotManager
    /// and RobotController.
    ///
    /// Supported Commands:
    /// - move_to_coordinate: Move robot end effector to target position
    ///
    /// Usage:
    /// 1. Attach this component to a GameObject in your scene
    /// 2. Ensure UnifiedPythonReceiver and RobotManager are active
    /// 3. Python operations will automatically send commands via ResultsServer
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

        private UnifiedPythonReceiver _resultsReceiver;
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
        /// Subscribe to UnifiedPythonReceiver events when component starts
        /// </summary>
        private void Start()
        {
            // Get UnifiedPythonReceiver instance
            _resultsReceiver = UnifiedPythonReceiver.Instance;
            if (_resultsReceiver == null)
            {
                Debug.LogError(
                    $"{_logPrefix} UnifiedPythonReceiver.Instance is null! "
                        + "Ensure UnifiedPythonReceiver GameObject is in the scene."
                );
                return;
            }

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

            // Subscribe to results from Python
            _resultsReceiver.OnLLMResultReceived += HandlePythonResult;

            Debug.Log($"{_logPrefix} Initialized and listening for Python commands");
        }

        /// <summary>
        /// Unsubscribe from events on destroy
        /// </summary>
        private void OnDestroy()
        {
            if (Instance == this)
            {
                if (_resultsReceiver != null)
                {
                    _resultsReceiver.OnLLMResultReceived -= HandlePythonResult;
                }
                Instance = null;
            }
        }

        #endregion

        #region Command Processing

        /// <summary>
        /// Handle incoming results from Python (may be LLM results or robot commands)
        /// </summary>
        private void HandlePythonResult(LLMResult result)
        {
            if (result == null)
                return;

            // Try to parse as robot command
            try
            {
                // Debug log the raw JSON for troubleshooting
                if (
                    _verboseLogging
                    && result.response != null
                    && result.response.Contains("command_type")
                )
                {
                    Debug.Log(
                        $"{_logPrefix} [req={result.request_id}] Received potential command: {result.response}"
                    );
                }

                RobotCommand command = JsonUtility.FromJson<RobotCommand>(result.response);

                if (command != null && !string.IsNullOrEmpty(command.command_type))
                {
                    // Set request_id from LLMResult (Protocol V2)
                    command.request_id = result.request_id;

                    ProcessCommand(command);
                }
                else if (_verboseLogging)
                {
                    Debug.Log(
                        $"{_logPrefix} Received non-command result (likely LLM response): {result.response}"
                    );
                }
            }
            catch (Exception ex)
            {
                // Not a command - likely an LLM text response, which is fine
                if (_verboseLogging)
                {
                    // Check if it looks like a command that failed to parse
                    if (result.response != null && result.response.Contains("command_type"))
                    {
                        Debug.LogError(
                            $"{_logPrefix} [req={result.request_id}] Failed to parse command JSON: {ex.Message}\n"
                                + $"JSON: {result.response}"
                        );
                    }
                    else
                    {
                        Debug.Log(
                            $"{_logPrefix} Received non-JSON or LLM text result: {result.response?.Substring(0, Math.Min(50, result.response?.Length ?? 0))}..."
                        );
                    }
                }
            }
        }

        /// <summary>
        /// Process a validated robot command
        /// </summary>
        private void ProcessCommand(RobotCommand command)
        {
            // Determine target type for logging
            string targetInfo = command.target_type == "camera"
                ? $"camera: {command.camera_id}"
                : $"robot: {command.robot_id}";

            if (_verboseLogging)
            {
                Debug.Log($"{_logPrefix} Processing command: {command.command_type} for {targetInfo}");
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

                // Camera-targeted commands (perception)
                case "calculate_object_coordinates":
                    ExecuteCalculateObjectCoordinates(command);
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
        /// </summary>
        /// <param name="robotId">Robot identifier</param>
        /// <param name="commandName">Command name for error messages</param>
        /// <param name="robotInstance">Output: robot instance if found</param>
        /// <param name="controller">Output: robot controller if found</param>
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

            // Get robot controller
            controller = robotInstance.controller;
            if (controller == null)
            {
                Debug.LogError(
                    $"{_logPrefix} {commandName}: Robot '{robotId}' has no RobotController component"
                );
                _failedCommands++;
                return false;
            }

            return true;
        }

        #endregion

        #region Command Implementations

        /// <summary>
        /// Execute move_to_coordinate command
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
                        out _,
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
                    controller.OnTargetReached -= onComplete;
                    if (_activeCommands.ContainsKey(commandKey))
                    {
                        _activeCommands.Remove(commandKey);
                        SendCommandCompletion(command.robot_id, "move_to_coordinate", true, command.request_id);
                    }
                };
                controller.OnTargetReached += onComplete;

                // Execute movement
                controller.SetTarget(targetPosition);

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
                GripperController gripperController = robotInstance.robotGameObject.GetComponentInChildren<GripperController>();
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
                        SendCommandCompletion(command.robot_id, "control_gripper", true, command.request_id);
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
        /// Execute return_to_start_position command - move robot joints back to initial positions
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
                if (robotInstance.startJointTargets == null || robotInstance.startJointTargets.Length == 0)
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

                // Start coroutine to smoothly interpolate joint targets
                StartCoroutine(ReturnToStartPositionCoroutine(
                    controller,
                    robotInstance.startJointTargets,
                    command.robot_id,
                    command.request_id,
                    2.0f
                ));

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
            float duration)
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

            if (_verboseLogging)
            {
                Debug.Log($"{_logPrefix} Return to start position completed for {robotId}");
            }

            // Send completion notification
            SendCommandCompletion(robotId, "return_to_start_position", true, requestId);
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
        /// Execute calculate_object_coordinates command - triggers stereo depth detection
        /// </summary>
        private void ExecuteCalculateObjectCoordinates(RobotCommand command)
        {
            try
            {
                // Validate camera_id
                string cameraId = command.camera_id;
                if (string.IsNullOrEmpty(cameraId))
                {
                    cameraId = "stereo_main"; // Default camera
                }

                // Find stereo camera controller
                StereoCameraController stereoCamera = FindStereoCameraById(cameraId);
                if (stereoCamera == null)
                {
                    Debug.LogError(
                        $"{_logPrefix} calculate_object_coordinates: Camera '{cameraId}' not found"
                    );
                    SendPerceptionErrorResponse(
                        cameraId,
                        "CAMERA_NOT_FOUND",
                        $"Stereo camera '{cameraId}' not found in scene",
                        command.request_id
                    );
                    _failedCommands++;
                    return;
                }

                // Extract parameters
                string[] objectTypes = command.parameters?.object_types;
                float minConfidence = command.parameters?.min_confidence ?? 0.5f;
                float maxDistance = command.parameters?.max_distance ?? 5.0f;

                if (_verboseLogging)
                {
                    string typesStr = objectTypes != null ? string.Join(", ", objectTypes) : "all";
                    Debug.Log(
                        $"{_logPrefix} Triggering depth detection on {cameraId}: "
                            + $"types=[{typesStr}], confidence>={minConfidence:F2}, distance<={maxDistance:F1}m"
                    );
                }

                // Trigger the stereo capture and detection
                // The results will be sent back via the normal detection pipeline (port 5007)
                stereoCamera.CaptureAndDetect(objectTypes, minConfidence, maxDistance);

                _successfulCommands++;
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"{_logPrefix} Error executing calculate_object_coordinates: {ex.Message}\n{ex.StackTrace}"
                );
                _failedCommands++;

                SendPerceptionErrorResponse(
                    command.camera_id ?? "unknown",
                    "EXCEPTION",
                    $"Error during depth detection: {ex.Message}",
                    command.request_id
                );
            }
        }

        /// <summary>
        /// Find a stereo camera controller by ID
        /// </summary>
        private StereoCameraController FindStereoCameraById(string cameraId)
        {
            // Find all stereo cameras in scene
            StereoCameraController[] cameras = FindObjectsByType<StereoCameraController>(FindObjectsSortMode.None);

            foreach (var camera in cameras)
            {
                // Check if camera name or ID matches
                if (camera.CameraId == cameraId || camera.gameObject.name == cameraId)
                {
                    return camera;
                }
            }

            // If only one camera and using default ID, return it
            if (cameras.Length == 1 && cameraId == "stereo_main")
            {
                return cameras[0];
            }

            return null;
        }

        /// <summary>
        /// Send perception error response to Python
        /// </summary>
        private void SendPerceptionErrorResponse(
            string cameraId,
            string errorCode,
            string errorMessage,
            uint requestId
        )
        {
            var errorResponse = new
            {
                success = false,
                camera_id = cameraId,
                detections = (object)null,
                error = new { code = errorCode, message = errorMessage },
            };

            string errorJson = JsonUtility.ToJson(errorResponse);

            // Send via depth results channel (port 5007) or status response
            bool sent = SendStatusResponseToPython(errorJson, requestId);

            if (!sent)
            {
                Debug.LogWarning(
                    $"{_logPrefix} [req={requestId}] Failed to send perception error response: {errorJson}"
                );
            }
        }

        /// <summary>
        /// Gather robot status data and return as JSON
        /// </summary>
        private string GatherRobotStatus(string robotId, RobotController controller, bool detailed)
        {
            Vector3 currentPosition =
                controller.endEffectorBase != null
                    ? controller.endEffectorBase.position
                    : Vector3.zero;

            Quaternion currentRotation =
                controller.endEffectorBase != null
                    ? controller.endEffectorBase.rotation
                    : Quaternion.identity;

            float[] jointAngles = GatherJointAngles(controller, detailed);
            Vector3 targetPosition = controller.GetCurrentTarget() ?? Vector3.zero;
            float distanceToTarget = controller.GetDistanceToTarget();
            bool isMoving = distanceToTarget > RobotConstants.MOVEMENT_THRESHOLD;

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
        private void SendCommandCompletion(string robotId, string commandType, bool success, uint requestId)
        {
            var completionData = new
            {
                type = "command_completion",
                robot_id = robotId,
                command_type = commandType,
                success = success,
                request_id = requestId,
                timestamp = Time.time
            };

            string json = JsonUtility.ToJson(completionData);

            if (_verboseLogging)
            {
                Debug.Log($"{_logPrefix} [req={requestId}] Sending completion for {commandType}: success={success}");
            }

            // Send via StatusResponseSender
            if (!SendStatusResponseToPython(json, requestId))
            {
                Debug.LogWarning($"{_logPrefix} [req={requestId}] Failed to send completion notification");
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
        /// Send status response (success or error) to Python StatusServer (Protocol V2)
        /// </summary>
        private bool SendStatusResponseToPython(string statusJson, uint requestId)
        {
            // Use StatusResponseSender to send response to StatusServer (port 5012)
            if (StatusResponseSender.Instance == null)
            {
                Debug.LogWarning(
                    $"{_logPrefix} StatusResponseSender not available. Ensure StatusResponseSender GameObject is in the scene."
                );
                return false;
            }

            return StatusResponseSender.Instance.SendStatusResponse(statusJson, requestId);
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
