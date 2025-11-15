using System;
using Robotics;
using UnityEngine;

namespace PythonCommunication
{
    /// <summary>
    /// Data structure for robot commands received from Python operations
    /// </summary>
    [System.Serializable]
    public class RobotCommand
    {
        public string command_type;
        public string robot_id;
        public CommandParameters parameters;
        public float timestamp;
    }

    [System.Serializable]
    public class CommandParameters
    {
        public TargetPosition target_position;
        public TargetPosition original_target;
        public float speed_multiplier;
        public float approach_offset;
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

            Debug.Log($"{_logPrefix} ✓ Initialized and listening for Python commands");
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
                RobotCommand command = JsonUtility.FromJson<RobotCommand>(result.response);

                if (command != null && !string.IsNullOrEmpty(command.command_type))
                {
                    ProcessCommand(command);
                }
                else if (_verboseLogging)
                {
                    Debug.Log(
                        $"{_logPrefix} Received non-command result (likely LLM response): {result.response}"
                    );
                }
            }
            catch (Exception)
            {
                // Not a command - likely an LLM text response, which is fine
                if (_verboseLogging)
                {
                    Debug.Log(
                        $"{_logPrefix} Received non-JSON or LLM text result: {result.response?.Substring(0, Math.Min(50, result.response?.Length ?? 0))}..."
                    );
                }
            }
        }

        /// <summary>
        /// Process a validated robot command
        /// </summary>
        private void ProcessCommand(RobotCommand command)
        {
            if (_verboseLogging)
            {
                Debug.Log(
                    $"{_logPrefix} 📨 Processing command: {command.command_type} for robot: {command.robot_id}"
                );
            }

            switch (command.command_type)
            {
                case "move_to_coordinate":
                    ExecuteMoveToCoordinate(command);
                    break;

                case "check_robot_status":
                    ExecuteCheckRobotStatus(command);
                    break;

                default:
                    Debug.LogWarning($"{_logPrefix} Unknown command type: {command.command_type}");
                    _failedCommands++;
                    break;
            }
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
                // Validate robot ID
                if (string.IsNullOrEmpty(command.robot_id))
                {
                    Debug.LogError($"{_logPrefix} move_to_coordinate: robot_id is null or empty");
                    _failedCommands++;
                    return;
                }

                // Find robot instance
                if (
                    !_robotManager.RobotInstances.TryGetValue(
                        command.robot_id,
                        out RobotInstance robotInstance
                    )
                )
                {
                    Debug.LogError(
                        $"{_logPrefix} move_to_coordinate: Robot '{command.robot_id}' not found in RobotManager. "
                            + $"Available robots: {string.Join(", ", _robotManager.RobotInstances.Keys)}"
                    );
                    _failedCommands++;
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

                // Get robot controller
                RobotController controller = robotInstance.controller;
                if (controller == null)
                {
                    Debug.LogError(
                        $"{_logPrefix} move_to_coordinate: Robot '{command.robot_id}' has no RobotController component"
                    );
                    _failedCommands++;
                    return;
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

                // Execute movement
                controller.SetTarget(targetPosition);

                if (_verboseLogging)
                {
                    string offsetInfo =
                        command.parameters.approach_offset > 0
                            ? $" (offset: {command.parameters.approach_offset:F3}m)"
                            : "";
                    Debug.Log(
                        $"{_logPrefix} ✓ Moving {command.robot_id} to position: "
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
        /// Execute get_robot_status command - gather robot state and send back to Python
        /// </summary>
        private void ExecuteCheckRobotStatus(RobotCommand command)
        {
            try
            {
                // Validate robot ID
                if (string.IsNullOrEmpty(command.robot_id))
                {
                    Debug.LogError($"{_logPrefix} get_robot_status: robot_id is null or empty");
                    _failedCommands++;
                    return;
                }

                // Find robot instance
                if (
                    !_robotManager.RobotInstances.TryGetValue(
                        command.robot_id,
                        out RobotInstance robotInstance
                    )
                )
                {
                    Debug.LogError(
                        $"{_logPrefix} get_robot_status: Robot '{command.robot_id}' not found in RobotManager. "
                            + $"Available robots: {string.Join(", ", _robotManager.RobotInstances.Keys)}"
                    );
                    _failedCommands++;

                    // Send error response to Python
                    SendStatusErrorResponse(
                        command.robot_id,
                        "ROBOT_NOT_FOUND",
                        $"Robot '{command.robot_id}' not found in RobotManager"
                    );
                    return;
                }

                // Get parameters
                bool detailed = false;
                if (
                    command.parameters != null
                    && command.parameters is CommandParameters cmdParams
                )
                {
                    // Parameters may contain a 'detailed' field but CommandParameters doesn't have it
                    // For now, we'll assume basic status unless we receive a more complex command structure
                    detailed = false; // Can be extended later
                }

                // Get robot controller
                RobotController controller = robotInstance.controller;
                if (controller == null)
                {
                    Debug.LogError(
                        $"{_logPrefix} get_robot_status: Robot '{command.robot_id}' has no RobotController component"
                    );
                    _failedCommands++;

                    SendStatusErrorResponse(
                        command.robot_id,
                        "NO_CONTROLLER",
                        $"Robot '{command.robot_id}' has no RobotController component"
                    );
                    return;
                }

                // Gather robot status data
                Vector3 currentPosition = controller.endEffectorBase != null
                    ? controller.endEffectorBase.position
                    : Vector3.zero;

                Quaternion currentRotation = controller.endEffectorBase != null
                    ? controller.endEffectorBase.rotation
                    : Quaternion.identity;

                // Get joint angles if detailed
                float[] jointAngles = null;
                if (detailed && controller.robotJoints != null)
                {
                    jointAngles = new float[controller.robotJoints.Length];
                    for (int i = 0; i < controller.robotJoints.Length; i++)
                    {
                        var joint = controller.robotJoints[i];
                        if (joint != null && joint.jointType == ArticulationJointType.RevoluteJoint)
                        {
                            // Get current joint position
                            jointAngles[i] = joint.jointPosition[0]; // Radians
                        }
                    }
                }
                else
                {
                    // Empty array for non-detailed
                    jointAngles = new float[0];
                }

                // Get target position from controller's GetCurrentTarget method
                Vector3? targetPositionNullable = controller.GetCurrentTarget();
                Vector3 targetPosition = targetPositionNullable ?? Vector3.zero;

                float distanceToTarget = controller.GetDistanceToTarget();
                bool isMoving = distanceToTarget > 0.01f; // Consider moving if distance > 1cm

                // Create status response
                var statusResponse = new
                {
                    success = true,
                    robot_id = command.robot_id,
                    detailed = detailed,
                    status = new
                    {
                        position = new { x = currentPosition.x, y = currentPosition.y, z = currentPosition.z },
                        rotation = new
                        {
                            x = currentRotation.x,
                            y = currentRotation.y,
                            z = currentRotation.z,
                            w = currentRotation.w
                        },
                        joint_angles = jointAngles,
                        target_position = new
                        {
                            x = targetPosition.x,
                            y = targetPosition.y,
                            z = targetPosition.z
                        },
                        distance_to_target = distanceToTarget,
                        is_moving = isMoving,
                        current_action = isMoving ? "moving_to_target" : "idle"
                    },
                    error = (object)null
                };

                // Convert to JSON
                string statusJson = JsonUtility.ToJson(statusResponse);

                if (_verboseLogging)
                {
                    Debug.Log(
                        $"{_logPrefix} ✓ Gathered status for {command.robot_id}:\n"
                            + $"  Position: ({currentPosition.x:F3}, {currentPosition.y:F3}, {currentPosition.z:F3})\n"
                            + $"  Target: ({targetPosition.x:F3}, {targetPosition.y:F3}, {targetPosition.z:F3})\n"
                            + $"  Distance: {distanceToTarget:F3}m, Moving: {isMoving}"
                    );
                }

                // Send status response back to Python StatusServer
                bool sent = SendStatusResponseToPython(statusJson);

                if (!sent)
                {
                    Debug.LogWarning($"{_logPrefix} Failed to send status response to Python");
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
                    $"Error gathering status: {ex.Message}"
                );
            }
        }

        /// <summary>
        /// Send status error response to Python
        /// </summary>
        private void SendStatusErrorResponse(string robotId, string errorCode, string errorMessage)
        {
            var errorResponse = new
            {
                success = false,
                robot_id = robotId,
                detailed = false,
                status = (object)null,
                error = new { code = errorCode, message = errorMessage }
            };

            string errorJson = JsonUtility.ToJson(errorResponse);

            // Send error response to Python
            bool sent = SendStatusResponseToPython(errorJson);

            if (!sent)
            {
                Debug.LogWarning($"{_logPrefix} Failed to send error response to Python: {errorJson}");
            }
        }

        /// <summary>
        /// Send status response (success or error) to Python StatusServer
        /// </summary>
        private bool SendStatusResponseToPython(string statusJson)
        {
            // Use StatusResponseSender to send response to StatusServer (port 5012)
            if (StatusResponseSender.Instance == null)
            {
                Debug.LogWarning(
                    $"{_logPrefix} StatusResponseSender not available. Ensure StatusResponseSender GameObject is in the scene."
                );
                return false;
            }

            return StatusResponseSender.Instance.SendStatusResponse(statusJson);
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
