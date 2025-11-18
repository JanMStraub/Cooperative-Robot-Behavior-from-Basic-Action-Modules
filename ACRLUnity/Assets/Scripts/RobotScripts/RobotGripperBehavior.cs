using Logging;
using UnityEngine;

namespace Robotics
{
    /// <summary>
    /// Handles gripper behavior in response to robot events.
    /// Subscribes to RobotController.OnTargetReached and executes grasp strategy when target is reached.
    /// Uses GraspPlanner to determine optimal grasp approach based on object characteristics.
    /// </summary>
    [RequireComponent(typeof(RobotController))]
    public class RobotGripperBehavior : MonoBehaviour
    {
        [Header("Configuration")]
        [Tooltip("Enable automatic grasp planning based on object heuristics")]
        [SerializeField]
        private bool _useGraspPlanning = true;

        [Tooltip("Log grasp actions to MainLogger")]
        [SerializeField]
        private bool _logGraspActions = true;

        private RobotController _robotController;
        private GripperController _gripperController;
        private MainLogger _logger;

        private GameObject _currentTarget;
        private string _currentActionId;

        // Helper variables
        private const string _logPrefix = "[ROBOT_GRIPPER_BEHAVIOR]";

        /// <summary>
        /// Unity Awake callback - find required components and subscribe to events.
        /// </summary>
        private void Awake()
        {
            // Get RobotController on this GameObject
            _robotController = GetComponent<RobotController>();
            if (_robotController == null)
            {
                Debug.LogError($"{_logPrefix} RobotController not found on GameObject!");
                return;
            }

            // Find GripperController in children
            _gripperController = GetComponentInChildren<GripperController>();
            if (_gripperController == null)
            {
                Debug.LogWarning(
                    $"{_logPrefix} No GripperController found in children of {gameObject.name}"
                );
                return;
            }

            // Get logger if available
            if (_logGraspActions)
            {
                _logger = MainLogger.Instance;
            }

            // Subscribe to the OnTargetReached event
            _robotController.OnTargetReached += HandleTargetReached;
            Debug.Log($"{_logPrefix} Subscribed to OnTargetReached event for {gameObject.name}");
        }

        /// <summary>
        /// Handles the OnTargetReached event by executing the planned grasp.
        /// </summary>
        private void HandleTargetReached()
        {
            if (_gripperController == null)
                return;

            // Apply gripper configuration based on grasp parameters
            if (_useGraspPlanning)
            {
                string targetName = _currentTarget != null ? _currentTarget.name : "target";
                Debug.Log(
                    $"{_logPrefix} Target reached");
            }

            // Close gripper
            _gripperController.CloseGrippers();

            // Log grasp execution
            if (_logger != null && _logGraspActions && !string.IsNullOrEmpty(_currentActionId))
            {
                _logger.CompleteAction(
                    _currentActionId,
                    success: true,
                    qualityScore: 1.0f
                );
            }

            Debug.Log(
                $"{_logPrefix} {gameObject.name} executing grasp after reaching target"
            );
        }

        /// <summary>
        /// Opens the gripper (e.g., to release an object).
        /// </summary>
        public void ReleaseGrasp()
        {
            if (_gripperController != null)
            {
                _gripperController.OpenGrippers();

                // Log release action
                if (_logger != null && _logGraspActions)
                {
                    string actionId = _logger.StartAction(
                        "release_grasp",
                        ActionType.Manipulation,
                        new[] { _robotController.robotId },
                        startPos: _gripperController.transform.position,
                        description: "Releasing object"
                    );

                    _logger.CompleteAction(
                        actionId,
                        success: true,
                        qualityScore: 1.0f
                    );
                }

                Debug.Log($"{_logPrefix} {gameObject.name} releasing grasp");
            }
        }

        /// <summary>
        /// Unity OnDestroy callback - unsubscribe from events to prevent memory leaks.
        /// </summary>
        private void OnDestroy()
        {
            if (_robotController != null)
            {
                _robotController.OnTargetReached -= HandleTargetReached;
            }
        }
    }
}
