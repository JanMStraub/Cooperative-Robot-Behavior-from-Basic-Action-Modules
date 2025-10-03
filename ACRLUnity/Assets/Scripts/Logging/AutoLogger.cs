using System.Collections.Generic;
using UnityEngine;

namespace Logging
{
    /// <summary>
    /// Optional auto-logger component for robots
    /// Attach to RobotController or GripperController for automatic logging
    /// Replaces: EnhancedRobotControllerIntegration + EnhancedGripperIntegration
    /// </summary>
    [RequireComponent(typeof(MonoBehaviour))]
    public class AutoLogger : MonoBehaviour
    {
        [Header("Auto-Logging Settings")]
        [Tooltip("Enable automatic action logging")]
        public bool enableAutoLogging = true;

        [Tooltip("Robot ID (auto-detected if empty)")]
        public string robotId;

        [Tooltip("Log movement actions")]
        public bool logMovement = true;

        [Tooltip("Log gripper actions")]
        public bool logGripper = true;

        [Tooltip("Auto-register objects in scene")]
        public bool autoRegisterObjects = true;

        // Component references
        private RobotLogger _logger;
        private RobotController _robotController;
        private GripperController _gripperController;

        // State tracking
        private Vector3 _lastTarget;
        private float _lastGripperPosition;
        private string _currentMovementActionId;
        private string _currentGripperActionId;
        private bool _isInitialized;

        private void Start()
        {
            Initialize();
        }

        private void Initialize()
        {
            _logger = RobotLogger.Instance;
            if (_logger == null)
            {
                Debug.LogWarning(
                    $"RobotLogger not found. Auto-logging disabled for {gameObject.name}"
                );
                enableAutoLogging = false;
                return;
            }

            // Auto-detect components
            _robotController = GetComponent<RobotController>();
            _gripperController = GetComponent<GripperController>();

            // Auto-detect robot ID
            if (string.IsNullOrEmpty(robotId))
            {
                robotId = _robotController != null ? _robotController.robotId : gameObject.name;
            }

            // Auto-register scene objects
            if (autoRegisterObjects)
            {
                RegisterSceneObjects();
            }

            _isInitialized = true;
            Debug.Log($"AutoLogger initialized for {robotId}");
        }

        private void Update()
        {
            if (!enableAutoLogging || !_isInitialized)
                return;

            // Monitor robot movement
            if (logMovement && _robotController != null)
            {
                MonitorRobotMovement();
            }

            // Monitor gripper
            if (logGripper && _gripperController != null)
            {
                MonitorGripper();
            }
        }

        private void MonitorRobotMovement()
        {
            Vector3 currentTarget = _robotController.GetCurrentTarget();

            // Target changed
            if (Vector3.Distance(currentTarget, _lastTarget) > 0.01f)
            {
                // Complete previous movement if exists
                if (!string.IsNullOrEmpty(_currentMovementActionId))
                {
                    CompleteMovement(false, "Target changed before completion");
                }

                // Start new movement action
                StartMovement(currentTarget);
                _lastTarget = currentTarget;
            }
            // Target reached
            else if (
                !string.IsNullOrEmpty(_currentMovementActionId)
                && _robotController.GetDistanceToTarget() < 0.1f
            )
            {
                CompleteMovement(true);
            }
        }

        private void MonitorGripper()
        {
            float currentPosition = _gripperController.CurrentPosition;

            // Gripper position changed significantly
            if (Mathf.Abs(currentPosition - _lastGripperPosition) > 0.05f)
            {
                // Complete previous gripper action if exists
                if (!string.IsNullOrEmpty(_currentGripperActionId))
                {
                    CompleteGripperAction();
                }

                // Start new gripper action
                StartGripperAction(currentPosition);
                _lastGripperPosition = currentPosition;
            }
            // Gripper reached target
            else if (
                !string.IsNullOrEmpty(_currentGripperActionId)
                && Mathf.Abs(_gripperController.targetPosition - currentPosition) < 0.01f
            )
            {
                CompleteGripperAction();
            }
        }

        private void StartMovement(Vector3 targetPosition)
        {
            _currentMovementActionId = _logger.StartAction(
                "move_to_target",
                ActionType.Movement,
                new[] { robotId },
                startPos: _robotController.endEffectorBase.position,
                targetPos: targetPosition,
                description: $"Moving to {targetPosition}"
            );
        }

        private void CompleteMovement(bool success, string error = null)
        {
            if (string.IsNullOrEmpty(_currentMovementActionId))
                return;

            float distance = _robotController.GetDistanceToTarget();
            float accuracy = Mathf.Max(0f, 1f - distance);

            var metrics = new Dictionary<string, float>
            {
                ["final_distance"] = distance,
                ["position_accuracy"] = accuracy,
            };

            _logger.CompleteAction(_currentMovementActionId, success, accuracy, error, metrics);
            _currentMovementActionId = null;
        }

        private void StartGripperAction(float targetPosition)
        {
            string actionName = DetermineGripperAction(targetPosition);

            _currentGripperActionId = _logger.StartAction(
                actionName,
                ActionType.Manipulation,
                new[] { robotId },
                description: $"Gripper {actionName}"
            );
        }

        private void CompleteGripperAction()
        {
            if (string.IsNullOrEmpty(_currentGripperActionId))
                return;

            float positionError = Mathf.Abs(
                _gripperController.targetPosition - _gripperController.CurrentPosition
            );
            float accuracy = Mathf.Max(0f, 1f - positionError);

            var metrics = new Dictionary<string, float>
            {
                ["position_error"] = positionError,
                ["final_position"] = _gripperController.CurrentPosition,
            };

            _logger.CompleteAction(_currentGripperActionId, true, accuracy, null, metrics);
            _currentGripperActionId = null;
        }

        private string DetermineGripperAction(float targetPosition)
        {
            float upperLimit = _gripperController.leftGripper?.xDrive.upperLimit ?? 1f;
            float lowerLimit = _gripperController.leftGripper?.xDrive.lowerLimit ?? 0f;

            if (Mathf.Abs(targetPosition - upperLimit) < 0.1f)
                return "open_gripper";
            else if (Mathf.Abs(targetPosition - lowerLimit) < 0.1f)
                return "close_gripper";
            else
                return "adjust_gripper";
        }

        private void RegisterSceneObjects()
        {
            // Find all objects with colliders (potential targets)
            var colliders = FindObjectsByType<Collider>(FindObjectsSortMode.None);

            foreach (var collider in colliders)
            {
                var obj = collider.gameObject;

                // Skip robot parts
                if (
                    obj.GetComponent<RobotController>() != null
                    || obj.GetComponent<ArticulationBody>() != null
                )
                    continue;

                // Skip too small objects
                if (collider.bounds.size.magnitude < 0.02f)
                    continue;

                // Register object
                bool isGraspable =
                    obj.GetComponent<Rigidbody>() != null && collider.bounds.size.magnitude < 0.5f;

                _logger.RegisterObject(obj, null, isGraspable);
            }
        }

        // Public methods for manual logging
        public string LogCustomAction(
            string actionName,
            ActionType type,
            string description = null,
            Vector3? targetPos = null,
            string[] objectIds = null
        )
        {
            if (!enableAutoLogging)
                return "";

            return _logger.StartAction(
                actionName,
                type,
                new[] { robotId },
                targetPos: targetPos,
                objectIds: objectIds,
                description: description
            );
        }

        public void CompleteCustomAction(string actionId, bool success, float quality = 0.8f)
        {
            if (!enableAutoLogging)
                return;

            _logger.CompleteAction(actionId, success, quality);
        }

        private void OnDestroy()
        {
            // Complete any active actions
            if (!string.IsNullOrEmpty(_currentMovementActionId))
            {
                CompleteMovement(false, "Component destroyed");
            }

            if (!string.IsNullOrEmpty(_currentGripperActionId))
            {
                CompleteGripperAction();
            }
        }

        // Editor helpers
#if UNITY_EDITOR
        [ContextMenu("Force Log Movement")]
        private void ForceLogMovement()
        {
            if (_robotController != null && _logger != null)
            {
                StartMovement(_robotController.GetCurrentTarget());
            }
        }

        [ContextMenu("Force Complete Movement")]
        private void ForceCompleteMovement()
        {
            if (!string.IsNullOrEmpty(_currentMovementActionId))
            {
                CompleteMovement(true);
            }
        }
#endif
    }
}
