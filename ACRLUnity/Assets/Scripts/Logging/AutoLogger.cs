using System.Collections.Generic;
using Core;
using Robotics;
using UnityEngine;
using Utilities;

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
        private MainLogger _logger;
        private RobotController _robotController;
        private GripperController _gripperController;

        // State tracking
        private Vector3 _lastTarget;
        private float _lastGripperPosition;
        private string _currentMovementActionId;
        private string _currentGripperActionId;
        private bool _isInitialized;

        // Helper variables
        private const string _logPrefix = "[AUTO_LOGGER]";

        private void Start()
        {
            Initialize();
        }

        private void Initialize()
        {
            _logger = MainLogger.Instance;
            if (_logger == null)
            {
                Debug.LogWarning(
                    $"{_logPrefix} MainLogger not found. Auto-logging disabled for {gameObject.name}"
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

            // Auto-register scene objects using ObjectRegistry
            if (autoRegisterObjects)
            {
                RegisterSceneObjectsViaRegistry();
            }

            _isInitialized = true;
            Debug.Log($"{_logPrefix} Initialized for {robotId}");
        }

        private void Update()
        {
            if (!enableAutoLogging || !_isInitialized)
                return;

            // Monitor robot movement
            if (
                logMovement
                && _robotController != null
                && _robotController.GetCurrentTarget().HasValue
            )
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
            Vector3? currentTargetNullable = _robotController.GetCurrentTarget();
            if (!currentTargetNullable.HasValue)
                return;

            Vector3 currentTarget = currentTargetNullable.Value;

            // Target changed
            if (Vector3.Distance(currentTarget, _lastTarget) > RobotConstants.MOVEMENT_THRESHOLD)
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
                && _robotController.GetDistanceToTarget() < RobotConstants.DEFAULT_CONVERGENCE_THRESHOLD
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
                && Mathf.Abs(_gripperController.targetPosition - currentPosition) < RobotConstants.MOVEMENT_THRESHOLD
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

            if (Mathf.Abs(targetPosition - upperLimit) < RobotConstants.DEFAULT_CONVERGENCE_THRESHOLD)
                return "open_gripper";
            else if (Mathf.Abs(targetPosition - lowerLimit) < RobotConstants.DEFAULT_CONVERGENCE_THRESHOLD)
                return "close_gripper";
            else
                return "adjust_gripper";
        }

        /// <summary>
        /// Registers scene objects for logging using centralized ObjectRegistry service.
        /// This eliminates code duplication with MainLogger.
        /// </summary>
        private void RegisterSceneObjectsViaRegistry()
        {
            // Ensure ObjectRegistry exists
            if (ObjectRegistry.Instance == null)
            {
                Debug.LogWarning($"{_logPrefix} ObjectRegistry not found in scene. Creating one.");
                var registryGO = new GameObject("ObjectRegistry");
                registryGO.AddComponent<ObjectRegistry>();
            }

            // Subscribe to registration events
            ObjectRegistry.Instance.OnObjectRegistered += HandleObjectRegistered;

            // Use centralized registry to find and register scene objects
            int count = ObjectRegistry.Instance.RegisterSceneObjects(
                includeColliders: true,
                includeTrackpoints: true
            );

            Debug.Log($"{_logPrefix} Registered {count} objects via ObjectRegistry");
        }

        /// <summary>
        /// Handles object registration events from ObjectRegistry.
        /// Delegates to MainLogger for actual tracking.
        /// </summary>
        private void HandleObjectRegistered(GameObject obj, ObjectRegistry.ObjectInfo info)
        {
            if (_logger == null || obj == null)
                return;

            // Delegate to MainLogger's registration
            _logger.RegisterObject(obj, info.ObjectType, info.IsGraspable);
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

            // Unsubscribe from ObjectRegistry events
            if (ObjectRegistry.Instance != null)
            {
                ObjectRegistry.Instance.OnObjectRegistered -= HandleObjectRegistered;
            }
        }

        // Editor helpers
#if UNITY_EDITOR
        [ContextMenu("Force Log Movement")]
        private void ForceLogMovement()
        {
            if (_robotController != null && _logger != null)
            {
                Vector3? currentTarget = _robotController.GetCurrentTarget();
                if (currentTarget.HasValue)
                {
                    StartMovement(currentTarget.Value);
                }
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
