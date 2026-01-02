/*
 * GripperController.cs
 *
 * Author: Fabian Kontor
 * Source: https://github.com/zebleck/AR4/blob/mlagents/Scripts/GripperController.cs
 * Modified by: Jan M. Straub
 *
 * Description:
 * Provides smooth control over AR4 gripper using ArticulationBody components.
 */

using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace Robotics
{
    using Grasp;
#if UNITY_EDITOR
    [CustomEditor(typeof(GripperController))]
    public class GripperControllerEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            var controller = (GripperController)target;

            if (controller.leftGripper == null || controller.rightGripper == null)
            {
                EditorGUILayout.HelpBox(
                    "Assign both gripper references to enable manual control.",
                    MessageType.Error
                );
                return;
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Gripper Control", EditorStyles.boldLabel);

            if (GUILayout.Button("Open Grippers"))
                controller.OpenGrippers();

            if (GUILayout.Button("Close Grippers"))
                controller.CloseGrippers();

            float lower = controller.leftGripper.xDrive.lowerLimit;
            float upper = controller.leftGripper.xDrive.upperLimit;

            float newPosition = GUILayout.HorizontalSlider(controller.targetPosition, lower, upper);

            if (!Mathf.Approximately(newPosition, controller.targetPosition))
            {
                controller.targetPosition = newPosition;
                EditorUtility.SetDirty(controller);
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Debug Info", EditorStyles.boldLabel);
            EditorGUILayout.LabelField(
                "Left Target",
                controller.leftGripper.xDrive.target.ToString("F2")
            );
            EditorGUILayout.LabelField(
                "Right Target",
                controller.rightGripper.xDrive.target.ToString("F2")
            );
        }
    }
#endif

    [RequireComponent(typeof(Transform))]
    public class GripperController : MonoBehaviour
    {
        [Header("Gripper References")]
        public ArticulationBody leftGripper;
        public ArticulationBody rightGripper;

        [Header("Control Parameters")]
        public float maxForce = 1000f;

        [Tooltip("Smooth interpolation time in seconds")]
        public float smoothTime = 0.3f;

        [Range(0f, 1f)]
        public float targetPosition = 0f;

        [Header("Gripper Geometry")]
        [Tooltip("Gripper geometry for grasp planning validation")]
        public GripperGeometry gripperGeometry = GripperGeometry.Default();

        /// <summary>
        /// Get gripper geometry for grasp planning.
        /// </summary>
        public GripperGeometry Geometry => gripperGeometry;

        // Helper variables
        private float _currentVelocity;
        private float _currentTarget = 0f;
        private const string _logPrefix = "[GRIPPER_CONTROLLER]";
        private bool _isMoving;
        private float _completionThreshold = 0.01f; // Threshold for considering gripper at target
        private bool _initialized = false; // Track if _currentTarget has been initialized

        public float CurrentPosition
        {
            get
            {
                if (leftGripper == null)
                    return 0f;

                // Try to read joint position directly (works if manually set or if physics is running)
                var jointPos = leftGripper.jointPosition;
                if (jointPos.dofCount > 0)
                {
                    return jointPos[0];
                }

                // Fallback: return drive target if joint position unavailable
                return leftGripper.xDrive.target;
            }
        }

        /// <summary>
        /// Event fired when gripper reaches its target position
        /// </summary>
        public event System.Action OnGripperActionComplete;

        /// <summary>
        /// Whether the gripper is currently moving to a target
        /// </summary>
        public bool IsMoving => _isMoving;

        /// <summary>
        /// Configures the articulation drive parameters for a gripper joint.
        /// </summary>
        /// <param name="gripper">The gripper ArticulationBody to configure</param>
        private void SetupDrive(ArticulationBody gripper)
        {
            var drive = gripper.xDrive;
            drive.forceLimit = maxForce;
            drive.stiffness = 2000f;
            drive.damping = 400f;
            gripper.xDrive = drive;
        }

        /// <summary>
        /// Applies a target position to both left and right grippers.
        /// </summary>
        /// <param name="target">The target position to apply</param>
        private void ApplyTargetToGrippers(float target)
        {
            ApplyDriveTarget(leftGripper, target);
            ApplyDriveTarget(rightGripper, target);
        }

        /// <summary>
        /// Applies a drive target to a specific gripper.
        /// </summary>
        /// <param name="gripper">The gripper ArticulationBody to apply the target to</param>
        /// <param name="target">The target position value</param>
        private void ApplyDriveTarget(ArticulationBody gripper, float target)
        {
            var drive = gripper.xDrive;
            drive.target = target;
            gripper.xDrive = drive;
        }

        /// <summary>
        /// Sets the gripper position using a normalized value (0 to 1).
        /// </summary>
        /// <param name="normalizedPosition">The normalized position (0 = closed, 1 = open)</param>
        public void SetGripperPosition(float normalizedPosition)
        {
            targetPosition = Mathf.Clamp01(normalizedPosition);
            // Reset velocity for immediate response to new command
            _currentVelocity = 0f;
            _isMoving = true;
        }

        /// <summary>
        /// Closes both grippers to their upper limit.
        /// </summary>
        public void CloseGrippers()
        {
            targetPosition = 1.0f;
            // Reset velocity for immediate response to new command
            _currentVelocity = 0f;
            _isMoving = true;
        }

        /// <summary>
        /// Opens both grippers to their lower limit.
        /// </summary>
        public void OpenGrippers()
        {
            targetPosition = 0.0f;
            // Reset velocity for immediate response to new command
            _currentVelocity = 0f;
            _isMoving = true;
        }

        /// <summary>
        /// Resets both grippers to their default position and clears their physics state.
        /// </summary>
        public void ResetGrippers()
        {
            targetPosition = 0f;
            ResetGripper(leftGripper, targetPosition);
            ResetGripper(rightGripper, targetPosition);
            _isMoving = false; // Reset is immediate, no need for smooth movement
        }

        /// <summary>
        /// Getter for the gripper target position to determine if it is open or closed
        /// </summary>
        public float GetTargetPosition()
        {
            return targetPosition;
        }

        /// <summary>
        /// Resets a single gripper to its default state.
        /// </summary>
        /// <param name="gripper">The gripper ArticulationBody to reset</param>
        /// <param name="normalizedPosition">The normalized position to reset to (0 = closed, 1 = open)</param>
        private void ResetGripper(ArticulationBody gripper, float normalizedPosition)
        {
            float lower = gripper.xDrive.lowerLimit;
            float upper = gripper.xDrive.upperLimit;
            // Invert mapping because AR4 gripper has lower=-0.015 (open) and upper=0.0 (closed)
            float targetValue = Mathf.Lerp(lower, upper, 1f - normalizedPosition);

            ApplyDriveTarget(gripper, targetValue);
            gripper.jointPosition = new ArticulationReducedSpace(targetValue);
            gripper.jointVelocity = new ArticulationReducedSpace(0f);
            gripper.jointForce = new ArticulationReducedSpace(0f);
        }

        /// <summary>
        /// Unity Awake callback - validates gripper references and sets up drive parameters.
        /// </summary>
        private void Awake()
        {
            if (leftGripper == null || rightGripper == null)
            {
                Debug.LogError($"{_logPrefix} Gripper references not assigned!");
                return;
            }

            SetupDrive(leftGripper);
            SetupDrive(rightGripper);
        }

        /// <summary>
        /// Unity Start callback - initializes the current target to match the initial target position.
        /// Also sets up drives if grippers were assigned after Awake (e.g., in tests).
        /// </summary>
        private void Start()
        {
            // Set up drives if grippers are now assigned (handles case where they're assigned after Awake)
            if (leftGripper != null && rightGripper != null)
            {
                SetupDrive(leftGripper);
                SetupDrive(rightGripper);

                // Map normalized position [0,1] to actual joint limits
                float lower = leftGripper.xDrive.lowerLimit;
                float upper = leftGripper.xDrive.upperLimit;

                float mappedTarget = Mathf.Lerp(lower, upper, targetPosition);

                // Check if joint position was manually set (for tests) - if so, use that as initial target
                var jointPos = leftGripper.jointPosition;
                if (jointPos.dofCount > 0 && jointPos[0] != 0f)
                {
                    _currentTarget = jointPos[0];
                    _initialized = true;
                    // If joint position differs from targetPosition, trigger movement
                    if (Mathf.Abs(_currentTarget - mappedTarget) > _completionThreshold)
                    {
                        _isMoving = true;
                    }
                    return;
                }

                // Otherwise initialize _currentTarget and grippers to the initial targetPosition
                _currentTarget = mappedTarget;
                ApplyTargetToGrippers(_currentTarget);
                _initialized = true;

                // Log the gripper range for debugging
                Debug.Log($"{_logPrefix} Initialized with range [{lower:F3}, {upper:F3}], target position: {targetPosition} -> {mappedTarget:F3}");
            }
        }

        /// <summary>
        /// Unity Update callback - smoothly interpolates gripper position towards target position using SmoothDamp.
        /// Continuously applies the target to ensure the gripper reaches its goal even under load.
        /// </summary>
        private void Update()
        {
            // Early exit if grippers are not properly initialized
            if (leftGripper == null || rightGripper == null)
                return;

            // Map normalized position [0,1] to actual joint limits
            float lower = leftGripper.xDrive.lowerLimit;
            float upper = leftGripper.xDrive.upperLimit;
            // Invert mapping because AR4 gripper has lower=-0.015 (open) and upper=0.0 (closed)
            float mappedTarget = Mathf.Lerp(lower, upper, 1f - targetPosition);

            // Initialize _currentTarget on first update
            if (!_initialized)
            {
                // Try to initialize from current joint position if available
                var jointPos = leftGripper.jointPosition;
                if (jointPos.dofCount > 0)
                {
                    _currentTarget = jointPos[0];
                }
                else
                {
                    // Joint position not yet initialized, use mapped target
                    _currentTarget = mappedTarget;
                }
                _initialized = true;
            }

            // Smooth interpolation using SmoothDamp for natural acceleration/deceleration
            float previousTarget = _currentTarget;
            _currentTarget = Mathf.SmoothDamp(
                _currentTarget,
                mappedTarget,
                ref _currentVelocity,
                smoothTime
            );

            // Always apply the target to ensure continuous force application
            ApplyTargetToGrippers(_currentTarget);

            // Get actual joint position to determine if we've reached the target
            float actualPosition = CurrentPosition;

            // Check for completion based on actual position vs mapped target
            bool wasMoving = _isMoving;
            bool reachedTarget = Mathf.Abs(actualPosition - mappedTarget) < _completionThreshold;

            // Update moving state: we're moving if the interpolated target hasn't converged
            // OR if the actual position is far from the target
            _isMoving = Mathf.Abs(_currentTarget - mappedTarget) >= _completionThreshold || !reachedTarget;

            // Fire completion event when we transition from moving to not moving
            if (wasMoving && !_isMoving)
            {
                OnGripperActionComplete?.Invoke();
            }
        }
    }
}
