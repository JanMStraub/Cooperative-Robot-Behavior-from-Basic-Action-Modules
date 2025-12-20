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
        public float maxForce = 100f;

        [Tooltip("Smooth interpolation time in seconds")]
        public float smoothTime = 0.3f;

        [Range(0f, 1f)]
        public float targetPosition = 1f;

        // Helper variables
        private float _currentVelocity;
        private float _currentTarget;
        private const string _logPrefix = "[GRIPPER_CONTROLLER]";
        private bool _isMoving;
        private float _completionThreshold = 0.01f; // Threshold for considering gripper at target

        public float CurrentPosition => leftGripper?.jointPosition[0] ?? 0f;

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
            drive.stiffness = 1000f;
            drive.damping = 100f;
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
        }

        /// <summary>
        /// Opens both grippers to their upper limit.
        /// </summary>
        public void OpenGrippers()
        {
            targetPosition = 1.0f;
            _isMoving = true;
        }

        /// <summary>
        /// Closes both grippers to their lower limit.
        /// </summary>
        public void CloseGrippers()
        {
            targetPosition = 0.0f;
            _isMoving = true;
        }

        /// <summary>
        /// Resets both grippers to their default position and clears their physics state.
        /// </summary>
        public void ResetGrippers()
        {
            targetPosition = 0f;
            ResetGripper(leftGripper);
            ResetGripper(rightGripper);
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
        private void ResetGripper(ArticulationBody gripper)
        {
            ApplyDriveTarget(gripper, 0f);
            gripper.jointPosition = new ArticulationReducedSpace(0f);
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
        /// Unity Update callback - smoothly interpolates gripper position towards target position using SmoothDamp.
        /// </summary>
        private void Update()
        {
            // Early exit if grippers are not properly initialized
            if (leftGripper == null || rightGripper == null)
                return;

            // Map normalized position [0,1] to actual joint limits
            float lower = leftGripper.xDrive.lowerLimit;
            float upper = leftGripper.xDrive.upperLimit;
            float mappedTarget = Mathf.Lerp(lower, upper, targetPosition);

            // Smooth interpolation using SmoothDamp for natural acceleration/deceleration
            _currentTarget = Mathf.SmoothDamp(
                _currentTarget,
                mappedTarget,
                ref _currentVelocity,
                smoothTime
            );

            ApplyTargetToGrippers(_currentTarget);

            // Check for completion
            if (_isMoving && Mathf.Abs(_currentTarget - mappedTarget) < _completionThreshold)
            {
                _isMoving = false;
                OnGripperActionComplete?.Invoke();
            }
        }
    }
}
