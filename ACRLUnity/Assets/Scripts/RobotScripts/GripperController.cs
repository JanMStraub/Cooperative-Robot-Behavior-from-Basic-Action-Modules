/*
 * GripperController.cs
 *
 * Optimized by: Gemini
 * Original Author: Fabian Kontor / Modified by Jan M. Straub
 *
 * Description:
 * Provides smooth control over AR4 gripper using ArticulationBody components.
 * Optimization: Reduced PhysX overhead, cached limits, and cleaner state management.
 */

using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace Robotics
{
    using Grasp;

    #region Editor
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
            EditorGUILayout.LabelField("Manual Control", EditorStyles.boldLabel);

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Open (1.0)"))
                controller.OpenGrippers();
            if (GUILayout.Button("Close (0.0)"))
                controller.CloseGrippers();
            EditorGUILayout.EndHorizontal();

            // Slider Control
            EditorGUI.BeginChangeCheck();
            float newPos = EditorGUILayout.Slider(
                "Target Position",
                controller.targetPosition,
                0f,
                1f
            );
            if (EditorGUI.EndChangeCheck())
            {
                Undo.RecordObject(controller, "Change Gripper Target");
                controller.SetGripperPosition(newPos);
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Debug Info", EditorStyles.boldLabel);
            GUI.enabled = false;
            EditorGUILayout.Toggle("Is Moving", controller.IsMoving);
            EditorGUILayout.FloatField("Current Drive Target", controller.CurrentDriveTarget);
            GUI.enabled = true;
        }
    }
#endif
    #endregion

    [RequireComponent(typeof(Transform))]
    public class GripperController : MonoBehaviour
    {
        [Header("Gripper References")]
        [Tooltip("Primary gripper joint (gripper_jaw1_joint in URDF)")]
        public ArticulationBody leftGripper;

        [Tooltip("Mimic gripper joint (gripper_jaw2_joint in URDF) - automatically follows leftGripper")]
        public ArticulationBody rightGripper;

        [Header("Control Parameters")]
        [Tooltip("Maximum force the gripper can apply.")]
        public float maxForce = 100f;

        [Tooltip("Stiffness of the grip. Higher = more rigid.")]
        public float stiffness = 2000f;

        [Tooltip("Damping of the grip. Higher = less oscillation.")]
        public float damping = 500f;

        [Tooltip("Smooth interpolation time in seconds.")]
        public float smoothTime = 0.3f;

        [Range(0f, 1f)]
        [SerializeField]
        public float targetPosition = 0f; // 0 = Closed, 1 = Open

        [Header("Gripper Geometry")]
        public GripperGeometry gripperGeometry = GripperGeometry.Default();
        public GripperGeometry Geometry => gripperGeometry;

        // Events
        public event System.Action OnGripperActionComplete;

        // State properties
        public bool IsMoving { get; private set; }
        public float CurrentDriveTarget => _currentDriveTarget;

        // Internals
        private float _currentVelocity;
        private float _currentDriveTarget = 0f;
        private float _cachedLowerLimit;
        private float _cachedUpperLimit;
        private bool _invertMapping;
        private bool _initialized;

        // Constants
        private const float MOVE_THRESHOLD = 0.001f; // Minimum delta to process movement
        private const float COMPLETION_THRESHOLD = 0.2f; // Distance to target to consider "Complete"
        private const string LOG_PREFIX = "[GRIPPER]";

        #region Unity Lifecycle

        private void Awake()
        {
            ValidateAndSetup();
        }

        private void Start()
        {
            if (leftGripper == null || rightGripper == null)
                return;

            CacheLimits();

            // Detect initial state
            float startJointPos =
                leftGripper.jointPosition.dofCount > 0 ? leftGripper.jointPosition[0] : 0f;

            // If the joint is not at 0 (meaning physics has settled or pre-configured), align target to it
            if (Mathf.Abs(startJointPos) > Mathf.Epsilon)
            {
                _currentDriveTarget = startJointPos;
                // Reverse calculate targetPosition (0-1) from physical position
                targetPosition = MapPhysicalToNormalized(startJointPos);
            }
            else
            {
                // Otherwise force the physical gripper to match the inspector setting
                _currentDriveTarget = MapNormalizedToPhysical(targetPosition);
                ApplyTargetToGrippersImmediate(_currentDriveTarget);
            }

            _initialized = true;
        }

        private void OnValidate()
        {
            // Ensure parameters are valid in Editor
            maxForce = Mathf.Max(0, maxForce);
            smoothTime = Mathf.Max(0, smoothTime);

            // Update drive params both in Edit and Play mode
            if (leftGripper != null && rightGripper != null)
            {
                SetupDrive(leftGripper);
                SetupDrive(rightGripper);
            }
        }

        private void Update()
        {
            if (!_initialized || leftGripper == null)
                return;

            // 1. Calculate the physical goal based on the 0-1 target
            float physicalGoal = MapNormalizedToPhysical(targetPosition);

            // 2. Determine if we need to process movement
            // We move if the current drive target is not yet at the goal
            bool needsMovement = Mathf.Abs(_currentDriveTarget - physicalGoal) > MOVE_THRESHOLD;

            if (needsMovement)
            {
                IsMoving = true;

                // Smoothly interpolate the drive target
                _currentDriveTarget = Mathf.SmoothDamp(
                    _currentDriveTarget,
                    physicalGoal,
                    ref _currentVelocity,
                    smoothTime
                );

                ApplyTargetToGrippers(_currentDriveTarget);
            }
            else if (IsMoving)
            {
                // 3. We were moving, but the drive target has settled near the goal.
                // Now check if the actual PHYSICS joints have caught up.
                float actualPosition =
                    leftGripper.jointPosition.dofCount > 0
                        ? leftGripper.jointPosition[0]
                        : _currentDriveTarget;

                if (Mathf.Abs(actualPosition - physicalGoal) <= COMPLETION_THRESHOLD)
                {
                    FinalizeMovement(physicalGoal);
                }
            }
        }

        #endregion

        #region Public Control API

        public void SetGripperPosition(float normalizedPosition)
        {
            targetPosition = Mathf.Clamp01(normalizedPosition);
            IsMoving = true;
            // Reset velocity allows for responsive direction changes
            _currentVelocity = 0f;
        }

        public void CloseGrippers() => SetGripperPosition(0f);

        public void OpenGrippers() => SetGripperPosition(1f);

        /// <summary>
        /// Hard reset of the grippers to Open state. Bypass smoothing.
        /// </summary>
        public void ResetGrippers()
        {
            targetPosition = 1f;
            float physicalOpen = MapNormalizedToPhysical(1f);

            ApplyTargetToGrippersImmediate(physicalOpen);

            // Reset ArticulationBody physics state
            ResetArticulationBody(leftGripper, physicalOpen);
            ResetArticulationBody(rightGripper, physicalOpen);

            _currentDriveTarget = physicalOpen;
            _currentVelocity = 0f;
            IsMoving = false;
        }

        #endregion

        #region Internal Logic

        private void ValidateAndSetup()
        {
            if (leftGripper == null || rightGripper == null)
            {
                Debug.LogError($"{LOG_PREFIX} References missing!");
                return;
            }
            SetupDrive(leftGripper);
            SetupDrive(rightGripper);
        }

        private void CacheLimits()
        {
            _cachedLowerLimit = leftGripper.xDrive.lowerLimit;
            _cachedUpperLimit = leftGripper.xDrive.upperLimit;

            // Detection rule: Invert for non-URDF (negative lower limit)
            _invertMapping = _cachedLowerLimit < 0f;
        }

        private float MapNormalizedToPhysical(float normalized01)
        {
            // 0 = Closed, 1 = Open
            // If Inverted: 0 -> Upper, 1 -> Lower (Logic: Lower is Open in non-URDF)
            // If Normal:   0 -> Lower, 1 -> Upper (Logic: Upper is Open in URDF)

            float t = _invertMapping ? (1f - normalized01) : normalized01;
            return Mathf.Lerp(_cachedLowerLimit, _cachedUpperLimit, t);
        }

        private float MapPhysicalToNormalized(float physicalPos)
        {
            float t = Mathf.InverseLerp(_cachedLowerLimit, _cachedUpperLimit, physicalPos);
            return _invertMapping ? (1f - t) : t;
        }

        /// <summary>
        /// Configures ArticulationBody drive parameters for gripper control.
        /// Note: URDF dynamics tags (damping/friction) provide baseline values that Unity imports.
        /// This method applies runtime overrides for fine-tuning gripper behavior.
        /// </summary>
        private void SetupDrive(ArticulationBody gripper)
        {
            // Get current drive configuration
            ArticulationDrive drive = gripper.xDrive;

            // Configure for position control with custom PD gains
            drive.driveType = ArticulationDriveType.Target;
            drive.stiffness = stiffness;
            drive.damping = damping;
            drive.forceLimit = maxForce;

            // Apply the modified drive back to the articulation body
            gripper.xDrive = drive;
        }

        private void ApplyTargetToGrippers(float target)
        {
            // Optimization: Only talk to physics engine if value changed significantly
            if (Mathf.Abs(leftGripper.xDrive.target - target) > Mathf.Epsilon)
            {
                SetDriveTarget(leftGripper, target);
                SetDriveTarget(rightGripper, target);
            }
        }

        private void ApplyTargetToGrippersImmediate(float target)
        {
            SetDriveTarget(leftGripper, target);
            SetDriveTarget(rightGripper, target);
        }

        private void SetDriveTarget(ArticulationBody body, float val)
        {
            var drive = body.xDrive;
            drive.target = val;
            body.xDrive = drive;
        }

        private void FinalizeMovement(float finalTarget)
        {
            // Snap to exact target to prevent micro-jitter
            ApplyTargetToGrippersImmediate(finalTarget);
            _currentDriveTarget = finalTarget;
            _currentVelocity = 0f;
            IsMoving = false;
            OnGripperActionComplete?.Invoke();
        }

        private void ResetArticulationBody(ArticulationBody body, float pos)
        {
            body.jointPosition = new ArticulationReducedSpace(pos);
            body.jointVelocity = new ArticulationReducedSpace(0f);
            body.jointForce = new ArticulationReducedSpace(0f);
        }

        #endregion
    }
}
