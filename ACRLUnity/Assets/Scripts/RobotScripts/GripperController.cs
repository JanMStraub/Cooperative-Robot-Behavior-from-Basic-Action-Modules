using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace Robotics
{
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
                return;

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Manual Control", EditorStyles.boldLabel);

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Open"))
                controller.OpenGrippers();
            if (GUILayout.Button("Close"))
                controller.CloseGrippers();
            EditorGUILayout.EndHorizontal();

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

            EditorGUILayout.LabelField($"Is Moving: {controller.IsMoving}");
        }
    }
#endif
    #endregion

    public class GripperController : MonoBehaviour
    {
        [Header("Gripper References")]
        public ArticulationBody leftGripper;
        public ArticulationBody rightGripper;

        [Header("Motion Parameters")]
        [Tooltip("Speed of gripper opening/closing in meters per second.")]
        public float gripSpeed = 0.05f; // Slower speed helps collision detection

        [Tooltip(
            "Maximum distance the target can be ahead of the physical finger. Prevents tunneling."
        )]
        public float maxTargetLead = 0.005f; // 5mm lead max

        [Tooltip("Stiffness (P-Gain).")]
        public float stiffness = 5000f; // Lowered slightly for stability

        [Tooltip("Damping (D-Gain).")]
        public float damping = 500f;

        [Tooltip("Force Limit.")]
        public float maxForce = 200f;

        [Range(0f, 1f)]
        public float targetPosition = 1f;

        // Internal State
        private float _currentPhysicalTarget;
        private float _cachedLowerLimit;
        private float _cachedUpperLimit;
        private bool _invertMapping;
        private bool _wasMoving;
        public bool IsMoving { get; private set; }

        public event System.Action OnGripperActionComplete;

        private void Start()
        {
            if (leftGripper == null || rightGripper == null)
                return;

            CacheLimits();
            SetupDrive(leftGripper);
            SetupDrive(rightGripper);

            float currentPos = leftGripper.jointPosition[0];
            _currentPhysicalTarget = currentPos;
            targetPosition = MapPhysicalToNormalized(currentPos);
        }

        private void Update()
        {
            if (leftGripper == null)
                return;

            float goalPhysicalPosition = MapNormalizedToPhysical(targetPosition);
            float currentRealPosition = leftGripper.jointPosition[0];

            // 1. Calculate where we WANT to go based on speed
            float nextTargetStep = Mathf.MoveTowards(
                _currentPhysicalTarget,
                goalPhysicalPosition,
                gripSpeed * Time.deltaTime
            );

            // 2. THE FIX: Clamp the target so it doesn't go too far past the REAL position
            // This prevents the "Tunneling" effect where the target goes deep inside the cube
            float minAllowed = currentRealPosition - maxTargetLead;
            float maxAllowed = currentRealPosition + maxTargetLead;

            // Apply clamp based on direction
            if (_invertMapping) // If 0 is Upper and 1 is Lower
            {
                // Logic varies by URDF, simple clamp usually suffices
                _currentPhysicalTarget = Mathf.Clamp(nextTargetStep, minAllowed, maxAllowed);
            }
            else
            {
                _currentPhysicalTarget = Mathf.Clamp(nextTargetStep, minAllowed, maxAllowed);
            }

            // Check if we are essentially at the goal (or stalled)
            bool isAtGoal = Mathf.Abs(_currentPhysicalTarget - goalPhysicalPosition) < 0.0001f;
            bool isStalled = Mathf.Abs(nextTargetStep - _currentPhysicalTarget) > 0.0001f; // We wanted to move but got clamped

            if (!isAtGoal && !isStalled)
            {
                IsMoving = true;
                SetDriveTarget(leftGripper, _currentPhysicalTarget);
                SetDriveTarget(rightGripper, _currentPhysicalTarget);
            }
            else
            {
                if (_wasMoving)
                    OnGripperActionComplete?.Invoke();
                IsMoving = false;
            }

            _wasMoving = IsMoving;
        }

        public void SetGripperPosition(float normalizedPosition)
        {
            targetPosition = Mathf.Clamp01(normalizedPosition);
        }

        public void CloseGrippers() => SetGripperPosition(0f);

        public void OpenGrippers() => SetGripperPosition(1f);

        public void ResetGrippers()
        {
            targetPosition = 1f;
            float openPos = MapNormalizedToPhysical(1f);
            _currentPhysicalTarget = openPos;
            SetDriveTarget(leftGripper, openPos);
            SetDriveTarget(rightGripper, openPos);
            ForceJointPosition(leftGripper, openPos);
            ForceJointPosition(rightGripper, openPos);
        }

        private void CacheLimits()
        {
            _cachedLowerLimit = leftGripper.xDrive.lowerLimit;
            _cachedUpperLimit = leftGripper.xDrive.upperLimit;
            _invertMapping = _cachedLowerLimit < 0f;
        }

        private float MapNormalizedToPhysical(float normalized01)
        {
            float t = _invertMapping ? (1f - normalized01) : normalized01;
            return Mathf.Lerp(_cachedLowerLimit, _cachedUpperLimit, t);
        }

        private float MapPhysicalToNormalized(float physical)
        {
            float t = Mathf.InverseLerp(_cachedLowerLimit, _cachedUpperLimit, physical);
            return _invertMapping ? (1f - t) : t;
        }

        private void SetupDrive(ArticulationBody gripper)
        {
            var drive = gripper.xDrive;
            drive.driveType = ArticulationDriveType.Target;
            drive.stiffness = stiffness;
            drive.damping = damping;
            drive.forceLimit = maxForce;
            gripper.xDrive = drive;
        }

        private void SetDriveTarget(ArticulationBody body, float val)
        {
            var drive = body.xDrive;
            drive.target = val;
            body.xDrive = drive;
        }

        private void ForceJointPosition(ArticulationBody body, float val)
        {
            body.jointPosition = new ArticulationReducedSpace(val);
            body.jointVelocity = new ArticulationReducedSpace(0f);
            body.jointForce = new ArticulationReducedSpace(0f);
        }
    }
}
