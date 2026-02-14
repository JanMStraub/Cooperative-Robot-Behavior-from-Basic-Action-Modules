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

    public class GripperController : MonoBehaviour
    {
        [Header("Gripper References")]
        public ArticulationBody leftGripper;
        public ArticulationBody rightGripper;

        [Header("Attachment Point")]
        [Tooltip("Transform to attach grasped objects to (usually end effector base)")]
        public Transform attachmentPoint;

        [Header("Motion Parameters")]
        [Tooltip("Speed of gripper opening/closing in meters per second.")]
        public float gripSpeed = 0.05f;

        [Tooltip(
            "Maximum distance the target can be ahead of the physical finger. Prevents tunneling."
        )]
        public float maxTargetLead = 0.005f;

        [Tooltip("Stiffness (P-Gain).")]
        public float stiffness = 5000f;

        [Tooltip("Damping (D-Gain).")]
        public float damping = 500f;

        [Tooltip("Force Limit.")]
        public float maxForce = 200f;

        [Range(0f, 1f)]
        public float targetPosition = 1f;

        private float _currentPhysicalTarget;
        private float _cachedLowerLimit;
        private float _cachedUpperLimit;
        private bool _invertMapping;
        private bool _wasMoving;
        private bool _isInitialized = false;
        public bool IsMoving { get; private set; }

        private GameObject _graspedObject;
        private Transform _graspedObjectOriginalParent;
        private bool _isHoldingObject = false;
        private GameObject _targetObjectToGrasp;
        private bool _shouldAttachOnClose = false;
        private const string _logPrefix = "[GRIPPER_CONTROLLER]";

        public event System.Action OnGripperActionComplete;

        /// <summary>
        /// Check if the gripper is currently holding an object.
        /// </summary>
        public bool IsHoldingObject => _isHoldingObject;

        /// <summary>
        /// Get the currently grasped object (if any).
        /// </summary>
        public GameObject GraspedObject => _graspedObject;

        private void Awake()
        {
            if (leftGripper == null || rightGripper == null)
            {
                Debug.LogError($"{_logPrefix} Gripper references not assigned!");
            }
        }

        private void Start()
        {
            if (leftGripper == null || rightGripper == null)
                return;

            CacheLimits();

            // Read the initial target BEFORE SetupDrive potentially changes it
            float initialTarget = leftGripper.xDrive.target;

            SetupDrive(leftGripper);
            SetupDrive(rightGripper);

            // Initialize from the saved initial target
            _currentPhysicalTarget = initialTarget;
            targetPosition = MapPhysicalToNormalized(initialTarget);

            _isInitialized = true;
        }

        private void Update()
        {
            if (!_isInitialized || leftGripper == null)
                return;

            if (leftGripper.jointPosition.dofCount == 0)
                return;

            float goalPhysicalPosition = MapNormalizedToPhysical(targetPosition);
            float currentRealPosition = leftGripper.jointPosition[0];

            float nextTargetStep = Mathf.MoveTowards(
                _currentPhysicalTarget,
                goalPhysicalPosition,
                gripSpeed * Time.deltaTime
            );

            bool isClosing = goalPhysicalPosition < currentRealPosition;

            if (isClosing)
            {
                float minAllowed = currentRealPosition - maxTargetLead;
                _currentPhysicalTarget = Mathf.Max(nextTargetStep, minAllowed);
            }
            else
            {
                _currentPhysicalTarget = nextTargetStep;
            }

            bool isAtGoal = Mathf.Abs(_currentPhysicalTarget - goalPhysicalPosition) < 0.0001f;
            bool isStalled = Mathf.Abs(nextTargetStep - _currentPhysicalTarget) > 0.0001f;

            if (!isAtGoal && !isStalled)
            {
                IsMoving = true;
                SetDriveTarget(leftGripper, _currentPhysicalTarget);
                SetDriveTarget(rightGripper, _currentPhysicalTarget);
            }
            else
            {
                if (_wasMoving)
                {
                    if (
                        _shouldAttachOnClose
                        && _targetObjectToGrasp != null
                        && targetPosition < 0.1f
                    )
                    {
                        AttachObject(_targetObjectToGrasp);
                        _targetObjectToGrasp = null;
                        _shouldAttachOnClose = false;
                    }

                    OnGripperActionComplete?.Invoke();
                }
                IsMoving = false;
            }

            _wasMoving = IsMoving;
        }

        public void SetGripperPosition(float normalizedPosition)
        {
            float newTarget = Mathf.Clamp01(normalizedPosition);

            if (Mathf.Abs(newTarget - targetPosition) > 0.0001f)
            {
                _wasMoving = false;
                IsMoving = true;
            }

            targetPosition = newTarget;
        }

        /// <summary>
        /// Close the grippers. If a target object was set, it will be attached when closing completes.
        /// </summary>
        public void CloseGrippers() => SetGripperPosition(0f);

        /// <summary>
        /// Open the grippers. Automatically detaches any held object first.
        /// </summary>
        public void OpenGrippers()
        {
            if (_isHoldingObject)
            {
                GameObject objectToMonitor = _graspedObject;
                DetachObject();
                StartCoroutine(MonitorObjectPosition(objectToMonitor));
            }

            SetGripperPosition(1f);
        }

        /// <summary>
        /// Debug coroutine to monitor if an object's position changes unexpectedly after release.
        /// </summary>
        private System.Collections.IEnumerator MonitorObjectPosition(GameObject obj)
        {
            if (obj == null)
                yield break;

            Vector3 releasePosition = obj.transform.position;
            Debug.Log($"{_logPrefix} Monitoring object '{obj.name}' released at {releasePosition}");

            for (int i = 0; i < 10; i++)
            {
                yield return null;
                Vector3 currentPos = obj.transform.position;
                float distance = Vector3.Distance(releasePosition, currentPos);

                if (distance > 0.01f)
                {
                    Debug.LogWarning(
                        $"{_logPrefix} Frame {i}: Object moved {distance:F3}m! "
                            + $"From {releasePosition} to {currentPos}"
                    );
                }
            }

            Debug.Log(
                $"{_logPrefix} Monitoring complete. Final position: {obj.transform.position}"
            );
        }

        public void ResetGrippers()
        {
            if (leftGripper == null || rightGripper == null)
                return;

            // Ensure limits are cached before mapping (handles call before Start())
            if (_cachedUpperLimit == 0f)
                CacheLimits();

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
            // Preserve the target value from the prefab
            // (don't reset it - the prefab target is the initial gripper position)
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

        #region Object Attachment

        /// <summary>
        /// Set the target object to grasp. This object will be automatically attached when CloseGrippers() completes.
        /// </summary>
        /// <param name="obj">GameObject to grasp</param>
        public void SetTargetObject(GameObject obj)
        {
            _targetObjectToGrasp = obj;
            _shouldAttachOnClose = true;
        }

        /// <summary>
        /// Clear the target object, preventing automatic attachment on gripper close.
        /// </summary>
        public void ClearTargetObject()
        {
            _targetObjectToGrasp = null;
            _shouldAttachOnClose = false;
        }

        /// <summary>
        /// Attach an object to the gripper (makes it a child of the attachment point).
        /// Disables physics on the object so it moves with the gripper.
        /// This is called automatically when the gripper closes if a target object was set.
        /// Handles handoff: if object is held by another gripper, detaches from that gripper first.
        /// </summary>
        /// <param name="obj">GameObject to attach</param>
        public void AttachObject(GameObject obj)
        {
            if (obj == null)
            {
                Debug.LogWarning($"{_logPrefix} Cannot attach null object");
                return;
            }

            if (attachmentPoint == null)
            {
                Debug.LogError($"{_logPrefix} No attachment point assigned! Cannot attach object.");
                return;
            }

            // Check if object is currently held by another gripper (handoff scenario)
            GripperController otherGripper = FindGripperHoldingObject(obj);
            if (otherGripper != null && otherGripper != this)
            {
                Debug.Log(
                    $"{_logPrefix} Handoff detected: transferring '{obj.name}' from another gripper"
                );
                // Force detach from other gripper without re-enabling physics
                otherGripper.ForceReleaseForHandoff();
            }

            // Store original parent for later release (only if not a handoff)
            if (otherGripper == null)
            {
                _graspedObjectOriginalParent = obj.transform.parent;
            }

            // Disable physics on the object so it moves with the gripper
            Rigidbody rb = obj.GetComponent<Rigidbody>();
            if (rb != null)
            {
                // Set velocities BEFORE making kinematic (can't set velocity on kinematic bodies)
                rb.linearVelocity = Vector3.zero;
                rb.angularVelocity = Vector3.zero;
                rb.isKinematic = true;
                rb.useGravity = false; // Disable gravity while held
            }

            // Parent to attachment point
            obj.transform.SetParent(attachmentPoint, worldPositionStays: true);
            _graspedObject = obj;
            _isHoldingObject = true;

            Debug.Log($"{_logPrefix} Object '{obj.name}' attached to gripper");
        }

        /// <summary>
        /// Find the GripperController that is currently holding the specified object.
        /// </summary>
        /// <param name="obj">Object to check</param>
        /// <returns>GripperController holding the object, or null if not held</returns>
        private static GripperController FindGripperHoldingObject(GameObject obj)
        {
            GripperController[] allGrippers = FindObjectsByType<GripperController>(
                FindObjectsSortMode.None
            );
            foreach (var gripper in allGrippers)
            {
                if (gripper._isHoldingObject && gripper._graspedObject == obj)
                {
                    return gripper;
                }
            }
            return null;
        }

        /// <summary>
        /// Release the object without re-enabling physics (used during handoff).
        /// The receiving gripper will handle the object's physics state.
        /// </summary>
        private void ForceReleaseForHandoff()
        {
            if (!_isHoldingObject || _graspedObject == null)
            {
                return;
            }

            Debug.Log($"{_logPrefix} Force releasing '{_graspedObject.name}' for handoff");

            // Just clear our references - don't unparent or re-enable physics
            // The receiving gripper will handle parenting
            _graspedObject = null;
            _isHoldingObject = false;
        }

        /// <summary>
        /// Detach the currently held object from the gripper.
        /// Places object in world space at its current position and re-enables physics.
        /// </summary>
        public void DetachObject()
        {
            if (!_isHoldingObject || _graspedObject == null)
            {
                return;
            }

            // Store current world position and rotation before changing anything
            Vector3 worldPosition = _graspedObject.transform.position;
            Quaternion worldRotation = _graspedObject.transform.rotation;

            Debug.Log(
                $"{_logPrefix} Detaching object '{_graspedObject.name}' at world position {worldPosition}"
            );

            // Get Rigidbody before making changes
            Rigidbody rb = _graspedObject.GetComponent<Rigidbody>();

            // CRITICAL: Unparent while still kinematic to avoid physics snap-back
            _graspedObject.transform.SetParent(null, worldPositionStays: true);

            // Re-apply position explicitly (in case SetParent didn't preserve it perfectly)
            _graspedObject.transform.position = worldPosition;
            _graspedObject.transform.rotation = worldRotation;

            // If there's a Rigidbody, handle physics carefully
            if (rb != null)
            {
                // While still kinematic, explicitly set the Rigidbody's position
                // This ensures the physics engine knows where the object is
                rb.position = worldPosition;
                rb.rotation = worldRotation;

                // Make it non-kinematic FIRST (required before setting velocities)
                rb.isKinematic = false;

                // Re-enable gravity (critical for object to fall when released)
                rb.useGravity = true;

                // NOW zero out velocities (only works on non-kinematic bodies)
                rb.linearVelocity = Vector3.zero;
                rb.angularVelocity = Vector3.zero;

                Debug.Log(
                    $"{_logPrefix} Re-enabled physics at position {rb.position}, gravity enabled, velocities zeroed"
                );
            }

            _graspedObject = null;
            _graspedObjectOriginalParent = null;
            _isHoldingObject = false;
        }

        /// <summary>
        /// Release the currently held object (alias for DetachObject).
        /// </summary>
        public void ReleaseObject() => DetachObject();

        #endregion
    }
}
