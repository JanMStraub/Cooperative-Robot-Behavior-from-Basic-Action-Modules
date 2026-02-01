using System.Collections;
using Core;
using MathNet.Numerics.LinearAlgebra;
using UnityEngine;
using Utilities;
using Configuration;

namespace Robotics
{
    /// <summary>
    /// Defines how the gripper should approach a target position.
    /// </summary>
    public enum GripperApproach
    {
        /// <summary>Gripper pointing straight down (for picking from above)</summary>
        TopDown,

        /// <summary>Gripper pointing horizontally toward robot base</summary>
        Front,

        /// <summary>Gripper approaching from the side</summary>
        Side,

        /// <summary>Keep current gripper orientation</summary>
        Current,
    }

    /// <summary>
    /// A simplified robot controller for basic IK-based movement.
    /// Uses the existing IKSolver for inverse kinematics computation.
    /// Suitable for single-arm scenarios without advanced grasp planning.
    /// </summary>
    [DefaultExecutionOrder(-10)] // Ensure this initializes before other scripts
    public class SimpleRobotController : MonoBehaviour
    {
        #region --- Configuration ---

        [Header("Robot Configuration")]
        [SerializeField]
        public string robotId = "Robot1";

        [SerializeField]
        public ArticulationBody[] robotJoints;

        [SerializeField]
        public Transform endEffectorBase;

        [SerializeField]
        public Transform IKReferenceFrame;

        [Header("IK Settings")]
        [SerializeField]
        private IKConfig _ikConfig;

        [SerializeField]
        private float _maxJointStepRad = 0.02f; // Reduced from 0.05 to prevent overshoot

        [Header("Speed Settings")]
        [SerializeField]
        private float _minStepSpeedNearTarget = 0.1f; // Reduced for gentler approach

        [SerializeField]
        private float _maxStepSpeed = 0.4f; // Reduced from 0.8 to prevent oscillation

        [Header("Velocity Damping (Anti-Oscillation)")]
        [SerializeField]
        private float _positionGain = 2f; // Kp - position error gain (low for stability)

        [SerializeField]
        private float _velocityGain = 4f; // Kd - velocity damping gain (high for stability)

        [SerializeField]
        private float _leashDistance = 0.10f; // Max distance to "carrot" target (10cm)

        [Header("Gripper")]
        [SerializeField]
        private GripperController _gripperController;

        [SerializeField]
        private bool _closeGripperOnReach = true; // Close gripper when target reached

        [SerializeField]
        private float _gripperCloseDelay = 0.2f; // Delay before closing gripper (seconds)

        [SerializeField]
        private bool _attachObjectOnGrasp = true; // Attach object to gripper when grasped

        [Header("Gripper Orientation")]
        [
            SerializeField,
            Tooltip(
                "Rotation offset to align gripper forward axis (adjust if gripper points wrong direction)"
            )
        ]
        private Vector3 _gripperRotationOffset = Vector3.zero;

        [Header("Operation Mode")]
        [SerializeField]
        [Tooltip("Enable autonomous operation (FixedUpdate loop). Disable when used as backup by RobotController.")]
        private bool _enableAutonomousMode = false;

        [Header("Debug")]
        [SerializeField]
        private bool enableDebugVisualization = true;

        #endregion

        #region --- Internal State ---

        // IK State
        private Vector3 _targetPosition;
        private Quaternion _targetRotation = Quaternion.identity;
        private Transform _ikFrame;
        private IKSolver _ikSolver;
        private JointInfo[] _cachedJointInfos;
        private bool _hasTarget = false;
        private bool _hasReachedTarget = true;
        private float _distanceToTarget;
        private float _sqrDistanceToTarget;

        // Position and orientation state (in IK frame)
        private Vector3 _endEffectorLocalPosition;
        private Quaternion _endEffectorLocalRotation;
        private Vector3 _targetLocalPosition;
        private Quaternion _targetLocalRotation;

        // Grasp State (tracked by object for attachment after reaching target)
        private GameObject _targetObject;

        // Constants
        private const string _logPrefix = "[SIMPLE_ROBOT_CONTROLLER]";
        private const float SqrEpsilon = 1e-6f;

        #endregion

        #region --- Properties & Events ---

        public event System.Action OnTargetReached;

        public bool HasReachedTarget => _hasReachedTarget;
        public float DistanceToTarget => _distanceToTarget;
        public bool HasTarget => _hasTarget;

        #endregion

        #region --- Unity Lifecycle ---

        /// <summary>
        /// Unity Start callback - initializes the robot controller.
        /// </summary>
        private void Start()
        {
            if (_ikConfig == null)
                _ikConfig = ScriptableObject.CreateInstance<IKConfig>();

            InitializeRobot();

            // Only run autonomous startup if enabled (when used standalone, not as backup)
            if (_enableAutonomousMode)
            {
                // Delay gripper and target setup to ensure GripperController has initialized
                StartCoroutine(DelayedStartup());
            }
        }

        /// <summary>
        /// Delayed startup to ensure GripperController is initialized first.
        /// Only runs when in autonomous mode.
        /// </summary>
        private IEnumerator DelayedStartup()
        {
            // Wait one frame for GripperController.Start() to complete
            yield return null;

            // Now open the gripper (this will auto-detach any held object)
            OpenGripper();
        }

        /// <summary>
        /// Unity FixedUpdate callback - performs IK step at physics rate.
        /// Only runs when in autonomous mode. When used as backup by RobotController,
        /// RobotController will call PerformInverseKinematicsStep() manually.
        /// </summary>
        private void FixedUpdate()
        {
            if (!_enableAutonomousMode)
                return;

            if (!_hasTarget || _hasReachedTarget)
                return;

            PerformInverseKinematicsStep();
            DrawDebugVisualization();
        }

        #endregion

        #region --- Initialization ---

        /// <summary>
        /// Initialize the robot controller, IK solver, and joint caches.
        /// </summary>
        private void InitializeRobot()
        {
            if (string.IsNullOrEmpty(robotId))
                robotId = gameObject.name;

            _ikFrame = IKReferenceFrame != null ? IKReferenceFrame : endEffectorBase?.root;

            if (robotJoints == null || robotJoints.Length == 0)
            {
                Debug.LogError($"{_logPrefix} [{robotId}] No robot joints assigned!");
                return;
            }

            if (endEffectorBase == null)
            {
                Debug.LogError($"{_logPrefix} [{robotId}] No end effector base assigned!");
                return;
            }

            // Initialize joint info cache
            int jointCount = robotJoints.Length;
            _cachedJointInfos = new JointInfo[jointCount];

            // Initialize IK solver
            _ikSolver = new IKSolver(jointCount, _ikConfig.dampingFactor);

            // Auto-find gripper controller if not assigned
            if (_gripperController == null)
            {
                _gripperController = GetComponentInChildren<GripperController>();
                if (_gripperController != null)
                    Debug.Log($"{_logPrefix} [{robotId}] Auto-found GripperController");
                else
                    Debug.LogWarning(
                        $"{_logPrefix} [{robotId}] GripperController NOT found! Assign manually in Inspector."
                    );
            }

            // Ensure gripper has attachment point set
            if (_gripperController != null && _gripperController.attachmentPoint == null)
            {
                _gripperController.attachmentPoint = endEffectorBase;
                Debug.Log(
                    $"{_logPrefix} [{robotId}] Set GripperController attachment point to endEffectorBase"
                );
            }

            // Tune joint drives for critical damping
            TuneJointDrivesForCriticalDamping();

            // CRITICAL: Set initial joint drive targets to current positions to prevent slumping
            for (int i = 0; i < robotJoints.Length; i++)
            {
                var joint = robotJoints[i];
                if (joint != null)
                {
                    // Get current joint position and set it as the drive target
                    float currentPosition = joint.jointPosition[0];
                    var drive = joint.xDrive;
                    drive.target = currentPosition * Mathf.Rad2Deg; // Convert radians to degrees
                    joint.xDrive = drive;
                }
            }

            // Initialize target to current position
            _targetPosition = endEffectorBase.position;
            _targetRotation = endEffectorBase.rotation;
        }

        /// <summary>
        /// Tune ArticulationBody drives using RobotConfig profile (if available) or critical damping.
        /// Uses formula: damping = 2 * sqrt(stiffness * inertia)
        /// </summary>
        private void TuneJointDrivesForCriticalDamping()
        {
            if (robotJoints == null || robotJoints.Length == 0)
                return;

            // Try to get robot config from RobotManager
            RobotManager robotManager = RobotManager.Instance;
            RobotInstance robotInstance = null;
            bool hasConfig = false;

            if (robotManager != null && !string.IsNullOrEmpty(robotId))
            {
                hasConfig = robotManager.RobotInstances.TryGetValue(robotId, out robotInstance);
            }

            for (int i = 0; i < robotJoints.Length; i++)
            {
                var joint = robotJoints[i];
                var drive = joint.xDrive;

                // Try to use config from RobotManager (like RobotController does)
                if (
                    hasConfig
                    && robotInstance.profile?.joints != null
                    && i < robotInstance.profile.joints.Length
                )
                {
                    var jointConfig = robotInstance.profile.joints[i];
                    drive.stiffness = jointConfig.stiffness;
                    drive.damping = jointConfig.damping;
                    drive.forceLimit = jointConfig.forceLimit;
                    drive.upperLimit = jointConfig.upperLimit;
                    drive.lowerLimit = jointConfig.lowerLimit;
                }
                else
                {
                    // Fallback: Use critical damping calculation
                    float inertia = joint.inertiaTensor.x;
                    if (inertia < 0.001f)
                        inertia = joint.mass * 0.1f; // Fallback estimate

                    float targetStiffness = 2000f;
                    float criticalDamping = 2f * Mathf.Sqrt(targetStiffness * inertia);

                    drive.stiffness = targetStiffness;
                    drive.damping = criticalDamping;
                }

                joint.matchAnchors = true;
                joint.xDrive = drive;
            }
        }

        #endregion

        #region --- IK Computation ---

        /// <summary>
        /// Update the cached joint info for Jacobian computation.
        /// Transforms joint positions and axes to IK frame coordinates.
        /// </summary>
        private void UpdateJointInfoCache()
        {
            Quaternion ikFrameInverseRot = Quaternion.Inverse(_ikFrame.rotation);

            for (int i = 0; i < robotJoints.Length; i++)
            {
                var joint = robotJoints[i];

                // Joint position in IK frame
                Vector3 jointLocalPos = _ikFrame.InverseTransformPoint(joint.transform.position);

                // Joint axis in world space, then transform to IK frame
                Vector3 axisWorld = joint.transform.rotation * joint.anchorRotation * Vector3.right;
                Vector3 axisLocal = ikFrameInverseRot * axisWorld;

                _cachedJointInfos[i] = new JointInfo(jointLocalPos, axisLocal.normalized);
            }
        }

        /// <summary>
        /// Update end effector and target state in IK frame coordinates.
        /// </summary>
        private void UpdateEndEffectorState()
        {
            if (_ikFrame == null || endEffectorBase == null)
                return;

            // End effector state in IK frame
            Vector3 eeWorldPos = endEffectorBase.position;
            Quaternion eeWorldRot = endEffectorBase.rotation;
            _endEffectorLocalPosition = _ikFrame.InverseTransformPoint(eeWorldPos);

            Quaternion invFrameRot = Quaternion.Inverse(_ikFrame.rotation);
            _endEffectorLocalRotation = invFrameRot * eeWorldRot;

            // Target state in IK frame
            _targetLocalPosition = _ikFrame.InverseTransformPoint(_targetPosition);
            _targetLocalRotation = invFrameRot * _targetRotation;

            // Calculate distance
            Vector3 distVec = _targetLocalPosition - _endEffectorLocalPosition;
            _sqrDistanceToTarget = distVec.sqrMagnitude;
            _distanceToTarget = Mathf.Sqrt(_sqrDistanceToTarget);
        }

        /// <summary>
        /// Perform one step of inverse kinematics computation and apply motor updates.
        /// Uses velocity-level IK with PD control to prevent oscillation.
        /// Public so RobotController can call it when using this as a backup.
        /// </summary>
        public void PerformInverseKinematicsStep()
        {
            if (robotJoints.Length == 0 || endEffectorBase == null)
                return;

            // Update state
            UpdateEndEffectorState();
            UpdateJointInfoCache();

            // Check convergence - use the Inspector threshold value
            float effectiveThreshold = _ikConfig.convergenceThreshold; // Use Inspector value (default 3cm)
            float sqrPosThreshold = effectiveThreshold * effectiveThreshold;
            Vector3 currentVelocity = GetEndEffectorVelocity();

            // Converge when position is close enough AND velocity is low (robot has settled)
            // Using a more relaxed velocity threshold than IKSolver to avoid getting stuck
            bool positionConverged = _sqrDistanceToTarget <= sqrPosThreshold;
            bool velocitySettled = currentVelocity.magnitude < 0.1f; // 10cm/s - relaxed from IKSolver's 5cm/s

            if (positionConverged && velocitySettled)
            {
                if (!_hasReachedTarget)
                {
                    Debug.Log(
                        $"{_logPrefix} [{robotId}] Target reached! Distance: {_distanceToTarget:F4}m, Velocity: {currentVelocity.magnitude:F4}m/s"
                    );
                    SetTargetReached(true);
                }
                return;
            }

            // If position is converged but velocity is high, keep running IK to let robot settle
            // This prevents premature convergence while robot is still moving
            if (positionConverged && !velocitySettled)
            {
                // Don't call IKSolver, just let physics settle the robot
                return;
            }

            // LEASH/CARROT APPROACH: Constrain immediate target to small distance ahead
            // This prevents aggressive corrections that cause oscillation
            Vector3 constrainedTargetPos;
            Quaternion constrainedTargetRot;

            // DISABLE orientation correction completely - it causes shaking
            // Only focus on position to ensure stable convergence
            float orientationWeight = 0f;

            if (_distanceToTarget > _leashDistance)
            {
                // Target is far - use a "carrot" a short distance ahead
                Vector3 directionToTarget = (
                    _targetLocalPosition - _endEffectorLocalPosition
                ).normalized;
                constrainedTargetPos =
                    _endEffectorLocalPosition + directionToTarget * _leashDistance;
                constrainedTargetRot = _endEffectorLocalRotation; // Keep current orientation
            }
            else
            {
                // Close enough - go directly to target position
                constrainedTargetPos = _targetLocalPosition;
                constrainedTargetRot = _endEffectorLocalRotation; // Still keep current orientation
            }

            // Compute IK using IKSolver
            IKState currentState = new IKState(
                _endEffectorLocalPosition,
                _endEffectorLocalRotation
            );
            IKState targetState = new IKState(constrainedTargetPos, constrainedTargetRot);

            // Use velocity-level IK with PD control for natural damping
            // Pass a very small convergence threshold to IKSolver so it never returns null prematurely
            // The controller handles convergence checking, not the IKSolver
            const float ikInternalThreshold = 0.001f; // 1mm - effectively disables IKSolver's convergence
            Vector<double> jointDeltas = _ikSolver.ComputeJointDeltasWithVelocity(
                currentState,
                targetState,
                currentVelocity,
                Vector3.zero, // Target velocity (stationary target)
                _cachedJointInfos,
                ikInternalThreshold,
                Kp: _positionGain,
                Kd: _velocityGain,
                orientationWeight: orientationWeight,
                orientationConvergenceThreshold: _ikConfig.orientationThresholdDegrees * Mathf.Deg2Rad,
                overrideDamping: _ikConfig.dampingFactor
            );

            if (jointDeltas != null)
            {
                ApplyMotorUpdates(jointDeltas);
            }
        }

        /// <summary>
        /// Get end effector velocity in IK frame coordinates.
        /// Computes velocity from ArticulationBody joint velocities using forward kinematics.
        /// </summary>
        private Vector3 GetEndEffectorVelocity()
        {
            if (robotJoints == null || robotJoints.Length == 0 || endEffectorBase == null)
                return Vector3.zero;

            Vector3 endEffectorWorldVelocity = Vector3.zero;

            // Compute velocity contribution from each joint: v = omega x r
            for (int i = 0; i < robotJoints.Length; i++)
            {
                var joint = robotJoints[i];

                // Get joint angular velocity (rad/sec)
                float jointAngularVel = joint.jointVelocity[0];

                // Get joint axis in world space
                Vector3 axisWorld = joint.transform.rotation * joint.anchorRotation * Vector3.right;

                // Get vector from joint to end effector
                Vector3 jointToEE = endEffectorBase.position - joint.transform.position;

                // Velocity contribution: v = omega x r
                Vector3 velContribution = Vector3.Cross(axisWorld * jointAngularVel, jointToEE);
                endEffectorWorldVelocity += velContribution;
            }

            // Transform to IK frame local coordinates
            return Quaternion.Inverse(_ikFrame.rotation) * endEffectorWorldVelocity;
        }

        /// <summary>
        /// Apply computed joint deltas to the robot's ArticulationBody drives.
        /// Uses adaptive speed based on distance to target.
        /// </summary>
        /// <param name="jointDeltas">Joint angle deltas in radians</param>
        private void ApplyMotorUpdates(Vector<double> jointDeltas)
        {
            // Adaptive gain: slow down near target
            float adaptiveGain = _maxStepSpeed;
            if (_distanceToTarget < 0.1f)
            {
                adaptiveGain = Mathf.Lerp(
                    _minStepSpeedNearTarget,
                    _maxStepSpeed,
                    _distanceToTarget / 0.1f
                );
            }

            for (int i = 0; i < robotJoints.Length; i++)
            {
                double delta = jointDeltas[i];

                // Handle NaN/Infinity
                if (double.IsNaN(delta) || double.IsInfinity(delta))
                    delta = 0.0;

                // Clamp joint step
                double clampedDeltaRad = System.Math.Clamp(
                    delta,
                    -_maxJointStepRad,
                    _maxJointStepRad
                );
                double finalDeltaRad = clampedDeltaRad * adaptiveGain;
                float deltaAngleDegree = (float)(finalDeltaRad * Mathf.Rad2Deg);

                // Apply to ArticulationBody drive
                ArticulationDrive drive = robotJoints[i].xDrive;
                float newTarget = Mathf.Clamp(
                    drive.target + deltaAngleDegree,
                    drive.lowerLimit,
                    drive.upperLimit
                );

                if (Mathf.Abs(newTarget - drive.target) > Mathf.Epsilon)
                {
                    drive.target = newTarget;
                    robotJoints[i].xDrive = drive;
                }
            }
        }

        #endregion

        #region --- Public API ---

        /// <summary>
        /// Set a new target position for the robot to move to.
        /// Uses top-down approach orientation (gripper pointing down).
        /// </summary>
        /// <param name="position">Target position in world coordinates</param>
        public void SetTarget(Vector3 position)
        {
            SetTarget(position, GripperApproach.TopDown);
        }

        /// <summary>
        /// Set a new target position with specified approach direction.
        /// </summary>
        /// <param name="position">Target position in world coordinates</param>
        /// <param name="approach">How the gripper should approach the target</param>
        public void SetTarget(Vector3 position, GripperApproach approach)
        {
            // Try to find a real object near this position
            if (ObjectFinder.Instance != null)
            {
                const float objectFindingRadius = 0.15f; // meters
                GameObject nearbyObject = ObjectFinder.Instance.FindClosestObject(
                    position,
                    objectFindingRadius
                );

                if (nearbyObject != null)
                {
                    const float objectDistanceThreshold = 0.1f; // meters
                    float distanceSqr = Vector3.SqrMagnitude(
                        position - nearbyObject.transform.position
                    );
                    if (
                        distanceSqr
                        < objectDistanceThreshold * objectDistanceThreshold
                    )
                    {
                        // Close enough - use the real object instead
                        Debug.Log(
                            $"{_logPrefix} [{robotId}] Found nearby object '{nearbyObject.name}' at distance {Mathf.Sqrt(distanceSqr):F3}m, snapping to it"
                        );
                        SetTarget(nearbyObject);
                        return;
                    }
                }
            }

            // No object found nearby - use the position directly
            _targetPosition = position;
            _targetRotation = CalculateApproachRotation(position, approach);

            _hasTarget = true;
            SetTargetReached(false);
            _ikSolver?.ResetIterationCount();

            Debug.Log(
                $"{_logPrefix} [{robotId}] New target: {position}, approach: {approach}, rot: {_targetRotation.eulerAngles}"
            );
        }

        /// <summary>
        /// Calculate gripper rotation for the specified approach direction.
        /// Properly handles targets at any position around the robot.
        /// </summary>
        private Quaternion CalculateApproachRotation(
            Vector3 targetPosition,
            GripperApproach approach
        )
        {
            Vector3 robotBase = _ikFrame != null ? _ikFrame.position : transform.position;

            // Direction from robot base to target (horizontal plane)
            Vector3 baseToTarget = targetPosition - robotBase;
            baseToTarget.y = 0;

            // If target is directly above/below robot, use forward as fallback
            if (baseToTarget.sqrMagnitude < 0.001f)
                baseToTarget = transform.forward;
            else
                baseToTarget.Normalize();

            Quaternion baseRotation;

            switch (approach)
            {
                case GripperApproach.TopDown:
                    // Gripper pointing straight down (-Y)
                    // Gripper's "forward" faces away from robot (toward target direction)
                    // This ensures fingers can close properly on the object
                    baseRotation = Quaternion.LookRotation(Vector3.down, baseToTarget);
                    break;

                case GripperApproach.Front:
                    // Gripper pointing horizontally toward the target
                    // Approaches from robot's side of the object
                    Vector3 approachDir = -baseToTarget; // Point toward robot (approach from robot's direction)
                    baseRotation = Quaternion.LookRotation(approachDir, Vector3.up);
                    break;

                case GripperApproach.Side:
                    // Gripper approaching from the side (perpendicular to robot-target line)
                    Vector3 sideDir = Vector3.Cross(Vector3.up, baseToTarget);
                    if (sideDir.sqrMagnitude < 0.001f)
                        sideDir = Vector3.right;
                    baseRotation = Quaternion.LookRotation(sideDir.normalized, Vector3.up);
                    break;

                case GripperApproach.Current:
                default:
                    // Keep current gripper orientation
                    return endEffectorBase != null ? endEffectorBase.rotation : Quaternion.identity;
            }

            // Apply user-configurable offset to align gripper axes
            return baseRotation * Quaternion.Euler(_gripperRotationOffset);
        }

        /// <summary>
        /// Set a new target position and rotation for the robot.
        /// </summary>
        /// <param name="position">Target position in world coordinates</param>
        /// <param name="rotation">Target rotation in world coordinates</param>
        public void SetTarget(Vector3 position, Quaternion rotation)
        {
            _targetPosition = position;
            _targetRotation = rotation;
            _hasTarget = true;
            SetTargetReached(false);
            _ikSolver?.ResetIterationCount();

            Debug.Log(
                $"{_logPrefix} [{robotId}] New target: {position}, rot: {rotation.eulerAngles}"
            );
        }

        /// <summary>
        /// Set a new target from a GameObject's transform.
        /// </summary>
        /// <param name="target">Target GameObject</param>
        public void SetTarget(GameObject target)
        {
            if (target == null)
                return;

            // Store the object for attachment after grasping
            _targetObject = target;

            SetTarget(target.transform.position, target.transform.rotation);
        }

        /// <summary>
        /// Set a new target from a Transform.
        /// </summary>
        /// <param name="target">Target Transform</param>
        public void SetTarget(Transform target)
        {
            if (target == null)
                return;

            SetTarget(target.position, target.rotation);
        }

        #endregion

        #region --- Gripper Control ---

        /// <summary>
        /// Update target reached state and fire events.
        /// </summary>
        /// <param name="reached">True if target has been reached</param>
        private void SetTargetReached(bool reached)
        {
            if (_hasReachedTarget != reached)
            {
                _hasReachedTarget = reached;
                if (reached)
                {
                    // Clear the target to stop all movement
                    _hasTarget = false;

                    // Close gripper after delay
                    if (_closeGripperOnReach && _gripperController != null)
                    {
                        StartCoroutine(CloseGripperAfterDelay());
                    }
                    else
                    {
                        OnTargetReached?.Invoke();
                    }
                }
            }
        }

        /// <summary>
        /// Close the gripper after a delay to allow the robot to settle.
        /// </summary>
        private IEnumerator CloseGripperAfterDelay()
        {
            yield return new WaitForSeconds(_gripperCloseDelay);

            // Set the target object so it will be attached automatically when gripper closes
            if (_attachObjectOnGrasp && _targetObject != null)
            {
                _gripperController.SetTargetObject(_targetObject);
                _targetObject = null; // Clear our reference
            }

            _gripperController.CloseGrippers();

            // Wait for gripper to finish closing (attachment happens automatically)
            yield return new WaitWhile(() => _gripperController.IsMoving);
            yield return new WaitForSeconds(0.2f); // Extra settle time

            OnTargetReached?.Invoke();
        }

        /// <summary>
        /// Release the currently held object by opening the gripper.
        /// </summary>
        public void ReleaseObject()
        {
            _gripperController?.ReleaseObject();
        }

        /// <summary>
        /// Check if the robot is currently holding an object.
        /// </summary>
        public bool IsHoldingObject => _gripperController?.IsHoldingObject ?? false;

        /// <summary>
        /// Manually open the gripper. Automatically detaches any held object.
        /// </summary>
        public void OpenGripper()
        {
            _gripperController?.OpenGrippers();
        }

        /// <summary>
        /// Manually close the gripper.
        /// </summary>
        public void CloseGripper()
        {
            _gripperController?.CloseGrippers();
        }

        #endregion

        #region --- Helpers ---

        /// <summary>
        /// Clear the current target and stop movement.
        /// </summary>
        public void ClearTarget()
        {
            _hasTarget = false;
            _hasReachedTarget = true;
        }

        /// <summary>
        /// Get current end effector position in world coordinates.
        /// </summary>
        public Vector3 GetCurrentEndEffectorPosition()
        {
            return endEffectorBase == null ? Vector3.zero : endEffectorBase.position;
        }

        /// <summary>
        /// Get current end effector rotation in world coordinates.
        /// </summary>
        public Quaternion GetCurrentEndEffectorRotation()
        {
            return endEffectorBase == null ? Quaternion.identity : endEffectorBase.rotation;
        }

        /// <summary>
        /// Get the current target position.
        /// </summary>
        public Vector3? GetCurrentTarget()
        {
            return _hasTarget ? _targetPosition : null;
        }

        /// <summary>
        /// Get the current target rotation.
        /// </summary>
        public Quaternion? GetCurrentTargetRotation()
        {
            return _hasTarget ? _targetRotation : null;
        }

        /// <summary>
        /// Reset all joint targets to zero.
        /// </summary>
        public void ResetJointTargets()
        {
            for (int i = 0; i < robotJoints.Length; i++)
            {
                var drive = robotJoints[i].xDrive;
                drive.target = 0;
                robotJoints[i].xDrive = drive;
                robotJoints[i].jointPosition = new ArticulationReducedSpace(0f);
                robotJoints[i].jointVelocity = new ArticulationReducedSpace(0f);
                robotJoints[i].jointForce = new ArticulationReducedSpace(0f);
            }
        }

        /// <summary>
        /// Check if tolerance (convergence threshold) has been reached.
        /// </summary>
        public bool IsToleranceReached()
        {
            return _distanceToTarget < _ikConfig.convergenceThreshold;
        }

        /// <summary>
        /// Draw debug visualization in the scene view.
        /// </summary>
        private void DrawDebugVisualization()
        {
            if (!enableDebugVisualization || endEffectorBase == null)
                return;

            Debug.DrawLine(endEffectorBase.position, _targetPosition, Color.red);
            Debug.DrawRay(endEffectorBase.position, Vector3.right * 0.1f, Color.blue);
        }

        /// <summary>
        /// Draw gizmos for target visualization in editor.
        /// </summary>
        private void OnDrawGizmos()
        {
            if (!enableDebugVisualization || !_hasTarget)
                return;

            Gizmos.color = _hasReachedTarget ? Color.green : Color.yellow;
            Gizmos.DrawWireSphere(_targetPosition, 0.02f);

            if (endEffectorBase != null)
            {
                Gizmos.color = Color.cyan;
                Gizmos.DrawLine(endEffectorBase.position, _targetPosition);
            }
        }

        #endregion
    }
}
