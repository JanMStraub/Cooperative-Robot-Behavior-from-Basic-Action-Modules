using System.Collections;
using Configuration;
using Core;
using Robotics.Grasp;
using Simulation;
using UnityEngine;
using Utilities;

namespace Robotics
{
    /// <summary>
    /// Controls a robotic arm using inverse kinematics (IK) with a damped least squares approach.
    /// Highly optimized for low Garbage Collection (GC) and minimal physics overhead.
    /// </summary>
    public class RobotController : MonoBehaviour
    {
        private SimulationManager _simulationManager;
        private RobotManager _robotManager;

        [Header("Robot Identity")]
        public string robotId = "Robot1";

        // --- Performance Caching ---
        private Transform _ikFrame;
        private JointInfo[] _cachedJointInfos;
        private ArticulationDrive[] _cachedDriveUpdates;
        private bool _cachedIsRobotActive = true;
        private float _lastActiveCheckTime = 0f;
        private const float ACTIVE_CHECK_CACHE_INTERVAL = 0.1f;

        // ---------------------------

        [Header("IK Parameters")]
        [SerializeField]
        private float _dampingFactorLambda = RobotConstants.DEFAULT_DAMPING_FACTOR;
        private float _minStepSpeedNearTarget = RobotConstants.MIN_STEP_SPEED_NEAR_TARGET;
        private float _maxStepSpeed = RobotConstants.MAX_STEP_SPEED;

        private IKSolver _ikSolver;

        // State variables
        private float _sqrDistanceToTarget; // Optimization: Use Squared Distance
        private float _distanceToTarget; // Cached for when actual distance is needed
        private bool _hasReachedTarget = true;
        private bool _isGraspingTarget = false;
        private Transform _targetTransform;

        // IK State cache (reusable structs)
        private Vector3 _endEffectorLocalPosition;
        private Quaternion _endEffectorLocalRotation;
        private Vector3 _targetLocalPosition;
        private Quaternion _targetLocalRotation;

        // Target Objects
        private GameObject _targetObject;
        public GameObject debugTarget;

        // Reusable Temporary Targets
        private GameObject _cachedTempTargetMove;
        private GameObject _cachedTempTargetGrasp;
        private GameObject _cachedTempTargetPre;
        private GameObject _cachedTempTargetRetreat;

        private Coroutine _activeGraspCoroutine;

        [Header("Robot Components")]
        public ArticulationBody[] robotJoints;
        public Transform endEffectorBase;
        public Transform IKReferenceFrame;
        public float[] jointDriveTargets;

        [Header("Velocity Control")]
        [SerializeField]
        private ArticulationBody endEffectorArticulationBody;
        private Vector3 _lastEndEffectorPosition;
        private Vector3 _endEffectorVelocity;

        [Header("Gripper Integration")]
        [SerializeField]
        private GripperController _gripperController;
        public bool _closeGripperAfterReach = false;
        private GraspApproach _currentGraspApproach;

        [SerializeField]
        [Tooltip("Automatically attach the target object to the gripper when grasped")]
        private bool _attachObjectOnGrasp = true;

        [SerializeField]
        [Tooltip("Delay in seconds before closing gripper after reaching target")]
        private float _gripperCloseDelay = 0.2f;

        [Header("Target Tracking")]
        [SerializeField]
        private bool _enableMovingTargetTracking = true;

        [SerializeField]
        private float _targetMovementThreshold = 0.01f;
        private float _sqrTargetMovementThreshold;
        private Vector3 _lastTrackedTargetPosition;
        private bool _isTrackingMovingTarget = false;

        // Moving target smoothing
        private Vector3 _smoothedTargetPosition;
        private Vector3 _targetPositionVelocity;
        private const float TARGET_SMOOTHING_TIME = 0.2f;

        [Header("Advanced Grasp Planning")]
        [SerializeField]
        private GraspConfig _graspConfig;
        private GraspPlanningPipeline _graspPipeline;

        [Header("Debug Visualization")]
        [SerializeField]
        private bool enableDebugVisualization = true;

        private const string _logPrefix = "[ROBOT_CONTROLLER]";

        public event System.Action OnTargetReached;
        public event System.Action<bool> OnCoordinationStateChanged;

        private void Start()
        {
            _simulationManager = SimulationManager.Instance;
            _robotManager = RobotManager.Instance;

            if (string.IsNullOrEmpty(robotId))
                robotId = gameObject.name;

            _ikFrame = IKReferenceFrame != null
                ? IKReferenceFrame
                : (endEffectorBase != null && endEffectorBase.root != null
                    ? endEffectorBase.root
                    : transform);

            if (_gripperController == null)
            {
                _gripperController = GetComponentInChildren<GripperController>();
                if (_gripperController == null)
                {
                    Debug.LogWarning($"{_logPrefix} No GripperController found in children of {robotId}");
                }
            }

            if (_robotManager != null && !_robotManager.IsRobotRegistered(robotId))
                _robotManager.RegisterRobot(robotId, gameObject);

            InitializeRobot();
            InitializeGraspPipeline();

            if (debugTarget != null)
                SetTarget(debugTarget, GraspOptions.Advanced);
        }

        private void InitializeRobot()
        {
            if (robotJoints == null || robotJoints.Length == 0)
            {
                Debug.LogWarning($"{_logPrefix} Robot joints are not assigned. Please assign ArticulationBodies.");
                return;
            }

            int jointCount = robotJoints.Length;

            jointDriveTargets = new float[jointCount];
            _cachedJointInfos = new JointInfo[jointCount];
            _cachedDriveUpdates = new ArticulationDrive[jointCount];

            for (int i = 0; i < jointCount; i++)
            {
                var drive = robotJoints[i].xDrive;

                if (
                    _robotManager != null
                    && _robotManager.RobotInstances.TryGetValue(robotId, out var robotInstance)
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

                robotJoints[i].xDrive = drive;
                jointDriveTargets[i] = drive.target;
            }

            _ikSolver = new IKSolver(jointCount, _dampingFactorLambda);
            _gripperController?.ResetGrippers();
            _sqrTargetMovementThreshold = _targetMovementThreshold * _targetMovementThreshold;

            // Find end effector ArticulationBody for velocity feedback
            if (endEffectorBase != null && endEffectorArticulationBody == null)
            {
                // Try to find ArticulationBody on end effector or parent chain
                endEffectorArticulationBody = endEffectorBase.GetComponentInParent<ArticulationBody>();
                if (endEffectorArticulationBody == null)
                {
                    Debug.LogWarning($"{_logPrefix} No ArticulationBody found for end effector velocity feedback. Using position-only IK.");
                }
            }

            _lastEndEffectorPosition = endEffectorBase != null ? endEffectorBase.position : Vector3.zero;
        }

        private void InitializeGraspPipeline()
        {
            if (
                _graspConfig != null
                && robotJoints != null
                && _ikFrame != null
                && endEffectorBase != null
            )
            {
                _graspPipeline = new GraspPlanningPipeline(
                    _graspConfig,
                    robotJoints,
                    _ikFrame,
                    endEffectorBase,
                    _dampingFactorLambda
                );
            }
        }

        // --- Core Helpers ---

        private GameObject GetCachedTempObject(ref GameObject cacheField, string suffix)
        {
            if (cacheField == null)
                cacheField = new GameObject($"{robotId}{suffix}");
            return cacheField;
        }

        private void StopActiveGraspCoroutine()
        {
            if (_activeGraspCoroutine != null)
            {
                StopCoroutine(_activeGraspCoroutine);
                _activeGraspCoroutine = null;
            }
        }

        public void SetTargetReached(bool setting)
        {
            if (_hasReachedTarget != setting)
            {
                _hasReachedTarget = setting;
                _simulationManager?.NotifyTargetReached(robotId, setting);

                if (setting)
                {
                    // If close gripper after reach is enabled, use coroutine to handle
                    // automatic object attachment before closing
                    if (_closeGripperAfterReach && _gripperController != null)
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
        /// Automatically attaches the target object if enabled.
        /// </summary>
        private IEnumerator CloseGripperAfterDelay()
        {
            yield return new WaitForSeconds(_gripperCloseDelay);

            // Set the target object so it will be attached automatically when gripper closes
            if (_attachObjectOnGrasp && _targetObject != null)
            {
                _gripperController.SetTargetObject(_targetObject);
            }

            _gripperController.CloseGrippers();

            // Wait for gripper to finish closing
            yield return new WaitWhile(() => _gripperController.IsMoving);
            yield return new WaitForSeconds(0.2f); // Extra settle time

            OnTargetReached?.Invoke();
        }

        // --- Optimized Core Loop ---

        private void FixedUpdate()
        {
            if (_simulationManager != null && _simulationManager.ShouldStopRobots)
                return;
            if (_simulationManager != null && !GetCachedIsRobotActive())
                return;

            if (!_hasReachedTarget)
            {
                PerformInverseKinematicsStep();
            }

            DrawDebugVisualization();
        }

        private void UpdateJointInfoCache()
        {
            Quaternion ikFrameInverseRot = Quaternion.Inverse(_ikFrame.rotation);

            for (int i = 0; i < robotJoints.Length; i++)
            {
                var joint = robotJoints[i];
                Vector3 jointLocalPos = _ikFrame.InverseTransformPoint(joint.transform.position);

                // Optimized axis rotation
                Vector3 axisWorld = joint.transform.rotation * joint.anchorRotation * Vector3.right;
                Vector3 axisLocal = ikFrameInverseRot * axisWorld;

                _cachedJointInfos[i] = new JointInfo(jointLocalPos, axisLocal.normalized);
            }
        }

        private void UpdateEndEffectorState()
        {
            // Update end effector velocity (using FixedUpdate time step for stability)
            if (endEffectorBase != null)
            {
                Vector3 currentPosition = endEffectorBase.position;
                float dt = Time.fixedDeltaTime;

                if (dt > 0f)
                {
                    // Calculate velocity in IK frame coordinates
                    Vector3 worldVelocity = (currentPosition - _lastEndEffectorPosition) / dt;
                    _endEffectorVelocity = _ikFrame.InverseTransformDirection(worldVelocity);
                }

                _lastEndEffectorPosition = currentPosition;
            }

            if (_targetTransform == null)
            {
                _sqrDistanceToTarget = 0f;
                _distanceToTarget = 0f;
                return;
            }

            bool hasGrippedTarget = _hasReachedTarget && _closeGripperAfterReach;
            if (
                _enableMovingTargetTracking
                && _isTrackingMovingTarget
                && _targetObject != null
                && !hasGrippedTarget
            )
            {
                Vector3 currentTargetPos = _targetObject.transform.position;

                _smoothedTargetPosition = Vector3.SmoothDamp(
                    _smoothedTargetPosition,
                    currentTargetPos,
                    ref _targetPositionVelocity,
                    TARGET_SMOOTHING_TIME
                );

                if (
                    Vector3.SqrMagnitude(_smoothedTargetPosition - _lastTrackedTargetPosition)
                    > 0.0004f
                ) // 0.02^2
                {
                    if (_targetTransform != _targetObject.transform)
                    {
                        _targetTransform.position = _smoothedTargetPosition;
                        _targetTransform.rotation = _targetObject.transform.rotation;
                    }
                    _lastTrackedTargetPosition = _smoothedTargetPosition;
                    if (_hasReachedTarget)
                        SetTargetReached(false);
                }
            }

            // Calculation optimizations
            Vector3 eeWorldPos = endEffectorBase.position;
            Quaternion eeWorldRot = endEffectorBase.rotation;
            Vector3 targetWorldPos = _targetTransform.position;
            Quaternion targetWorldRot = _targetTransform.rotation;

            _endEffectorLocalPosition = _ikFrame.InverseTransformPoint(eeWorldPos);
            _targetLocalPosition = _ikFrame.InverseTransformPoint(targetWorldPos);

            Quaternion invFrameRot = Quaternion.Inverse(_ikFrame.rotation);
            _endEffectorLocalRotation = invFrameRot * eeWorldRot;
            _targetLocalRotation = invFrameRot * targetWorldRot;

            Vector3 distVec = _targetLocalPosition - _endEffectorLocalPosition;
            _sqrDistanceToTarget = distVec.sqrMagnitude;
            _distanceToTarget = Mathf.Sqrt(_sqrDistanceToTarget); // Only calc sqrt once per frame
        }

        public void PerformInverseKinematicsStep()
        {
            if (robotJoints == null || robotJoints.Length == 0 || endEffectorBase == null || _targetTransform == null)
                return;

            UpdateEndEffectorState();

            // Thresholds
            float posThreshold = _isGraspingTarget
                ? RobotConstants.DEFAULT_CONVERGENCE_THRESHOLD * RobotConstants.GRASP_CONVERGENCE_MULTIPLIER
                : RobotConstants.DEFAULT_CONVERGENCE_THRESHOLD;

            // Optimization: Compare Squares to avoid SQRT in logic
            float sqrPosThreshold = posThreshold * posThreshold;

            float angleError = Quaternion.Angle(_endEffectorLocalRotation, _targetLocalRotation);
            bool isOrientationCorrect = angleError < RobotConstants.DEFAULT_ORIENTATION_THRESHOLD_DEGREES;

            // Stop Condition
            if (_sqrDistanceToTarget <= sqrPosThreshold && isOrientationCorrect)
            {
                if (!_hasReachedTarget)
                {
                    SetTargetReached(true);
                    OnTargetReached?.Invoke();
                }
                return;
            }
            else if (
                _hasReachedTarget && (_distanceToTarget > posThreshold * 1.5f || angleError > 10f)
            )
            {
                SetTargetReached(false);
            }

            UpdateJointInfoCache();

            // --- "Carrot" Logic (Linear Motion) ---
            Vector3 vectorToTarget = _targetLocalPosition - _endEffectorLocalPosition;

            // Blend between carrot and direct based on distance (smooth transition 5-15cm)
            float blendFactor = Mathf.Clamp01((_distanceToTarget - 0.05f) / 0.10f);
            Vector3 carrotPos = _endEffectorLocalPosition + (vectorToTarget / _distanceToTarget) * 0.05f;
            Vector3 constrainedTargetPos = Vector3.Lerp(_targetLocalPosition, carrotPos, blendFactor);

            // --- KEY FIX: ALWAYS target the final rotation, but vary the weight ---
            // This allows orientation ramping to work correctly
            Quaternion constrainedTargetRot = _targetLocalRotation;

            // --- Orientation Ramping ---
            float baseWeight = GetOrientationWeight(_currentGraspApproach);
            float orientationRamp = 0f;

            // Start ramping at 30cm instead of 20cm with gentler curve
            if (_distanceToTarget < 0.30f)
            {
                // Between 30cm and 10cm - gradually ramp up orientation
                // At 30cm: ramp = 0 (ignore rotation)
                // At 10cm: ramp = 1 (full rotation control)
                float t = Mathf.Clamp01((_distanceToTarget - 0.10f) / 0.20f);
                orientationRamp = 1.0f - (t * t); // Smooth curve from 0.3m to 0.1m
            }
            // else: > 30cm, orientationRamp stays 0 (ignore rotation completely)

            float finalWeight = baseWeight * orientationRamp;

            // --- Dynamic Damping (Precision) ---
            float dynamicLambda = _dampingFactorLambda;
            if (_distanceToTarget < 0.1f)
            {
                // Increase damping when closer for stability (not decrease)
                dynamicLambda = Mathf.Lerp(
                    _dampingFactorLambda,
                    _dampingFactorLambda * 1.5f,
                    1f - (_distanceToTarget / 0.1f)
                );
            }

            // Input States
            IKState currentState = new IKState(
                _endEffectorLocalPosition,
                _endEffectorLocalRotation
            );
            IKState targetState = new IKState(constrainedTargetPos, constrainedTargetRot);

            // Target velocity (zero for now - future enhancement: use TrajectoryController)
            Vector3 targetVelocity = Vector3.zero;

            // PD gains for velocity-level IK (tuned to eliminate oscillation)
            // Kp: Position gain - how strongly to correct position error
            // Kd: Velocity gain (damping) - prevents overshoot and oscillation
            float Kp = 1.0f;  // Standard position correction
            float Kd = 0.5f;  // Damping to prevent oscillation

            // Use velocity-aware IK to eliminate oscillation
            var jointDeltas = _ikSolver.ComputeJointDeltasWithVelocity(
                currentState,
                targetState,
                _endEffectorVelocity,      // Current velocity for damping
                targetVelocity,            // Target velocity (zero = stop at target)
                _cachedJointInfos,
                posThreshold,
                Kp,                        // Position gain
                Kd,                        // Velocity gain (damping)
                finalWeight,               // Orientation weight
                RobotConstants.DEFAULT_ORIENTATION_THRESHOLD_DEGREES * Mathf.Deg2Rad,
                dynamicLambda              // Dynamic damping for pseudo-inverse
            );

            if (jointDeltas == null)
            {
                SetTargetReached(true);
                OnTargetReached?.Invoke();
                return;
            }

            // Speed Calculation
            RobotConfig robotProfile =
                _robotManager?.GetRobotProfile(robotId) ?? _robotManager?.RobotProfile;
            float maxStep =
                robotProfile != null
                    ? robotProfile.maxJointStepRad
                    : RobotConstants.DEFAULT_MAX_JOINT_STEP_RAD;
            float adjSpeed = robotProfile != null ? robotProfile.adjustmentSpeed : 1.0f;
            float globalSpeed = _robotManager != null ? _robotManager.globalSpeedMultiplier : 1.0f;

            float adaptiveGain = _maxStepSpeed;
            if (_distanceToTarget < 0.1f)
            {
                adaptiveGain = Mathf.Lerp(
                    _minStepSpeedNearTarget,
                    _maxStepSpeed,
                    _distanceToTarget / 0.1f
                );
            }

            float stepScale = adjSpeed * globalSpeed * adaptiveGain * Mathf.Rad2Deg;

            // Apply Updates
            bool hasUpdates = false;

            for (int i = 0; i < robotJoints.Length; i++)
            {
                double rawDelta = System.Math.Clamp(jointDeltas[i], -maxStep, maxStep);
                ArticulationDrive drive = robotJoints[i].xDrive;
                float deltaAngleDegree = (float)rawDelta * stepScale;

                float newTarget = Mathf.Clamp(
                    drive.target + deltaAngleDegree,
                    drive.lowerLimit,
                    drive.upperLimit
                );

                // Epsilon check to minimize PhysX overhead
                if (Mathf.Abs(newTarget - drive.target) > Mathf.Epsilon)
                {
                    drive.target = newTarget;
                    _cachedDriveUpdates[i] = drive;
                    hasUpdates = true;
                }
                else
                {
                    _cachedDriveUpdates[i] = drive;
                }
            }

            if (hasUpdates)
            {
                for (int i = 0; i < robotJoints.Length; i++)
                {
                    if (
                        Mathf.Abs(robotJoints[i].xDrive.target - _cachedDriveUpdates[i].target)
                        > Mathf.Epsilon
                    )
                    {
                        robotJoints[i].xDrive = _cachedDriveUpdates[i];
                        if (i < jointDriveTargets.Length)
                            jointDriveTargets[i] = _cachedDriveUpdates[i].target;
                    }
                }
            }

            // Velocity Damping - REMOVED: Let IK solver control motion
            // The physics velocity damping was conflicting with IK commands
            // causing poor convergence. IK solver now handles all motion control.
        }

        // --- Target Setting (Refactored) ---

        private void SetTargetInternal(
            Transform targetTransform,
            GameObject originalObject,
            GraspOptions options
        )
        {
            _targetObject = originalObject;
            _targetTransform = targetTransform;
            _closeGripperAfterReach = options.closeGripperOnReach;
            SetTargetReached(false);

            _isTrackingMovingTarget = (originalObject != null);
            if (_isTrackingMovingTarget)
            {
                _lastTrackedTargetPosition = originalObject.transform.position;
            }

            _ikSolver?.ResetIterationCount();
        }

        public void SetTarget(GameObject target, GraspOptions options = default)
        {
            if (target == null)
                return;
            StopActiveGraspCoroutine();

            if (options.Equals(default(GraspOptions)))
                options = GraspOptions.Default;

            // Check if object is held by another gripper (handoff scenario)
            GripperController holdingGripper = FindGripperHoldingObject(target);
            if (holdingGripper != null && holdingGripper != _gripperController)
            {
                Debug.Log($"{_logPrefix} {robotId} detected handoff scenario - object '{target.name}' held by another gripper");
                _activeGraspCoroutine = StartCoroutine(ExecuteHandoffGrasp(target, holdingGripper, options));
                return;
            }

            // Check Advanced/Simple Planning
            if (_graspPipeline != null)
            {
                GraspCandidate? candidate = null;

                if (options.useAdvancedPlanning)
                {
                    if (options.graspConfig == null)
                        options.graspConfig = _graspConfig;
                    candidate = _graspPipeline.PlanGrasp(target, endEffectorBase.position, options);
                    if (candidate.HasValue)
                    {
                        _activeGraspCoroutine = StartCoroutine(
                            ExecuteThreeWaypointGrasp(candidate.Value, target, options)
                        );
                        return;
                    }
                }
                else if (options.useGraspPlanning)
                {
                    // Create minimal options for simple planner
                    var simpleOpts = new GraspOptions
                    {
                        useGraspPlanning = false,
                        approach = options.approach,
                        graspConfig = options.graspConfig,
                    };
                    candidate = _graspPipeline.PlanGrasp(
                        target,
                        endEffectorBase.position,
                        simpleOpts
                    );
                    if (candidate.HasValue)
                    {
                        _activeGraspCoroutine = StartCoroutine(
                            ExecuteTwoWaypointGrasp(candidate.Value, target, options)
                        );
                        return;
                    }
                }
            }

            // Fallback: Direct Move
            if (options.openGripperOnSet)
                _gripperController?.OpenGrippers();
            _isGraspingTarget = false;
            SetTargetInternal(target.transform, target, options);
        }

        public void SetTarget(Vector3 position, GraspOptions options = default)
        {
            StopActiveGraspCoroutine();
            if (options.Equals(default(GraspOptions)))
                options = GraspOptions.MoveOnly;

            // Object Finder Logic
            if (ObjectFinder.Instance != null)
            {
                GameObject realObject = ObjectFinder.Instance.FindClosestObject(
                    position,
                    RobotConstants.OBJECT_FINDING_RADIUS
                );
                if (realObject != null)
                {
                    if (
                        Vector3.SqrMagnitude(position - realObject.transform.position)
                        < RobotConstants.OBJECT_DISTANCE_THRESHOLD
                            * RobotConstants.OBJECT_DISTANCE_THRESHOLD
                    )
                    {
                        SetTarget(realObject, options);
                        return;
                    }
                }
            }

            GameObject temp = GetCachedTempObject(
                ref _cachedTempTargetMove,
                RobotConstants.TEMP_TARGET_SUFFIX
            );
            temp.transform.position = position;

            options.useGraspPlanning = false;
            if (options.openGripperOnSet)
                _gripperController?.OpenGrippers();

            _isGraspingTarget = false;
            SetTargetInternal(temp.transform, null, options);
        }

        public void SetTarget(Vector3 position, Quaternion rotation, GraspOptions options = default)
        {
            StopActiveGraspCoroutine();
            if (options.Equals(default(GraspOptions)))
                options = GraspOptions.MoveOnly;

            GameObject temp = GetCachedTempObject(
                ref _cachedTempTargetMove,
                RobotConstants.TEMP_TARGET_SUFFIX
            );
            temp.transform.SetPositionAndRotation(position, rotation);

            options.useGraspPlanning = false;
            if (options.openGripperOnSet)
                _gripperController?.OpenGrippers();

            _isGraspingTarget = false;
            SetTargetInternal(temp.transform, null, options);
        }

        // --- Grasp Coroutines ---

        /// <summary>
        /// Helper coroutine that waits for target to be reached with timeout protection.
        /// Prevents infinite waiting when target is unreachable.
        /// </summary>
        /// <param name="timeoutSeconds">Maximum time to wait before giving up</param>
        /// <param name="onTimeout">Optional callback to invoke on timeout</param>
        private IEnumerator WaitForTargetWithTimeout(float timeoutSeconds, System.Action onTimeout = null)
        {
            float startTime = Time.time;
            while (!_hasReachedTarget)
            {
                if (Time.time - startTime > timeoutSeconds)
                {
                    Debug.LogWarning($"{_logPrefix} {robotId} reached target timeout after {timeoutSeconds}s at distance {_distanceToTarget:F3}m");
                    onTimeout?.Invoke();
                    yield break;
                }
                yield return null;
            }
        }

        private IEnumerator ExecuteTwoWaypointGrasp(
            GraspCandidate candidate,
            GameObject targetObject,
            GraspOptions options
        )
        {
            _currentGraspApproach = candidate.approachType;
            if (_gripperController != null)
                _gripperController.SetGripperPosition(candidate.preGraspGripperWidth);

            // 1. Pre-Grasp
            GameObject pre = GetCachedTempObject(ref _cachedTempTargetPre, "_pre");
            pre.transform.SetPositionAndRotation(
                candidate.preGraspPosition,
                candidate.preGraspRotation
            );

            _isGraspingTarget = false;
            SetTargetInternal(
                pre.transform,
                targetObject,
                new GraspOptions { closeGripperOnReach = false }
            );
            yield return StartCoroutine(WaitForTargetWithTimeout(RobotConstants.DEFAULT_GRASP_TIMEOUT_SECONDS));
            if (!_hasReachedTarget) yield break; // Abort if timed out
            yield return new WaitForSeconds(0.2f);

            // 2. Grasp
            GameObject main = GetCachedTempObject(
                ref _cachedTempTargetGrasp,
                RobotConstants.GRASP_TARGET_SUFFIX
            );
            main.transform.SetPositionAndRotation(candidate.graspPosition, candidate.graspRotation);

            _isGraspingTarget = true;
            SetTargetInternal(
                main.transform,
                targetObject,
                new GraspOptions { closeGripperOnReach = false }
            );
            yield return StartCoroutine(WaitForTargetWithTimeout(RobotConstants.DEFAULT_GRASP_TIMEOUT_SECONDS));
            if (!_hasReachedTarget) yield break; // Abort if timed out

            if (options.closeGripperOnReach && _gripperController != null)
            {
                yield return new WaitForSeconds(0.3f);
                // Set target object for attachment before closing gripper
                _gripperController.SetTargetObject(targetObject);
                _gripperController.SetGripperPosition(candidate.graspGripperWidth);
            }

            _activeGraspCoroutine = null;
            OnTargetReached?.Invoke();
        }

        private IEnumerator ExecuteThreeWaypointGrasp(
            GraspCandidate candidate,
            GameObject targetObject,
            GraspOptions options
        )
        {
            _currentGraspApproach = candidate.approachType;
            if (_gripperController != null)
                _gripperController.SetGripperPosition(candidate.preGraspGripperWidth);

            // 1. Pre
            GameObject pre = GetCachedTempObject(ref _cachedTempTargetPre, "_pre");
            pre.transform.SetPositionAndRotation(
                candidate.preGraspPosition,
                candidate.preGraspRotation
            );

            _isGraspingTarget = false;
            SetTargetInternal(
                pre.transform,
                targetObject,
                new GraspOptions { closeGripperOnReach = false }
            );
            yield return StartCoroutine(WaitForTargetWithTimeout(RobotConstants.DEFAULT_GRASP_TIMEOUT_SECONDS));
            if (!_hasReachedTarget) yield break; // Abort if timed out
            yield return new WaitForSeconds(0.1f);

            // 2. Grasp
            GameObject main = GetCachedTempObject(
                ref _cachedTempTargetGrasp,
                RobotConstants.GRASP_TARGET_SUFFIX
            );
            main.transform.SetPositionAndRotation(candidate.graspPosition, candidate.graspRotation);

            _isGraspingTarget = true;
            SetTargetInternal(
                main.transform,
                targetObject,
                new GraspOptions { closeGripperOnReach = false }
            );
            yield return StartCoroutine(WaitForTargetWithTimeout(RobotConstants.DEFAULT_GRASP_TIMEOUT_SECONDS));
            if (!_hasReachedTarget) yield break; // Abort if timed out

            if (options.closeGripperOnReach && _gripperController != null)
            {
                yield return new WaitForSeconds(0.1f);
                // Set target object for attachment before closing gripper
                _gripperController.SetTargetObject(targetObject);
                _gripperController.SetGripperPosition(candidate.graspGripperWidth);
                yield return new WaitWhile(() => _gripperController.IsMoving);
                yield return new WaitForSeconds(1.1f);
            }

            // 3. Retreat
            if (options.graspConfig != null && options.graspConfig.enableRetreat)
            {
                GameObject retreat = GetCachedTempObject(ref _cachedTempTargetRetreat, "_retreat");
                retreat.transform.SetPositionAndRotation(
                    candidate.retreatPosition,
                    candidate.retreatRotation
                );

                _isGraspingTarget = false;
                SetTargetInternal(
                    retreat.transform,
                    targetObject,
                    new GraspOptions { closeGripperOnReach = false }
                );
                yield return StartCoroutine(WaitForTargetWithTimeout(RobotConstants.DEFAULT_GRASP_TIMEOUT_SECONDS));
                if (!_hasReachedTarget) yield break; // Abort if timed out
            }

            _activeGraspCoroutine = null;
            OnTargetReached?.Invoke();
        }

        /// <summary>
        /// Execute a handoff grasp - used when the target object is already held by another gripper.
        /// Moves directly to the object's current position and closes the gripper to receive it.
        /// </summary>
        private IEnumerator ExecuteHandoffGrasp(
            GameObject targetObject,
            GripperController sourceGripper,
            GraspOptions options
        )
        {
            Debug.Log($"{_logPrefix} {robotId} executing handoff grasp for '{targetObject.name}'");

            // Open gripper first if needed
            if (options.openGripperOnSet && _gripperController != null)
            {
                _gripperController.OpenGrippers();
                yield return new WaitWhile(() => _gripperController.IsMoving);
            }

            // Move directly to the object's current world position (where the other gripper is holding it)
            // Use a slight offset to approach from the side
            Vector3 objectPosition = targetObject.transform.position;

            // Create a temporary target at the object's position
            GameObject handoffTarget = GetCachedTempObject(ref _cachedTempTargetGrasp, "_handoff");
            handoffTarget.transform.position = objectPosition;
            handoffTarget.transform.rotation = endEffectorBase.rotation; // Keep current orientation

            _isGraspingTarget = true;
            SetTargetInternal(
                handoffTarget.transform,
                targetObject,
                new GraspOptions { closeGripperOnReach = false }
            );

            // Wait for robot to reach the handoff position
            yield return StartCoroutine(WaitForTargetWithTimeout(RobotConstants.DEFAULT_GRASP_TIMEOUT_SECONDS));
            if (!_hasReachedTarget)
            {
                Debug.LogWarning($"{_logPrefix} {robotId} failed to reach handoff position");
                _activeGraspCoroutine = null;
                yield break;
            }

            // Small delay to settle
            yield return new WaitForSeconds(0.2f);

            // Close gripper and attach the object (this will trigger handoff transfer in GripperController)
            if (options.closeGripperOnReach && _gripperController != null)
            {
                _gripperController.SetTargetObject(targetObject);
                _gripperController.CloseGrippers();
                yield return new WaitWhile(() => _gripperController.IsMoving);

                // Wait for grip to stabilize
                yield return new WaitForSeconds(0.5f);
            }

            Debug.Log($"{_logPrefix} {robotId} handoff grasp complete for '{targetObject.name}'");
            _activeGraspCoroutine = null;
            OnTargetReached?.Invoke();
        }

        /// <summary>
        /// Find the GripperController that is currently holding the specified object.
        /// </summary>
        /// <param name="obj">Object to check</param>
        /// <returns>GripperController holding the object, or null if not held by any gripper</returns>
        private static GripperController FindGripperHoldingObject(GameObject obj)
        {
            GripperController[] allGrippers = FindObjectsByType<GripperController>(FindObjectsSortMode.None);
            foreach (var gripper in allGrippers)
            {
                if (gripper.IsHoldingObject && gripper.GraspedObject == obj)
                {
                    return gripper;
                }
            }
            return null;
        }

        // --- Utilities ---

        public float GetDistanceToTarget()
        {
            if (_targetTransform == null)
                return 0f;
            // Return the cached value if this frame already updated, otherwise quick calc
            return _distanceToTarget;
        }

        public Vector3? GetCurrentTarget() => _targetTransform?.position;

        public Quaternion? GetCurrentTargetRotation() => _targetTransform?.rotation;

        public bool HasTarget => _targetTransform != null;

        public GameObject GetTargetObject() => _targetObject;

        public void SetMovingTargetTracking(bool enable) => _enableMovingTargetTracking = enable;

        public void SetTargetMovementThreshold(float threshold)
        {
            _targetMovementThreshold = Mathf.Max(0.001f, threshold);
            _sqrTargetMovementThreshold = _targetMovementThreshold * _targetMovementThreshold;
        }

        public bool IsTargetTrackingEnabled() => _enableMovingTargetTracking;

        public Vector3 GetCurrentEndEffectorPosition() =>
            endEffectorBase == null ? Vector3.zero : endEffectorBase.position;

        public void ResetJointTargets()
        {
            for (int i = 0; i < robotJoints.Length; i++)
            {
                var drive = robotJoints[i].xDrive;
                drive.target = 0;
                robotJoints[i].xDrive = drive;
                robotJoints[i].jointPosition = new ArticulationReducedSpace(0f);
                robotJoints[i].jointForce = new ArticulationReducedSpace(0f);
                robotJoints[i].jointVelocity = new ArticulationReducedSpace(0f);
                if (i < jointDriveTargets.Length)
                    jointDriveTargets[i] = 0;
            }
        }

        private bool GetCachedIsRobotActive()
        {
            float t = Time.time;
            if (t - _lastActiveCheckTime >= ACTIVE_CHECK_CACHE_INTERVAL)
            {
                bool newState = _simulationManager.IsRobotActive(robotId);
                if (newState != _cachedIsRobotActive)
                    OnCoordinationStateChanged?.Invoke(newState);
                _cachedIsRobotActive = newState;
                _lastActiveCheckTime = t;
            }
            return _cachedIsRobotActive;
        }

        private float GetOrientationWeight(GraspApproach approach)
        {
            switch (approach)
            {
                case GraspApproach.Top:
                    return 1.5f;
                case GraspApproach.Side:
                    return 0.7f;
                default:
                    return 1.0f;
            }
        }

        private void DrawDebugVisualization()
        {
            if (enableDebugVisualization && _targetObject != null && endEffectorBase != null)
                Debug.DrawLine(
                    endEffectorBase.position,
                    _targetObject.transform.position,
                    Color.blue
                );
        }

        private void OnDestroy()
        {
            if (_cachedTempTargetMove)
                Destroy(_cachedTempTargetMove);
            if (_cachedTempTargetGrasp)
                Destroy(_cachedTempTargetGrasp);
            if (_cachedTempTargetPre)
                Destroy(_cachedTempTargetPre);
            if (_cachedTempTargetRetreat)
                Destroy(_cachedTempTargetRetreat);
        }
    }
}
